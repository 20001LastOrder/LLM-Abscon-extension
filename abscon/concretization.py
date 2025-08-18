import time
from collections import Counter, defaultdict

import networkx as nx
import numpy as np
import pulp
from loguru import logger
from networkx import DiGraph

from abscon.base import Concretizer
from abscon.utils import maximum_spanning_branch


def most_frequent(items: list[str]):
    # first identify whether the label should be empty
    empty_labels = []
    non_empty_labels = []
    for item in items:
        if item.strip() == "":
            empty_labels.append(item)
        else:
            non_empty_labels.append(item)
    if len(empty_labels) > len(non_empty_labels):
        return empty_labels[0]

    occurence_count = Counter(non_empty_labels)
    return occurence_count.most_common(1)[0][0]


class MajorityVotingConcretizer:
    def concretize(self, partial_model: DiGraph, **kwargs) -> DiGraph:
        partial_model.remove_nodes_from(list(nx.isolates(partial_model)))
        merged_graph = DiGraph()

        for node, data in partial_model.nodes(data=True):
            merged_graph.add_node(node, label=data["label"])

        for i, j, data in partial_model.edges(data=True):
            if data["weight"] <= 0.5:
                continue

            if "labels" in data:
                label = most_frequent(data["labels"])
                merged_graph.add_edge(i, j, label=label)
            else:
                merged_graph.add_edge(i, j)
        return merged_graph


class TaxonomyConcretizer(Concretizer):
    MAXIMUM_PATH_TO_SAMPLE: int = 1000

    def concretize(self, partial_model: DiGraph, one_parent_constraint=True) -> DiGraph:
        if len(partial_model.edges) == 0:
            logger.warning("Empty partial model!")
            return partial_model
        # remove isolated nodes and self loops
        partial_model.remove_nodes_from(list(nx.isolates(partial_model)))
        partial_model.remove_edges_from(list(nx.selfloop_edges(partial_model)))

        spanning_tree = maximum_spanning_branch(partial_model, attr="weight")
        source = next(nx.topological_sort(spanning_tree))

        edge_result, _ = self.concretize_taxonomy(
            partial_model, source, one_parent_constraint=one_parent_constraint
        )

        # construct the new graph
        node_result = set()

        for i, j in edge_result:
            if edge_result[(i, j)]:
                node_result.add(i)
                node_result.add(j)

        new_graph = nx.DiGraph()
        for node, data in partial_model.nodes(data=True):
            if node in node_result:
                new_graph.add_node(node, label=data["label"])

        for i, j, data in partial_model.edges(data=True):
            if edge_result[(i, j)]:
                new_graph.add_edge(i, j)

        return new_graph

    def concretize_taxonomy(
        self, partial_model: DiGraph, source_node: int, one_parent_constraint=True
    ) -> dict[tuple[int, int], int]:
        """
        Concretize the edges using constraint optimization

        Optimizing the cross entropy of an edge existence and make sure the solution is
        a valid activity diagram
        """
        start = time.time()
        edge_variables = pulp.LpVariable.dicts(
            "edges",
            list(partial_model.edges),
            lowBound=0,
            upBound=1,
            cat=pulp.LpInteger,
        )

        problem = pulp.LpProblem("Maximum Taxonomy", pulp.LpMaximize)
        # One should be able to reach any node from the source node (formatted as a
        # *phantom flow problem*)
        reachability_variables = pulp.LpVariable.dicts(
            "reachability", list(partial_model.edges), lowBound=0, cat=pulp.LpInteger
        )

        n = len(partial_model.nodes)
        for i, j in partial_model.edges:
            problem += (
                reachability_variables[(i, j)] <= (n - 1) * edge_variables[(i, j)]
            )

        for i in partial_model.nodes:
            if i == source_node:
                continue

            incoming_sum = pulp.lpSum(
                [reachability_variables[u, i] for u, i in partial_model.in_edges(i)]
            )
            outgoing_sum = pulp.lpSum(
                [reachability_variables[i, v] for i, v in partial_model.out_edges(i)]
            )

            # Each node consumes one unit
            problem += incoming_sum == outgoing_sum + 1

        # There should be no cycles in the taxonomy
        for nodes in nx.simple_cycles(partial_model):
            edges = [edge_variables[(nodes[-1], nodes[0])]]
            for i in range(len(nodes) - 1):
                edges.append(edge_variables[(nodes[i], nodes[i + 1])])

            problem += pulp.lpSum(edges) <= len(edges) - 1

        # each node (except root should only have one parent)
        if one_parent_constraint:
            for node in partial_model.nodes:
                if node == source_node:
                    continue

                edges = []
                for source, _ in partial_model.in_edges(node):
                    edges.append(edge_variables[(source, node)])
                if len(edges) == 0:
                    continue

                problem += pulp.lpSum(edges) <= 1

        # There is no skip edges in the graph
        for source in partial_model.nodes:
            for target in partial_model.nodes:
                if source == target or (source, target) not in partial_model.edges:
                    continue

                paths = []

                for path in nx.all_simple_paths(partial_model, source, target):
                    paths.append(path)
                    if len(paths) > self.MAXIMUM_PATH_TO_SAMPLE:
                        break
                for path in paths:
                    if len(path) < 3:
                        continue

                    edges = [edge_variables[(source, target)]]
                    for i in range(len(path) - 1):
                        edges.append(edge_variables[(path[i], path[i + 1])])
                    problem += pulp.lpSum(edges) <= len(edges) - 1

        # Maximize the cross entropy of edge variables
        edge_weights = {}
        for i, j, data in partial_model.edges(data=True):
            edge_weights[(i, j)] = min(data["weight"], 1)
            if edge_weights[(i, j)] == 0:
                edge_weights[(i, j)] += 1e-5
            elif edge_weights[(i, j)] == 1:
                edge_weights[(i, j)] -= 1e-5

        problem += pulp.lpSum(
            [
                edge_variables[(i, j)] * np.log(edge_weights[(i, j)])
                + (1 - edge_variables[(i, j)]) * np.log(1 - edge_weights[(i, j)])
                for (i, j) in edge_weights
            ]
        )
        self.problem_definition_runtime = time.time() - start

        start = time.time()
        status = problem.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=120))
        self.problem_solve_runtime = time.time() - start

        logger.debug(problem)
        logger.debug(f"Solving satus {pulp.LpStatus[status]}")
        logger.debug(f"optimum objective value {problem.objective.value()}")
        result_map = {}
        for i, j in partial_model.edges:
            if edge_variables[i, j].value() == 1.0:
                result_map[(i, j)] = True
            else:
                result_map[(i, j)] = False

        return result_map, problem.objective.value()


class ClevrConcretizer(Concretizer):
    MAXIMUM_CYCLES_TO_SAMPLE = 1000
    COLORS = ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"]
    SIZES = ["large", "small"]
    SHAPES = ["cube", "sphere", "cylinder"]
    MATERIALS = ["rubber", "metal"]
    ATTRIBUTES = ["color", "size", "shape", "material"]

    def __init__(self, seed: int = 42):
        self.seed = seed

    def label_to_num_previous(self, label: str) -> int:
        label = label.lower().strip()
        if "scene" in label:
            return 0
        if label in ["and", "or"]:
            return 2
        if "equal" in label or "less" in label or "greater" in label:
            return 2

        return 1

    def is_termination_node(self, label: str) -> bool:
        label = label.lower().strip()
        return (
            "count" in label
            or "exist" in label
            or "query" in label
            or "equal" in label
            or "greater_than" in label
            or "less_than" in label
        )

    def get_input_type(self, label: str) -> str:
        label = label.lower().strip()
        if "scene" in label:
            return ""
        elif (
            "relate" in label
            or "query" in label
            or "same" in label
            or ("equal" in label and "integer" not in label)
        ):
            return "object"
        elif (
            "unique" in label
            or "count" in label
            or "exist" in label
            or "filter" in label
            or label in ["and", "or"]
        ):
            return "object_set"
        else:
            return "integer"

    def get_output_type(self, label: str) -> str:
        label = label.lower().strip()
        if (
            "scene" in label
            or "relate" in label
            or "filter" in label
            or "same" in label
            or label in ["and", "or"]
        ):
            return "object_set"
        elif "unique" in label:
            return "object"
        elif "count" in label:
            return "integer"
        elif "query" in label:
            return "string"
        else:
            return "boolean"

    def is_label_valid(self, label: str) -> bool:
        label = label.strip().lower()

        if label in ["equal_integer", "less_than", "greater_than"]:
            return True

        labels = label.split("_")
        if labels[0] in [
            "scene",
            "count",
            "unique",
            "exist",
            "and",
            "or",
        ]:
            return True
        elif labels[0] == "filter" and len(labels) >= 3:
            if labels[1] == "color" and labels[2] in self.COLORS:
                return True
            if labels[1] == "size" and labels[2] in self.SIZES:
                return True
            if labels[1] == "shape" and labels[2] in self.SHAPES:
                return True
            if labels[1] == "material" and labels[2] in self.MATERIALS:
                return True
        elif labels[0] == "query" and len(labels) >= 2 and labels[1] in self.ATTRIBUTES:
            return True
        elif labels[0] == "same" and len(labels) >= 2 and labels[1] in self.ATTRIBUTES:
            return True
        elif labels[0] == "equal" and len(labels) >= 2 and labels[1] in self.ATTRIBUTES:
            return True
        elif (
            labels[0] == "relate"
            and len(labels) >= 2
            and (
                (labels[1] in ["left", "right", "behind"])
                or (len(labels) >= 3 and labels[1] == "in" and labels[2] == "front")
            )
        ):
            return True
        else:
            return False

    def concretize(self, partial_model: DiGraph) -> DiGraph:
        if len(partial_model.edges) == 0:
            logger.warning("Empty partial model!")
            return partial_model
        # remove isolated nodes and self loops
        partial_model.remove_nodes_from(list(nx.isolates(partial_model)))
        partial_model.remove_edges_from(list(nx.selfloop_edges(partial_model)))

        spanning_tree = maximum_spanning_branch(partial_model, attr="weight")
        source = next(nx.topological_sort(spanning_tree))

        edge_result, _ = self.concretize_model(partial_model, source)

        # construct the new graph
        node_result = set()

        for i, j in edge_result:
            if edge_result[(i, j)]:
                node_result.add(i)
                node_result.add(j)

        new_graph = nx.DiGraph()
        for node, data in partial_model.nodes(data=True):
            if node in node_result:
                new_graph.add_node(node, label=data["label"])

        for i, j, data in partial_model.edges(data=True):
            if edge_result[(i, j)]:
                new_graph.add_edge(i, j)

        return new_graph

    def concretize_model(
        self, partial_model: DiGraph, source_node: int
    ) -> dict[tuple[int, int], int]:
        """
        Concretize the edges using constraint optimization

        Optimizing the cross entropy of an edge existence and make sure the solution is
        a valid activity diagram
        """
        start = time.time()
        edge_variables = pulp.LpVariable.dicts(
            "edges",
            list(partial_model.edges),
            lowBound=0,
            upBound=1,
            cat=pulp.LpInteger,
        )

        node_variables = pulp.LpVariable.dicts(
            "nodes",
            list(partial_model.nodes),
            lowBound=0,
            upBound=1,
            cat=pulp.LpInteger,
        )

        problem = pulp.LpProblem("Maximum_Clevr_Program", pulp.LpMaximize)

        # One should be able to reach any node from the source node (formatted as a
        # *phantom flow problem*)
        reachability_variables = pulp.LpVariable.dicts(
            "reachability", list(partial_model.edges), lowBound=0, cat=pulp.LpInteger
        )

        n = len(partial_model.nodes)
        for i, j in partial_model.edges:
            problem += (
                reachability_variables[(i, j)] <= (n - 1) * edge_variables[(i, j)]
            )

        for i in partial_model.nodes:
            if i == source_node:
                continue

            incoming_sum = pulp.lpSum(
                [reachability_variables[u, i] for u, i in partial_model.in_edges(i)]
            )
            outgoing_sum = pulp.lpSum(
                [reachability_variables[i, v] for i, v in partial_model.out_edges(i)]
            )

            # Only consumes the unit when the node is selected
            problem += incoming_sum == outgoing_sum + node_variables[i]

        # There should be no cycles in the program graph
        num_cycles = 0
        for nodes in nx.simple_cycles(partial_model):
            edges = [edge_variables[(nodes[-1], nodes[0])]]
            for i in range(len(nodes) - 1):
                edges.append(edge_variables[(nodes[i], nodes[i + 1])])

            problem += pulp.lpSum(edges) <= len(edges) - 1
            num_cycles += 1
            if num_cycles > self.MAXIMUM_CYCLES_TO_SAMPLE:
                break

        termination_candidates = []
        for i in partial_model.nodes:
            label = partial_model.nodes[i]["label"]
            num_previous = self.label_to_num_previous(label)
            in_edges = partial_model.in_edges(i)
            out_edges = partial_model.out_edges(i)

            # Constrain the number of incoming edge using the program type
            problem += (
                pulp.lpSum(edge_variables[s, t] for s, t in in_edges)
                == num_previous * node_variables[i]
            )

            # The node has outgoing edges only if it is selected
            problem += (
                pulp.lpSum(edge_variables[s, t] for s, t in out_edges)
                <= len(list(out_edges)) * node_variables[i]
            )

            # Non-termination node must have at least one outgoing edge
            if not self.is_termination_node(label):
                problem += (
                    pulp.lpSum(edge_variables[s, t] for s, t in out_edges)
                    >= node_variables[i]
                )
            else:
                termination_candidates.append(i)

        termination_variables = pulp.LpVariable.dicts(
            "termination",
            termination_candidates,
            lowBound=0,
            upBound=1,
            cat=pulp.LpInteger,
        )

        # There can only be one termination node
        for node in termination_candidates:
            out_edges = partial_model.out_edges(node)
            # When the node has no out edges, then the termination variable should be 1
            if len(out_edges) >= 1:
                problem += (len(out_edges)) * termination_variables[
                    node
                ] >= 1 - pulp.lpSum(
                    [edge_variables[s, t] for s, t in out_edges]
                    + (1 - node_variables[node])
                )
            else:
                problem += termination_variables[node] == node_variables[node]
        problem += pulp.lpSum(termination_variables) == 1

        # Disable impossible edges
        for i, j in edge_variables:
            source_label = partial_model.nodes[i]["label"]
            target_label = partial_model.nodes[j]["label"]
            if self.get_output_type(source_label) != self.get_input_type(target_label):
                problem += edge_variables[(i, j)] == 0

        # Disable impossible nodes
        for i in node_variables:
            if not self.is_label_valid(partial_model.nodes[i]["label"]):
                # print(partial_model.nodes[i]["label"])
                problem += node_variables[i] == 0

        # Maximize the cross entropy of node variables
        node_weights = {}
        for i, data in partial_model.nodes(data=True):
            node_weights[i] = min(data.get("weight", 1), 1)
            if node_weights[i] == 0:
                node_weights[i] += 1e-5
            elif node_weights[i] == 1:
                node_weights[i] -= 1e-5

        # Maximize the cross entropy of edge variables
        edge_weights = {}
        for i, j, data in partial_model.edges(data=True):
            edge_weights[(i, j)] = min(data["weight"], 1)
            if edge_weights[(i, j)] == 0:
                edge_weights[(i, j)] += 1e-5
            elif edge_weights[(i, j)] == 1:
                edge_weights[(i, j)] -= 1e-5
        problem += pulp.lpSum(
            [
                edge_variables[(i, j)] * np.log(edge_weights[(i, j)])
                + (1 - edge_variables[(i, j)]) * np.log(1 - edge_weights[(i, j)])
                for (i, j) in edge_weights
            ]
        ) + pulp.lpSum(
            [
                node_variables[i] * np.log(node_weights[i])
                + (1 - node_variables[i]) * np.log(1 - node_weights[i])
                for i in node_weights
            ]
        )

        self.problem_definition_runtime = time.time() - start

        start = time.time()
        status = problem.solve(
            pulp.PULP_CBC_CMD(msg=0, timeLimit=120, options=[f"RandomS {self.seed}"])
        )
        self.problem_solve_runtime = time.time() - start

        # logger.debug(problem)
        logger.debug(f"Solving satus {pulp.LpStatus[status]}")
        logger.debug(f"optimum objective value {problem.objective.value()}")
        result_map = {}
        for i, j in partial_model.edges:
            if edge_variables[i, j].value() == 1.0:
                result_map[(i, j)] = True
            else:
                result_map[(i, j)] = False

        return result_map, problem.objective.value()


class ActivityDiagramConcretizer(Concretizer):
    seed: int = 42

    def __init__(self, seed: int = 42):
        self.seed = seed

    def concretize(self, partial_model: DiGraph) -> DiGraph:

        # remove isolated nodes and self loops
        partial_model.remove_nodes_from(list(nx.isolates(partial_model)))
        partial_model.remove_edges_from(list(nx.selfloop_edges(partial_model)))

        spanning_tree = maximum_spanning_branch(partial_model, attr="weight")
        source = next(nx.topological_sort(spanning_tree))
        # logger.info(f"source node: {source}")

        edge_result, status = self.concretize_flow_chart(partial_model, source)
        num_selected_edges = sum(edge_result.values())
        if num_selected_edges == 0 or status == pulp.LpStatusInfeasible:
            mv_concretizer = MajorityVotingConcretizer()
            return mv_concretizer.concretize(partial_model)

        node_result = set()

        for i, j in edge_result:
            if edge_result[(i, j)]:
                node_result.add(i)
                node_result.add(j)

        new_graph = nx.DiGraph()
        for node, data in partial_model.nodes(data=True):
            if node in node_result:
                new_graph.add_node(node, label=data["label"])

        for i, j, data in partial_model.edges(data=True):
            if edge_result[(i, j)]:
                new_graph.add_edge(
                    i, j, label=partial_model.edges[i, j].get("label", "")
                )

        return new_graph

    def add_contradictory_edge_constraints(
        self, problem, partial_model: DiGraph, edge_variables: dict
    ):
        for u in partial_model.nodes:
            candidate_edges = []
            for u, v, data in partial_model.out_edges(u, data=True):
                if data["label"] == "":
                    candidate_edges.append((u, v))

            for i in range(len(candidate_edges)):
                for j in range(i + 1, len(candidate_edges)):
                    a = candidate_edges[i][1]
                    b = candidate_edges[j][1]

                    a_children = nx.descendants(partial_model, a)
                    a_children.add(a)
                    b_children = nx.descendants(partial_model, b)
                    b_children.add(b)

                    if (
                        len(a_children - b_children) == 0
                        or len(b_children - a_children) == 0
                    ):
                        problem += edge_variables[(u, a)] + edge_variables[(u, b)] <= 1

    def concretize_flow_chart(
        self, partial_model: DiGraph, source_node: int
    ) -> dict[tuple[int, int], int]:
        """
        Concretize the edges using constraint optimization

        Optimzing the cross entropy of an edge existance and make sure the solution is
        a valid activity diagram
        """
        start = time.time()
        termination_candidate = set()
        for node, data in partial_model.nodes(data=True):
            if data["termination_weight"] >= 0.5:
                termination_candidate.add(node)

        edge_variables = pulp.LpVariable.dicts(
            "edges",
            list(partial_model.edges),
            lowBound=0,
            upBound=1,
            cat=pulp.LpInteger,
        )

        node_variables = pulp.LpVariable.dicts(
            "nodes",
            list(partial_model.nodes),
            lowBound=0,
            upBound=1,
            cat=pulp.LpInteger,
        )

        problem = pulp.LpProblem("Maximum_activity_diagram", pulp.LpMaximize)

        # One should be able to reach any node from the source node (formatted as a
        # *phantom flow problem*)
        reachability_variables = pulp.LpVariable.dicts(
            "reachability", list(partial_model.edges), lowBound=0, cat=pulp.LpInteger
        )

        n = len(partial_model.nodes)
        for i, j in partial_model.edges:
            problem += (
                reachability_variables[(i, j)] <= (n - 1) * edge_variables[(i, j)]
            )

        for i in partial_model.nodes:
            if i == source_node:
                continue

            incoming_sum = pulp.lpSum(
                [reachability_variables[u, i] for u, i in partial_model.in_edges(i)]
            )
            outgoing_sum = pulp.lpSum(
                [reachability_variables[i, v] for i, v in partial_model.out_edges(i)]
            )

            # Only consumes the unit when the node is selected
            problem += incoming_sum == outgoing_sum + node_variables[i]

        # 1. Selected non-termination candidate nodes must have at least one outgoing
        # edge
        # 2. A node may have outgoing edges only when it is seleccted in the final model
        # 3. if a node has either in edge or out edge, then the node must be selected
        for i in partial_model.nodes:
            out_edges = partial_model.out_edges(i)
            in_edges = partial_model.in_edges(i)
            if i not in termination_candidate:
                problem += (
                    pulp.lpSum([edge_variables[i, v] for i, v in out_edges])
                    >= node_variables[i]
                )

            problem += (
                pulp.lpSum([edge_variables[i, v] for i, v in out_edges])
                <= len(out_edges) * node_variables[i]
            )

            problem += (
                pulp.lpSum([edge_variables[i, v] for i, v in in_edges])
                <= len(in_edges) * node_variables[i]
            )

        # self.add_contradictory_edge_constraints(problem, partial_model,
        # edge_variables)

        # The outgoing edges should be either labeled or unlabeled
        labeled_variables = pulp.LpVariable.dicts(
            "labeled",
            list(partial_model.nodes),
            lowBound=0,
            upBound=1,
            cat=pulp.LpInteger,
        )

        unlabled_variables = pulp.LpVariable.dicts(
            "unlabled",
            list(partial_model.nodes),
            lowBound=0,
            upBound=1,
            cat=pulp.LpInteger,
        )

        for i in partial_model.nodes:
            labled_edges = []
            unlabled_edges = []

            label_map = defaultdict(list)
            for i, j, data in partial_model.out_edges(i, data=True):
                if data.get("label", "") == "":
                    unlabled_edges.append((i, j))
                else:
                    labled_edges.append((i, j))
                    label_map[data.get("label")].append((i, j))

            if len(labled_edges) != 0:
                problem += (
                    pulp.lpSum([edge_variables[(i, j)] for i, j in labled_edges])
                    >= 2 * labeled_variables[i]
                )
                problem += (
                    pulp.lpSum([edge_variables[(i, j)] for i, j in labled_edges])
                    <= len(labled_edges) * labeled_variables[i]
                )

            for _, edges in label_map.items():
                if len(edges) >= 2:
                    problem += pulp.lpSum([edge_variables[i, j] for i, j in edges]) <= 1

            if len(unlabled_edges) == 0 or len(labled_edges) == 0:
                continue

            problem += (
                pulp.lpSum([edge_variables[(i, j)] for i, j in unlabled_edges])
                >= unlabled_variables[i]
            )
            problem += labeled_variables[i] + unlabled_variables[i] == 1

        # Maximize the cross entropy of node varaiables
        node_weights = {}
        for i, data in partial_model.nodes(data=True):
            node_weights[i] = min(data.get("weight", 1), 1)
            if node_weights[i] == 0:
                node_weights[i] += 1e-5
            elif node_weights[i] == 1:
                node_weights[i] -= 1e-5

        # Maximize the cross entropy of edge variables
        edge_weights = {}
        for i, j, data in partial_model.edges(data=True):
            edge_weights[(i, j)] = min(data["weight"], 1)
            if edge_weights[(i, j)] == 0:
                edge_weights[(i, j)] += 1e-5
            elif edge_weights[(i, j)] == 1:
                edge_weights[(i, j)] -= 1e-5
        problem += pulp.lpSum(
            [
                edge_variables[(i, j)] * np.log(edge_weights[(i, j)])
                + (1 - edge_variables[(i, j)]) * np.log(1 - edge_weights[(i, j)])
                for (i, j) in edge_weights
            ]
        ) + pulp.lpSum(
            [
                node_variables[i] * np.log(node_weights[i])
                + (1 - node_variables[i]) * np.log(1 - node_weights[i])
                for i in node_weights
            ]
        )
        self.problem_definition_runtime = time.time() - start

        start = time.time()
        status = problem.solve(
            pulp.PULP_CBC_CMD(msg=0, timeLimit=120, options=[f"RandomS {self.seed}"])
        )
        self.problem_solve_runtime = time.time() - start

        logger.debug(problem)
        logger.debug(f"Solving satus {pulp.LpStatus[status]}")
        logger.debug(f"optimum objective value {problem.objective.value()}")
        result_map = {}
        for i, j in partial_model.edges:
            if edge_variables[i, j].value() == 1.0:
                result_map[(i, j)] = True
            else:
                result_map[(i, j)] = False

        return result_map, status
