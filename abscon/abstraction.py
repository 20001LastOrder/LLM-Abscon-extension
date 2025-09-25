import copy
import time
from collections import Counter
from typing import Any

import networkx as nx
import numpy as np
from loguru import logger
from numpy import dot
from numpy.linalg import norm
from pydantic import Field
from sentence_transformers import SentenceTransformer
from thefuzz import fuzz, process

from abscon.base import Abstractor


def find_match(text, candidates):
    match, score = process.extractOne(text, candidates, scorer=fuzz.ratio)

    if score < 100:
        logger.debug(f"{text} not exist in the node")
        return None

    return match


class TaxonomyAbstractor(Abstractor):
    nodes: list[str]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        node_map = {}

        for i, node in enumerate(self.nodes):
            self.partial_model.add_node(i, label=node)
            node_map[node] = i

        self.partial_model.graph["node_map"] = node_map

    def add_concrete_model(self, model: nx.DiGraph):
        self.candidate_count += 1

        label_node_map = self.partial_model.graph["node_map"]
        node_node_map = {}
        # Map node first
        start_time = time.time()
        for node, data in model.nodes(data=True):
            label = data["label"]
            matched_label = find_match(label, self.nodes)

            if matched_label is None:
                continue
            node_node_map[node] = label_node_map[matched_label]
        self.embedding_runtime.append(time.time() - start_time)

        # Map relations termination_weight
        start_time = time.time()
        for source, target, data in model.edges(data=True):
            if source not in node_node_map or target not in node_node_map:
                continue

            source = node_node_map[source]
            target = node_node_map[target]

            if not self.partial_model.has_edge(source, target):
                self.partial_model.add_edge(source, target, weight=0)

            edge_data = self.partial_model.edges[(source, target)]

            edge_data["weight"] = edge_data["weight"] + 1
        self.graph_matching_runtime.append(time.time() - start_time)


class ClevrAbstractor(Abstractor):
    encoder: Any
    edit_distance_timeout: int = 5
    node_embeddings: list = Field(default_factory=list)
    similarities: np.ndarray = None

    def set_seed_model(self, model: nx.DiGraph):
        self.candidate_count += 1
        source_model = copy.deepcopy(model)
        nx.set_node_attributes(source_model, 1, "weight")
        nx.set_edge_attributes(source_model, 1, "weight")

        self.partial_model = source_model
        source_nodes = [
            data["label"].lower().strip()
            for _, data in self.partial_model.nodes(data=True)
        ]

        start_time = time.time()
        self.node_embeddings = self.encoder.encode(source_nodes).tolist()
        self.embedding_runtime.append(time.time() - start_time)

    def get_node_distance_matrix(
        self, source_embeddings: list[list[float]], target_embeddings: list[list[float]]
    ):
        m = len(source_embeddings)
        n = len(target_embeddings)

        embeddings1 = np.array(source_embeddings)
        embeddings2 = np.array(target_embeddings)

        embeddings1 = np.repeat(embeddings1.reshape(m, 1, -1), n, axis=1)
        embeddings2 = np.repeat(embeddings2.reshape(1, n, -1), m, axis=0)

        return cosine_similarity_batch(embeddings1, embeddings2)

    def add_concrete_model(self, model: nx.DiGraph):
        if self.candidate_count == 0:
            self.set_seed_model(model)
            return

        self.candidate_count += 1

        if len(model.nodes) == 0:
            return

        target_nodes = [
            data["label"].lower().strip() for _, data in model.nodes(data=True)
        ]

        start_time = time.time()
        target_node_embeddings = self.encoder.encode(target_nodes).tolist()
        self.embedding_runtime.append(time.time() - start_time)

        start_time = time.time()
        self.similarities = self.get_node_distance_matrix(
            self.node_embeddings, target_node_embeddings
        )

        paths = nx.optimize_edit_paths(
            self.partial_model,
            model,
            node_subst_cost=self.node_subst_cost,
            node_del_cost=self.node_del_cost,
            node_ins_cost=self.node_ins_cost,
            edge_subst_cost=self.edge_subst_cost,
            edge_ins_cost=self.edge_ins_cost,
            edge_del_cost=self.edge_del_cost,
            strictly_decreasing=True,
            timeout=self.edit_distance_timeout,
        )
        for p in paths:
            best_path = p

        node_id_map = self.process_nodes(best_path[0], model, target_node_embeddings)
        self.process_edges(best_path[1], model, node_id_map)
        self.graph_matching_runtime.append(time.time() - start_time)

    def process_nodes(self, node_match, target_model, target_node_embeddings):
        target_node_id_map = {node_id: node_id for node_id in target_model.nodes}
        next_idx = len(self.partial_model.nodes) + 1

        for source_node, target_node in node_match:
            if source_node is not None and target_node is not None:
                self.partial_model.nodes[source_node]["weight"] += 1

                source_label = (
                    self.partial_model.nodes[source_node]["label"].lower().strip()
                )
                target_label = target_model.nodes[target_node]["label"].lower().strip()
                if source_label != target_label:
                    logger.warning(
                        f"Two node labels {source_label} and {target_label} is not exactly the same"  # noqa: E501
                    )
                target_node_id_map[target_node] = source_node
            elif target_node is not None:
                node_data = target_model.nodes[target_node]
                node_data["weight"] = 1
                node_data["id"] = next_idx
                self.partial_model.add_node(next_idx, **node_data)
                target_node_id_map[target_node] = next_idx
                self.node_embeddings.append(target_node_embeddings[target_node - 1])

                next_idx += 1

        return target_node_id_map

    def process_edges(self, edge_match, target_model, target_node_id_map):
        for source_edge, target_edge in edge_match:
            if source_edge is not None and target_edge is not None:
                edge_data = self.partial_model.edges[source_edge]
                edge_data["weight"] += 1
            elif target_edge is not None:
                edge_data = target_model.edges[target_edge]

                # get new node idx fro mthe source node
                target_edge = (
                    target_node_id_map[target_edge[0]],
                    target_node_id_map[target_edge[1]],
                )
                edge_data["weight"] = 1
                self.partial_model.add_edge(target_edge[0], target_edge[1], **edge_data)

    def node_match_func(self, source_node, target_node):
        source_id = int(source_node["id"]) - 1
        target_id = int(target_node["id"]) - 1
        return np.isclose(self.similarities[source_id, target_id], 1)

    def node_subst_cost(self, source_node, target_node):
        source_id = int(source_node["id"]) - 1
        target_id = int(target_node["id"]) - 1
        return 1 - self.similarities[source_id, target_id]

    def node_del_cost(self, source_node):
        return 0.01

    def node_ins_cost(self, target_model):
        return 0.01

    def edge_subst_cost(self, source_edge, target_edge):
        return 0.01

    def edge_del_cost(self, source_edge):
        return 0.01

    def edge_ins_cost(self, source_edge):
        return 0.01


class ActivityDiagramAbstractor(Abstractor):
    encoder: SentenceTransformer
    edit_distance_timeout: int = 5
    similarities: np.ndarray = None
    node_embeddings: list = Field(default_factory=list)

    def get_partial_model(self) -> nx.DiGraph:
        graph = nx.DiGraph()
        for node, data in self.partial_model.nodes(data=True):
            for key in data:
                data = data.copy()
                if "termination" in key:
                    data[key] = data[key] / len(data["labels"])
                elif "weight" in key:
                    data[key] = data[key] / self.candidate_count
            graph.add_node(node, **data)

        for source, target, data in self.partial_model.edges(data=True):
            for key in data:
                data = data.copy()
                if "weight" in key:
                    data[key] = data[key] / self.candidate_count
            graph.add_edge(source, target, **data)

        return graph

    def set_seed_model(self, model: nx.DiGraph):
        """
        Set seed model for activity diagrams:
        1. set labels of nodes and edges as a list of candidates
        2. set 1 to the weight of each node and edges
        3. Calculate current source node embeddings
        """
        self.candidate_count += 1
        source_model = copy.deepcopy(model)

        nx.set_node_attributes(source_model, 1, "weight")
        for _, data in source_model.nodes(data=True):
            data["labels"] = [data["label"]]

        nx.set_edge_attributes(source_model, 1, "weight")
        for _, _, data in source_model.edges(data=True):
            data["labels"] = [data.get("label", "")]

        self.partial_model = source_model
        source_nodes = [
            data["label"] for _, data in self.partial_model.nodes(data=True)
        ]

        for node, out_degree in source_model.out_degree:
            if out_degree > 0:
                source_model.nodes[node]["termination_weight"] = 0
            else:
                source_model.nodes[node]["termination_weight"] = 1

        start_time = time.time()
        self.node_embeddings = self.encoder.encode(source_nodes).tolist()
        self.embedding_runtime.append(time.time() - start_time)

    def get_node_distance_matrix(
        self, source_embeddings: list[list[float]], target_embeddings: list[list[float]]
    ):
        m = len(source_embeddings)
        n = len(target_embeddings)

        embeddings1 = np.array(source_embeddings)
        embeddings2 = np.array(target_embeddings)

        embeddings1 = np.repeat(embeddings1.reshape(m, 1, -1), n, axis=1)
        embeddings2 = np.repeat(embeddings2.reshape(1, n, -1), m, axis=0)

        return cosine_similarity_batch(embeddings1, embeddings2)

    def add_concrete_model(self, model: nx.DiGraph):
        # Skip empty graphs
        if len(model.nodes) == 0:
            return
        if self.candidate_count == 0:
            self.set_seed_model(model)
            return

        self.candidate_count += 1

        target_nodes = [data["label"].strip() for _, data in model.nodes(data=True)]

        start_time = time.time()
        target_node_embeddings = self.encoder.encode(target_nodes).tolist()
        self.embedding_runtime.append(time.time() - start_time)

        start_time = time.time()
        self.similarities = self.get_node_distance_matrix(
            self.node_embeddings, target_node_embeddings
        )

        paths = nx.optimize_edit_paths(
            self.partial_model,
            model,
            node_subst_cost=self.node_subst_cost,
            node_del_cost=self.node_del_cost,
            node_ins_cost=self.node_ins_cost,
            edge_subst_cost=self.edge_subst_cost,
            edge_ins_cost=self.edge_ins_cost,
            edge_del_cost=self.edge_del_cost,
            strictly_decreasing=True,
            timeout=self.edit_distance_timeout,
        )
        for p in paths:
            best_path = p

        node_id_map = self.process_nodes(best_path[0], model, target_node_embeddings)
        self.process_edges(best_path[1], model, node_id_map)

        # add termination weight to the termination nodes of this concrete model
        for node, out_degree in model.out_degree():
            if out_degree != 0:
                continue

            self.partial_model.nodes[node_id_map[node]]["termination_weight"] += 1
        self.graph_matching_runtime.append(time.time() - start_time)

    def process_nodes(self, node_match, target_model, target_node_embeddings):
        target_node_id_map = {node_id: node_id for node_id in target_model.nodes}
        next_idx = len(self.partial_model.nodes)

        for source_node, target_node in node_match:
            if source_node is not None and target_node is not None:
                self.partial_model.nodes[source_node]["weight"] += 1
                self.partial_model.nodes[source_node]["labels"].append(
                    target_model.nodes[target_node]["label"]
                )
                self.partial_model.nodes[source_node]["label"] = most_frequent(
                    self.partial_model.nodes[source_node]["labels"]
                )
                target_node_id_map[target_node] = source_node
            elif target_node is not None:
                node_data = target_model.nodes[target_node]
                # print(node_data["label"])
                # print(next_idx)
                node_data["weight"] = 1
                node_data["id"] = next_idx
                node_data["labels"] = [node_data["label"]]
                node_data["termination_weight"] = 0
                self.partial_model.add_node(next_idx, **node_data)
                target_node_id_map[target_node] = next_idx

                self.node_embeddings.append(target_node_embeddings[target_node])

                next_idx += 1

        return target_node_id_map

    def process_edges(self, edge_match, target_model, target_node_id_map):
        for source_edge, target_edge in edge_match:
            if source_edge is not None and target_edge is not None:
                edge_data = self.partial_model.edges[source_edge]
                edge_data["weight"] += 1

                target_edge_data = target_model.edges[target_edge]
                edge_data.get("labels", "").append(target_edge_data.get("label", ""))

            elif target_edge is not None:
                edge_data = target_model.edges[target_edge]

                # get new node idx fro mthe source node
                target_edge = (
                    target_node_id_map[target_edge[0]],
                    target_node_id_map[target_edge[1]],
                )
                edge_data["weight"] = 1
                edge_data["labels"] = [edge_data.get("label", "")]
                self.partial_model.add_edge(target_edge[0], target_edge[1], **edge_data)

    def node_subst_cost(self, source_node, target_node):
        source_id = int(source_node["id"])
        target_id = int(target_node["id"])
        # print(source_node["label"])
        # print(target_node["label"])
        # print(1 - self.similarities[source_id, target_id])
        return 1 - self.similarities[source_id, target_id]

    def node_del_cost(self, source_node):
        return 0.1

    def node_ins_cost(self, target_model):
        return 0.1

    def edge_subst_cost(self, source_edge, target_edge):
        if source_edge.get("label", "") == target_edge.get("label", ""):
            return 0
        else:
            return 0.1

    def edge_del_cost(self, source_edge):
        return 0.1

    def edge_ins_cost(self, source_edge):
        return 0.1

    def edge_match_func(self, source_edge, target_edge):
        return source_edge["label"] == target_edge["label"]


def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))


def cosine_similarity_batch(a, b):
    return (a * b).sum(axis=2) / (norm(a, axis=2) * norm(b, axis=2))


def most_frequent(items: list[str]):
    occurence_count = Counter(items)
    return occurence_count.most_common(1)[0][0]


def longest(items: list[str]):
    result = items[0]
    for item in items[1:]:
        if len(item) > len(result):
            result = item
    return result
