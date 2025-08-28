import re

import networkx as nx
from loguru import logger


class ActivityParser:
    relation_pattern = re.compile(r".+-->.+")
    node_definition_pattern = re.compile(r'(\|.+\|)?\s*([^\s\[\"\]]+)[\{\[]\"*(.+)\"*[\}\]]$')
    condition_pattern = re.compile(r'\|["]*([^"]+)["]*\|')
    node_pattern = re.compile(r"(\|.+\|)?\s*([^\s\[\"\]]+)$")

    def parse(self, mermaid_text: str) -> nx.DiGraph:
        # unify the edge type
        mermaid_text = mermaid_text.replace("-.->", "-->")

        graph = nx.DiGraph()
        relations = self.relation_pattern.findall(mermaid_text)
        node_set = set()
        for relation in relations:
            self.process_relation(relation, graph, node_set)

        for node, data in graph.nodes(data=True):
            if "label" not in data:
                data["label"] = str(node)
                data["id"] = node

        remapping = {node: count for count, node in enumerate(graph.nodes)}
        graph: nx.DiGraph = nx.relabel_nodes(graph, remapping)
        for node_id, data in graph.nodes(data=True):
            data["id"] = node_id

        # identify node types
        for node, data in graph.nodes(data=True):
            out_edges = graph.out_edges(node, data=True)
            is_decision = False
            for _, _, edge_data in out_edges:
                if edge_data.get("label", "") != "":
                    is_decision = True
                    break
            if is_decision:
                data["node_type"] = "decision"
            else:
                data["node_type"] = "activity"

        return graph

    def process_relation(self, relation_text: str, graph: nx.DiGraph, node_set: set):
        nodes = relation_text.split("-->")

        if len(nodes) != 2:
            print(relation_text)
            raise ValueError("Each relation must have exactly two nodes")

        processed_nodes = []
        for node in nodes:
            node = node.strip()
            if self.node_pattern.match(node):
                match = self.node_pattern.match(node)
                node_label = match.group(2)

                if node_label not in node_set:
                    graph.add_node(node_label, label=node_label)
                    node_set.add(node_label)
                processed_nodes.append(match.group(2))
            elif self.node_definition_pattern.match(node):
                match = self.node_definition_pattern.match(node)
                node = match.group(2).rstrip('"') # remove trailing quote if any
                label = match.group(3)
                graph.add_node(node, label=label)
                node_set.add(node)
                processed_nodes.append(node)
            else:
                logger.error(node)
                raise ValueError("Node definition is malformed")

        condition_match = self.condition_pattern.search(relation_text)
        if not condition_match:
            graph.add_edge(processed_nodes[0], processed_nodes[1])
        else:
            graph.add_edge(
                processed_nodes[0], processed_nodes[1], label=condition_match.group(1)
            )
