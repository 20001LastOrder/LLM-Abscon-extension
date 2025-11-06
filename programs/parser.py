import networkx as nx

from abscon.mermaid.parser import MermaidGraphParser
import re


class ClevrParser:
    def parse(self, mermaid_text: str) -> nx.DiGraph:
        parser = MermaidGraphParser()
        graph = parser.parse(mermaid_text)

        remapping = {node: count + 1 for count, node in enumerate(graph.nodes)}
        graph = nx.relabel_nodes(graph, remapping)
        for node_id, data in graph.nodes(data=True):
            data["id"] = node_id
        return graph


# class ClevrParser:
#     relation_pattern = re.compile(r".+-->.+")
#     node_definition_pattern = re.compile(r'(\d+)\["(.+)"\]$')
#     node_pattern = re.compile(r"\d+$")

#     def parse(self, mermaid_text: str) -> nx.DiGraph:
#         # unify the edge type
#         mermaid_text = mermaid_text.replace("-.->", "-->")

#         graph = nx.DiGraph()
#         relations = self.relation_pattern.findall(mermaid_text)
#         for relation in relations:
#             self.process_relation(relation, graph)

#         for node, data in graph.nodes(data=True):
#             if "label" not in data:
#                 data["label"] = str(node)
#                 data["id"] = node

#         remapping = {node: count + 1 for count, node in enumerate(graph.nodes)}
#         graph = nx.relabel_nodes(graph, remapping)
#         for node_id, data in graph.nodes(data=True):
#             data["id"] = node_id
#         return graph

#     def process_relation(self, relation_text: str, graph: nx.DiGraph):
#         nodes = relation_text.split("-->")

#         if len(nodes) != 2:
#             raise ValueError("Each relation must have exactly two nodes")

#         processed_nodes = []
#         for node in nodes:
#             node = node.strip()
#             if self.node_pattern.match(node):
#                 processed_nodes.append(int(node))
#             elif self.node_definition_pattern.match(node):
#                 match = self.node_definition_pattern.match(node)
#                 node = int(match.group(1))
#                 label = match.group(2)
#                 graph.add_node(node, label=label, id=node)
#                 processed_nodes.append(node)
#             else:
#                 raise ValueError("Node definition is malformed")
#         graph.add_edge(processed_nodes[0], processed_nodes[1])
