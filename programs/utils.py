import re
import networkx as nx
import numpy as np
from programs.parser import ClevrParser


def process_scene(scene: dict) -> dict:
    """
    Remove unnecessary information from the scene
    """
    scene.pop("image_index", None)
    scene.pop("image_filename", None)
    scene.pop("split", None)
    scene.pop("directions", None)
    for obj in scene["objects"]:
        obj.pop("rotation", None)
        obj.pop("3d_coords", None)
        obj.pop("pixel_coords", None)
    return scene


def extract_mermaid(result: str) -> str:
    parser = ClevrParser()
    pattern = re.compile("```mermaid(.+?)```", re.DOTALL)

    match = pattern.search(result)

    if match is not None:
        text = match.group(1)
        # only keep the ones that are syntactically correct
        try:
            parser.parse(text)
            return text
        except ValueError:
            return None
    else:
        return None


def partial_model_to_mermaid(
    partial_model: nx.DiGraph, orientation: bool = "LR"
) -> str:
    def process_node_partial(node_id: str, node_set: set[str]):
        if node_id in node_set:
            return node_id

        node_set.add(node_id)
        node_label = partial_model.nodes[node_id]["label"]
        weight = np.round(partial_model.nodes[node_id].get("weight", 0), 4)

        if weight != 0:
            return f'{node_id}["{node_label}, {weight}"]'
        else:
            return f'{node_id}["{node_label}"]'

    def process_edge_partial(source: str, target: str, node_set: set[str]):
        weight = np.round(partial_model.edges[source, target].get("weight", 0), 4)
        source_text = process_node_partial(source, node_set)
        target_text = process_node_partial(target, node_set)

        if weight != 0:
            return f'{source_text} --> |"{weight}"|{target_text}'
        else:
            return f"{source_text} --> {target_text}"

    header = f"graph {orientation}"
    processed_nodes = set()
    body = ""
    for source, target in partial_model.edges:
        edge_text = process_edge_partial(source, target, processed_nodes)
        body += f"    {edge_text}\n"

    return f"{header}\n{body}"
