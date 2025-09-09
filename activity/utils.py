import re
from loguru import logger
import networkx as nx
import numpy as np
from typing import Callable
from activity.parser import ActivityParser


def partial_model_to_mermaid(
    partial_model: nx.DiGraph, orientation: bool = "LR"
) -> str:
    def process_node_partial(node_id: str, node_set: set[str]):
        if node_id in node_set:
            return node_id

        node_set.add(node_id)
        node_label = partial_model.nodes[node_id].get(
            "labels", partial_model.nodes[node_id]["label"]
        )
        weight = np.round(partial_model.nodes[node_id].get("weight", 0), 4)
        node_type = partial_model.nodes[node_id].get("node_type")

        if node_type == "decision":
            if weight != 0:
                return f'{node_id}{{"{node_label}, {weight}"}}'
            else:
                return f'{node_id}{{"{node_label}"}}'
        else:
            if weight != 0:
                return f'{node_id}["{node_label}, {weight}"]'
            else:
                return f'{node_id}["{node_label}"]'

    def process_edge_partial(source: str, target: str, node_set: set[str]):
        condition = partial_model.edges[source, target].get("label", "")
        weight = np.round(partial_model.edges[source, target].get("weight", 0), 4)
        source_text = process_node_partial(source, node_set)
        target_text = process_node_partial(target, node_set)

        if weight != 0:
            return f'{source_text} --> |"{weight}, {condition}"|{target_text}'
        elif condition != "":
            return f'{source_text} --> |"{condition}"|{target_text}'
        else:
            return f"{source_text} --> {target_text}"

    header = f"flowchart {orientation}"
    processed_nodes = set()
    body = ""
    for source, target in partial_model.edges:
        edge_text = process_edge_partial(source, target, processed_nodes)
        body += f"{edge_text}\n"

    return f"{header}\n{body}".strip()


def extract_mermaid(result: str) -> str:
    parser = ActivityParser()
    pattern = re.compile("```mermaid(.+?)```", re.DOTALL)

    match = pattern.search(result)

    if match is not None:
        text = match.group(1)
        # only keep the ones that are syntactically correct
        try:
            parser.parse(text)
            return text
        except Exception as e:
            logger.warning(f"Error parsing mermaid text: {e}")
            return None
    else:
        return None


def graph_to_relation_set(graph: nx.DiGraph) -> set[str]:
    results = []
    for s, t, data in graph.edges(data=True):
        source_label = graph.nodes[s]["label"]
        target_label = graph.nodes[t]["label"]
        edge_label = data.get("label", "")
        if edge_label != "":
            results.append(f"{source_label} {edge_label} {target_label}")
        else:
            results.append(f"{source_label} {target_label}")
    return set(results)


def soft_cardinality(
    relation_set: set[str], similarity_func: Callable[[str, str], float]
) -> float:
    result = 0
    for relation in relation_set:
        similarity = 0
        for other in relation_set:
            similarity += similarity_func(relation, other)
        result += 1 / similarity

    return result


def soft_metrics(
    predicted_relations: set[str],
    gt_relations: set[str],
    similarity_func: Callable[[str, str], float],
) -> dict[str, float]:
    predicted_cardinality = soft_cardinality(predicted_relations, similarity_func)
    gt_cardinality = soft_cardinality(gt_relations, similarity_func)
    union_cardinality = soft_cardinality(
        list(predicted_relations) + list(gt_relations), similarity_func
    )
    intersect_cardinality = predicted_cardinality + gt_cardinality - union_cardinality

    if predicted_cardinality != 0:
        precision = intersect_cardinality / predicted_cardinality
    else:
        precision = 0
    recall = intersect_cardinality / gt_cardinality
    f1 = 2 * intersect_cardinality / (predicted_cardinality + gt_cardinality)

    return {"precision": precision, "recall": recall, "f1": f1}
