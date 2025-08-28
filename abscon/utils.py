import networkx as nx
import pandas as pd


def maximum_spanning_branch(
    G, attr="weight", default=1, preserve_attrs=False, partition=None
):
    # In order to use the same algorithm is the maximum branching, we need to adjust
    # the weights of the graph. The branching algorithm can choose to not include an
    # edge if it doesn't help find a branching, mainly triggered by edges with negative
    # weights.
    #
    # To prevent this from happening while trying to find a spanning arborescence, we
    # just have to tweak the edge weights so that they are all positive and cannot
    # become negative during the branching algorithm, find the maximum branching and
    # then return them to their original values.
    INF = float("inf")

    min_weight = INF
    max_weight = -INF
    for _, _, w in G.edges(data=attr, default=default):
        if w < min_weight:
            min_weight = w
        if w > max_weight:
            max_weight = w

    for _, _, d in G.edges(data=True):
        d[attr] = d.get(attr, default) - min_weight + 1 - (min_weight - max_weight)

    B = nx.maximum_branching(G, attr, default, preserve_attrs, partition)

    for _, _, d in G.edges(data=True):
        d[attr] = d.get(attr, default) + min_weight - 1 + (min_weight - max_weight)

    for _, _, d in B.edges(data=True):
        d[attr] = d.get(attr, default) + min_weight - 1 + (min_weight - max_weight)

    return B


def serialize_output(results, results_raw, args):
    llm_name = args.llm_name.split("/")[-1]

    output_path = f"{args.output_folder}/{llm_name}/{args.dataset}/results_{args.output_suffix}.csv"
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path)

    output_path_raw = f"{args.output_folder}/{llm_name}/{args.dataset}/results_{args.output_suffix}_raw.csv"
    results_raw_df = pd.DataFrame(results_raw)
    results_raw_df.to_csv(output_path_raw)
