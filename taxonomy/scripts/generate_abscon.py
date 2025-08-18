import pickle
import sys
from argparse import ArgumentParser

import networkx as nx
import pandas as pd
from evaluation_utils import TaxonomyEvaluator
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")


def build_runtime(evaluator: TaxonomyEvaluator):
    runtime_obj = {
        "embedding_runtime": evaluator.runtime_statistics.embedding_runtime,
        "graph_matching_runtime": evaluator.runtime_statistics.graph_matching_runtime,
        "problem_build_runtimes": evaluator.runtime_statistics.problem_build_runtimes,
        "problem_solve_runtimes": evaluator.runtime_statistics.problem_solve_runtimes,
    }

    runtime_df = pd.DataFrame(runtime_obj)
    evaluator.reset_runtime_statistics()
    return runtime_df


def get_partial_models(abstractors) -> list[nx.DiGraph]:
    models = []
    for abstractor in abstractors:
        if abstractor.partial_model is not None:
            models.append(abstractor.partial_model)
        else:
            logger.error(f"Abstractor {abstractor.name} does not have a partial model.")
    return models


def main(args):
    evaluator = TaxonomyEvaluator(
        args.folder_path,
        args.dataset_name,
        f"{args.ground_truth_path}/{args.dataset_name}.csv",
        args.num_generations,
    )

    abscon_result_df, abstractors = evaluator.generate_merged_results(
        args.num_generations,
        concretization_method="solver",
        dataset=args.dataset_name,
        verbose=True,
        return_abstractors=True,
    )
    abscon_runtime = build_runtime(evaluator)

    mv_result_df = evaluator.generate_merged_results(
        args.num_generations, concretization_method="mv", dataset=args.dataset_name
    )

    mv_result_df.to_csv(
        f"{args.folder_path}/{args.dataset_name}/results_mv_{args.num_generations}.csv"
    )
    abscon_result_df.to_csv(
        f"{args.folder_path}/{args.dataset_name}/results_abscon_{args.num_generations}.csv"  # noqa: E501
    )

    abscon_runtime.to_csv(
        f"{args.folder_path}/{args.dataset_name}/runtime_abscon_{args.num_generations}.csv"  # noqa: E501
    )

    if args.save_partial_models:
        partial_models = get_partial_models(abstractors)
        with open(
            f"{args.folder_path}/{args.dataset_name}/partial_models_{args.num_generations}.pkl",  # noqa: E501
            "wb",
        ) as f:
            pickle.dump(partial_models, f)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--folder_path", type=str, required=True)
    parser.add_argument("--dataset_name", choices=["ccs", "wordnet"], required=True)
    parser.add_argument("--ground_truth_path", type=str, default="data")
    parser.add_argument(
        "--save_partial_models",
        action="store_true",
        help="Save partial models to disk",
        default=False,
    )
    parser.add_argument("--num_generations", type=int, required=True)

    args = parser.parse_args()
    main(args)
