import pickle
import sys
from argparse import ArgumentParser

import networkx as nx
import pandas as pd
from evaluation_utils import ActivityEvaluator
from loguru import logger
from sentence_transformers import SentenceTransformer

logger.remove()
logger.add(sys.stderr, level="INFO")


def build_runtime(evaluator: ActivityEvaluator):

    runtime_obj = {
        "embedding_runtime": evaluator.runtime_statistics.embedding_runtime,
        "graph_matching_runtime": evaluator.runtime_statistics.graph_matching_runtime,
        "problem_build_runtimes": evaluator.runtime_statistics.problem_build_runtimes,
        "problem_solve_runtimes": evaluator.runtime_statistics.problem_solve_runtimes,
    }

    runtime_df = pd.DataFrame(runtime_obj)
    evaluator.reset_runtime_statistics()
    return runtime_df


def get_partial_models(evaluator: ActivityEvaluator) -> list[nx.DiGraph]:
    models = []
    for abstractor in evaluator.abstractors:
        if abstractor.partial_model is not None:
            models.append(abstractor.partial_model)
        else:
            logger.error(f"Abstractor {abstractor.name} does not have a partial model.")
    return models


def main(args):
    encoder = SentenceTransformer(args.encoder)
    evaluator = ActivityEvaluator(
        args.folder_path,
        args.dataset_name,
        data_folder=args.ground_truth_path,
        encoder=encoder,
        seed=args.seed,
    )

    for num_candidates in range(args.num_candidates_start, args.num_candidates_end + 1):
        logger.info(f"Processing {num_candidates} candidates...")
        abscon_results = evaluator.combine_solutions(
            num_candidates, concretization_method="solver", verbose=True
        )
        abscon_runtime = build_runtime(evaluator)

        if args.num_processes == 1:
            mv_results = evaluator.combine_solutions(
                num_candidates, concretization_method="mv", verbose=True
            )
        else:
            mv_results = evaluator.combine_solutions_concurrent(
                num_candidates,
                concretization_method="mv",
                verbose=True,
                num_processes=args.num_processes,
            )

        pd.DataFrame(mv_results).to_csv(
            f"{args.folder_path}/{args.dataset_name}/results_mv_{num_candidates}.csv"
        )
        pd.DataFrame(abscon_results).to_csv(
            f"{args.folder_path}/{args.dataset_name}/results_abscon_{num_candidates}.csv"  # noqa: E501
        )
        abscon_runtime.to_csv(
            f"{args.folder_path}/{args.dataset_name}/runtime_abscon_{num_candidates}.csv"  # noqa: E501
        )

        if args.save_partial_models:
            # Save the partial models for each case
            partial_models = get_partial_models(evaluator)
            with open(
                f"{args.folder_path}/{args.dataset_name}/partial_models_{num_candidates}.pkl",  # noqa: E501
                "wb",
            ) as f:
                pickle.dump(partial_models, f)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--folder_path", type=str, required=True)
    parser.add_argument("--dataset_name", choices=["paged"], default="paged")
    parser.add_argument("--ground_truth_path", type=str, default="data")
    parser.add_argument(
        "--encoder", type=str, default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument(
        "--num_candidates_start",
        help="Starting number of candidates to abstract, this is included",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--num_candidates_end",
        help="Ending number of candidates to abstract (inclusive). If -1 then it will be set the same number as num_candidates_start",  # noqa: E501
        type=int,
        default=False,
    )
    parser.add_argument(
        "--seed",
        help="The random seed to control randomness in the approximation algorithms",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--save_partial_models",
        help="If true, it will save the partial models for each candidate number",
        action="store_true",
        default=False,
    )
    parser.add_argument("--num_processes", type=int, default=1)

    args = parser.parse_args()
    main(args)
