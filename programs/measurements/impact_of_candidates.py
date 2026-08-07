from argparse import ArgumentParser
from tqdm import tqdm
import pandas as pd
from evaluation_utils import ClevrEvaluator
import json
from loguru import logger
import sys
import os

logger.remove()
logger.add(sys.stderr, level="INFO")


def get_result(
    dataset: str,
    llms: list[str],
    num_generation: int,
    approaches: list[str],
    folder: str,
    evaluators: dict[str, ClevrEvaluator],
) -> dict:
    result = {}
    for approach in approaches:
        result[approach] = {}
        for llm in llms:
            folder_path = f"{folder}/{llm}"
            evaluator = evaluators[llm]

            if approach == "greedy":
                metrics = evaluator.evaluate_greedy_result()
            elif approach == "esc":
                metrics = evaluator.evaluate_execution_sc(num_generation)
            elif approach == "escf":
                metrics = evaluator.evaluate_execution_sc(
                    num_generation, exclude_error=True
                )
            elif approach == "best":
                metrics = evaluator.evaluate_execution_sc(
                    num_generation, exclude_error=True, best_answer=True
                )
            else:
                df = pd.read_csv(
                    f"{folder_path}/{dataset}/results_{approach}_{num_generation}.csv"
                )["0"].tolist()
                metrics = evaluator.evaluate_solutions(df)
            result[approach][llm] = metrics
    return result


def main(args):
    results = []
    num_generations = range(args.num_generations[0], args.num_generations[1] + 1)

    evaluators = {
        llm: ClevrEvaluator(
            folder_path=f"{args.folder}/{llm}",
            dataset_name=args.dataset,
            data_folder=args.ground_truth_folder,
        )
        for llm in args.llms
    }

    for num_generation in tqdm(num_generations):
        print(f"Process results of {num_generation} candidates")
        result = get_result(
            args.dataset,
            args.llms,
            num_generation,
            args.approaches,
            args.folder,
            evaluators,
        )
        results.append(result)

    with open(
        os.path.join(args.output_path, f"candidates_{args.dataset}.json"), "w"
    ) as f:
        json.dump(results, f)


def delimited_list(s: str) -> list[str]:
    return s.split(",")


def delimited_list_int(s: str) -> list[int]:
    return [int(value) for value in s.split(",")]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--folder", type=str, default="results")
    parser.add_argument("--ground_truth_folder", type=str, default="data")
    parser.add_argument(
        "--approaches", type=delimited_list, default="mv,greedy,abscon,esc,escf,best"
    )
    parser.add_argument(
        "--llms",
        type=delimited_list,
        default="Meta-Llama-3.1-8B-Instruct,Meta-Llama-3.1-70B-Instruct",
    )
    parser.add_argument("--num_generations", type=delimited_list_int, default="1,20")
    parser.add_argument("--dataset", type=str, default="clevr")
    parser.add_argument(
        "--output_path",
        type=str,
        default="../measurements/results/programs/",
    )

    args = parser.parse_args()

    main(args)
