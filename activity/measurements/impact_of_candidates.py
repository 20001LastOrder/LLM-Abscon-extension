from argparse import ArgumentParser
from tqdm import tqdm
import pandas as pd
from evaluation_utils import ActivityEvaluator
import json
from loguru import logger
import sys
import os
import numpy as np

logger.remove()
logger.add(sys.stderr, level="INFO")


def get_result(
    dataset: str,
    llm: str,
    num_generation: int,
    approaches: list[str],
    folder: str,
    ground_truth_folder: str,
) -> dict:
    result = {}
    folder_path = f"{folder}/{llm}"
    for approach in approaches:
        if approach == "greedy":
            evaluator = ActivityEvaluator(
                folder_path,
                dataset,
                data_folder=ground_truth_folder,
            )
            metrics = evaluator.evaluate_greedy_result()
        else:
            evaluator = ActivityEvaluator(
                folder_path, dataset, data_folder=ground_truth_folder
            )
            df = pd.read_csv(
                f"{folder_path}/{dataset}/results_{approach}_{num_generation}.csv"
            )["0"].tolist()
            metrics = evaluator.evaluate_solutions(df)
        result[approach] = metrics

    return result


def main(args):
    num_generations = list(range(args.num_generations[0], args.num_generations[1] + 1))
    results = [{} for _ in num_generations]

    for llm in args.llms:
        logger.info(f"Process {llm}")
        for i, num_generation in enumerate(tqdm(num_generations)):
            logger.info(f"Process {num_generation} candidates")
            result = get_result(
                args.dataset,
                llm,
                num_generation,
                args.approaches,
                args.folder,
                args.ground_truth_folder,
            )
            for approach, metrics in result.items():
                if approach not in results[i]:
                    results[i][approach] = {}
                results[i][approach][llm] = metrics

    # Initialize the approach dictionary
    for llm in args.llms:
        for i in range(len(results)):
            results[i]["max"] = {}
            results[i]["median"] = {}
    
    for llm in args.llms:
        logger.info(f"Calculating the best and median performance for {llm}")
        folder_path = f"{args.folder}/{llm}"
        evaluator = ActivityEvaluator(
            folder_path, args.dataset, data_folder=args.ground_truth_folder
        )

        for i in tqdm(range(len(results))):
            results[i]["max"][llm] = evaluator.evaluate_individual(
                i + 1, dataset=args.dataset, aggregator=max
            )
            results[i]["median"][llm] = evaluator.evaluate_individual(
                i + 1, dataset=args.dataset, aggregator=np.median
            )

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
    parser.add_argument("--approaches", type=delimited_list, default="mv,greedy,abscon")
    parser.add_argument(
        "--llms",
        type=delimited_list,
        default="Meta-Llama-3.1-8B-Instruct,Meta-Llama-3.1-70B-Instruct",
    )
    parser.add_argument("--num_generations", type=delimited_list_int, default="1,20")
    parser.add_argument("--dataset", type=str, default="paged")
    parser.add_argument(
        "--output_path",
        type=str,
        default="../measurements/results/activity/",
    )

    args = parser.parse_args()

    main(args)
