from evaluation_utils import TaxonomyEvaluator
import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm
from loguru import logger
import sys
import os
import json
import numpy as np

logger.remove()
logger.add(sys.stderr, level="INFO")


def get_result(
    dataset: str,
    llms: list[str],
    num_generation: int,
    approaches: list[str],
    folder: str,
    ground_truth_path: str,
) -> dict:
    result = {}
    for approach in approaches:
        result[approach] = {}
        for llm in llms:
            folder_path = f"{folder}/{llm}"
            if approach == "greedy":
                evaluator = TaxonomyEvaluator(
                    folder_path,
                    dataset,
                    ground_truth_path,
                    num_generation,
                    evaluate_greedy=True,
                )
                metrics = evaluator.evaluate_abstraction(
                    num_generation, concretization_method="mv", dataset=dataset
                )
            else:
                evaluator = TaxonomyEvaluator(
                    folder_path, dataset, ground_truth_path, num_generation
                )
                df = pd.read_csv(
                    f"{folder_path}/{dataset}/results_{approach}_{num_generation}.csv"
                )
                metrics = evaluator.evaluate_taxonomies(df, dataset)
            result[approach][llm] = metrics
    
    return result


def main(args):
    results = []
    num_generations = list(range(args.num_generations[0], args.num_generations[1] + 1))
    ground_truth_path = f"{args.ground_truth_folder}/{args.dataset}.csv"

    for num_generation in tqdm(num_generations):
        logger.info(f"Process {num_generation} candidates")
        result = get_result(
            args.dataset,
            args.llms,
            num_generation,
            args.approaches,
            args.folder,
            ground_truth_path,
        )
        results.append(result)

    # Initialize the approach dictionary
    for i in range(len(results)):
        results[i]["max"] = {}
        results[i]["median"] = {}

    for i in tqdm(range(len(results))):
        for llm in args.llms:
            folder_path = f"{args.folder}/{llm}"
            evaluator = TaxonomyEvaluator(
                folder_path,
                args.dataset,
                ground_truth_path,
                num_generations=num_generations[i]
            )

            results[i]["max"][llm] = evaluator.evaluate_individual(num_generations[i], args.dataset, aggregator=max)
            results[i]["median"][llm] = evaluator.evaluate_individual(num_generations[i], args.dataset, aggregator=np.median)

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
    parser.add_argument(
        "--dataset", type=str, default="wordnet", choices=["ccs", "wordnet"]
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="../measurements/results/taxonomy/",
    )

    args = parser.parse_args()

    main(args)
