from argparse import ArgumentParser
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
    temperature: str,
    num_generation: int,
    folder: str,
    ground_truth_folder: str,
) -> dict:
    result = {}
    for llm in llms:
        folder_path = f"{folder}/{temperature}/{llm}"
        result_path = f"{folder_path}/{dataset}/results_abscon_{num_generation}.csv"
        if not os.path.exists(result_path):
            logger.warning(f"{result_path} does not exist, skipping")
            continue
        evaluator = ClevrEvaluator(
            folder_path=folder_path,
            dataset_name=dataset,
            data_folder=ground_truth_folder,
        )
        df = pd.read_csv(result_path)["0"].tolist()
        metrics = evaluator.evaluate_solutions(df)
        logger.info(
            f"Temperature {temperature}, {llm}: accuracy = {metrics['accuracy']:.4f}"
        )
        result[llm] = metrics

    return result


def main(args):
    results = {}
    for temperature in args.temperatures:
        logger.info(f"Process temperature {temperature}")
        results[temperature] = get_result(
            args.dataset,
            args.llms,
            temperature,
            args.num_generation,
            args.folder,
            args.ground_truth_folder,
        )

    os.makedirs(args.output_path, exist_ok=True)
    with open(
        os.path.join(args.output_path, f"temperature_{args.dataset}.json"), "w"
    ) as f:
        json.dump(results, f)


def delimited_list(s: str) -> list[str]:
    return s.split(",")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--folder", type=str, default="results/temperature")
    parser.add_argument("--ground_truth_folder", type=str, default="data")
    parser.add_argument(
        "--llms",
        type=delimited_list,
        default="llama-3.1-8b-instruct,llama-3.1-70b-instruct",
    )
    parser.add_argument(
        "--temperatures", type=delimited_list, default="0.1,0.4,0.7,1.0"
    )
    parser.add_argument("--num_generation", type=int, default=10)
    parser.add_argument("--dataset", type=str, default="clevr")
    parser.add_argument(
        "--output_path",
        type=str,
        default="../measurements/results/programs/",
    )

    args = parser.parse_args()

    main(args)
