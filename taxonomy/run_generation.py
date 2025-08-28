import json
import os
import sys
from argparse import ArgumentParser
from multiprocessing import Pool

import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from output_parsers import TaxonomyOutputParser
from prompts import get_prompt, get_relation
from tqdm import tqdm
from utils import construct_input, gather_concept_groups

from abscon.llms import get_llm
from abscon.utils import serialize_output

load_dotenv()


logger.remove()
logger.add(sys.stderr, level="INFO")


base_url = os.environ.get("OPENAI_BASE_URL")


def run_single_input_openai(data):
    args, concepts, group = data

    template = get_prompt(args.dataset, args.prompt_type)
    llm = get_llm(args)
    chain = template | llm

    relation = get_relation(args.dataset)
    output_parser = TaxonomyOutputParser(
        pattern=r"```taxonomy\n((.|\n)+?)\n```", relation=relation
    )
    text_input = construct_input(concepts)

    result = None
    raw_content = None
    while result is None:
        try:
            result_raw = chain.invoke(input={"user_input": text_input})
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for group {group}: {e}")
            continue
        result = output_parser.parse(result_raw.content, group)
        if result is None or len(result) == 0:
            print("result is None!")

        raw_content = result_raw.content
        if "reasoning_content" in result_raw.additional_kwargs:
            raw_content = f"<think>\n{result_raw.additional_kwargs['reasoning_content']}\n</think>\n\n{raw_content}"  # noqa: E501

    return group, result, raw_content


def run_gpt(
    groups,
    group_concepts,
    args,
    results,
    results_raw,
):
    all_data = [(args, group_concepts[group], group) for group in groups]
    # outputs = [run_single_input_openai(all_data[0])]

    with Pool(processes=args.num_processes) as pool:
        outputs = list(
            tqdm(
                pool.imap_unordered(run_single_input_openai, all_data),
                total=len(all_data),
            )
        )

    group_result_map = {r[0]: (r[1], r[2]) for r in outputs}

    for group in groups:
        output_result = group_result_map[group]
        results.extend(output_result[0])
        results_raw.append(output_result[1])

    return results, results_raw


def load_cache(args):
    llm_name = args.llm_name.split("/")[-1]
    output_path = f"{args.output_folder}/{llm_name}/{args.dataset}/results_{args.output_suffix}.csv"  # noqa: E501
    output_path_raw = f"{args.output_folder}/{llm_name}/{args.dataset}/results_{args.output_suffix}_raw.csv"  # noqa: E501

    if os.path.exists(output_path):
        results = pd.read_csv(output_path, index_col=0)
        results_raw = pd.read_csv(output_path_raw, index_col=0)
        return results.to_dict(orient="records"), results_raw["0"].tolist()
    else:
        return [], []


def main(args):
    test_df = pd.read_csv(f"{args.input_folder}/{args.dataset}.csv")

    results, results_raw = load_cache(args)

    processed_groups = set([str(result["group"]) for result in results])

    group_concepts = gather_concept_groups(test_df)

    groups = set(group_concepts.keys()).difference(processed_groups)
    groups = sorted(list(groups))

    logger.info(f"Number of groups: {len(groups)}")

    results, results_raw = run_gpt(
        groups,
        group_concepts,
        args,
        results,
        results_raw,
    )

    serialize_output(results=results, results_raw=results_raw, args=args)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--input_folder", default="data")
    parser.add_argument("--dataset", type=str, choices=["wordnet"], required=True)
    parser.add_argument("--output_folder", default="results")
    parser.add_argument("--output_suffix", type=str, required=True)
    parser.add_argument("--llm_type", choices=["reasoning", "regular"], required=True)
    parser.add_argument("--llm_name", type=str, required=True)
    parser.add_argument("--num_processes", type=int, default=8)
    parser.add_argument(
        "--prompt_type", choices=["simple", "fewshot"], type=str, required=True
    )

    parser.add_argument("--temperature", type=float, default=0.7)

    args = parser.parse_args()
    main(args)
