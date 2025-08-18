import os
import sys
from argparse import ArgumentParser

import pandas as pd
from langchain_openai import ChatOpenAI
from loguru import logger
from output_parsers import TaxonomyOutputParser
from prompts import get_prompt, get_relation
from tqdm import tqdm
from utils import construct_input, gather_concept_groups

from abscon.llms import SelfHostedLLM
from abscon.utils import construct_messages, serialize_output

logger.remove()
logger.add(sys.stderr, level="INFO")


proxy = os.environ.get("OPENAI_PROXY", None)
base_url = os.environ.get("OPENAI_BASE_URL")


def run_llama(
    template,
    llm,
    output_parser,
    groups,
    group_concepts,
    args,
    results,
    results_raw,
    serialize_frequency=1,
):

    for i, group in enumerate(tqdm(groups)):
        text_input = construct_input(group_concepts[group])
        messages = construct_messages(template, text_input)

        result = []
        while result is None or len(result) == 0:
            response = llm(prompt="", messages=messages)
            result = output_parser.parse(response, group)
            if result is None or len(result) == 0:
                logger.info("Result is none or empty!")

        results_raw.append(response)
        results.extend(result)

        if i % serialize_frequency == 0:
            serialize_output(results, results_raw, args)
    return results, results_raw


def run_gpt(
    template,
    llm,
    output_parser,
    groups,
    group_concepts,
    args,
    results,
    results_raw,
    serialize_frequency=10,
):
    chain = template | llm

    # results = []
    # results_raw = []
    for i, group in enumerate(tqdm(groups)):
        text_input = construct_input(group_concepts[group])

        result = None
        while result is None:
            result_raw = chain.invoke(input={"user_input": text_input})
            result = output_parser.parse(result_raw.content, group)
            if result is None or len(result) == 0:
                print("result is None!")

        results.extend(result)
        results_raw.append(result_raw.content)

        if i % serialize_frequency == 0:
            serialize_output(results, results_raw, args)

    return results, results_raw


def load_cache(args):
    output_path = f"{args.output_folder}/{args.llm_name}/{args.dataset}/results_{args.output_suffix}.csv"
    output_path_raw = f"{args.output_folder}/{args.llm_name}/{args.dataset}/results_{args.output_suffix}_raw.csv"

    if os.path.exists(output_path):
        results = pd.read_csv(output_path, index_col=0)
        results_raw = pd.read_csv(output_path_raw, index_col=0)
        return results.to_dict(orient="records"), results_raw["0"].tolist()
    else:
        return [], []


def main(args):
    template = get_prompt(args.dataset)

    if args.llm_type == "llama":
        llm = SelfHostedLLM(
            default_model=f"{args.llm_name}",
            llm_config={"temperature": args.temperature},
        )
    else:
        llm = ChatOpenAI(
            base_url=base_url,
            openai_proxy=proxy,
            model=args.llm_name,
            temperature=args.temperature,
        )

    relation = get_relation(args.dataset)
    output_parser = TaxonomyOutputParser(
        pattern=r"```taxonomy\n((.|\n)+?)\n```", relation=relation
    )

    test_df = pd.read_csv(f"{args.input_folder}/{args.dataset}.csv")

    results, results_raw = load_cache(args)

    processed_groups = set([str(result["group"]) for result in results])

    group_concepts = gather_concept_groups(test_df)

    groups = set(group_concepts.keys()).difference(processed_groups)
    groups = sorted(list(groups))

    logger.info(f"Number of groups: {len(groups)}")

    if args.llm_type == "llama":
        results, results_raw = run_llama(
            template,
            llm,
            output_parser,
            groups,
            group_concepts,
            args,
            results,
            results_raw,
        )
    else:
        results, results_raw = run_gpt(
            template,
            llm,
            output_parser,
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
    parser.add_argument(
        "--dataset", type=str, choices=["wordnet", "ccs"], required=True
    )
    parser.add_argument("--output_folder", default="results")
    parser.add_argument("--output_suffix", type=str, required=True)
    parser.add_argument("--llm_type", choices=["llama", "gpt"], required=True)
    parser.add_argument("--llm_name", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.7)

    args = parser.parse_args()
    main(args)
