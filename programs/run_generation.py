import json
import os
from multiprocessing import Pool

import pandas as pd
from loguru import logger
from programs.prompts import get_prompt
from tqdm import tqdm
from programs.utils import extract_mermaid

from abscon.llms import get_llm
from abscon.utils import serialize_output


def run_single_input(data):
    index, question, args = data
    template = get_prompt()
    llm = get_llm(args)
    chain = template | llm

    text_input = question["question"]

    result = None
    raw_content = None
    while result is None:
        try:
            result_raw = chain.invoke(input={"user_input": text_input})
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            continue
        result = extract_mermaid(result_raw.content)
        if result is None or len(result) == 0:
            logger.warning("result is None!")

        # Handle reasoning content
        raw_content = result_raw.content
        if "reasoning_content" in result_raw.additional_kwargs:
            raw_content = f"<think>\n{result_raw.additional_kwargs['reasoning_content']}\n</think>\n\n{raw_content}"  # noqa: E501

    return index, result, raw_content


def run_llm(
    questions,
    args,
    results,
    results_raw,
):
    batch_size = args.batch_size

    for i in tqdm(range(0, len(questions), batch_size)):
        batch = questions[i : min(i + batch_size, len(questions))]  # noqa: E203

        batch = [(idx, question, args) for idx, question in enumerate(batch)]

        with Pool(processes=args.num_processes) as pool:
            results_batch = list(
                tqdm(
                    pool.imap_unordered(run_single_input, batch),
                    total=len(batch),
                    leave=False,
                )
            )

        results_batch = sorted(results_batch, key=lambda x: x[0])
        for _, res, raw in results_batch:
            results.append(res)
            results_raw.append(raw)
        serialize_output(results, results_raw, args)

    return results, results_raw


def load_cache(args):
    llm_name = args.llm_name.split("/")[-1]
    output_path = f"{args.output_folder}/{llm_name}/{args.dataset}/results_{args.output_suffix}.csv"  # noqa: E501
    output_path_raw = f"{args.output_folder}/{llm_name}/{args.dataset}/results_{args.output_suffix}_raw.csv"  # noqa: E501

    if os.path.exists(output_path):
        results = pd.read_csv(output_path, index_col=0)
        results_raw = pd.read_csv(output_path_raw, index_col=0)
        return results["0"].to_list(), results_raw["0"].tolist()
    else:
        return [], []


def generate_program_diagrams(args):
    with open(f"{args.input_folder}/{args.dataset}.json") as f:
        questions = json.load(f)["questions"]

    results, results_raw = load_cache(args)
    num_processed = len(results)

    logger.info(f"Number of questions: {len(questions) - num_processed}")

    results, results_raw = run_llm(
        questions[num_processed:],
        args,
        results,
        results_raw,
    )

    return results, results_raw
