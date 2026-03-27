import json
import os
from multiprocessing import Pool

import pandas as pd
from loguru import logger
from activity.prompts import get_prompt
from tqdm import tqdm
from activity.utils import extract_mermaid
from abscon.utils import serialize_output

from abscon.llms import get_llm


def run_single_result(data):
    idx, sample, args = data
    llm = get_llm(args)
    template = get_prompt(args.prompt_type)
    chain = template | llm
    text_input = sample["paragraph"]

    result = None
    raw_content = None
    tries = 0
    while result is None:
        tries += 1
        try:
            result_raw = chain.invoke(input={"user_input": text_input})
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            continue
        except Exception as e:
            logger.error(f"Error during LLM invocation: {e}")
            continue

        result = extract_mermaid(result_raw.content)
        if result is None or len(result) == 0:
            logger.warning(result_raw.content)
            logger.warning("result is None!")
            if tries >= args.tolerance:
                logger.warning("Max retries reached, return an empty graph.")
                result = "graph TD\n"

        raw_content = result_raw.content
        if "reasoning_content" in result_raw.additional_kwargs:
            raw_content = f"<think>\n{result_raw.additional_kwargs['reasoning_content']}\n</think>\n\n{raw_content}"  # noqa: E501

    return idx, result, raw_content


def run_gpt(
    samples,
    args,
    results,
    results_raw,
):

    batch_size = args.batch_size

    for i in tqdm(range(0, len(samples), batch_size)):
        batch = samples[i : min(i + batch_size, len(samples))]  # noqa: E203
        batch = [(idx, sample, args) for idx, sample in enumerate(batch)]

        with Pool(processes=args.num_processes) as pool:
            results_batch = list(
                tqdm(
                    pool.imap_unordered(run_single_result, batch),
                    desc=f"Processing batch {i // batch_size}",
                    total=len(batch),
                    leave=False,
                )
            )

        results_batch = sorted(results_batch, key=lambda r: r[0])
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


def generate_activity_diagrams(args):

    with open(f"{args.input_folder}/{args.dataset}.json") as f:
        samples = json.load(f)

    results, results_raw = load_cache(args)
    num_processed = len(results)

    logger.info(f"Number of questions: {len(samples) - num_processed}")

    results, results_raw = run_gpt(samples[num_processed:], args, results, results_raw)

    return results, results_raw
