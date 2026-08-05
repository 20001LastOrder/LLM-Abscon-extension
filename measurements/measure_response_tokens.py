import os
import re
from argparse import ArgumentParser

import pandas as pd
from loguru import logger
from tqdm import tqdm
from transformers import AutoTokenizer

USE_CASE_DATASETS = {
    "activity": "paged",
    "programs": "clevr",
    "taxonomy": "wordnet",
}

THINK_PATTERN = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def measure_run(tokenizer, path: str) -> dict:
    responses = pd.read_csv(path)["0"].tolist()
    tokens_with_think = []
    tokens_without_think = []
    num_missing_think = 0
    for response in responses:
        response = str(response)
        tokens_with_think.append(count_tokens(tokenizer, response))
        stripped, num_replaced = THINK_PATTERN.subn("", response)
        if num_replaced == 0:
            num_missing_think += 1
        tokens_without_think.append(count_tokens(tokenizer, stripped.strip()))
    return {
        "n_responses": len(responses),
        "tokens_with_think": tokens_with_think,
        "tokens_without_think": tokens_without_think,
        "num_missing_think": num_missing_think,
    }


def make_row(
    use_case: str,
    run: str,
    tokens_with_think: list[int],
    tokens_without_think: list[int],
    num_missing_think: int,
    throughput: float,
) -> dict:
    total_with_think = sum(tokens_with_think)
    total_without_think = sum(tokens_without_think)
    return {
        "use_case": use_case,
        "run": run,
        "n_responses": len(tokens_with_think),
        "avg_tokens_with_think": pd.Series(tokens_with_think).mean(),
        "avg_tokens_without_think": pd.Series(tokens_without_think).mean(),
        "total_tokens_with_think": total_with_think,
        "total_tokens_without_think": total_without_think,
        "avg_time_with_think_s": pd.Series(tokens_with_think).mean() / throughput,
        "avg_time_without_think_s": pd.Series(tokens_without_think).mean() / throughput,
        "num_missing_think": num_missing_think,
    }


def main(args):
    logger.info(f"Loading tokenizer {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    rows = []
    for use_case, dataset in USE_CASE_DATASETS.items():
        all_with_think = []
        all_without_think = []
        total_missing_think = 0
        for run in tqdm(range(1, args.num_runs + 1), desc=use_case):
            path = os.path.join(
                args.root, use_case, "results", args.llm, dataset, f"results_{run}_raw.csv"
            )
            run_result = measure_run(tokenizer, path)
            all_with_think.extend(run_result["tokens_with_think"])
            all_without_think.extend(run_result["tokens_without_think"])
            total_missing_think += run_result["num_missing_think"]
            rows.append(
                make_row(
                    use_case,
                    str(run),
                    run_result["tokens_with_think"],
                    run_result["tokens_without_think"],
                    run_result["num_missing_think"],
                    args.throughput,
                )
            )
        rows.append(
            make_row(
                use_case,
                "overall",
                all_with_think,
                all_without_think,
                total_missing_think,
                args.throughput,
            )
        )

    df = pd.DataFrame(rows)
    os.makedirs(args.output_path, exist_ok=True)
    output_file = os.path.join(args.output_path, f"token_counts_{args.llm}.csv")
    df.to_csv(output_file, index=False)
    logger.info(f"Saved per-run results to {output_file}")

    summary = df[df["run"] == "overall"]
    print("\nAverage tokens per raw response (runs 1-{}):".format(args.num_runs))
    print(
        summary[
            ["use_case", "n_responses", "avg_tokens_with_think", "avg_tokens_without_think"]
        ].to_string(index=False)
    )
    print(f"\nTime estimates at {args.throughput} tokens/s:")
    print(
        summary[
            ["use_case", "avg_time_with_think_s", "avg_time_without_think_s"]
        ].to_string(index=False)
    )
    if summary["num_missing_think"].sum() > 0:
        print("\nResponses without <think></think> tags:")
        print(summary[["use_case", "num_missing_think"]].to_string(index=False))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--llm", type=str, default="qwq-32b")
    parser.add_argument("--tokenizer", type=str, default="Qwen/QwQ-32B")
    parser.add_argument("--num_runs", type=int, default=10)
    parser.add_argument("--throughput", type=float, default=40.0)
    parser.add_argument("--output_path", type=str, default="measurements/results")

    args = parser.parse_args()

    main(args)
