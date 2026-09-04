from argparse import ArgumentParser
import json
import sys

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")


def main(args):
    combined = None
    for input_file in args.input_files:
        logger.info(f"Reading {input_file}")
        with open(input_file) as f:
            results = json.load(f)

        if combined is None:
            combined = [{} for _ in results]
        elif len(results) != len(combined):
            raise ValueError(
                f"{input_file} has {len(results)} entries "
                f"but previous files have {len(combined)}"
            )

        for entry, combined_entry in zip(results, combined):
            for approach, llm_metrics in entry.items():
                for llm in llm_metrics:
                    if llm in combined_entry.get(approach, {}):
                        raise ValueError(
                            f"{llm} appears in multiple input files for {approach}"
                        )
                combined_entry.setdefault(approach, {}).update(llm_metrics)

    with open(args.output_file, "w") as f:
        json.dump(combined, f)
    logger.info(f"Saved combined results to {args.output_file}")


def delimited_list(s: str) -> list[str]:
    return s.split(",")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--input_files",
        type=delimited_list,
        default="measurements/results/activity/candidates_paged_8b.json,"
        "measurements/results/activity/candidates_paged_70b.json,"
        "measurements/results/activity/candidates_paged_qwq.json",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="measurements/results/activity/candidates_paged.json",
    )

    args = parser.parse_args()

    main(args)
