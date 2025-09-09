from argparse import ArgumentParser
from activity.run_generation import generate_activity_diagrams
from programs.run_generation import generate_program_diagrams
from dotenv import load_dotenv
from loguru import logger
import sys
from abscon.utils import serialize_output
from taxonomy.run_generation import generate_taxonomies


load_dotenv()


logger.remove()
logger.add(sys.stderr, level="INFO")


def main(args):
    if args.dataset == "paged":
        root_folder = "activity"
        generation_function = generate_activity_diagrams
    elif args.dataset == "clevr":
        root_folder = "programs"
        generation_function = generate_program_diagrams
    elif args.dataset in ["wordnet", "ccs"]:
        root_folder = "taxonomy"
        generation_function = generate_taxonomies
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    args.input_folder = f"{root_folder}/{args.input_folder}"
    args.output_folder = f"{root_folder}/{args.output_folder}"
    results, results_raw = generation_function(args)

    serialize_output(results=results, results_raw=results_raw, args=args)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--input_folder", default="data")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["paged", "clevr", "wordnet", "ccs"],
        required=True,
    )
    parser.add_argument("--output_folder", default="results")
    parser.add_argument("--output_suffix", type=str, required=True)
    parser.add_argument(
        "--prompt_type", type=str, default="fewshot", choices=["fewshot", "simple"]
    )
    parser.add_argument("--llm_type", choices=["regular", "reasoning"], required=True)
    parser.add_argument("--llm_name", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--num_processes", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)

    args = parser.parse_args()
    main(args)
