# LLM-AbsCon

This repository contains the code and data for **“Characteristics of Constraint-Aware Self-Consistency for LLM-based Graph Model Generation.”** It extends **“Accurate and Consistent Graph Model Generation from Text with Large Language Models,”** published at the ACM/IEEE 28th International Conference on Model Driven Engineering Languages and Systems (MODELS 2025).

AbsCon combines several candidate outputs from a large language model into one graph-based model while enforcing constraints from the target domain. The repository covers three use cases: activity diagrams, executable programs, and taxonomies.

## Repository layout

- [`abscon`](abscon): shared implementation of the AbsCon approach
- [`activity`](activity): activity-diagram generation with the PAGED dataset
- [`programs`](programs): executable-program generation with the CLEVR dataset
- [`taxonomy`](taxonomy): taxonomy construction with the WordNet and CCS datasets
- [`measurements`](measurements): analysis shared across the three use cases

Each use-case folder has its own README with preparation, generation, and evaluation instructions.

## What the extension adds

The extension builds on the MODELS 2025 version with:

- **Reasoning models.** `run_generation.py` accepts `--llm_type regular` or `--llm_type reasoning`. Reasoning models such as QwQ-32B are called through `ChatDeepSeek`, which preserves their `<think>` blocks in the raw output. Existing QwQ-32B results are under `<use_case>/results/qwq-32b`.
- **Prompt variants and retries.** The flowchart and taxonomy generators support `--prompt_type fewshot` and `--prompt_type simple`. Use `--tolerance` to control how many times an unparsable response is retried.
- **Temperature experiments (RQ3).** Candidate sets for temperatures 0.1, 0.4, 0.7, 1.0, and 1.3 live under `<use_case>/results/temperature/<temperature>/<llm>/`. Each use case includes `measurements/impact_of_temperature.py`; the shared `measurements/plot_temperature.py` script plots the results.
- **Runtime analysis.** Each `scripts/generate_abscon.py` records the time spent on embedding, graph matching, problem construction, and problem solving in `runtime_abscon_<n>.csv`. Add `--save_partial_models` to keep the abstracted models. See `measurements/runtime/analysis.ipynb` for the analysis.
- **Consistency analysis.** Each use case includes `influence_of_consistency.ipynb`, which compares consistent and inconsistent candidates with a Wilcoxon rank-sum test and Cliff's delta.
- **Program-generation baselines.** The program use case adds execution-based self-consistency (`esc`), a version that filters programs that fail to run (`escf`), and an oracle upper bound (`best`).
- **The native Mermaid parser.** `abscon/mermaid` uses Mermaid's JavaScript parser from Python through PythonMonkey. Build instructions are in [`abscon/js/README.md`](abscon/js/README.md).
- **Reasoning-token measurements.** `measurements/measure_response_tokens.py` counts response tokens with and without the `<think>` block.

## Setup

The project requires Python 3.10 or later and uses [Poetry](https://python-poetry.org/):

```bash
poetry install
```

Copy `.env_template` to `.env`, then add the settings for your OpenAI-compatible endpoint:

- `OPENAI_BASE_URL`
- `OPENAI_API_KEY`
- `OPENAI_PROXY`, if your setup requires one

Both regular and reasoning models use this endpoint.

## Generate candidates

Run the top-level generator from the repository root. It chooses the correct use-case folder from the dataset name.

```bash
python run_generation.py --dataset paged --output_suffix 1 --llm_type regular --llm_name gpt-4o-mini --temperature 0.7
python run_generation.py --dataset clevr --output_suffix 1 --llm_type reasoning --llm_name qwq-32b --temperature 0.7
```

The main options are:

- `--dataset`: `paged`, `clevr`, `wordnet`, or `ccs`
- `--output_suffix`: usually `1` through `20` for candidate runs; use a descriptive name for a direct-generation run
- `--llm_type`: `regular` or `reasoning`
- `--llm_name`: the model name exposed by your configured endpoint
- `--temperature`: typically `0.7` for candidate generation and `0.01` for direct generation
- `--prompt_type`: `fewshot` (the default) or `simple`
- `--output_folder`: output folder relative to the selected use case; defaults to `results`
- `--num_processes` and `--batch_size`: generation parallelism and batching
- `--tolerance`: maximum number of retries for an unparsable response; defaults to `5`

For a temperature experiment, an output folder might be `results/temperature/0.4`; the generator adds the model and dataset directories itself.

Generated models are written to:

```text
<use_case>/<output_folder>/<llm_name>/<dataset>/results_<output_suffix>.csv
```

The corresponding unprocessed model responses are saved as `results_<output_suffix>_raw.csv` in the same directory.
