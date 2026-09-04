# Activity Diagram Generation

This use case generates activity diagrams from the PAGED dataset with AbsCon. Original and processed data are stored in `data/`.

The commands below assume that you are in the `activity/` directory unless they explicitly use the repository-level `run_generation.py`.

## Prepare the dataset

Create the sampled PAGED dataset with:

```bash
python -m scripts.sample_dataset
```

Run `python -m scripts.sample_dataset --help` to see the input, output, and sampling options.

## Generate candidates

Candidate generation uses the top-level script, so run this command from the repository root:

```bash
python run_generation.py --dataset paged --output_suffix 1 --llm_type regular --llm_name gpt-4o-mini --temperature 0.7
```

Configure your OpenAI-compatible endpoint in the root `.env` file before running the command. The relevant options are:

- `--output_suffix`: usually `1` through `20` for candidate runs; use a descriptive name for direct generation
- `--dataset`: `paged`
- `--llm_type`: `regular` or `reasoning`; reasoning-model `<think>` content is retained in the raw output
- `--llm_name`: the model name exposed by the configured endpoint
- `--temperature`: typically `0.7` for candidates and `0.01` for direct generation
- `--output_folder`: output directory relative to `activity/`; defaults to `results`
- `--prompt_type`: `fewshot` (the default) or `simple`
- `--tolerance`: maximum retries for an unparsable response; defaults to `5`

## Merge candidates with AbsCon

From the `activity/` directory, merge 10 candidates with:

```bash
python -m scripts.generate_abscon --folder_path results/gpt-4o-mini --num_candidates_start 10 --num_candidates_end 10
```

Useful options include:

- `--folder_path`: directory containing the model's candidate files
- `--num_candidates_start`: first candidate count to process
- `--num_candidates_end`: last candidate count to process, inclusive; pass the same value as `--num_candidates_start` to process a single count
- `--seed`: random seed used by the approximation algorithms
- `--save_partial_models`: save each abstracted model as `partial_models_<n>.pkl` for runtime analysis
- `--num_processes`: number of abstraction workers

Along with the majority-vote and AbsCon outputs, the script records a phase-by-phase runtime breakdown in `runtime_abscon_<n>.csv`. The batch command in `scripts/generate_abscon.sh` shows how to process candidate counts 1 through 20.

## Run the measurements

From `activity/`, measure how results change with the number of candidates:

```bash
python -m measurements.impact_of_candidates
```

The result is written to `../measurements/results/activity/candidates_paged.json`. Use `--llms` to select models and `--num_generations` to select a range such as `1,20`.

For the temperature study, first generate candidates under `results/temperature/<temperature>/<llm>/paged` and run `scripts.generate_abscon` for each set. Then run:

```bash
python -m measurements.impact_of_temperature --temperatures 0.1,0.4,0.7,1.0
```

This writes `../measurements/results/activity/temperature_paged.json`.

## Analysis notebooks

- `evaluation.ipynb` compares direct generation, majority vote, and AbsCon (RQ1), then studies candidate count (RQ2) and temperature (RQ3).
- `influence_of_consistency.ipynb` compares consistent and inconsistent candidates with a Wilcoxon rank-sum test and Cliff's delta.
