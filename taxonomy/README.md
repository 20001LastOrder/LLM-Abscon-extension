# Taxonomy Construction

This use case builds taxonomies for the WordNet and CCS datasets with AbsCon. Original and processed data are stored in `data/`.

The commands below assume that you are in the `taxonomy/` directory unless they explicitly use the repository-level `run_generation.py`.

## Prepare the datasets

Sample the full WordNet data with:

```bash
python -m scripts.sample_wordnet --input_path data/wordnet_full.csv --output_path data/wordnet.csv
```

The filtered CCS dataset contains 75 taxonomies, each with fewer than 70 terms.

## Generate candidates

Candidate generation uses the top-level script, so run this command from the repository root:

```bash
python run_generation.py --dataset ccs --output_suffix 1 --llm_type regular --llm_name gpt-4o-mini --temperature 0.7
```

Configure your OpenAI-compatible endpoint in the root `.env` file before running the command. The relevant options are:

- `--dataset`: `ccs` or `wordnet`
- `--output_suffix`: usually `1` through `20` for candidate runs; use a descriptive name for direct generation
- `--llm_type`: `regular` or `reasoning`; reasoning-model `<think>` content is retained in the raw output
- `--llm_name`: the model name exposed by the configured endpoint
- `--temperature`: typically `0.7` for candidates and `0.01` for direct generation

## Merge candidates with AbsCon

From the `taxonomy/` directory, merge 10 CCS candidates with:

```bash
python -m scripts.generate_abscon --folder_path results/gpt-4o-mini --dataset_name ccs --num_generations 10
```

The required options are `--folder_path`, `--dataset_name` (`ccs` or `wordnet`), and `--num_generations`. Add `--save_partial_models` to save the abstracted models as `partial_models_<n>.pkl` for runtime analysis.

Along with the majority-vote and AbsCon outputs, the script records a phase-by-phase runtime breakdown in `runtime_abscon_<n>.csv`. The loop in `scripts/generate_abscon.sh` shows how to process candidate counts 1 through 20.

## Run the measurements

From `taxonomy/`, measure how results change with the number of candidates:

```bash
python -m measurements.impact_of_candidates
```

Results are written to `../measurements/results/taxonomy/candidates_<dataset>.json`. Use `--llms` to select models and `--num_generations` to select a range such as `1,20`.

For the temperature study, first generate candidates under `results/temperature/<temperature>/<llm>/<dataset>` and run `scripts.generate_abscon` for each set. Then run:

```bash
python -m measurements.impact_of_temperature --temperatures 0.1,0.4,0.7,1.0
```

This writes `../measurements/results/taxonomy/temperature_<dataset>.json`.

## Analysis notebooks

- `evaluation.ipynb` compares direct generation, majority vote, and AbsCon (RQ1), then studies candidate count (RQ2) and temperature (RQ3).
- `influence_of_consistency.ipynb` compares consistent and inconsistent candidates with a Wilcoxon rank-sum test and Cliff's delta.
