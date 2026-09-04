# Program Induction

This use case generates executable programs for the CLEVR dataset with AbsCon. Original and processed data are stored in `data/`.

The commands below assume that you are in the `programs/` directory unless they explicitly use the repository-level `run_generation.py`.

## Prepare the dataset

Create the sampled dataset with:

```bash
python -m scripts.sample_dataset
```

Run `python -m scripts.sample_dataset --help` to see the input, output, and sampling options.

## Generate candidates

Candidate generation uses the top-level script, so run this command from the repository root:

```bash
python run_generation.py --dataset clevr --output_suffix 1 --llm_type regular --llm_name gpt-4o-mini --temperature 0.7
```

Configure your OpenAI-compatible endpoint in the root `.env` file before running the command. The relevant options are:

- `--output_suffix`: usually `1` through `20` for candidate runs; use a descriptive name for direct generation
- `--dataset`: `clevr`
- `--llm_type`: `regular` or `reasoning`; reasoning-model `<think>` content is retained in the raw output
- `--llm_name`: the model name exposed by the configured endpoint
- `--temperature`: typically `0.7` for candidates and `0.01` for direct generation

## Merge candidates with AbsCon

From the `programs/` directory, merge 10 candidates from `gpt-4o-mini` with:

```bash
python -m scripts.generate_abscon --folder_path results/gpt-4o-mini --num_candidates_start 10 --num_candidates_end 10
```

Useful options include:

- `--folder_path`: directory containing the model's candidate files
- `--num_candidates_start`: first candidate count to process
- `--num_candidates_end`: last candidate count to process, inclusive; use `-1` or pass the starting count again to process a single count
- `--seed`: random seed used by the approximation algorithms
- `--save_partial_models`: save each abstracted model as `partial_models_<n>.pkl` for runtime analysis

Along with the majority-vote and AbsCon outputs, the script records a phase-by-phase runtime breakdown in `runtime_abscon_<n>.csv`. The batch command in `scripts/generate_abscon.sh` shows how to process candidate counts 1 through 20.

## Run the measurements

From `programs/`, measure how results change with the number of candidates:

```bash
python -m measurements.impact_of_candidates
```

The result is written to `../measurements/results/programs/candidates_clevr.json`. Use `--llms` to select models, `--num_generations` to select a range such as `1,20`, and `--approaches` to choose the methods being compared.

The default approaches are:

- `mv`: majority vote over candidate programs
- `greedy`: direct generation
- `abscon`: AbsCon
- `esc`: majority vote over the candidates' execution results
- `escf`: execution-based self-consistency after filtering programs that fail to run
- `best`: an oracle upper bound that returns the correct answer whenever any candidate produces it

For the temperature study, first generate candidates under `results/temperature/<temperature>/<llm>/clevr` and run `scripts.generate_abscon` for each set. Then run:

```bash
python -m measurements.impact_of_temperature --temperatures 0.1,0.4,0.7,1.0
```

This writes `../measurements/results/programs/temperature_clevr.json`.

## Analysis notebooks

- `evaluation.ipynb` compares direct generation, majority vote, execution-based self-consistency, and AbsCon (RQ1), then studies candidate count (RQ2) and temperature (RQ3).
- `influence_of_consistency.ipynb` compares consistent and inconsistent candidates with a Wilcoxon rank-sum test and Cliff's delta.

## Reproducibility notes

Some AbsCon steps use approximation algorithms. The script sets a seed so that ties and other ambiguous cases are handled consistently.

Abstraction and concretization also have time limits. On a slower or faster machine, the solver may occasionally return a different best-so-far result before the limit. These cases should be uncommon and are not expected to materially affect the reported results.
