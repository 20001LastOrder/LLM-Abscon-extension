# Measurements and Analysis

This folder contains analyses that span all three use cases. Each use case produces its own measurements with `<use_case>/measurements/impact_of_candidates.py` and `impact_of_temperature.py`; the scripts and notebooks here combine or visualize those results.

## Files and directories

- `results/<use_case>/candidates_<dataset>.json`: metrics for candidate counts 1 through 20 (RQ2)
- `results/<use_case>/temperature_<dataset>.json`: AbsCon metrics at each sampling temperature (RQ3)
- `results/token_counts_qwq-32b.csv`: response-token counts for the reasoning model
- `runtime/data/<use_case>/`: per-model runtime breakdowns (`<llm>_runtime_abscon_10.csv`) and saved partial models (`<llm>_partial_models_10.pkl`)
- `plots/temperature/`: figures created by `plot_temperature.py`

## Combine results from separate model runs

You can run a use case's `impact_of_candidates.py` separately for each model. `combine_candidates.py` then merges those JSON files into one. It stops with an error if the same model appears in more than one input, which prevents one run from silently overwriting another.

From the repository root:

```bash
python measurements/combine_candidates.py \
  --input_files measurements/results/activity/candidates_paged_8b.json,measurements/results/activity/candidates_paged_70b.json,measurements/results/activity/candidates_paged_qwq.json \
  --output_file measurements/results/activity/candidates_paged.json
```

## Plot the temperature study

```bash
python measurements/plot_temperature.py --temperatures 0.1,0.4,0.7,1.0
```

By default, the script reads `measurements/results/<use_case>/temperature_<dataset>.json` for all three use cases and writes the figures to `measurements/plots/temperature/`.

## Measure reasoning-model response tokens

This script counts tokens in raw reasoning-model responses both with and without the `<think>` block. It also estimates generation time from a supplied throughput in tokens per second.

```bash
python measurements/measure_response_tokens.py --llm qwq-32b --tokenizer Qwen/QwQ-32B --num_runs 10 --throughput 40
```

It requires the `transformers` package and writes its output to `measurements/results/token_counts_<llm>.csv` by default.

## Notebooks

- `rq2.ipynb` plots the effect of candidate count across all use cases using `results/<use_case>/candidates_<dataset>.json`.
- `runtime/analysis.ipynb` breaks AbsCon runtime into embedding, graph matching, problem construction, and problem solving. It also relates total runtime to the number of decisions (entropy) in each partial model. Figures are saved beside the notebook.
