from argparse import ArgumentParser
import json
import os

import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

plt.style.use(['science', "ieee"])

USE_CASES = {
    "activity": {
        "dataset": "paged",
        "title": "Paged",
        "metrics": [("f1", "F1"), ("consistency", "Consistency")],
    },
    "programs": {
        "dataset": "clevr",
        "title": "Clevr",
        "metrics": [("accuracy", "Accuracy"), ("success_rate", "Success Rate")],
    },
    "taxonomy": {
        "dataset": "wordnet",
        "title": "WordNet",
        "metrics": [("f1", "F1"), ("consistency", "Consistency")],
    },
}

models = ["llama-3.1-8b-instruct", "llama-3.1-70b-instruct"]
model_names = ["Llama3.1 8b", "Llama3.1 70b"]
lines = ["-", "-"]
markers = ['*', '.']

colors = [[33, 25, 24], [195, 56, 40]]
colors = [[c / 255 for c in color] for color in colors]


def main(args):
    os.makedirs(args.output_path, exist_ok=True)
    x = [float(temperature) for temperature in args.temperatures]

    for use_case, config in USE_CASES.items():
        result_path = os.path.join(
            args.results_path, use_case, f"temperature_{config['dataset']}.json"
        )
        with open(result_path) as f:
            results = json.load(f)

        for metric, metric_name in config["metrics"]:
            plt.figure(figsize=(4, 1.5))
            for i, llm in enumerate(models):
                values = [
                    results[temperature][llm][metric]
                    for temperature in args.temperatures
                ]
                plt.plot(
                    x,
                    values,
                    color=colors[i],
                    linestyle=lines[i],
                    label=model_names[i],
                    marker=markers[i],
                )
            plt.legend(shadow=True, ncol=2)
            plt.title(config["title"])
            plt.ylabel(metric_name)
            plt.xlabel("Temperature")
            plt.savefig(
                os.path.join(args.output_path, f"{config['dataset']}_{metric}.png"),
                dpi=300,
            )
            plt.close()


def delimited_list(s: str) -> list[str]:
    return s.split(",")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--results_path", type=str, default="measurements/results")
    parser.add_argument(
        "--temperatures", type=delimited_list, default="0.1,0.4,0.7,1.0"
    )
    parser.add_argument(
        "--output_path", type=str, default="measurements/plots/temperature"
    )

    args = parser.parse_args()

    main(args)
