import json
import os
from collections import Counter
from parser import ClevrParser

import pandas as pd
from loguru import logger
from program_executor import evaluate, programs_from_networkx, set_scene
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from utils import partial_model_to_mermaid

from abscon.abstraction import ClevrAbstractor
from abscon.concretization import ClevrConcretizer, MajorityVotingConcretizer
from abscon.statistics import RuntimeStatistics


def evaluate_graph_with_scene(mermaid_text: str, scene: dict, throw_error=False) -> str:
    parser = ClevrParser()
    graph = parser.parse(mermaid_text)

    try:
        program = programs_from_networkx(graph)
        set_scene(program, scene)
        result = evaluate(program)

        if type(result) is int:
            result = str(result)
        elif type(result) is bool:
            if result:
                result = "yes"
            else:
                result = "no"
        return result
    except Exception as e:
        if throw_error:
            raise e
        return "error"


def evaluate_prediction(
    predicted_answers: list[str], gt_answers: list[str], return_value="mean"
) -> dict[str, float]:
    errors = [
        1 if predicted_answer == "error" else 0
        for predicted_answer in predicted_answers
    ]

    corrects = [
        1 if predicted_answer == gt_answer else 0
        for predicted_answer, gt_answer in zip(predicted_answers, gt_answers)
    ]

    if return_value == "mean":
        return {
            "success_rate": 1 - sum(errors) / len(errors),
            "accuracy": sum(corrects) / len(corrects),
        }
    else:
        return {"success_rate": [1 - error for error in errors], "accuracy": corrects}


def get_most_frequent(items):
    data = Counter(items)
    return data.most_common(1)[0][0]


class ClevrEvaluator:
    def __init__(
        self,
        folder_path: str,
        dataset_name: str,
        data_folder: str = "data",
        scene_file: str = "scenes.json",
        seed: int = 42,
    ):
        self.result_dir = os.path.join(folder_path, dataset_name)

        # read ground_truth
        with open(os.path.join(data_folder, scene_file)) as f:
            self.scenes = json.load(f)["scenes"]

        with open(os.path.join(data_folder, f"{dataset_name}.json")) as f:
            self.questions = json.load(f)["questions"]

        self.num_abstracted_candidates = 0
        self.abstractors = []
        self.seed = seed

        self.runtime_statistics = RuntimeStatistics()

    def reset_runtime_statistics(self):
        self.runtime_statistics = RuntimeStatistics()

    def evaluate_greedy_result(self) -> dict[str, float]:
        results_greedy = pd.read_csv(
            os.path.join(self.result_dir, "results_greedy.csv")
        )["0"].tolist()

        return self.evaluate_solutions(results_greedy)

    def get_abstractor(self, candidates):
        abstractor = ClevrAbstractor(encoder=self.encoder)
        parser = ClevrParser()
        candidate_graphs = [parser.parse(text) for text in candidates]
        for candidate_graph in candidate_graphs:
            abstractor.add_concrete_model(candidate_graph)
        return abstractor

    def combine_solutions(
        self,
        num_candidates: int,
        encoder: SentenceTransformer,
        concretization_method: str = "solver",
        verbose: bool = False,
    ) -> list[str]:
        if self.num_abstracted_candidates > num_candidates:
            raise ValueError(
                f"{self.num_abstracted_candidates} has already been abstracted but {num_candidates} candidates are required. "  # noqa: E501
                "Cannot remove candidates from the abstractor. Create a new evaluator."
            )

        self.encoder = encoder
        # read candidates for self-consistency
        result_candidates = []
        for run in range(num_candidates):
            result_candidates.append(
                pd.read_csv(os.path.join(self.result_dir, f"results_{run + 1}.csv"))[
                    "0"
                ].tolist()
            )

        iter_list = (
            tqdm(range(len(result_candidates[0])))
            if verbose
            else range(len(result_candidates[0]))
        )

        if concretization_method == "solver":
            concretizer = ClevrConcretizer(seed=self.seed)
        else:
            concretizer = MajorityVotingConcretizer()

        logger.info("Abstracting")

        if (
            len(self.abstractors) == 0
            or self.num_abstracted_candidates < num_candidates
        ):
            for i in iter_list:
                candidates = [
                    result_candidates[run][i] for run in range(num_candidates)
                ]

                if len(self.abstractors) <= i:
                    abstractor = ClevrAbstractor(encoder=encoder)
                    self.abstractors.append(abstractor)

                abstractor = self.abstractors[i]
                parser = ClevrParser()
                for candidate_graph in candidates[
                    self.num_abstracted_candidates : num_candidates  # noqa: E203
                ]:
                    candidate_graph = parser.parse(candidate_graph)
                    abstractor.add_concrete_model(candidate_graph)

                # Record the runtime statistics for abstraction
                self.runtime_statistics.embedding_runtime.append(
                    abstractor.embedding_runtime
                )
                self.runtime_statistics.graph_matching_runtime.append(
                    abstractor.graph_matching_runtime
                )

        logger.info("Concretizing")
        iter_list = tqdm(self.abstractors) if verbose else self.abstractors
        concretized_results = []
        for abstractor in iter_list:
            concretized_graph = concretizer.concretize(abstractor.get_partial_model())
            concretized_results.append(partial_model_to_mermaid(concretized_graph))

            # Record the runtime statistics for concretization
            if concretization_method == "solver":
                self.runtime_statistics.problem_build_runtimes.append(
                    concretizer.problem_definition_runtime
                )
                self.runtime_statistics.problem_solve_runtimes.append(
                    concretizer.problem_solve_runtime
                )

        self.num_abstracted_candidates = num_candidates

        return concretized_results

    def evaluate_execution_sc(
        self,
        num_candidates,
        exclude_error=False,
        return_answer=False,
        best_answer=False,
    ):
        # read candidates for self-consistency
        results = []
        for run in range(num_candidates):
            results.append(
                pd.read_csv(os.path.join(self.result_dir, f"results_{run + 1}.csv"))[
                    "0"
                ].tolist()
            )

        # transpose the results
        result_candidates = [[] for _ in range(len(results[0]))]
        for run in range(num_candidates):
            for idx in range(len(result_candidates)):
                result_candidates[idx].append(results[run][idx])

        individual_results = []
        gt_answers = [question["answer"] for question in self.questions]
        for i, candidates in enumerate(result_candidates):
            individual_result = []
            for mermaid_text in candidates:
                scene = self.scenes[self.questions[i]["image_index"]]
                result = evaluate_graph_with_scene(mermaid_text, scene)

                if type(result) is list:
                    result = str(result)

                if not exclude_error or (exclude_error and result != "error"):
                    individual_result.append(result)
            individual_results.append(individual_result)

        predicted_answers = []
        for i, candidates in enumerate(individual_results):
            if len(candidates) == 0:
                candidates = ["error"]
            if best_answer and gt_answers[i] in candidates:
                predicted_answers.append(gt_answers[i])
            else:
                predicted_answers.append(get_most_frequent(candidates))

        if not return_answer:
            return evaluate_prediction(predicted_answers, gt_answers)
        else:
            return predicted_answers, evaluate_prediction(predicted_answers, gt_answers)

    def evaluate_solutions(
        self, predicted_results: list[str], return_value="mean"
    ) -> dict[str, float]:
        gt_answers = []
        predicted_answers = []

        for question, result in zip(self.questions, predicted_results):
            scene = self.scenes[question["image_index"]]
            gt_answers.append(question["answer"])
            result = evaluate_graph_with_scene(result, scene)
            predicted_answers.append(result)

        return evaluate_prediction(predicted_answers, gt_answers, return_value)
