import json
import os
from multiprocessing import Pool
from parser import ActivityParser

import editdistance
import networkx as nx
import numpy as np
import pandas as pd
import tiktoken
from loguru import logger
from numpy import dot
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
from thefuzz import fuzz
from tqdm import tqdm
from utils import graph_to_relation_set, partial_model_to_mermaid, soft_metrics

from abscon.abstraction import ActivityDiagramAbstractor
from abscon.concretization import (ActivityDiagramConcretizer,
                                   MajorityVotingConcretizer)
from abscon.statistics import RuntimeStatistics


class ActivityEvaluator:
    def __init__(
        self,
        folder_path: str,
        dataset_name: str,
        encoder: SentenceTransformer = None,
        data_folder: str = "data",
        seed: int = 42,
    ):
        self.result_dir = os.path.join(folder_path, dataset_name)

        # read the ground truth
        with open(os.path.join(data_folder, f"{dataset_name}.json")) as f:
            self.gt_graphs = json.load(f)

        self.abstractors = []
        self.encoder = encoder
        self.evaluation_cache = {}
        self.num_abstracted_candidates = 0
        self.seed = seed

        self.runtime_statistics = RuntimeStatistics()

    def reset_runtime_statistics(self):
        self.runtime_statistics = RuntimeStatistics()

    def evaluate_greedy_result(self) -> dict[str, float]:
        results_greedy = pd.read_csv(
            os.path.join(self.result_dir, "results_greedy.csv")
        )["0"].tolist()

        return self.evaluate_solutions(results_greedy)

    def combine_solutions(
        self,
        num_candidates,
        concretization_method: str = "solver",
        verbose: bool = False,
    ) -> list[str]:
        if self.num_abstracted_candidates > num_candidates:
            raise ValueError(
                f"{self.num_abstracted_candidates} has already been abstracted but {num_candidates} candidates are required. "  # noqa: E501
                "Cannot remove candidates from the abstractor. Create a new evaluator."
            )

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
                    abstractor = ActivityDiagramAbstractor(encoder=self.encoder)
                    self.abstractors.append(abstractor)

                abstractor = self.abstractors[i]
                parser = ActivityParser()
                for candidate in candidates[
                    self.num_abstracted_candidates : num_candidates  # noqa: E203
                ]:
                    graph = parser.parse(candidate)
                    abstractor.add_concrete_model(graph)

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
            if concretization_method == "solver":
                concretizer = ActivityDiagramConcretizer(seed=self.seed)
            else:
                concretizer = MajorityVotingConcretizer()
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

    def abstract_single_solution(self, args) -> ActivityDiagramAbstractor:
        abstractor, candidates = args
        parser = ActivityParser()
        for candidate in candidates[
            self.num_abstracted_candidates : len(candidates)  # noqa: E203
        ]:
            graph = parser.parse(candidate)
            abstractor.add_concrete_model(graph)
        return abstractor

    def combine_solutions_concurrent(
        self,
        num_candidates,
        concretization_method: str = "solver",
        num_processes: int = 16,
        verbose: bool = False,
    ) -> list[str]:
        if self.num_abstracted_candidates > num_candidates:
            raise ValueError(
                f"{self.num_abstracted_candidates} has already been abstracted but {num_candidates} candidates are required. "  # noqa: E501
                "Cannot remove candidates from the abstractor. Create a new evaluator."
            )

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

        logger.info("Abstracting")

        if len(self.abstractors) == 0:
            for i in range(len(result_candidates[0])):
                self.abstractors.append(ActivityDiagramAbstractor(encoder=self.encoder))

        if self.num_abstracted_candidates < num_candidates:
            all_candidates = []
            for i in range(len(result_candidates[0])):
                candidates = [
                    result_candidates[run][i] for run in range(num_candidates)
                ]
                all_candidates.append(candidates)

            logger.info(f"Number of abstractors {len(self.abstractors)}")
            logger.info(f"Number of samples {len(all_candidates)}")
            logger.info(f"Starting with {num_processes} processes")

            with Pool(processes=num_processes) as p:
                self.abstractors = list(
                    tqdm(
                        p.imap(
                            self.abstract_single_solution,
                            [
                                (abstractor, candidates)
                                for abstractor, candidates in zip(
                                    self.abstractors, all_candidates
                                )
                            ],
                        ),
                        total=len(all_candidates),
                        desc="Abstracting...",
                    )
                )

        # collect the runtime result
        for abstractor in self.abstractors:
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
            if concretization_method == "solver":
                concretizer = ActivityDiagramConcretizer(seed=self.seed)
            else:
                concretizer = MajorityVotingConcretizer()
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

    def evaluate_abstractor(
        self, concretization_method: str = "solver", verbose: bool = False
    ) -> dict[str, float]:
        concretized_results = self.combine_solutions(concretization_method, verbose)
        return self.evaluate_solutions(concretized_results)

    def evaluate_individual(self, runs, dataset, aggregator=np.mean):
        results = []
        for run in range(1, runs + 1):
            filename = f"results_{run}.csv"
            if filename not in self.evaluation_cache:
                graphs = pd.read_csv(os.path.join(self.result_dir, filename))[
                    "0"
                ].tolist()
                result = self.evaluate_solutions(graphs, return_value="all")
                results.append(result)
                self.evaluation_cache[filename] = result
            else:
                results.append(self.evaluation_cache[filename])

        metrics = ["recall", "precision", "f1", "consistency"]
        metric_result = {}
        for metric in metrics:
            metric_values = [
                aggregator([results[run][metric][sample_id] for run in range(runs)])
                for sample_id in range(len(results[0][metric]))
            ]
            metric_result[metric] = np.mean(metric_values)
        return metric_result

    def evaluate_solutions(
        self, predicted_results: list[str], return_value="avg", verbose: bool = False
    ) -> dict[str, float]:
        results = []
        similarity = TokenBasedSimilarity()
        parser = ActivityParser()
        results_progress = tqdm(predicted_results) if verbose else predicted_results

        for i, predicted_result in enumerate(results_progress):
            gt_result = self.gt_graphs[i]["mermaid_text"]
            predicted_graph = parser.parse(predicted_result)
            predicted = graph_to_relation_set(predicted_graph)
            gt = graph_to_relation_set(parser.parse(gt_result))
            evaluation_result = soft_metrics(
                predicted, gt, similarity.measure_similarity
            )
            consistency_result = evaluate_consistency(predicted_graph)
            evaluation_result["consistency"] = consistency_result["consistent"]
            evaluation_result["num_violations"] = consistency_result["num_violations"]
            results.append(evaluation_result)

        if return_value == "avg":
            return {
                "recall": sum(r["recall"] for r in results) / len(results),
                "precision": sum(r["precision"] for r in results) / len(results),
                "f1": sum(r["f1"] for r in results) / len(results),
                "consistency": sum([r["consistency"] for r in results]) / len(results),
            }
        else:
            return {
                "recall": [r["recall"] for r in results],
                "precision": [r["precision"] for r in results],
                "f1": [r["f1"] for r in results],
                "consistency": [r["consistency"] for r in results],
                "num_violations": [r["num_violations"] for r in results],
            }


class TokenBasedSimilarity:
    def __init__(self, tokenizer: str = "o200k_base"):
        self.enc = tiktoken.get_encoding(tokenizer)

    def measure_similarity(self, s1: str, s2: str) -> float:
        s1_tokens = set(self.enc.encode(s1.lower()))
        s2_tokens = set(self.enc.encode(s2.lower()))

        result = len(s1_tokens.intersection(s2_tokens)) / len(
            s1_tokens.union(s2_tokens)
        )

        return result


class TokenRatioSimilarity:
    def __init__(self, tokenizer: str = "o200k_base"):
        self.enc = tiktoken.get_encoding(tokenizer)

    def measure_similarity(self, s1: str, s2: str) -> float:
        s1_tokens = self.enc.encode(s1.lower())
        s2_tokens = self.enc.encode(s2.lower())

        # sm = difflib.SequenceMatcher(None, s1_tokens, s2_tokens)
        edit_distance = editdistance.eval(s1_tokens, s2_tokens)
        return 1 - edit_distance / max(len(s1_tokens), len(s2_tokens))


class ExactMatchSimilarity:
    def measure_similarity(self, s1: str, s2: str) -> float:
        return int(s1.lower() == s2.lower())


class StringEditDistanceSimilarity:
    def measure_similarity(self, s1: str, s2: str) -> float:
        s1 = s1.lower()
        s2 = s2.lower()
        return fuzz.ratio(s1, s2) / 100


class EmbeddingSimilarity:
    def __init__(self):
        # self.encoder = RemoteEncoder(url=os.getenv("ENCODER_URL"))
        self.cache = {}

    def cosine_similarity(self, a, b):
        return dot(a, b) / (norm(a) * norm(b))

    def measure_similarity(self, s1: str, s2: str) -> float:
        s1 = s1.lower()
        s2 = s2.lower()

        if s1 not in self.cache and s1 not in self.cache:
            embeddings = self.encoder.encode([s1, s2])
            self.cache[s1] = embeddings[0, :]
            self.cache[s1] = embeddings[1, :]
        elif s1 not in self.cache:
            embeddings = self.encoder.encode([s1])
            self.cache[s1] = embeddings[0, :]
        elif s2 not in self.cache:
            embeddings = self.encoder.encode([s2])
            self.cache[s2] = embeddings[0, :]

        s1_embedding, s2_embedding = self.cache[s1], self.cache[s2]

        return self.cosine_similarity(s1_embedding, s2_embedding)


def evaluate_consistency(graph: nx.DiGraph) -> dict[str, bool]:
    results = {}
    consistent = True
    num_violations = 0

    # 1. check if the graph contains more than one source
    source_nodes = []
    for node in graph.nodes:
        if graph.in_degree(node) == 0:
            source_nodes.append(node)
    single_source = len(source_nodes) == 1
    if len(source_nodes) > 1:
        num_violations = len(source_nodes) - 1

    results["single_source"] = single_source
    consistent = consistent and single_source

    # 2. Check if the source can reach all nodes
    if len(source_nodes) == 0:
        results["reachability"] = False
        consistent = False
    else:
        source = source_nodes[0]
        reachability = True
        for target in graph.nodes:
            if target == source:
                continue
            if not nx.has_path(graph, source, target):
                reachability = False
                num_violations += 1
        results["reachability"] = reachability
        consistent = consistent and reachability

    # 3. Decision should have at least two outgoing relations
    valid_decisions = True
    for source, target, data in graph.edges(data=True):
        if data.get("label", "") != "" and graph.out_degree(source) < 2:
            valid_decisions = False
            num_violations += 1
    results["valid_decisions"] = valid_decisions
    consistent = consistent and valid_decisions

    # 4. Decisions should have non-empty conditions
    valid_conditions = True

    decision_nodes = set()
    for source, target, data in graph.edges(data=True):
        if data.get("label", "") != "":
            decision_nodes.add(source)

    for node in decision_nodes:
        for _, _, data in graph.out_edges(node, data=True):
            if data.get("label", "") == "":
                valid_conditions = False
                break
        if not valid_conditions:
            num_violations += 1

    results["valid_conditions"] = valid_conditions
    consistent = consistent and valid_conditions

    # 5. No self-cycles
    no_self_cycles = True
    for node in graph.nodes:
        if (node, node) in graph.edges:
            no_self_cycles = False
            num_violations += 1

    results["no_self_cycles"] = no_self_cycles
    consistent = consistent and no_self_cycles

    results["consistent"] = consistent
    results["num_violations"] = num_violations
    return results
