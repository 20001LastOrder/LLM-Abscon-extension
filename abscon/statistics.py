from pydantic import Field
from pydantic.dataclasses import dataclass


@dataclass
class RuntimeStatistics:
    """
    A class to store runtime statistics for various components of the evaluation process.

    Attributes:
        embedding_runtime (list[list[float]]): A list of lists containing runtime statistics for embeddings
            of each candidate when adding to the partial model
        graph_matching_runtime (list[list[float]]): A list of lists containing runtime statistics for graph
            matching of each candidate when adding to the partial model
        problem_build_runtimes (list[float]): A list containing the runtime statistics for building the
            optimization problem for concretization
        problem_solve_runtimes (list[float]): A list containing the runtime statistics for solving the
            optimization problem
    """  # noqa: E501

    embedding_runtime: list[list[float]] = Field(default_factory=list)
    graph_matching_runtime: list[list[float]] = Field(default_factory=list)
    problem_build_runtimes: list[float] = Field(default_factory=list)
    problem_solve_runtimes: list[float] = Field(default_factory=list)
