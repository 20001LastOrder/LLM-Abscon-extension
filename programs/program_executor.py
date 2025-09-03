from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Literal, Optional, Type

import networkx as nx
from loguru import logger
from pydantic import BaseModel, Field


class Property(Enum):
    color = "color"
    size = "size"
    shape = "shape"
    material = "material"


class ProgramNode(ABC, BaseModel):
    output: Any = None
    previous: list[ProgramNode] = Field(default=[], repr=False)
    next: list[ProgramNode] = Field(default=[], repr=False)
    scene: dict = Field(default=None, repr=False)

    @abstractmethod
    def execute(self):
        pass

    @classmethod
    @abstractmethod
    def is_type(cls, name: str) -> bool:
        """
        Check if a name is belong to this type of program node

        Args:
            name (str): The name to check
        Returns:
            bool: True if the name should be considered as this type of node, False otherwise
        """  # noqa: E501
        pass

    @classmethod
    @abstractmethod
    def from_name(cls, name: str) -> Optional[ProgramNode]:
        """
        Create a program node from the name. The node will only be created if `cls.is_type(name)` is true

        Args:
            name (str): the name to check
        Returns:
            Optional[ProgramNode]: the created program node from the name if `cls.is_type(name)` is true,
            None otherwise.
        """  # noqa: E501
        pass

    @property
    @abstractmethod
    def label(self) -> str:
        pass


class Scene(ProgramNode):
    output: list[int] = None

    def execute(self):
        assert len(self.previous) == 0, "Scene must does not have any previous programs"
        self.output = list(range(len(self.scene["objects"])))

    @classmethod
    def is_type(cls, name: str) -> bool:
        return name == "scene"

    @classmethod
    def from_name(cls, name: str) -> Optional[Scene]:
        if not (cls.is_type(name)):
            return None
        return cls()

    @property
    def label(self) -> str:
        return "scene"


class Unique(ProgramNode):
    output: int = None

    def execute(self):
        assert len(self.previous) == 1, "Unique must have exactly one previous program"

        previous_program = self.previous[0]
        assert isinstance(
            previous_program.output, list
        ), "The output of the previous program for Unique must be a list"
        # assert (
        #     len(previous_program.output) <= 1
        # ), "The list of objects in the list is not unique"

        if len(previous_program.output) > 1:
            logger.debug("The list of objects in the list is not unique")

        if len(previous_program.output) > 0:
            self.output = previous_program.output[0]
        else:
            self.output = -1

    @classmethod
    def is_type(cls, name: str) -> bool:
        return name == "unique"

    @classmethod
    def from_name(cls, name: str) -> Optional[Unique]:
        if not cls.is_type(name):
            return None
        return cls()

    @property
    def label(self) -> str:
        return "unique"


class Relate(ProgramNode):
    name: str = "relate"
    output: list[int] = None
    value: Literal["left", "right", "in_front", "behind"]

    def execute(self):
        assert len(self.previous) == 1, "Relate must have exactly one previous program"
        assert isinstance(
            self.previous[0], Unique
        ), "Relate must follow right after unique"

        obj_idx = self.previous[0].output

        if self.value == "in_front":
            value = "front"
        else:
            value = self.value

        if obj_idx >= 0:
            self.output = self.scene["relationships"][value][obj_idx]
        else:
            self.output = []

    @classmethod
    def is_type(cls, name: str) -> bool:
        values = name.split("_", 1)
        return (
            len(values) == 2
            and values[0] == "relate"
            and values[1] in ["left", "right", "in_front", "behind"]
        )

    @classmethod
    def from_name(cls, name: str) -> Optional[Relate]:
        if not cls.is_type(name):
            return None

        _, value = name.split("_", 1)
        return cls(value=value)

    @property
    def label(self) -> str:
        return f"relate_{self.value}"


class Count(ProgramNode):
    output: int = None

    def execute(self):
        assert len(self.previous) == 1, "Count must have exactly one previous program"

        previous_program = self.previous[0]
        assert isinstance(
            previous_program.output, list
        ), "The output of the previous program for Count must be a list"

        self.output = len(previous_program.output)

    @classmethod
    def is_type(cls, name: str) -> bool:
        return name == "count"

    @classmethod
    def from_name(cls, name: str) -> Optional[Count]:
        if not cls.is_type(name):
            return None
        return Count()

    @property
    def label(self) -> str:
        return "count"


class Exist(ProgramNode):
    output: bool = None

    def execute(self):
        assert len(self.previous) == 1, "Exist must have exactly one previous program"

        previous_program = self.previous[0]
        assert isinstance(
            previous_program.output, list
        ), "The output of the previous program for Exist must be a list"

        self.output = len(previous_program.output) != 0

    @classmethod
    def is_type(cls, name: str) -> bool:
        return name == "exist"

    @classmethod
    def from_name(cls, name: str) -> Optional[Exist]:
        return Exist()

    @property
    def label(self) -> str:
        return "exist"


class Filter(ProgramNode):
    output: list[int] = None
    property: Property
    value: str

    def execute(self):
        if self.property == Property.color:
            assert self.value in [
                "gray",
                "red",
                "blue",
                "green",
                "brown",
                "purple",
                "cyan",
                "yellow",
            ], "Value for color must be in the list of values"
        elif self.property == Property.shape:
            assert self.value in [
                "cube",
                "sphere",
                "cylinder",
            ], "Value for shape must be in the list of values"
        elif self.property == Property.size:
            assert self.value in [
                "large",
                "small",
            ], "value for size must be in the list of values"
        else:
            assert self.value in [
                "rubber",
                "metal",
            ], "value for material must be in the list of values"

        assert len(self.previous) == 1, "Filter must have exactly one previous program"

        previous_program = self.previous[0]
        assert isinstance(
            previous_program.output, list
        ), "The output of the previous program for Filter must be a list"

        candidates = previous_program.output
        results = []
        property_name = self.property.name
        for candidate in candidates:
            if self.scene["objects"][candidate][property_name] == self.value:
                results.append(candidate)

        self.output = results

    @classmethod
    def is_type(cls, name: str) -> bool:
        values = name.split("_", 2)
        return (
            len(values) == 3
            and values[0] == "filter"
            and values[1] in ["size", "color", "material", "shape"]
        )

    @classmethod
    def from_name(cls, name: str) -> Optional[Filter]:
        if not cls.is_type(name):
            return None

        _, property, value = name.split("_", 2)
        return cls(property=property, value=value)

    @property
    def label(self) -> str:
        return f"filter_{self.property.value}_{self.value}"


class Query(ProgramNode):
    output: str = None
    property: Property

    def execute(self):
        assert len(self.previous) == 1, "Query must have exactly one previous program"

        previous_program = self.previous[0]
        assert isinstance(
            previous_program, Unique
        ), "Query must follow right after unique"

        obj_index = previous_program.output

        if obj_index >= 0:
            self.output = self.scene["objects"][obj_index][self.property.value]
        else:
            self.output = self.scene["objects"][0][self.property.value]

    @classmethod
    def is_type(cls, name: str) -> bool:
        values = name.split("_", 1)
        return (
            len(values) == 2
            and values[0] == "query"
            and values[1] in ["size", "color", "material", "shape"]
        )

    @classmethod
    def from_name(cls, name: str) -> Optional[Query]:
        _, property = name.split("_", 1)

        return cls(property=property)

    @property
    def label(self) -> str:
        return f"query_{self.property.value}"


class And(ProgramNode):
    output: list[int] = None

    def execute(self):
        assert len(self.previous) == 2, "And must have exactly two previous programs"

        left_program = self.previous[0]
        right_program = self.previous[1]
        assert isinstance(
            left_program.output, list
        ), "The output of the previous program for And must be a list"
        assert isinstance(
            right_program.output, list
        ), "The output of the previous program for And must be a list"

        left = set(left_program.output)
        right = set(right_program.output)

        self.output = list(left.intersection(right))

    @classmethod
    def is_type(cls, name: str) -> bool:
        return name == "and"

    @classmethod
    def from_name(cls, name: str) -> Optional[And]:
        if not cls.is_type(name):
            return None

        return And()

    @property
    def label(self) -> str:
        return "AND"


class Or(ProgramNode):
    output: list[int] = None

    def execute(self):
        assert len(self.previous) == 2, "Or must have exactly two previous programs"

        left_program = self.previous[0]
        right_program = self.previous[1]
        assert isinstance(
            left_program.output, list
        ), "The output of the previous program for Or must be a list"
        assert isinstance(
            right_program.output, list
        ), "The output of the previous program for Or must be a list"

        left = set(left_program.output)
        right = set(right_program.output)

        self.output = list(left.union(right))

    @classmethod
    def is_type(cls, name: str) -> bool:
        return name == "or"

    @classmethod
    def from_name(cls, name: str) -> Optional[Or]:
        if not cls.is_type(name):
            return None
        return Or()

    @property
    def label(self) -> str:
        return "OR"


class SameAttribute(ProgramNode):
    output: list[int] = None
    property: Property

    def execute(self):
        assert (
            len(self.previous) == 1
        ), "SameAttribute must have exactly one previous program"

        previous_program = self.previous[0]
        assert isinstance(
            previous_program, Unique
        ), "SameAttribute must follow right after unique"

        obj_idx = previous_program.output

        if obj_idx < 0:
            self.output = []
            return

        property_name = self.property.name
        property_value = self.scene["objects"][obj_idx][property_name]

        results = []
        for i, obj in enumerate(self.scene["objects"]):
            if i == obj_idx:
                continue

            if obj[property_name] == property_value:
                results.append(i)

        self.output = results

    @classmethod
    def is_type(cls, name: str) -> bool:
        values = name.split("_", 1)
        return (
            len(values) == 2
            and values[0] == "same"
            and values[1] in ["size", "color", "material", "shape"]
        )

    @classmethod
    def from_name(cls, name: str) -> Optional[SameAttribute]:
        _, property = name.split("_", 1)

        return cls(property=property)

    @property
    def label(self) -> str:
        return f"same_{self.property.value}"


class IntegerComp(ProgramNode):
    output: bool = None
    comp: Literal["equal", "less_than", "greater_than"]

    def execute(self):
        assert (
            len(self.previous) == 2
        ), "IntegerComp must have exactly two previous programs"

        left_program = self.previous[0]
        right_program = self.previous[1]

        assert isinstance(
            left_program, Count
        ), "IntegerComp must follow right after Count"
        assert isinstance(
            right_program, Count
        ), "IntegerComp must follow right after Count"

        left = left_program.output
        right = right_program.output

        if self.comp == "equal":
            self.output = left == right
        elif self.comp == "greater_than":
            self.output = left > right
        else:
            self.output = left < right

    @classmethod
    def is_type(cls, name: str) -> bool:
        return name in ["equal_integer", "equal", "less_than", "greater_than"]

    @classmethod
    def from_name(cls, name: str) -> Optional[IntegerComp]:
        if not cls.is_type(name):
            return None

        if name == "equal_integer" or name == "equal":
            return cls(comp="equal")
        elif name == "less_than":
            return cls(comp="less_than")
        elif name == "greater_than":
            return cls(comp="greater_than")

    @property
    def label(self) -> str:
        if self.comp == "equal":
            return "equal_integer"
        else:
            return self.comp


class AttrComp(ProgramNode):
    output: bool = None
    property: Property

    def execute(self):
        assert (
            len(self.previous) == 2
        ), "AttrComp must have exactly two previous programs"

        left_program = self.previous[0]
        right_program = self.previous[1]

        assert isinstance(
            left_program, Unique
        ), "AttrComp must follow right after Unique"
        assert isinstance(
            right_program, Unique
        ), "AttrComp must follow right after Unique"

        left = left_program.output
        right = right_program.output

        if left < 0 or right < 0:
            self.output = False
            return

        property_name = self.property.name

        self.output = (
            self.scene["objects"][left][property_name]
            == self.scene["objects"][right][property_name]
        )

    @classmethod
    def is_type(cls, name: str) -> bool:
        values = name.split("_", 1)
        return (
            len(values) == 2
            and values[0] == "equal"
            and values[1] in ["size", "color", "material", "shape"]
        )

    @classmethod
    def from_name(cls, name: str) -> Optional[AttrComp]:
        _, property = name.split("_", 1)

        return cls(property=property)

    @property
    def label(self) -> str:
        return f"equal_{self.property.value}"


PROGRAM_TYPES: list[Type[ProgramNode]] = [
    Scene,
    Unique,
    Relate,
    Count,
    Exist,
    Filter,
    Query,
    And,
    Or,
    SameAttribute,
    IntegerComp,
    AttrComp,
]


def set_scene(node: ProgramNode, scene: dict):
    node.scene = scene

    for next in node.next:
        set_scene(next, scene)


def evaluate(node: ProgramNode) -> Any:
    queue = []
    queue.append(node)

    while not len(queue) == 0:
        node = queue.pop(0)

        # Check if all prerequisite for this node has been executed
        prerequisite_checked = True
        for prev in node.previous:
            prerequisite_checked = prerequisite_checked and prev.output is not None

        if not prerequisite_checked:
            continue

        node.execute()
        for next in node.next:
            queue.append(next)

    return node.output


def programs_from_networkx(graph: nx.DiGraph) -> Scene:
    node_map = {}
    nodes = list(nx.topological_sort(graph))
    first_node = nodes[0]

    for node in nodes:
        program_node = process_node(node, graph, node_map)
        node_map[node] = program_node

    if not isinstance(node_map[first_node], Scene):
        raise ValueError("The program must start with a Scene program")

    return node_map[first_node]


def networkx_from_programs(program: ProgramNode) -> nx.DiGraph:
    graph = nx.DiGraph()
    processed_programs = {}

    def create_graph_rec(node: ProgramNode, count: int) -> int:
        if id(node) in processed_programs:
            return

        for prev in node.previous:
            if id(prev) not in processed_programs:
                # Continue on another branch
                return count

        graph.add_node(count, label=node.label)

        for prev in node.previous:
            graph.add_edge(processed_programs[id(prev)], count)

        processed_programs[id(node)] = count
        count += 1

        for next in node.next:
            count = create_graph_rec(next, count)
        return count

    create_graph_rec(program, 1)
    return graph


def process_node(
    node: int, graph: nx.DiGraph, node_map: dict[int, ProgramNode]
) -> ProgramNode:
    label = graph.nodes[node]["label"].strip().lower()
    program_node = None
    for node_type in PROGRAM_TYPES:
        if node_type.is_type(label):
            program_node = node_type.from_name(label)
            break

    if program_node is None:
        raise ValueError(f"Node {label} does not belong to any category")

    for prev, _ in graph.in_edges(node):
        prev_program = node_map[prev]
        prev_program.next.append(program_node)
        program_node.previous.append(prev_program)

    return program_node
