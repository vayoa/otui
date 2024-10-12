import base64
from dataclasses import dataclass, field
import io
from typing import (
    Dict,
    List,
    NamedTuple,
    Tuple,
    Union,
    TypeVar,
    NewType,
    Type,
    Generic,
    Union,
    TypeVarTuple,
    get_origin,
    get_args,
    get_type_hints,
)
import json
import itertools
from PIL.Image import Image

T = TypeVar("T")
_Out = Tuple[str, int]
In = Union[T, _Out]
Model = NewType("Model", _Out)
Clip = NewType("Clip", _Out)


class Out(Generic[T]):
    pass


# @dataclass
# class Node:
#     id: int = field(init=False, default_factory=lambda: next(Node._id_counter))
#     _id_counter = itertools.count(0)
#     _title: str = field(init=False, default="")
#     _outputs: Tuple = field(init=False, default=())
#     outputs: Dict[str, Input] = field(init=False)

#     def __post_init__(self):
#         if not self._title:
#             self._title = self.__class__.__name__

#         self.outputs = {
#             output: (str(self.id), i) for i, output in enumerate(self._outputs)
#         }

#     def json(self) -> Dict:
#         return {
#             self.id: {
#                 "inputs": {
#                     key: self.__dict__[key]
#                     for key in self.__dict__
#                     if key not in ("id", "_outputs", "outputs")
#                 },
#                 "class_type": self.__class__.__name__,
#                 "_meta": {"title": self._title},
#             }
#         }

#     @staticmethod
#     def from_json(j: dict):

#         inputs = [
#             f"{key}: Plug" if isinstance(val, list) else f"{key}: Input = {repr(val)}"
#             for key, val in j["inputs"].items()
#         ]
#         inputs = "\n\t".join(inputs)

#         name = j["class_type"]
#         title = j["_meta"]["title"]
#         title = f'\t_title = "{title}"'

#         c = f"""
# @dataclass
# class {name}(Node):
# \t{inputs}

# {title}

#  (\t_outputs = ()
#     """
#         return c


from functools import wraps

Ts = TypeVarTuple("Ts")


@dataclass
class Node(Generic[*Ts]):
    id: str = field(init=False, default_factory=lambda: str(next(Node._id_counter)))
    _id_counter = itertools.count(0)

    @property
    def outputs(self) -> Tuple[*Ts]:
        ts_types = get_args(self.__orig_bases__[0])[1:]  # type: ignore
        return tuple(t((self.id, i)) for i, t in enumerate(ts_types))


@dataclass
class NodeExample(Node[NamedTuple("NodeExampleOutputs", [("a", int)])]):
    x: int


if __name__ == "__main__":
    pass
