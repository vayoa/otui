from dataclasses import dataclass, field
from typing import (
    Generic,
    Literal,
    Sequence,
    Optional,
    TypeVar,
    TypedDict,
    Union,
    Mapping,
    Iterator,
    Any,
    overload,
    Callable,
)
from rich import print

tool_type_mapper: Mapping[type, str] = {
    str: "string",
    int: "number",
    float: "number",
    bool: "boolean",
    list: "array",
    Sequence: "array",
    dict: "object",
    Mapping: "object",
    type(None): "null",
}


class ToolFunction(TypedDict):
    func: Callable
    description: str
    parameter_descriptions: Sequence[str]


Message = TypeVar("Message")
Tool = TypeVar("Tool")


@dataclass
class Brain(Generic[Message, Tool]):
    model: str = field(init=False)
    messages: list[Message] = field(default_factory=list)

    def clear_last_messages(self, n, keep=None):
        messages = self.messages
        kept_messages = [messages[-keep]] if keep is not None else []
        messages = messages[:-n] + kept_messages

        self.messages = messages

    @overload
    def chat(
        self,
        input: str | Sequence[Message] = [],
        model: str = model,
        messages: Optional[Sequence[Message]] = None,
        stream: Literal[False] = False,
        format: Literal["", "json"] = "",
        keep_alive: Optional[Union[float, str]] = None,
        tools: Optional[list[Tool]] = None,
    ) -> Mapping[str, Any]: ...

    @overload
    def chat(
        self,
        input: str | Sequence[Message] = [],
        model: str = model,
        messages: Optional[Sequence[Message]] = None,
        stream: Literal[True] = True,
        format: Literal["", "json"] = "",
        keep_alive: Optional[Union[float, str]] = None,
        tools: Optional[list[Tool]] = None,
    ) -> Iterator[Mapping[str, Any]]: ...

    def chat(
        self,
        input: str | Sequence[Message] = [],
        model: str = model,
        messages: Optional[Sequence[Message]] = None,
        stream: bool = False,
        format: Literal["", "json"] = "",
        keep_alive: Optional[Union[float, str]] = None,
        tools: Optional[list[Tool]] = None,
    ) -> Union[Mapping[str, Any], Iterator[Mapping[str, Any]]]: ...
