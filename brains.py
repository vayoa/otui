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
from vstore import VStore
import json

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


class BaseMessage(TypedDict, total=False):
    """A generic chat message used across brain implementations."""

    role: str
    content: str


class BaseToolParam(TypedDict, total=False):
    """Generic tool parameter definition."""

    type: Literal["function"]
    function: Mapping[str, Any]


Message = TypeVar("Message", bound=BaseMessage)
Tool = TypeVar("Tool", bound=BaseToolParam)


@dataclass
class Brain(Generic[Message, Tool]):
    model: str = field(init=False)
    messages: list[Message] = field(default_factory=list)
    default_tools: Optional[list[Tool]] = None
    message_limit: int = 25
    tool_message_limit: int = 1
    query_message_limit: int = 4
    vstore: VStore = field(init=False, default_factory=VStore)
    latest_rag_context: Optional[list[Message]] = None
    rag_tool_name: str = "generate_scene_image"

    def clear_last_messages(self, n, keep=None):
        messages = self.messages
        kept_messages = messages[-keep:] if keep is not None else []
        messages = messages[:-n] + kept_messages

        self.vstore.delete_last(len(self.messages), n, keep)
        self.messages = messages

    def set_messages(self, messages: list[Message]):
        self.messages = messages
        self.vstore.purge()
        self.vstore.add_messages(messages)

    def add_messages(self, messages: list[Message], on_index: Optional[int] = None):
        on_index = on_index if on_index is not None else len(self.messages)
        self.messages.extend(messages)
        self.vstore.add_messages(messages, on_index)

    def update_message_content(self, content: str, index: int):
        message = self.messages[index]
        if isinstance(message, Mapping) and "content" in message:
            message["content"] = content
            if message.get("role") != "tool":
                self.vstore.update_content(content, index)

    def change_system(self, content: str):
        if self.messages:
            self.messages[0] = {"role": "system", "content": content}  # type: ignore
        else:
            self.messages.append({"role": "system", "content": content})  # type: ignore
        self.vstore.change_system(content)

    def prepare_messages(
        self,
        input_messages: Sequence[Message],
        base_messages: Optional[Sequence[Message]] = None,
        *,
        rag: bool = True,
    ) -> list[Message]:
        """Combine existing messages with the input and optionally apply RAG."""

        messages = list(base_messages or self.messages)
        messages = messages + list(input_messages)

        if rag:
            user_messages = [m for m in input_messages if m.get("role") == "user"]
            if user_messages:
                user_message = user_messages[-1]["content"]

                filtered_messages = [(i, m) for i, m in enumerate(messages)]
                last_tool_call_indices = [
                    i
                    for i, m in filtered_messages
                    if "tool_calls" in m
                    and m["tool_calls"][0]["function"]["name"] == self.rag_tool_name
                ][-self.tool_message_limit :]
                last_tool_call_indices += [i + 1 for i in last_tool_call_indices]

                filtered_messages = [
                    o
                    for o in filtered_messages
                    if ("content" in o[1] and o[1].get("role") != "tool")
                    or (o[0] in last_tool_call_indices)
                ]

                filtered_count = (
                    self.message_limit
                    - self.tool_message_limit
                    - self.query_message_limit
                    - 1
                )
                new_filtered_messages = filtered_messages[-filtered_count:]
                first_index = new_filtered_messages[0][0] if new_filtered_messages else 0
                new_filtered_messages = [o[1] for o in new_filtered_messages]

                if first_index > 0:
                    query_messages = self.vstore.query(
                        user_message,
                        self.query_message_limit,
                        first_index,
                    )
                    message = [
                        {
                            "role": "user",
                            "content": "This story is loading from the middle to save memory. We already started the adventure and are in the middle of it.\nHere are some previous messages I picked for you for context:\n"
                            + "\n".join(
                                [f"{m['role']}: {m['content']}" for m in query_messages]
                            )
                            + "\nAlso, do not forget to generate an image using your tool like you were ordered! You generated one for every response but I cut it to save memory and only kept the last one."
                            + "\nWe will now resume our story from our last point.",
                        }
                    ]
                    new_filtered_messages = message + new_filtered_messages

                    new_filtered_messages = [filtered_messages[0][1]] + new_filtered_messages

                messages = new_filtered_messages
                self.latest_rag_context = messages
                with open(
                    r"C:\Users\ew0nd\Documents\otui\chats\_context_log.json",
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(self.latest_rag_context, f, indent=4)

        return messages

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
