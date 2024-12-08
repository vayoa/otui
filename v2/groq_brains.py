from dataclasses import dataclass, field
from typing import (
    Literal,
    Sequence,
    Optional,
    TypedDict,
    Union,
    Mapping,
    Iterator,
    Any,
    overload,
    Callable,
    get_type_hints,
    get_origin,
)
import inspect
from groq import Groq
from groq.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionToolParam,
)
from groq._streaming import Stream
from rich import print
from brains import Brain

Message = ChatCompletionMessageParam


@dataclass
class GroqBrain(Brain[Message, ChatCompletionToolParam]):
    model: str = "llama-3.3-70b-versatile"
    client: Groq = field(
        init=False,
        default_factory=lambda: Groq(
            api_key="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"
        ),
    )
    messages: list[Message] = field(default_factory=list)
    default_tools: Optional[list[ChatCompletionToolParam]] = None

    @overload
    def chat(
        self,
        input: str | Sequence[Message] = [],
        model: str = model,
        messages: Optional[Sequence[Message]] = None,
        stream: Literal[False] = False,
        format: Literal["", "json"] = "",
        keep_alive: Optional[Union[float, str]] = None,
        tools: Optional[list[ChatCompletionToolParam]] = default_tools,
    ) -> ChatCompletion: ...

    @overload
    def chat(
        self,
        input: str | Sequence[Message] = [],
        model: str = model,
        messages: Optional[Sequence[Message]] = None,
        stream: Literal[True] = True,
        format: Literal["", "json"] = "",
        keep_alive: Optional[Union[float, str]] = None,
        tools: Optional[list[ChatCompletionToolParam]] = default_tools,
    ) -> Stream[ChatCompletionChunk]: ...

    def chat(
        self,
        input: str | Sequence[Message] = [],
        model: str = model,
        messages: Optional[Sequence[Message]] = None,
        stream: bool = False,
        format: Literal["", "json"] = "",
        keep_alive: Optional[Union[float, str]] = None,
        tools: Optional[list[ChatCompletionToolParam]] = default_tools,
    ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        if isinstance(input, str):
            input = [ChatCompletionUserMessageParam(role="user", content=input)]

        self.messages.extend(input)

        model = model or self.model
        messages = messages or self.messages

        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=stream,  # type: ignore
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stop=None,
            tools=tools,
            tool_choice="auto",
        )
