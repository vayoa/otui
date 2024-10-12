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
    ChatCompletion,
    ChatCompletionChunk,
)
from groq._streaming import Stream
from rich import print
from brains import Brain

Message = ChatCompletionMessageParam


@dataclass
class GroqBrain(Brain[Message]):
    model: str = "llama-3.1-70b-versatile"
    client: Groq = field(
        init=False,
        default_factory=lambda: Groq(
            api_key="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"
        ),
    )
    messages: list[Message] = field(default_factory=list)

    @overload
    def chat(
        self,
        input: str | Sequence[Message] = [],
        model: str = model,
        messages: Optional[Sequence[Message]] = None,
        stream: Literal[False] = False,
        format: Literal["", "json"] = "",
        keep_alive: Optional[Union[float, str]] = None,
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
    ) -> Stream[ChatCompletionChunk]: ...

    def chat(
        self,
        input: str | Sequence[Message] = [],
        model: str = model,
        messages: Optional[Sequence[Message]] = None,
        stream: bool = False,
        format: Literal["", "json"] = "",
        keep_alive: Optional[Union[float, str]] = None,
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
        )
