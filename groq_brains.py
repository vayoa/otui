from dataclasses import dataclass, field
import json
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
    ChatCompletionSystemMessageParam,
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionToolParam,
)
from groq._types import NOT_GIVEN
from groq._streaming import Stream
from rich import print
from brains import Brain
from vstore import VStore

Message = ChatCompletionMessageParam


@dataclass
class GroqBrain(Brain[Message, ChatCompletionToolParam]):
    model: str = "llama-3.3-70b-versatile"
    client: Groq = field(
        init=False,
        default_factory=lambda: Groq(
            api_key=open(
                r"C:\Users\ew0nd\Documents\otui\secrets\groq.txt", "r", encoding="utf-8"
            )
            .read()
            .strip()
        ),
    )
    messages: list[Message] = field(default_factory=list)
    default_tools: Optional[list[ChatCompletionToolParam]] = None

    @overload
    def chat(
        self,
        input: str | Sequence[Message] = [],
        model: str | None = None,
        messages: Optional[Sequence[Message]] = None,
        stream: Literal[False] = False,
        format: Literal["", "json"] = "",
        keep_alive: Optional[Union[float, str]] = None,
        tools: Optional[list[ChatCompletionToolParam]] = default_tools,
        rag=True,
    ) -> ChatCompletion: ...

    @overload
    def chat(
        self,
        input: str | Sequence[Message] = [],
        model: str | None = None,
        messages: Optional[Sequence[Message]] = None,
        stream: Literal[True] = True,
        format: Literal["", "json"] = "",
        keep_alive: Optional[Union[float, str]] = None,
        tools: Optional[list[ChatCompletionToolParam]] = default_tools,
        rag=True,
    ) -> Stream[ChatCompletionChunk]: ...

    def chat(
        self,
        input: str | Sequence[Message] = [],
        model: str | None = None,
        messages: Optional[Sequence[Message]] = None,
        stream: bool = False,
        format: Literal["", "json"] = "",
        keep_alive: Optional[Union[float, str]] = None,
        tools: Optional[list[ChatCompletionToolParam]] = default_tools,
        rag=True,
    ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        if isinstance(input, str):
            input = [ChatCompletionUserMessageParam(role="user", content=input)]
        input = list(input)

        model = model or self.model
        messages = self.prepare_messages(input, messages, rag=rag)

        self.add_messages(input)

        return self.client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore
            stream=stream,  # type: ignore
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stop=None,
            tools=tools,
            tool_choice="auto",
        )

    def change_system(self, content: str):
        self.messages[0] = ChatCompletionSystemMessageParam(
            role="system", content=content
        )
        self.vstore.change_system(content)

    def quick_format(self, input: str, model=None) -> dict:
        new_messages = []

        # whenever a message has other attributes besides 'role' and 'content', we'll flatten those attrinbutes and their values into 'content' as strings
        for message in self.messages:
            new_message = message
            if message["role"] not in ["user", "assistant", "system"]:
                continue
            if set(message.keys()) != {"role", "content"}:
                for key, value in message.items():
                    if key not in ["role", "content"]:
                        new_message = {
                            "role": message["role"],
                            "content": str(message.get("content", ""))
                            + f"\n{key}: {value}",
                        }
            new_messages.append(new_message)

        new_messages.extend(
            [ChatCompletionUserMessageParam(role="user", content=input)]
        )

        new_messages = new_messages[-self.message_limit :]

        return json.loads(
            self.client.chat.completions.create(
                model=model or self.model,
                messages=new_messages,
                stream=False,
                temperature=1,
                max_tokens=1024,
                top_p=1,
                stop=None,
                response_format={"type": "json_object"},
            )
            .choices[0]
            .message.content  # type: ignore
        )
