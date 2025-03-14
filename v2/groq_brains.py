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
            api_key="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"
        ),
    )
    messages: list[Message] = field(default_factory=list)
    default_tools: Optional[list[ChatCompletionToolParam]] = None
    message_limit: int = 25
    tool_message_limit: int = 1
    query_message_limit: int = 4
    vstore: VStore = field(init=False, default_factory=VStore)

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
        messages = list(messages or []) or self.messages
        messages = messages + input

        if rag:
            user_message = [message for message in input if message["role"] == "user"][
                -1
            ]["content"]

            filtered_messages = [(i, message) for i, message in enumerate(messages)]
            last_tool_call_indices = [
                o[0]
                for o in filtered_messages
                if "tool_calls" in o[1]  # type: ignore
                and o[1]["tool_calls"][0]["function"]["name"]  # type: ignore
                == "generate_scene_image"
            ][-self.tool_message_limit :]
            last_tool_call_indices += [i + 1 for i in last_tool_call_indices]

            filtered_messages = [
                o
                for o in filtered_messages
                if ("content" in o[1] and o[1]["role"] != "tool")
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
                            [
                                f"{message['role']}: {message['content']}"
                                for message in query_messages
                            ]
                        )
                        + "\nAlso, do not forget to generate an image using your tool like you were ordered! You generated one for every response but I cut it to save memory and only kept the last one."
                        + "\nWe will now resume our story from our last point.",
                    }
                ]
                new_filtered_messages = message + new_filtered_messages

                new_filtered_messages = [
                    filtered_messages[0][1]
                ] + new_filtered_messages

            messages: list[Message] = new_filtered_messages  # type: ignore
            with open(
                r"C:\Users\ew0nd\Documents\otui\v2\chats\_context_log.json", "w"
            ) as f:
                json.dump(messages, f, indent=4)

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

    def clear_last_messages(self, n, keep=None):
        self.vstore.delete_last(len(self.messages), n, keep)
        super().clear_last_messages(n, keep)

    def set_messages(self, messages: list[Message]):
        self.messages = messages
        self.vstore.purge()
        self.vstore.add_messages(messages)

    def add_messages(self, messages: list[Message]):
        on_index = len(self.messages)
        self.messages.extend(messages)
        self.vstore.add_messages(messages, on_index)

    def update_message_content(self, content, index):
        if "content" in self.messages[index]:
            self.messages[index]["content"] = content
            if self.messages[index]["role"] != "tool":
                self.vstore.update_content(content, index)

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

        new_messages = new_messages[self.message_limit :]

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
