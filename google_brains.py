from dataclasses import dataclass, field
import json
from typing import (
    Literal,
    Sequence,
    Optional,
    Union,
    Mapping,
    Iterator,
    Any,
    overload,
)
from types import SimpleNamespace
from google import genai
from rich import print
from brains import Brain
from vstore import VStore

Message = dict[str, str]


def _to_google_messages(messages: Sequence[Message]):
    g_messages = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "tool":
            tool_id = m.get("tool_call_id", "")
            g_messages.append(
                {
                    "role": "function",
                    "parts": [
                        {
                            "function_response": {
                                "name": tool_id,
                                "response": content,
                            }
                        }
                    ],
                }
            )
        else:
            g_messages.append({"role": role, "parts": [content]})
    return g_messages


def _struct_to_dict(struct):
    if struct is None:
        return {}
    return json.loads(struct.to_json())


def _convert_chunk(chunk) -> Mapping[str, Any]:
    cand = chunk.candidates[0]
    parts = cand.content.parts
    text = ""
    tool_calls = []
    for part in parts:
        if hasattr(part, "text") and part.text:
            text += part.text
        elif hasattr(part, "function_call") and part.function_call:
            fc = part.function_call
            tool_calls.append(
                {
                    "id": fc.id or "",
                    "function": {
                        "name": fc.name,
                        "arguments": json.dumps(_struct_to_dict(fc.args)),
                    },
                    "type": "function",
                }
            )
    delta = SimpleNamespace(content=text or None, tool_calls=tool_calls or None)
    choice = SimpleNamespace(delta=delta)
    return SimpleNamespace(choices=[choice])


def _convert_response(resp) -> Mapping[str, Any]:
    cand = resp.candidates[0]
    parts = cand.content.parts
    text = ""
    tool_calls = []
    for part in parts:
        if hasattr(part, "text") and part.text:
            text += part.text
        elif hasattr(part, "function_call") and part.function_call:
            fc = part.function_call
            tool_calls.append(
                {
                    "id": fc.id or "",
                    "function": {
                        "name": fc.name,
                        "arguments": json.dumps(_struct_to_dict(fc.args)),
                    },
                    "type": "function",
                }
            )
    message = {"role": "assistant", "content": text, "tool_calls": tool_calls or None}
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


@dataclass
class GoogleBrain(Brain[Message, Mapping[str, Any]]):
    model: str = "gemini-2.0-flash-001"
    client: genai.GenerativeModel = field(init=False)
    messages: list[Message] = field(default_factory=list)
    default_tools: Optional[list[Mapping[str, Any]]] = None
    message_limit: int = 25
    tool_message_limit: int = 1
    query_message_limit: int = 4
    vstore: VStore = field(init=False, default_factory=VStore)
    latest_rag_context: Optional[list[Message]] = None

    def __post_init__(self):
        genai.configure(
            api_key=open(
                r"C:\\Users\\ew0nd\\Documents\\otui\\secrets\\google.txt", "r", encoding="utf-8"
            )
            .read()
            .strip()
        )
        self.client = genai.GenerativeModel(self.model)

    @overload
    def chat(
        self,
        input: str | Sequence[Message] = [],
        model: str | None = None,
        messages: Optional[Sequence[Message]] = None,
        stream: Literal[False] = False,
        tools: Optional[list[Mapping[str, Any]]] = None,
        rag: bool = True,
    ) -> Mapping[str, Any]:
        ...

    @overload
    def chat(
        self,
        input: str | Sequence[Message] = [],
        model: str | None = None,
        messages: Optional[Sequence[Message]] = None,
        stream: Literal[True] = True,
        tools: Optional[list[Mapping[str, Any]]] = None,
        rag: bool = True,
    ) -> Iterator[Mapping[str, Any]]:
        ...

    def chat(
        self,
        input: str | Sequence[Message] = [],
        model: str | None = None,
        messages: Optional[Sequence[Message]] = None,
        stream: bool = False,
        tools: Optional[list[Mapping[str, Any]]] = None,
        rag: bool = True,
    ) -> Mapping[str, Any] | Iterator[Mapping[str, Any]]:
        if isinstance(input, str):
            input = [{"role": "user", "content": input}]
        input = list(input)

        model = model or self.model
        messages = list(messages or []) or self.messages
        messages = messages + input

        if rag:
            user_messages = [m for m in input if m["role"] == "user"]
            if user_messages:
                user_message = user_messages[-1]["content"]
                filtered_messages = [(i, m) for i, m in enumerate(messages)]
                last_tool_indices = [
                    o[0]
                    for o in filtered_messages
                    if "tool_calls" in o[1]
                    and o[1]["tool_calls"][0]["function"]["name"] == "generate_scene_image"
                ][-self.tool_message_limit :]
                last_tool_indices += [i + 1 for i in last_tool_indices]

                filtered_messages = [
                    o
                    for o in filtered_messages
                    if ("content" in o[1] and o[1]["role"] != "tool") or (o[0] in last_tool_indices)
                ]

                filtered_count = (
                    self.message_limit - self.tool_message_limit - self.query_message_limit - 1
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
                            + "\n".join([
                                f"{m['role']}: {m['content']}" for m in query_messages
                            ])
                            + "\nAlso, do not forget to generate an image using your tool like you were ordered! You generated one for every response but I cut it to save memory and only kept the last one."
                            + "\nWe will now resume our story from our last point.",
                        }
                    ]
                    new_filtered_messages = message + new_filtered_messages
                    new_filtered_messages = [filtered_messages[0][1]] + new_filtered_messages

                messages = new_filtered_messages
                self.latest_rag_context = messages
                with open(r"C:\\Users\\ew0nd\\Documents\\otui\\chats\\_context_log.json", "w") as f:
                    json.dump(self.latest_rag_context, f, indent=4)

        self.add_messages(input)

        g_messages = _to_google_messages(messages)
        resp = self.client.generate_content(g_messages, stream=stream, tools=tools)
        if stream:
            def _gen():
                for chunk in resp:
                    yield _convert_chunk(chunk)
            return _gen()
        return _convert_response(resp)

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
        self.messages[0] = {"role": "system", "content": content}
        self.vstore.change_system(content)

    def quick_format(self, input: str, model=None) -> dict:
        new_messages = []
        for message in self.messages:
            new_message = message
            if message["role"] not in ["user", "assistant", "system"]:
                continue
            if set(message.keys()) != {"role", "content"}:
                for key, value in message.items():
                    if key not in ["role", "content"]:
                        new_message = {
                            "role": message["role"],
                            "content": str(message.get("content", "")) + f"\n{key}: {value}",
                        }
            new_messages.append(new_message)

        new_messages.append({"role": "user", "content": input})
        new_messages = new_messages[-self.message_limit :]
        resp = self.client.generate_content(_to_google_messages(new_messages))
        return json.loads(resp.text)
