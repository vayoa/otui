from dataclasses import dataclass, field
from typing import Literal, Sequence, Optional, Iterator, Mapping, Any, overload
from google import generativeai as genai
from brains import Brain, BaseMessage, BaseToolParam

Message = BaseMessage
Tool = BaseToolParam

@dataclass
class GoogleBrain(Brain[Message, Tool]):
    """Brain implementation using Google's generative models."""

    model: str = "gemini-pro"
    api_key_path: str = r"C:\\Users\\ew0nd\\Documents\\otui\\secrets\\google.txt"
    client: genai.GenerativeModel = field(init=False)

    def __post_init__(self):
        genai.configure(api_key=open(self.api_key_path, "r", encoding="utf-8").read().strip())
        self.client = genai.GenerativeModel(self.model)

    def _to_google_messages(self, messages: Sequence[Message]):
        return [
            {"role": "user" if m["role"] == "user" else "model", "parts": [m.get("content", "")]}
            for m in messages
        ]

    @overload
    def chat(
        self,
        input: str | Sequence[Message] = [],
        model: str | None = None,
        messages: Optional[Sequence[Message]] = None,
        stream: Literal[False] = False,
        format: Literal["", "json"] = "",
        keep_alive: Optional[float | str] = None,
        tools: Optional[list[Tool]] = None,
        rag: bool = True,
    ) -> Mapping[str, Any]: ...

    @overload
    def chat(
        self,
        input: str | Sequence[Message] = [],
        model: str | None = None,
        messages: Optional[Sequence[Message]] = None,
        stream: Literal[True] = True,
        format: Literal["", "json"] = "",
        keep_alive: Optional[float | str] = None,
        tools: Optional[list[Tool]] = None,
        rag: bool = True,
    ) -> Iterator[Mapping[str, Any]]: ...

    def chat(
        self,
        input: str | Sequence[Message] = [],
        model: str | None = None,
        messages: Optional[Sequence[Message]] = None,
        stream: bool = False,
        format: Literal["", "json"] = "",
        keep_alive: Optional[float | str] = None,
        tools: Optional[list[Tool]] = None,
        rag: bool = True,
    ) -> Mapping[str, Any] | Iterator[Mapping[str, Any]]:
        if isinstance(input, str):
            input = [{"role": "user", "content": input}]
        input = list(input)

        model = model or self.model
        messages = self.prepare_messages(input, messages, rag=rag)

        self.add_messages(input)

        google_messages = self._to_google_messages(messages)
        return self.client.generate_content(google_messages, stream=stream, tools=tools)
