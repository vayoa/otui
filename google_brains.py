from dataclasses import dataclass, field
from typing import Any, Iterator, Literal, Mapping, Optional, Sequence, Union, overload
from google import genai
from brains import Brain
from llm_types import LLMMessage

@dataclass
class GoogleBrain(Brain[LLMMessage, Any]):
    model: str = "gemini-2.0-flash"
    client: genai.Client = field(
        init=False,
        default_factory=lambda: genai.Client(
            api_key=open("/workspace/secrets/google.txt", "r", encoding="utf-8").read().strip()
        ),
    )
    messages: list[LLMMessage] = field(default_factory=list)

    @overload
    def chat(
        self,
        input: str | Sequence[LLMMessage] = [],
        model: str | None = None,
        messages: Optional[Sequence[LLMMessage]] = None,
        stream: Literal[False] = False,
        format: Literal["", "json"] = "",
        keep_alive: Optional[Union[float, str]] = None,
        tools: Optional[list[Any]] = None,
    ) -> Mapping[str, Any]: ...

    @overload
    def chat(
        self,
        input: str | Sequence[LLMMessage] = [],
        model: str | None = None,
        messages: Optional[Sequence[LLMMessage]] = None,
        stream: Literal[True] = True,
        format: Literal["", "json"] = "",
        keep_alive: Optional[Union[float, str]] = None,
        tools: Optional[list[Any]] = None,
    ) -> Iterator[Mapping[str, Any]]: ...

    def chat(
        self,
        input: str | Sequence[LLMMessage] = [],
        model: str | None = None,
        messages: Optional[Sequence[LLMMessage]] = None,
        stream: bool = False,
        format: Literal["", "json"] = "",
        keep_alive: Optional[Union[float, str]] = None,
        tools: Optional[list[Any]] = None,
    ) -> Mapping[str, Any] | Iterator[Mapping[str, Any]]:
        if isinstance(input, str):
            input = [{"role": "user", "content": input}]
        input = list(input)

        model = model or self.model
        messages = list(messages or self.messages) + input

        google_messages = [
            {"role": m["role"], "parts": [{"text": m.get("content", "")}]}
            for m in messages
            if "content" in m
        ]

        self.add_messages(input)

        if stream:
            def gen():
                for chunk in self.client.models.generate_content_stream(
                    model=model,
                    contents=google_messages,
                ):
                    text = ""
                    if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                        text = chunk.candidates[0].content.parts[0].text or ""
                    yield {"choices": [{"delta": {"content": text}}]}
            return gen()
        else:
            resp = self.client.models.generate_content(
                model=model,
                contents=google_messages,
            )
            text = ""
            if resp.candidates and resp.candidates[0].content and resp.candidates[0].content.parts:
                text = resp.candidates[0].content.parts[0].text or ""
            return {"choices": [{"message": {"content": text}}]}
