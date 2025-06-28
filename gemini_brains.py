from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Sequence, Optional, Literal
import google.generativeai as genai

from brains import Brain, MessageDict

Message = MessageDict

@dataclass
class GeminiBrain(Brain[Message, dict]):
    model: str = "gemini-2.0-flash"
    client: genai.GenerativeModel = field(init=False)
    default_tools: Optional[list[dict]] = None

    def __post_init__(self):
        genai.configure(
            api_key=open(
                r"C:\Users\ew0nd\Documents\otui\secrets\gemini.txt", "r", encoding="utf-8"
            ).read().strip()
        )
        self.client = genai.GenerativeModel(self.model)

    def _convert_messages(self, messages: Sequence[Message]):
        return [
            {"role": m.get("role", "user"), "parts": [m.get("content", "")]} for m in messages
        ]

    def chat(
        self,
        input: str | Sequence[Message] = [],
        model: str | None = None,
        messages: Optional[Sequence[Message]] = None,
        stream: bool = False,
        format: Literal["", "json"] = "",
        keep_alive: Optional[float | str] = None,
        tools: Optional[list[dict]] = None,
    ) -> Mapping[str, Any] | Iterator[Mapping[str, Any]]:
        if isinstance(input, str):
            input = [{"role": "user", "content": input}]
        input = list(input)

        model = model or self.model
        messages = list(messages or []) or self.messages
        messages = messages + input
        gen_messages = self._convert_messages(messages)
        gen_tools = [t.get("function", t) for t in tools] if tools else None

        self.add_messages(input)

        if stream:
            resp = self.client.generate_content(
                gen_messages, stream=True, tools=gen_tools
            )
            collected = ""
            for chunk in resp:
                text = getattr(chunk, "text", "")
                collected += text
                yield {"choices": [{"delta": {"content": text}}]}
            self.messages.append({"role": "assistant", "content": collected})
        else:
            resp = self.client.generate_content(
                gen_messages, stream=False, tools=gen_tools
            )
            text = resp.text
            self.messages.append({"role": "assistant", "content": text})
            return {"choices": [{"message": {"content": text}}]}

    def clear_last_messages(self, n, keep=None):
        super().clear_last_messages(n, keep)

    def set_messages(self, messages: list[Message]):
        self.messages = messages

    def add_messages(self, messages: list[Message]):
        self.messages.extend(messages)

    def change_system(self, content: str):
        if self.messages:
            self.messages[0] = {"role": "system", "content": content}

    def quick_format(self, input: str, model=None) -> dict:
        msgs = self._convert_messages(self.messages + [{"role": "user", "content": input}])
        resp = self.client.generate_content(msgs, stream=False)
        import json
        return json.loads(resp.text)
