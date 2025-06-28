from dataclasses import dataclass, field
from typing import Literal, Sequence, Optional, Iterator, Mapping, Any, overload
from pathlib import Path
from google import genai
from google.genai import types
from types import SimpleNamespace
import json
from brains import Brain, BaseMessage, BaseToolParam

Message = BaseMessage
Tool = BaseToolParam


@dataclass
class GoogleBrain(Brain[Message, Tool]):
    """Brain implementation using Google's generative models."""

    model: str = "gemini-2.0-flash"
    api_key_path: str = r"C:\Users\ew0nd\Documents\otui\secrets\google.txt"
    client: genai.Client = field(init=False)

    def __post_init__(self):
        api_key_path = Path(self.api_key_path)
        api_key = api_key_path.read_text(encoding="utf-8").strip()
        self.client = genai.Client(api_key=api_key)

    def _to_contents(self, messages: Sequence[Message]):
        """Convert messages to Google genai Content objects."""

        return [
            types.Content(
                role=m.get("role", "user"),
                parts=[types.Part.from_text(text=m.get("content", ""))],
            )
            for m in messages
        ]

    def _to_tool(self, tool: Tool) -> types.Tool:
        """Convert a tool definition to the genai Tool format."""

        func = tool.get("function", {})
        return types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=func.get("name"),
                    description=func.get("description"),
                    parameters_json_schema=func.get("parameters"),
                )
            ]
        )

    def _convert_chunk(self, chunk: genai.types.GenerateContentResponse):
        """Convert a streaming chunk to a ChatCompletionChunk-like object."""

        content = chunk.text or ""

        tool_calls = None
        if chunk.function_calls:
            tool_calls = [
                SimpleNamespace(
                    id=fc.id or "",  # type: ignore
                    function=SimpleNamespace(
                        name=fc.name,
                        arguments=json.dumps(fc.args or {}),
                    ),
                )
                for fc in chunk.function_calls
            ]

        delta = SimpleNamespace(content=content, tool_calls=tool_calls)
        choice = SimpleNamespace(delta=delta)
        return SimpleNamespace(choices=[choice])

    def _convert_response(self, resp: genai.types.GenerateContentResponse):
        """Convert the full response to match Groq's ChatCompletion interface."""

        chunk = self._convert_chunk(resp)
        message = SimpleNamespace(
            role="assistant",
            content=resp.text or "",
            tool_calls=chunk.choices[0].delta.tool_calls,
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

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

        contents = self._to_contents(messages)

        if tools is None:
            tools = self.default_tools

        genai_tools = [self._to_tool(t) for t in tools] if tools else None

        config = types.GenerateContentConfig(
            tools=genai_tools,
            response_mime_type="application/json" if format == "json" else "text/plain",
        )

        if stream:
            for chunk in self.client.models.generate_content_stream(
                model=model, contents=contents, config=config
            ):
                yield self._convert_chunk(chunk)
            return

        return self._convert_response(
            self.client.models.generate_content(
                model=model, contents=contents, config=config
            )
        )
