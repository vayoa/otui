from dataclasses import dataclass, field
from typing import Literal, Sequence, Optional, Union, Mapping, Iterator, Any, overload
import ollama
import ollama._types as ot


def tool(
    name: str,
    description: str,
    properties: Mapping[str, ot.Property],
    required: Sequence[str],
):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


@dataclass
class Brain:
    model: str = "llama3.1"
    messages: Sequence[ot.Message] = field(default_factory=lambda: [])
    tools: Optional[Sequence[ot.Tool]] = None

    @overload
    def chat(
        self,
        input: str | Sequence[ot.Message] = [],
        model: str = model,
        messages: Optional[Sequence[ot.Message]] = None,
        tools: Optional[Sequence[ot.Tool]] = tools,
        stream: Literal[False] = False,
        format: Literal["", "json"] = "",
        options: Optional[ot.Options] = None,
        keep_alive: Optional[Union[float, str]] = None,
    ) -> Mapping[str, Any]: ...

    @overload
    def chat(
        self,
        input: str | Sequence[ot.Message] = [],
        model: str = model,
        messages: Optional[Sequence[ot.Message]] = None,
        tools: Optional[Sequence[ot.Tool]] = tools,
        stream: Literal[True] = True,
        format: Literal["", "json"] = "",
        options: Optional[ot.Options] = None,
        keep_alive: Optional[Union[float, str]] = None,
    ) -> Iterator[Mapping[str, Any]]: ...

    def chat(
        self,
        input: str | Sequence[ot.Message] = [],
        model: str = model,
        messages: Optional[Sequence[ot.Message]] = None,
        tools: Optional[Sequence[ot.Tool]] = tools,
        stream: bool = False,
        format: Literal["", "json"] = "",
        options: Optional[ot.Options] = None,
        keep_alive: Optional[Union[float, str]] = None,
    ) -> Union[Mapping[str, Any], Iterator[Mapping[str, Any]]]:
        if isinstance(input, str):
            input = [ot.Message(role="user", content=input)]

        self.messages.extend(input)

        model = model or self.model
        messages = messages or self.messages
        tools = tools or self.tools

        return ollama.chat(
            model=model,
            messages=messages,
            tools=tools,
            stream=stream,
            format=format,
            options=options,
            keep_alive=keep_alive,
        )


if __name__ == "__main__":
    tools = [
        tool(
            name="get_current_weather",
            description="Get the current weather for a city",
            required=["city"],
            properties={
                "city": {
                    "type": "string",
                    "description": "The name of the city",
                },
            },
        )
    ]

    brain = Brain(tools=tools)

    # print(brain.chat("What is the weather in Toronto?"))

    for i in brain.chat("What is the weather in Toronto?", stream=True):
        print(i)
