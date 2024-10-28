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
import ollama
import ollama._types as ot
from rich import print
from brains import Brain

tool_type_mapper: Mapping[type, str] = {
    str: "string",
    int: "number",
    float: "number",
    bool: "boolean",
    list: "array",
    Sequence: "array",
    dict: "object",
    Mapping: "object",
    type(None): "null",
}


class ToolFunction(TypedDict):
    func: Callable
    description: str
    parameter_descriptions: Sequence[str]


@dataclass
class OllamaBrain(Brain[ot.Message, ot.Tool]):
    model: str = "llama3.1"
    messages: list[ot.Message] = field(default_factory=list)
    tools: list[ot.Tool] = field(default_factory=list)
    options: Optional[ot.Options] = None
    functions: Optional[Sequence[ToolFunction]] = None

    def __post_init__(self):
        if self.functions is not None:
            for tool_func in self.functions:
                self.register_function(tool_func)

    @staticmethod
    def tool(
        name: str,
        description: str,
        properties: Mapping[str, ot.Property],
        required: Sequence[str],
    ) -> ot.Tool:
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

    @staticmethod
    def toolify(
        func: Callable,
        description: str,
        parameter_descriptions: Sequence[str],
    ) -> ot.Tool:

        signature = inspect.signature(func)
        type_hints = get_type_hints(func)

        description = description.strip()
        parameter_descriptions = [desc.strip() for desc in parameter_descriptions]

        assert len(signature.parameters) == len(parameter_descriptions)
        # assert descriptions are not empty...
        assert description
        assert all(desc for desc in parameter_descriptions)

        properties: Mapping[str, ot.Property] = {}
        required: list[str] = []

        for (name, param), desc in zip(
            signature.parameters.items(), parameter_descriptions
        ):
            param_type = type_hints.get(name, str)

            # if the parameter is required
            if param.default is inspect.Parameter.empty and param_type is not Optional:
                required.append(name)

            param_type = tool_type_mapper.get(
                param_type, tool_type_mapper.get(get_origin(param_type), "string")
            )
            properties[name] = ot.Property(type=param_type, description=desc)
            if get_origin(param.annotation) is Literal:
                properties[name]["enum"] = list(param.annotation.__args__)

        return OllamaBrain.tool(
            name=func.__name__,
            description=description,
            properties=properties,
            required=required,
        )

    def register_function(self, tool_func: ToolFunction):
        self.tools.append(OllamaBrain.toolify(**tool_func))

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
        tools: Optional[Sequence[ot.Tool]] = None,
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
        tools = tools or (None if not self.tools else self.tools)
        options = options or self.options

        return ollama.chat(
            model=model,
            messages=messages,
            tools=tools,
            stream=stream,  # type: ignore
            format=format,
            options=options,
            keep_alive=keep_alive,
        )


if __name__ == "__main__":
    from eyes import Eyes

    eyes = Eyes()

    def generate_character(name: str, prompt: str):
        print(name)
        img, _ = eyes.generate(positive=prompt, lcm=True)
        if img is not None:
            img.show()

    brain = OllamaBrain(
        functions=[
            ToolFunction(
                func=generate_character,
                description="Use if you introduced a new character to the story.",
                parameter_descriptions=[
                    "The name of the new character",
                    """
A comma separated Stable Diffusion / DALL-E prompt, detailed and medium length describing how the character looks.
Pay attention to age, ethnicity, country of origin, eye and hair color, skin color, clothes, emotion etc...""",
                ],
            )
        ]
    )

    content = ""
    for chunk in brain.chat(
        [
            {
                "role": "user",
                "content": "Lets roleplay, start me off in front of the ice king, Steffen!",
            },
            {
                "role": "assistant",
                "content": '{"name": "',
            },
        ],
        stream=True,
    ):
        content += chunk["message"]["content"]
        print(chunk["message"]["content"], end="")

    print()

    import json

    print(json.loads(content))

    for chunk in brain.chat(input=[{"role": "assistant", "content": " "}], stream=True):
        print(chunk["message"]["content"], end="")

    print()

    eyes.close()
