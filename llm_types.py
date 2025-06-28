from typing import TypedDict, Literal, Optional, Any

class LLMToolCall(TypedDict, total=False):
    id: str
    function: dict[str, Any]
    type: Literal["function"]

class LLMMessage(TypedDict, total=False):
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str]
    tool_calls: list[LLMToolCall]
    tool_call_id: str
