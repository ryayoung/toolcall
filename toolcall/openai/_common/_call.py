from typing import Self, Literal
from dataclasses import dataclass
from openai.types.chat import (
    ChatCompletionMessageCustomToolCall,
    ChatCompletionMessageFunctionToolCall,
)
from openai.types.responses import ResponseFunctionToolCall, ResponseCustomToolCall

__all__ = [
    "StandardToolCall",
    "AnyToolCall",
    "AnyFunctionToolCall",
    "AnyCustomToolCall",
]

type AnyToolCall = (
    StandardToolCall
    | ChatCompletionMessageFunctionToolCall
    | ChatCompletionMessageCustomToolCall
    | ResponseFunctionToolCall
    | ResponseCustomToolCall
)

type AnyFunctionToolCall = (
    StandardToolCall | ChatCompletionMessageFunctionToolCall | ResponseFunctionToolCall
)

type AnyCustomToolCall = (
    StandardToolCall | ChatCompletionMessageCustomToolCall | ResponseCustomToolCall
)


@dataclass(slots=True, frozen=True, kw_only=True)
class StandardToolCall:
    """
    A common structure to store the data from a tool call from various API types.
    """

    type: Literal["function", "custom"]
    id: str
    name: str
    input: str

    @classmethod
    def from_any_call(
        cls,
        call: Self
        | ChatCompletionMessageCustomToolCall
        | ChatCompletionMessageFunctionToolCall
        | ResponseFunctionToolCall
        | ResponseCustomToolCall,
    ) -> Self:
        if isinstance(call, ResponseFunctionToolCall):
            id = call.call_id
            return cls(type="function", id=id, name=call.name, input=call.arguments)
        if isinstance(call, ResponseCustomToolCall):
            id = call.call_id
            return cls(type="custom", id=id, name=call.name, input=call.input)
        if isinstance(call, ChatCompletionMessageFunctionToolCall):
            id = call.id
            func = call.function
            return cls(type="function", id=id, name=func.name, input=func.arguments)
        if isinstance(call, ChatCompletionMessageCustomToolCall):
            id = call.id
            cust = call.custom
            return cls(type="custom", id=id, name=cust.name, input=cust.input)
        return call
