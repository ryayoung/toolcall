from typing import Self
from dataclasses import dataclass
from openai.types.chat import ChatCompletionMessageToolCall
from openai.types.responses import ResponseFunctionToolCall

__all__ = ["StandardToolCall", "AnyToolCall"]

type AnyToolCall = (
    StandardToolCall | ChatCompletionMessageToolCall | ResponseFunctionToolCall
)


@dataclass(slots=True, frozen=True, kw_only=True)
class StandardToolCall:
    """
    A common structure to store the data from a tool call from various API types.
    """

    id: str
    name: str
    arguments: str

    @classmethod
    def from_any_call(
        cls, call: Self | ChatCompletionMessageToolCall | ResponseFunctionToolCall
    ) -> Self:
        if isinstance(call, ChatCompletionMessageToolCall):
            func = call.function
            return cls(id=call.id, name=func.name, arguments=func.arguments)
        if isinstance(call, ResponseFunctionToolCall):
            return cls(id=call.call_id, name=call.name, arguments=call.arguments)
        return call
