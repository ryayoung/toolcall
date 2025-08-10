from typing import Annotated, Literal, NamedTuple
from dataclasses import dataclass
import pydantic
from openai.types.chat import ChatCompletionToolMessageParam
from openai.types.responses.response_input_param import FunctionCallOutput
from openai.types.responses import ResponseCustomToolCallOutputParam

__all__ = [
    "ErrorForLLMToSee",
    "HandlerResult",
    "ToolCallFailReason",
    "ToolCallSuccess",
    "ToolCallFailure",
    "ToolCallResult",
]


class ErrorForLLMToSee(Exception):
    """
    Raise one of these in your handler, and it will automatically be stringified
    and placed in the tool response message.
    """

    pass


class HandlerResult[ContextOut](NamedTuple):
    """
    Result of a tool's user-defined `model_tool_handler()`
    """

    result_content: str
    context: ContextOut


type ToolCallFailReason = Literal[
    "invalid_name", "invalid_arguments", "explicit_handler_error"
]


@dataclass(slots=True, frozen=True, kw_only=True)
class _BaseToolCallResult:
    type: Literal["custom", "function"]
    call_id: str
    result_content: str

    @property
    def tool_message(self) -> ChatCompletionToolMessageParam:
        """
        For the Chat Completions API: A role='tool' message param.
        """
        return {
            "role": "tool",
            "tool_call_id": self.call_id,
            "content": self.result_content,
        }

    @property
    def output_item(self) -> FunctionCallOutput | ResponseCustomToolCallOutputParam:
        """
        For the Responses API: Function call output item param.
        """
        if self.type == "function":
            return {
                "type": "function_call_output",
                "call_id": self.call_id,
                "output": self.result_content,
            }
        return {
            "type": "custom_tool_call_output",
            "call_id": self.call_id,
            "output": self.result_content,
        }


@dataclass(slots=True, frozen=True, kw_only=True)
class ToolCallSuccess[ContextOut](_BaseToolCallResult):
    """
    Result of handling a tool call successfully. Includes the output context.
    """

    call_id: str
    result_content: str
    fail_reason: None = None
    exception: None = None
    context: ContextOut


@dataclass(slots=True, frozen=True, kw_only=True)
class ToolCallFailure(_BaseToolCallResult):
    """
    Result of handling a tool call unsuccessfully. Includes the output context.
    """

    call_id: str
    result_content: str
    fail_reason: ToolCallFailReason
    exception: pydantic.ValidationError | ErrorForLLMToSee
    context: None = None


type ToolCallResult[ContextOut] = Annotated[
    ToolCallSuccess[ContextOut] | ToolCallFailure, pydantic.Discriminator("fail_reason")
]
