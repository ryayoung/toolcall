from typing import Annotated, Iterable, Literal, NamedTuple
from functools import cached_property
from dataclasses import dataclass
from openai.types.chat import ChatCompletionToolParam
from openai.types.responses import FunctionToolParam
from openai.types.shared_params import FunctionDefinition
import pydantic
from openai.types.chat.chat_completion_content_part_text_param import (
    ChatCompletionContentPartTextParam,
)
from openai.types.chat.chat_completion_tool_message_param import (
    ChatCompletionToolMessageParam,
)
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)
from openai.types.responses.response_input_param import FunctionCallOutput
from openai.types.responses.response_function_tool_call import (
    ResponseFunctionToolCall,
)


class ToolErrorMessageForLLMToSee(Exception):
    """
    Raise one of these in your handler, and it will automatically be stringified
    and placed in the tool response message.
    """

    pass


class ToolHandlerResult[ContextOut](NamedTuple):
    """
    Result of a tool's user-defined `model_tool_handler()`
    """

    result_content: str
    context: ContextOut


type ToolCallFailReason = Literal[
    "invalid_name",
    "invalid_arguments",
    "explicit_handler_error",
]

type ToolMessageContent = str | Iterable[ChatCompletionContentPartTextParam]


@dataclass(frozen=True, kw_only=True)
class BaseToolCallResult:
    call_id: str
    result_content: ToolMessageContent

    @cached_property
    def tool_message(self) -> ChatCompletionToolMessageParam:
        """
        For the Chat Completions API: A role='tool' message param.
        """
        return {
            "role": "tool",
            "tool_call_id": self.call_id,
            "content": self.result_content,
        }

    @cached_property
    def output_item(self) -> FunctionCallOutput:
        """
        For the Responses API: Function call output item param.
        """
        assert isinstance(self.result_content, str), "Responses API requires str"
        return {
            "call_id": self.call_id,
            "type": "function_call_output",
            "output": self.result_content,
        }


@dataclass(frozen=True, kw_only=True)
class ToolCallSuccess[ContextOut](BaseToolCallResult):
    """
    Result of handling a tool call successfully. Includes the output context.
    """

    fail_reason: None = None
    context: ContextOut


@dataclass(frozen=True, kw_only=True)
class ToolCallFailure(BaseToolCallResult):
    """
    Result of handling a tool call unsuccessfully. Includes the output context.
    """

    fail_reason: ToolCallFailReason
    context: None = None


type ToolCallResult[T] = Annotated[
    ToolCallSuccess[T] | ToolCallFailure, pydantic.Discriminator("fail_reason")
]


@dataclass(slots=True, frozen=True, kw_only=True)
class StandardToolCall:
    """
    A common structure to store the data from a tool call from various API types.
    """

    id: str
    name: str
    arguments: str


def standardize_tool_call(
    call: ChatCompletionMessageToolCall | ResponseFunctionToolCall | StandardToolCall,
) -> StandardToolCall:
    """
    Standardize tool call from different API types, to a common structure.
    """
    if isinstance(call, StandardToolCall):
        return call
    if isinstance(call, ChatCompletionMessageToolCall):
        return StandardToolCall(
            id=call.id,
            name=call.function.name,
            arguments=call.function.arguments,
        )
    if isinstance(call, ResponseFunctionToolCall):
        return StandardToolCall(
            id=call.call_id,
            name=call.name,
            arguments=call.arguments,
        )
    raise TypeError(f"Unsupported tool call type: {type(call)}")


def tool_def_for_chat_completions_api(
    name: str, description: str | None, strict: bool, schema: dict
) -> ChatCompletionToolParam:
    """
    Create a tool definition for the `tools` array in the Chat Completions API
    """
    function: FunctionDefinition = {
        "name": name,
        "description": description or "",
        "parameters": schema,
        "strict": strict,
    }
    return {"type": "function", "function": function}


def tool_def_for_responses_api(
    name: str, description: str | None, strict: bool, schema: dict
) -> FunctionToolParam:
    """
    Create a tool definition for the `tools` array in the Responses API
    """
    return {
        "type": "function",
        "name": name,
        "description": description or "",
        "parameters": schema,
        "strict": strict,
    }
