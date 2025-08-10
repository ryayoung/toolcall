# File generated from its async equivalent, toolcall/openai/aio/_group.py
from typing import Literal, Sequence, overload
from openai.types.chat import (
    ChatCompletionCustomToolParam,
    ChatCompletionFunctionToolParam,
)
from openai.types.responses import CustomToolParam, FunctionToolParam
from ._tool import ToolModelType
from .._common import (
    StandardToolCall,
    AnyToolCall,
    ToolCallResult,
    ToolCallFailure,
    ErrorForLLMToSee,
)

__all__ = ["ToolGroup"]


class ToolGroup[ContextIn, ContextOut](
    dict[str, type[ToolModelType[ContextIn, ContextOut]]]
):
    """
    A group of tools that have the same input and output context types, enabling
    automatic dispatching of tool calls.
    """

    @classmethod
    def from_list[CtxIn, CtxOut](
        cls, tools: Sequence[type[ToolModelType[CtxIn, CtxOut]]]
    ) -> "ToolGroup[CtxIn, CtxOut]":
        """
        Create from a list of tool classes.
        """
        group = ToolGroup({t.model_tool_name(): t for t in tools})
        if len(group) != len(tools):
            raise ValueError(
                f"Cannot create {cls.__name__}: duplicate tool names in {tools}"
            )
        return group

    def run_tool_call(
        self, call: AnyToolCall, context: ContextIn
    ) -> ToolCallResult[ContextOut]:
        """
        Dispatch a tool call to the tool with a matching name.
        """
        call = StandardToolCall.from_any_call(call)

        if tool := self.get(call.name):
            if tool.model_tool_type != call.type:
                # This should go uncaught, because it indicates a user error,
                # NOT an LLM mistake.
                raise ValueError(
                    f"Received a tool call of type '{call.type}' to a tool named, "
                    f"'{call.name}', but this name in the tool group points to "
                    f"'{type(tool).__name__}', which is a '{tool.model_tool_type}' tool."
                )
            return tool.model_tool_run_tool_call(call, context)

        exception = ErrorForLLMToSee(f"Tool, `{call.name}` not found.")
        return ToolCallFailure(
            type=call.type,
            call_id=call.id,
            result_content=str(exception),
            fail_reason="invalid_name",
            exception=exception,
        )

    def run_tool_calls(
        self, calls: Sequence[AnyToolCall], context: ContextIn
    ) -> list[ToolCallResult[ContextOut]]:
        """
        Dispatch multiple tool calls to the tool with a matching name.
        If using async, the calls will run concurrently.
        """
        return [self.run_tool_call(c, context) for c in calls]

    @overload
    def tool_definitions(
        self, api: Literal["responses"]
    ) -> list[CustomToolParam | FunctionToolParam]: ...

    @overload
    def tool_definitions(
        self, api: Literal["chat.completions"]
    ) -> list[ChatCompletionCustomToolParam | ChatCompletionFunctionToolParam]: ...

    @overload
    def tool_definitions(
        self, api: Literal["chat.completions", "responses"]
    ) -> (
        list[ChatCompletionCustomToolParam | ChatCompletionFunctionToolParam]
        | list[CustomToolParam | FunctionToolParam]
    ): ...

    def tool_definitions(
        self, api: Literal["chat.completions", "responses"]
    ) -> Sequence[
        ChatCompletionCustomToolParam
        | ChatCompletionFunctionToolParam
        | CustomToolParam
        | FunctionToolParam
    ]:
        """
        Tool definitions for the `tools` array parameter in the API.
        """
        return [tool.model_tool_definition(api) for tool in self.values()]
