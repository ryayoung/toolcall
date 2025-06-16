# File generated from its async equivalent, toolcall/openai/aio/_group.py
from typing import Literal, Sequence, overload
from openai.types.chat import ChatCompletionToolParam
from openai.types.responses import FunctionToolParam
from ._tool import BaseFunctionToolModel
from .._result import ToolCallResult, ToolCallFailure, ErrorForLLMToSee
from .._call import StandardToolCall, AnyToolCall

__all__ = ["FunctionToolGroup"]


class FunctionToolGroup[ContextIn, ContextOut](
    dict[str, type[BaseFunctionToolModel[ContextIn, ContextOut]]]
):
    """
    A group of tools that have the same input and output context types, enabling
    automatic dispatching of tool calls.
    """

    @classmethod
    def from_list[CtxIn, CtxOut](
        cls, tools: Sequence[type[BaseFunctionToolModel[CtxIn, CtxOut]]]
    ) -> "FunctionToolGroup[CtxIn, CtxOut]":
        """
        Create from a list of tool classes.
        """
        return FunctionToolGroup({t.model_tool_name(): t for t in tools})

    def add_tool(self, tool: type[BaseFunctionToolModel[ContextIn, ContextOut]]):
        """
        Add a tool to the group.
        """
        self[tool.model_tool_name()] = tool
        return tool

    def run_tool_call(
        self, call: AnyToolCall, context: ContextIn
    ) -> ToolCallResult[ContextOut]:
        """
        Dispatch a tool call to the tool with a matching name.
        """
        call = StandardToolCall.from_any_call(call)
        if tool := self.get(call.name):
            return tool.model_tool_run_tool_call(call, context)
        exception = ErrorForLLMToSee(f"Function `{call.name}` not found.")
        return ToolCallFailure(
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
        If using async, the calls will be run in parallel.
        """
        return [self.run_tool_call(c, context) for c in calls]

    @overload
    def tool_definitions(
        self, api: Literal["responses"]
    ) -> list[FunctionToolParam]: ...

    @overload
    def tool_definitions(
        self, api: Literal["chat.completions"]
    ) -> list[ChatCompletionToolParam]: ...

    @overload
    def tool_definitions(
        self, api: Literal["chat.completions", "responses"]
    ) -> list[ChatCompletionToolParam] | list[FunctionToolParam]: ...

    def tool_definitions(
        self, api: Literal["chat.completions", "responses"]
    ) -> Sequence[ChatCompletionToolParam | FunctionToolParam]:
        """
        Tool definitions for the `tools` array parameter in the API.
        """
        return [tool.model_tool_definition(api) for tool in self.values()]

    def pretty_definition(self) -> str:
        """
        For development, get a pretty representation of the tool definitions.
        """
        from textwrap import indent

        definitions = [c.model_tool_pretty_definition() for c in self.values()]
        definition = indent(",\n".join(definitions), " " * 4)
        return f"{type(self).__name__}([\n{definition}\n])"
