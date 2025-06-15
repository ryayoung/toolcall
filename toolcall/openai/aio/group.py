from typing import Literal, Sequence, cast, overload
import asyncio
from openai.types.chat import ChatCompletionToolParam
from openai.types.responses import FunctionToolParam
from .tool import LLMFunctionTool
from ..result import ToolCallResult, ToolCallFailure, ErrorForLLMToSee
from ..call import StandardToolCall, AnyToolCall

__all__ = ["LLMFunctionToolGroup"]


class LLMFunctionToolGroup[ContextIn, ContextOut](
    dict[str, type[LLMFunctionTool[ContextIn, ContextOut]]]
):
    """
    A group of tools that have the same input and output context types, enabling
    automatic dispatching of tool calls.
    """

    @classmethod
    def from_list[CtxIn, CtxOut](
        cls, tools: Sequence[type[LLMFunctionTool[CtxIn, CtxOut]]]
    ) -> "LLMFunctionToolGroup[CtxIn, CtxOut]":
        """
        Create from a list of tool classes.
        """
        return LLMFunctionToolGroup({t.model_tool_name(): t for t in tools})

    def add_tool(self, tool: type[LLMFunctionTool[ContextIn, ContextOut]]):
        """
        Can either be used alone, or as a decorator over a tool class.
        """
        self[tool.model_tool_name()] = tool
        # This intentionally-incorrect use of `cast()` is **necessary**, because:
        #   1. We're restricting func arg type to a **specialized** generic type.
        #   2. We want to let this function be used as a decorator on a class
        #      without type checkers thinking the class's type has changed.
        # It works by causing type checkers to drop the return type entirely.
        return cast(..., tool)  # pyright: ignore[reportInvalidTypeForm]

    async def run_tool_call(
        self, call: AnyToolCall, context: ContextIn
    ) -> ToolCallResult[ContextOut]:
        """
        Dispatch a tool call to the tool with a matching name.
        """
        call = StandardToolCall.from_any_call(call)
        if tool := self.get(call.name):
            return await tool.model_tool_run_tool_call(call, context)
        exception = ErrorForLLMToSee(f"Function `{call.name}` not found.")
        return ToolCallFailure(
            call_id=call.id,
            result_content=str(exception),
            fail_reason="invalid_name",
            exception=exception,
        )

    async def run_tool_calls(
        self, calls: Sequence[AnyToolCall], context: ContextIn
    ) -> list[ToolCallResult[ContextOut]]:
        """
        Dispatch multiple tool calls to the tool with a matching name.
        If using async, the calls will be run in parallel.
        """
        return await asyncio.gather(*[self.run_tool_call(c, context) for c in calls])

    @overload
    def tool_definitions(  # pragma: no cover
        self, api: Literal["responses"]
    ) -> list[FunctionToolParam]: ...

    @overload
    def tool_definitions(  # pragma: no cover
        self, api: Literal["chat.completions"]
    ) -> list[ChatCompletionToolParam]: ...

    @overload
    def tool_definitions(  # pragma: no cover
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
