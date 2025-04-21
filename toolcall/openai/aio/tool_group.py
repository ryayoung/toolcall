from typing import Literal, Self, Sequence, cast, overload
import asyncio
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)
from openai.types.responses.function_tool_param import (
    FunctionToolParam,
)
from openai.types.responses.response_function_tool_call import (
    ResponseFunctionToolCall,
)
from .tool import LLMFunctionTool
from ..common import (
    StandardToolCall,
    ToolCallFailure,
    ToolCallResult,
    standardize_tool_call,
)


class LLMFunctionToolGroup[ContextIn, ContextOut](
    dict[str, type[LLMFunctionTool[ContextIn, ContextOut]]]
):
    """
    A group of tools that have the same input and output context types, enabling
    automatic dispatching of tool calls.
    """

    def add_tool(self, tool: type[LLMFunctionTool[ContextIn, ContextOut]]):
        """
        Can either be used alone, or as a decorator over a tool class.
        """
        self[tool.model_tool_name()] = tool
        # This is dark magic, but it works.
        return cast(..., tool)  # pyright: ignore[reportInvalidTypeForm]

    @property
    def tool_names(self) -> list[str]:
        return [tool.model_tool_name() for tool in self.values()]

    @property
    def tool_classes(self) -> list[type[LLMFunctionTool[ContextIn, ContextOut]]]:
        return list(self.values())

    @overload
    def tool_definitions(  # pragma: no cover
        self, api: Literal["responses"]
    ) -> list[FunctionToolParam]: ...

    @overload
    def tool_definitions(  # pragma: no cover
        self, api: Literal["chat.completions"]
    ) -> list[ChatCompletionToolParam]: ...

    def tool_definitions(
        self, api: Literal["chat.completions", "responses"]
    ) -> list[ChatCompletionToolParam] | list[FunctionToolParam]:
        """
        Tool definitions for the `tools` array in the API request.
        """
        if api == "responses":
            return [tool.model_tool_definition(api) for tool in self.tool_classes]
        return [tool.model_tool_definition(api) for tool in self.tool_classes]

    async def run_tool_call(
        self,
        call: (
            ChatCompletionMessageToolCall | ResponseFunctionToolCall | StandardToolCall
        ),
        context: ContextIn,
    ) -> ToolCallResult[ContextOut]:
        """
        Dispatch a tool call to the tool with a matching name.
        """
        call = standardize_tool_call(call)
        if tool := self.get(call.name):
            return await tool.model_tool_run_tool_call(call, context)
        return ToolCallFailure(
            call_id=call.id,
            result_content=f"Function `{call.name}` not found.",
            fail_reason="invalid_name",
        )

    async def run_tool_calls(
        self,
        calls: (
            list[ChatCompletionMessageToolCall]
            | list[ResponseFunctionToolCall]
            | list[StandardToolCall]
        ),
        context: ContextIn,
    ) -> list[ToolCallResult[ContextOut]]:
        """
        Dispatch multiple tool calls to the tool with a matching name.
        If using async, the calls will be run in parallel.
        """
        coros = [self.run_tool_call(call, context) for call in calls]
        return await asyncio.gather(*coros)

    @classmethod
    def from_list(
        cls, tools: Sequence[type[LLMFunctionTool[ContextIn, ContextOut]]]
    ) -> Self:
        """
        Create from a list of tool classes.
        """
        return cls({tool.model_tool_name(): tool for tool in tools})

    def __contains__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, item: str | type[LLMFunctionTool]
    ):
        if not isinstance(item, str):
            item = item.model_tool_name()
        return super().__contains__(item)

    def pretty_definition(self, api: Literal["chat.completions", "responses"]) -> str:
        """
        For development, get a pretty representation of the tool definitions.
        """
        from textwrap import indent

        definitions = [c.model_tool_pretty_definition(api) for c in self.tool_classes]
        definition = indent(",\n".join(definitions), " " * 4)
        return f"{type(self).__name__}([\n{definition}\n])"
