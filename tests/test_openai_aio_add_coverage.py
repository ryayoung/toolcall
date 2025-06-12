import asyncio
import pytest
from toolcall.openai.aio import (
    LLMFunctionToolGroup,
    LLMFunctionTool,
    standardize_tool_call,
)
from openai.types.chat.chat_completion_message_tool_call import (
    Function,
    ChatCompletionMessageToolCall,
)


def test_add_more_coverage():
    async def main():
        class MyEmptyTool(LLMFunctionTool):
            model_tool_strict = True
            model_tool_custom_json_schema = {"foo": "bar"}
            model_tool_name_generator = lambda name: name

        invalid_tool_call = make_tc_chat("MyEmptyTool", "{Invalid JSON")
        # Won't raise due to missing handler, because we won't even reach handler
        await MyEmptyTool.model_tool_run_tool_call(invalid_tool_call, None)

        with pytest.raises(TypeError):
            standardize_tool_call(1)  # pyright: ignore[reportArgumentType]

        with pytest.raises(NotImplementedError):
            tool_call = make_tc_chat("MyEmptyTool", "{}")
            await MyEmptyTool.model_tool_run_tool_call(tool_call, None)

        class BadHandlerTool(LLMFunctionTool):
            async def model_tool_handler(self, context: None) -> tuple:
                return "invalid"  # type: ignore

        class BadHandlerTool2(LLMFunctionTool):
            async def model_tool_handler(self, context: None) -> tuple:
                return ("x", "y", "z")  # type: ignore

        with pytest.raises(TypeError):
            tool_call = make_tc_chat("BadHandlerTool", "{}")
            await BadHandlerTool.model_tool_run_tool_call(tool_call, None)
            await BadHandlerTool2.model_tool_run_tool_call(tool_call, None)

        group = LLMFunctionToolGroup[None, None].from_list(
            [MyEmptyTool, BadHandlerTool]
        )
        group.tool_names
        group.tool_classes
        group.tool_definitions("chat.completions")
        group.tool_definitions("responses")
        group.pretty_definition("chat.completions")
        group.pretty_definition("responses")
        assert MyEmptyTool in group
        assert "MyEmptyTool" in group

        await group.run_tool_call(make_tc_chat("NotATool", "{}"), None)

    asyncio.run(main())


def make_tc_chat(name: str, args: str) -> ChatCompletionMessageToolCall:
    return ChatCompletionMessageToolCall(
        id="123", type="function", function=Function(name=name, arguments=args)
    )
