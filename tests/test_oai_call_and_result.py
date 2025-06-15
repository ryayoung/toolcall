import pytest
import asyncio
from typing import TYPE_CHECKING
from toolcall.openai.call import StandardToolCall
from toolcall.openai.result import ToolCallSuccess
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageToolCall
from openai.types.responses import ResponseFunctionToolCall
from openai.types.chat.chat_completion_message_tool_call import Function


def test_call():
    std_call = StandardToolCall(
        id="call_123",
        name="example_tool",
        arguments='{"param1": "value1", "param2": 42}',
    )
    call = StandardToolCall.from_any_call(std_call)
    assert call is std_call

    chat_call = ChatCompletionMessageToolCall(
        id="123", type="function", function=Function(name="x", arguments="{}")
    )
    call = StandardToolCall.from_any_call(chat_call)

    resp_call = ResponseFunctionToolCall(
        id="123",
        call_id="call_123",
        type="function_call",
        name="x",
        arguments="{}",
    )
    call = StandardToolCall.from_any_call(resp_call)


def test_result():
    result = ToolCallSuccess(
        call_id="123",
        result_content="Test content",
        context=None,
    )

    tool_message = result.tool_message
    output_item = result.output_item

    if TYPE_CHECKING:
        client = OpenAI()

        client.chat.completions.create(model="gpt-4o", messages=[tool_message])
        client.responses.create(model="gpt-4o", input=[output_item])


def test_call_handler_core():
    def main():
        from toolcall.openai.core import (
            LLMFunctionTool,
            LLMFunctionToolGroup,
            ErrorForLLMToSee,
        )

        class NotImplementedTool(LLMFunctionTool[None, None]):
            x: int = 0

        group = LLMFunctionToolGroup.from_list([NotImplementedTool])

        @group.add_tool
        class Tool(LLMFunctionTool[None, None]):
            x: int = 0

            def model_tool_handler(self, _) -> tuple[str, None]:
                if self.x == 1:
                    # except branch
                    raise ErrorForLLMToSee("hi from error")
                if self.x == 2:
                    # TypeError: not a tuple
                    return "hi"  # type: ignore
                if self.x == 3:
                    # TypeError: too long
                    return "hi", None, None  # type: ignore
                return "hi", None

        group.add_tool(Tool)

        def make_call(args: str) -> StandardToolCall:
            return StandardToolCall(id="", name="Tool", arguments=args)

        group.run_tool_calls(
            [
                make_call("{}"),
                StandardToolCall(id="", name="INVALID", arguments="{}"),
                make_call('{"x": "foo"}'),
                make_call('{"x": 1}'),
            ],
            None,
        )
        with pytest.raises(NotImplementedError):
            group.run_tool_call(
                StandardToolCall(id="", name="NotImplementedTool", arguments="{}"),
                None,
            )
        with pytest.raises(TypeError):
            group.run_tool_call(make_call('{"x": 2}'), None)
        with pytest.raises(TypeError):
            group.run_tool_call(make_call('{"x": 3}'), None)

    main()


def test_call_handler_aio():
    async def main():
        from toolcall.openai.aio import (
            LLMFunctionTool,
            LLMFunctionToolGroup,
            ErrorForLLMToSee,
        )

        class NotImplementedTool(LLMFunctionTool[None, None]):
            x: int = 0

        group = LLMFunctionToolGroup.from_list([NotImplementedTool])

        @group.add_tool
        class Tool(LLMFunctionTool[None, None]):
            x: int = 0

            async def model_tool_handler(self, _) -> tuple[str, None]:
                if self.x == 1:
                    # except branch
                    raise ErrorForLLMToSee("hi from error")
                if self.x == 2:
                    # TypeError: not a tuple
                    return "hi"  # type: ignore
                if self.x == 3:
                    # TypeError: too long
                    return "hi", None, None  # type: ignore
                return "hi", None

        group.add_tool(Tool)

        def make_call(args: str) -> StandardToolCall:
            return StandardToolCall(id="", name="Tool", arguments=args)

        await group.run_tool_calls(
            [
                make_call("{}"),
                StandardToolCall(id="", name="INVALID", arguments="{}"),
                make_call('{"x": "foo"}'),
                make_call('{"x": 1}'),
            ],
            None,
        )
        with pytest.raises(NotImplementedError):
            await group.run_tool_call(
                StandardToolCall(id="", name="NotImplementedTool", arguments="{}"),
                None,
            )
        with pytest.raises(TypeError):
            await group.run_tool_call(make_call('{"x": 2}'), None)
        with pytest.raises(TypeError):
            await group.run_tool_call(make_call('{"x": 3}'), None)

    asyncio.run(main())
