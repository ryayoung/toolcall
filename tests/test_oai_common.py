from typing import TYPE_CHECKING, Any
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageCustomToolCall,
)
from openai.types.chat.chat_completion_message_custom_tool_call import Custom
from openai.types.responses import ResponseFunctionToolCall, ResponseCustomToolCall
from openai.types.chat.chat_completion_message_tool_call import Function
from toolcall.openai._common import (
    StandardCustomToolDefinition,
    StandardCustomToolTextFormat,
    StandardCustomToolGrammarFormat,
    StandardToolCall,
    ToolCallSuccess,
    ToolCallResult,
)


def test_call():
    calls: list[
        StandardToolCall
        | ChatCompletionMessageCustomToolCall
        | ChatCompletionMessageFunctionToolCall
        | ResponseFunctionToolCall
        | ResponseCustomToolCall
    ] = [
        StandardToolCall(
            type="function",
            id="call_123",
            name="example_tool",
            input='{"param1": "value1", "param2": 42}',
        ),
        ResponseFunctionToolCall(
            type="function_call",
            call_id="call_123",
            name="x",
            arguments="{}",
        ),
        ResponseCustomToolCall(
            type="custom_tool_call",
            call_id="call_123",
            name="x",
            input="hello",
        ),
        ChatCompletionMessageFunctionToolCall(
            id="123", type="function", function=Function(name="x", arguments="{}")
        ),
        ChatCompletionMessageCustomToolCall(
            id="123", type="custom", custom=Custom(name="x", input="{}")
        ),
    ]
    for call in calls:
        _ = StandardToolCall.from_any_call(call)


def test_definition():
    definitions: list[StandardCustomToolDefinition] = [
        StandardCustomToolDefinition(
            name="1",
            description="1",
            format=None,
        ),
        StandardCustomToolDefinition(
            name="1",
            description="1",
            format=StandardCustomToolTextFormat(),
        ),
        StandardCustomToolDefinition(
            name="1",
            description="1",
            format=StandardCustomToolGrammarFormat(
                syntax="lark",
                definition="123",
            ),
        ),
    ]
    for dfn in definitions:
        _ = dfn.tool_def_for_chat_completions_api()
        _ = dfn.tool_def_for_responses_api()


def test_result():
    results: list[ToolCallResult[Any]] = [
        ToolCallSuccess(
            type="function",
            call_id="123",
            result_content="Test content",
            context=None,
        ),
        ToolCallSuccess(
            type="custom",
            call_id="123",
            result_content="Test content",
            context=None,
        ),
    ]
    for result in results:
        tool_message = result.tool_message
        output_item = result.output_item

        if TYPE_CHECKING:
            client = OpenAI()

            client.chat.completions.create(model="gpt-4o", messages=[tool_message])
            client.responses.create(model="gpt-4o", input=[output_item])
