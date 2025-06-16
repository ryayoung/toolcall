from typing import TYPE_CHECKING
from openai import OpenAI
from toolcall.openai._call import StandardToolCall
from toolcall.openai._result import ToolCallSuccess
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
