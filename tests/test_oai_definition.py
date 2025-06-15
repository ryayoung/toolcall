from typing import TYPE_CHECKING


def test_definition_aio():
    from toolcall.openai.aio import LLMFunctionTool, LLMFunctionToolGroup

    class Tool(LLMFunctionTool[None, None]):
        foo: str

    tool_def_chat = Tool.model_tool_definition(api="chat.completions")
    tool_def_resp = Tool.model_tool_definition(api="responses")
    tool_def_json_chat = Tool.model_tool_json_format_definition(api="chat.completions")
    tool_def_json_resp = Tool.model_tool_json_format_definition(api="responses")
    _ = Tool.model_tool_pretty_definition()

    group = LLMFunctionToolGroup[None, None].from_list([Tool])

    tool_defs_chat = group.tool_definitions(api="chat.completions")
    tool_defs_resp = group.tool_definitions(api="responses")
    _ = group.pretty_definition()

    if TYPE_CHECKING:
        from openai import OpenAI

        client = OpenAI(api_key="123")

        client.chat.completions.create(
            model="gpt-4o", messages=[], tools=[tool_def_chat]
        )
        client.responses.create(model="gpt-4o", input=[], tools=[tool_def_resp])
        client.chat.completions.create(
            model="gpt-4o", messages=[], tools=tool_defs_chat
        )
        client.responses.create(model="gpt-4o", input=[], tools=tool_defs_resp)
        client.chat.completions.create(
            model="gpt-4o", messages=[], response_format=tool_def_json_chat
        )
        client.responses.create(
            model="gpt-4o", input=[], text={"format": tool_def_json_resp}
        )


def test_definition_core():
    from toolcall.openai.core import LLMFunctionTool, LLMFunctionToolGroup

    class Tool(LLMFunctionTool[None, None]):
        foo: str

    tool_def_chat = Tool.model_tool_definition(api="chat.completions")
    tool_def_resp = Tool.model_tool_definition(api="responses")
    tool_def_json_chat = Tool.model_tool_json_format_definition(api="chat.completions")
    tool_def_json_resp = Tool.model_tool_json_format_definition(api="responses")
    _ = Tool.model_tool_pretty_definition()

    group = LLMFunctionToolGroup[None, None].from_list([Tool])

    tool_defs_chat = group.tool_definitions(api="chat.completions")
    tool_defs_resp = group.tool_definitions(api="responses")
    _ = group.pretty_definition()

    if TYPE_CHECKING:
        from openai import OpenAI

        client = OpenAI(api_key="123")

        client.chat.completions.create(
            model="gpt-4o", messages=[], tools=[tool_def_chat]
        )
        client.responses.create(model="gpt-4o", input=[], tools=[tool_def_resp])
        client.chat.completions.create(
            model="gpt-4o", messages=[], tools=tool_defs_chat
        )
        client.responses.create(model="gpt-4o", input=[], tools=tool_defs_resp)
        client.chat.completions.create(
            model="gpt-4o", messages=[], response_format=tool_def_json_chat
        )
        client.responses.create(
            model="gpt-4o", input=[], text={"format": tool_def_json_resp}
        )
