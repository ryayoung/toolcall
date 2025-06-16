from typing import TYPE_CHECKING
import pytest
import asyncio
from openai import OpenAI
from toolcall.openai.aio import (
    StandardToolCall,
    BaseFunctionToolModel,
    FunctionToolGroup,
    ErrorForLLMToSee,
)


def test_definition():
    class Tool(BaseFunctionToolModel[None, None]):
        foo: str

    tool_def_chat = Tool.model_tool_definition(api="chat.completions")
    tool_def_resp = Tool.model_tool_definition(api="responses")
    tool_def_json_chat = Tool.model_tool_json_format_definition(api="chat.completions")
    tool_def_json_resp = Tool.model_tool_json_format_definition(api="responses")
    _ = Tool.model_tool_pretty_definition()

    group = FunctionToolGroup[None, None].from_list([Tool])

    tool_defs_chat = group.tool_definitions(api="chat.completions")
    tool_defs_resp = group.tool_definitions(api="responses")
    _ = group.pretty_definition()

    if TYPE_CHECKING:
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


def test_config():
    class RegularTool(BaseFunctionToolModel[None, None]):
        """A docstring"""

        pass

    assert RegularTool.model_tool_name() == "RegularTool"
    schema = RegularTool.model_tool_json_schema()
    assert schema["properties"] == {}
    definition = RegularTool.model_tool_standard_definition()
    assert definition.description == "A docstring"

    class BaseTool(BaseFunctionToolModel[None, None]):
        model_tool_name_generator = lambda name: name.upper()

    assert BaseTool.model_tool_name() == "BASETOOL"

    class Tool(BaseTool):
        """A docstring"""

        model_tool_custom_name = "custom_tool_name"
        model_tool_custom_description = "custom description"
        model_tool_custom_json_schema = {"1": "1"}

    assert Tool.model_tool_name() == "custom_tool_name"
    assert Tool.model_tool_json_schema() == {"1": "1"}
    definition = Tool.model_tool_standard_definition()
    assert definition.description == "custom description"


def test_handler():
    async def main():
        class NotImplementedTool(BaseFunctionToolModel[None, None]):
            x: int = 0

        group = FunctionToolGroup.from_list([NotImplementedTool])

        @group.add_tool
        class Tool(BaseFunctionToolModel[None, None]):
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


# Keeping these imports right next to where they're used, for easy visual
# confirmation we aren't missing anything.
from examples.aio import (
    chat_group,
    chat_output,
    chat_tool,
    resp_group,
    resp_output,
    resp_tool,
)


def test_examples():
    async def main():
        runs = [
            chat_group.main(),
            chat_output.main(),
            chat_tool.main(),
            resp_group.main(),
            resp_output.main(),
            resp_tool.main(),
        ]
        _ = await asyncio.gather(*runs)

    asyncio.run(main())
