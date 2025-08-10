from typing import Literal
from toolcall.openai.aio import (
    BaseFunctionToolModel,
    BaseCustomToolModel,
    ToolGroup,
)


# A `type="custom"` tool that takes arbitrary text input
class bio(BaseCustomToolModel[None, None]):
    """Saves a memory about the user."""

    async def model_tool_handler(self, _: None):
        print(f"LLM gave us '{self.input}'...")
        return f"Memory updated.", None


# A `type="custom"` tool that takes text input, constrained by regex.
class timestamp(BaseCustomToolModel[None, None]):
    """Saves a timestamp in ISO 24-hr format."""

    model_tool_format = {
        "type": "grammar",
        "syntax": "regex",
        "definition": r"^\d{4}-\d{2}-\d{2}T\d{2}$",
    }

    async def model_tool_handler(self, _: None):
        print(f'"LLM gave us {self.input}..."')
        return f"Timestamp saved.", None


# A `type="function"` tool that takes a JSON object input.
class SayHelloFunctionTool(BaseFunctionToolModel[None, None]):
    """Say hello to someone."""

    name: Literal["Alice", "Jeff"]
    """Name of the person to greet."""

    model_tool_custom_name = "say_hello"

    async def model_tool_handler(self, _: None):
        return f"Message delivered to {self.name}.", None


# Container for our tools that can generate the `tools` array for API calls,
# and dispatch tool calls to the correct tool in a type-safe manner.
tool_group = ToolGroup.from_list([bio, timestamp, SayHelloFunctionTool])

import json

for tool_def in tool_group.tool_definitions(api="responses"):
    print(json.dumps(tool_def, indent=2))


from openai import AsyncOpenAI


async def main():
    client = AsyncOpenAI()
    response = await client.responses.create(
        input="Use the timestamp tool to save a timestamp for August 7th 2025 at 10AM.",
        model="gpt-5",
        tools=tool_group.tool_definitions(api="responses"),
    )
    # Blindly assuming it gave us a tool call...
    tool_call = response.output[-1]
    assert tool_call.type == "custom_tool_call" or tool_call.type == "function_call"

    tool_results = await tool_group.run_tool_calls([tool_call], None)
    print(json.dumps(tool_results[0].output_item, indent=2))


import asyncio

asyncio.run(main())
