# toolcall

[![PyPI](https://img.shields.io/pypi/v/toolcall)](https://pypi.org/project/toolcall/)
[![Tests](https://github.com/ryayoung/toolcall/actions/workflows/tests.yml/badge.svg)](https://github.com/ryayoung/toolcall/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/ryayoung/toolcall/branch/main/graph/badge.svg)](https://codecov.io/gh/ryayoung/toolcall)
[![License](https://img.shields.io/github/license/ryayoung/toolcall)](https://github.com/ryayoung/toolcall/blob/main/LICENSE)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Pyright](https://img.shields.io/badge/type%20checker-pyright-blue)](https://github.com/microsoft/pyright)


*The agentic framework for building without an agentic framework.*

```
pip install toolcall
```

`toolcall` provides function-tool primitives that enhance (**without
loss of control** over) the process of manually defining your own function tools,
handling calls to them, and dynamically dispatching calls across a group of tools, with
**static type safety**.

---

*Who are you?*

- You want a framework to handle tool orchestration and control-flow for you:
    - ⛔ Do **not** use `toolcall`. There are plenty of good frameworks out there.
- You define, dispatch, and handle function tool calls yourself:
    - ✅ You **should** be using `toolcall`. It will make your life easier.


## Learn-by-example Documentation

Below is one end-to-end tool-calling workflow built with `toolcall`, in several variations.

### 1. Manual tool call -> handler dispatch (no tool group)

> [!CAUTION]
> 
> This is **not** recommended, unless you only have 1 tool, or each tool's handler has different
input/output context type requirements, which is unlikely for most use cases.

<details><summary>OpenAI - Chat Completions API</summary>

```python
import os
import json
from typing import Literal
from openai import OpenAI
from openai.types.chat.chat_completion_message_param import (
    ChatCompletionMessageParam,
)
from toolcall.openai.core import (
    LLMFunctionTool,
    ToolHandlerResult,
    ToolErrorMessageForLLMToSee,
)

openai_client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
)

SYSTEM_PROMPT = """
You are a helpful assistant. You have several function tools available. Use as needed.

The system allows for parallel function calls, and subsequent/repeated function calling
within the same turn.
""".strip()

USER_PROMPT = """
What's the weather in san francisco, and apple's stock price?
""".strip()

def main():
    conversation: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ]
    conversation = assistant_take_turn(conversation)
    for msg in conversation:
        print("-" * 80 + "\n" + json.dumps(msg, indent=2).strip("{}"))

# Simple style of tool definition:
class get_weather(LLMFunctionTool):  # <-- Pydantic BaseModel extension
    """Get the weather somewhere."""

    city: str
    """City to get the weather for."""

    state: Literal["California", "New York", "Texas"]
    """State where the city is. Only a few are available."""

    # Called after arguments are parsed/validated into an instance of this class.
    # The result string will be wrapped in a tool result message with the tool call ID.
    def model_tool_handler(self, context: None) -> tuple[str, float]:
        if self.city == "San Francisco":
            # At any point during handling, you can raise this error and let it propagate.
            # It will be caught and used as the result tool message's content. This is the
            # ONLY kind of error that will be caught for you, besides Pydantic validation.
            raise ToolErrorMessageForLLMToSee(
                "Weather unavailable for San Francisco. Please get the weather for a "
                "nearby city, before responding to the user. Don't ask first. Just call "
                "this function again."
            )

        result = f"It's currently 30 degrees in {self.city}, {self.state}."
        # Result for the LLM, plus any context you want to pass back to yourself.
        return result, 1.234

# More explicit style:
class StockPriceTool(LLMFunctionTool[None, float]):
    ticker: str
    exchange: Literal["NASDAQ", "NYSE"]

    # By default, the class name is used. You can override it:
    model_tool_custom_name = "get_stock_price"

    # By default, the class docstring is used. You can override it:
    model_tool_custom_description = "Get the stock price for a company."

    # By default, Pydantic generates the JSON Schema. You can override it:
    model_tool_custom_json_schema = {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "Ticker symbol of the company.",
            },
            "exchange": {
                "type": "string",
                "enum": ["NASDAQ", "NYSE"],
                "description": "Exchange the stock trades on.",
            },
        },
        "required": ["ticker", "exchange"],
    }

    def model_tool_handler(self, context: None) -> ToolHandlerResult[float]:
        result = f"{self.ticker} is currently trading at $100."
        return ToolHandlerResult(result_content=result, context=1.234)

def assistant_take_turn(
    messages: list[ChatCompletionMessageParam],
) -> list[ChatCompletionMessageParam]:
    response = openai_client.chat.completions.create(
        messages=messages,
        model="gpt-4.1-mini",
        tools=[
            get_weather.model_tool_definition(api="chat.completions"),
            StockPriceTool.model_tool_definition(api="chat.completions"),
        ],
    )
    message = response.choices[0].message

    message_param = message.model_dump(exclude_none=True, exclude_unset=True)
    messages.append(message_param)  # pyright: ignore[reportArgumentType]

    if not message.tool_calls:
        return messages

    for call in message.tool_calls:

        if call.function.name == get_weather.model_tool_name():
            result = get_weather.model_tool_run_tool_call(call, None)
            tool_message = result.tool_message

            if result.fail_reason == "explicit_handler_error":
                print("get_weather raised a ToolErrorMessageForLLMToSee()")

        elif call.function.name == StockPriceTool.model_tool_name():
            result = StockPriceTool.model_tool_run_tool_call(call, None)
            tool_message = result.tool_message

            if result.fail_reason is None:
                context: float = result.context

        else:
            tool_message: ChatCompletionMessageParam = {
                "role": "tool",
                "tool_call_id": call.id,
                "content": f"Function `{call.function.name}` not found.",
            }
        messages.append(tool_message)

    return assistant_take_turn(messages)

main()
```

Output:

```
--------------------------------------------------------------------------------

  "role": "user",
  "content": "What's the weather in san francisco, and apple's stock price?"

--------------------------------------------------------------------------------

  "role": "assistant",
  "annotations": [],
  "tool_calls": [
    {
      "id": "call_eC0HJwZsnk6N2SzaZBViJlah",
      "function": {
        "arguments": "{\"city\": \"San Francisco\", \"state\": \"California\"}",
        "name": "get_weather"
      },
      "type": "function"
    },
    {
      "id": "call_MgtnVah214eFRSe5qJWnFqkz",
      "function": {
        "arguments": "{\"ticker\": \"AAPL\", \"exchange\": \"NASDAQ\"}",
        "name": "get_stock_price"
      },
      "type": "function"
    }
  ]

--------------------------------------------------------------------------------

  "role": "tool",
  "tool_call_id": "call_eC0HJwZsnk6N2SzaZBViJlah",
  "content": "Weather unavailable for San Francisco. Please get the weather for a nearby city, before responding to the user. Don't ask first. Just call this function again."

--------------------------------------------------------------------------------

  "role": "tool",
  "tool_call_id": "call_MgtnVah214eFRSe5qJWnFqkz",
  "content": "AAPL is currently trading at $100."

--------------------------------------------------------------------------------

  "role": "assistant",
  "annotations": [],
  "tool_calls": [
    {
      "id": "call_woAZj5hAd7i21lolxoYhCYrL",
      "function": {
        "arguments": "{\"city\":\"Oakland\",\"state\":\"California\"}",
        "name": "get_weather"
      },
      "type": "function"
    }
  ]

--------------------------------------------------------------------------------

  "role": "tool",
  "tool_call_id": "call_woAZj5hAd7i21lolxoYhCYrL",
  "content": "It's currently 30 degrees in Oakland, California."

--------------------------------------------------------------------------------

  "content": "The weather in San Francisco is currently not available, but nearby in Oakland, California, it is 30 degrees. Apple's stock price (AAPL) is currently trading at $100.",
  "role": "assistant",
  "annotations": []
```

</details>


<details><summary>OpenAI - Chat Completions API - Async</summary>

```python
import os
import json
import asyncio
from typing import Literal
from openai import AsyncOpenAI
from openai.types.chat.chat_completion_message_param import (
    ChatCompletionMessageParam,
)
from toolcall.openai.aio import (
    LLMFunctionTool,
    ToolHandlerResult,
    ToolErrorMessageForLLMToSee,
)

openai_client = AsyncOpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
)

SYSTEM_PROMPT = """
You are a helpful assistant. You have several function tools available. Use as needed.

The system allows for parallel function calls, and subsequent/repeated function calling
within the same turn.
""".strip()

USER_PROMPT = """
What's the weather in san francisco, and apple's stock price?
""".strip()

async def main():
    conversation: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ]
    conversation = await assistant_take_turn(conversation)
    for msg in conversation:
        print("-" * 80 + "\n" + json.dumps(msg, indent=2).strip("{}"))

# Simple style of tool definition:
class get_weather(LLMFunctionTool):  # <-- Pydantic BaseModel extension
    """Get the weather somewhere."""

    city: str
    """City to get the weather for."""

    state: Literal["California", "New York", "Texas"]
    """State where the city is. Only a few are available."""

    # Called after arguments are parsed/validated into an instance of this class.
    # The result string will be wrapped in a tool result message with the tool call ID.
    async def model_tool_handler(self, context: None) -> tuple[str, float]:
        if self.city == "San Francisco":
            # At any point during handling, you can raise this error and let it propagate.
            # It will be caught and used as the result tool message's content. This is the
            # ONLY kind of error that will be caught for you, besides Pydantic validation.
            raise ToolErrorMessageForLLMToSee(
                "Weather unavailable for San Francisco. Please get the weather for a "
                "nearby city, before responding to the user. Don't ask first. Just call "
                "this function again."
            )

        result = f"It's currently 30 degrees in {self.city}, {self.state}."
        # Result for the LLM, plus any context you want to pass back to yourself.
        return result, 1.234

# More explicit style:
class StockPriceTool(LLMFunctionTool[None, float]):
    ticker: str
    exchange: Literal["NASDAQ", "NYSE"]

    # By default, the class name is used. You can override it:
    model_tool_custom_name = "get_stock_price"

    # By default, the class docstring is used. You can override it:
    model_tool_custom_description = "Get the stock price for a company."

    # By default, Pydantic generates the JSON Schema. You can override it:
    model_tool_custom_json_schema = {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "Ticker symbol of the company.",
            },
            "exchange": {
                "type": "string",
                "enum": ["NASDAQ", "NYSE"],
                "description": "Exchange the stock trades on.",
            },
        },
        "required": ["ticker", "exchange"],
    }

    async def model_tool_handler(self, context: None) -> ToolHandlerResult[float]:
        result = f"{self.ticker} is currently trading at $100."
        return ToolHandlerResult(result_content=result, context=1.234)

async def assistant_take_turn(
    messages: list[ChatCompletionMessageParam],
) -> list[ChatCompletionMessageParam]:
    response = await openai_client.chat.completions.create(
        messages=messages,
        model="gpt-4.1-mini",
        tools=[
            get_weather.model_tool_definition(api="chat.completions"),
            StockPriceTool.model_tool_definition(api="chat.completions"),
        ],
    )
    message = response.choices[0].message

    message_param = message.model_dump(exclude_none=True, exclude_unset=True)
    messages.append(message_param)  # pyright: ignore[reportArgumentType]

    if not message.tool_calls:
        return messages

    for call in message.tool_calls:

        if call.function.name == get_weather.model_tool_name():
            result = await get_weather.model_tool_run_tool_call(call, None)
            tool_message = result.tool_message

            if result.fail_reason == "explicit_handler_error":
                print("get_weather raised a ToolErrorMessageForLLMToSee()")

        elif call.function.name == StockPriceTool.model_tool_name():
            result = await StockPriceTool.model_tool_run_tool_call(call, None)
            tool_message = result.tool_message

            if result.fail_reason is None:
                context: float = result.context

        else:
            tool_message: ChatCompletionMessageParam = {
                "role": "tool",
                "tool_call_id": call.id,
                "content": f"Function `{call.function.name}` not found.",
            }
        messages.append(tool_message)

    return await assistant_take_turn(messages)

asyncio.run(main())
```

Output:

```
--------------------------------------------------------------------------------

  "role": "user",
  "content": "What's the weather in san francisco, and apple's stock price?"

--------------------------------------------------------------------------------

  "role": "assistant",
  "annotations": [],
  "tool_calls": [
    {
      "id": "call_eC0HJwZsnk6N2SzaZBViJlah",
      "function": {
        "arguments": "{\"city\": \"San Francisco\", \"state\": \"California\"}",
        "name": "get_weather"
      },
      "type": "function"
    },
    {
      "id": "call_MgtnVah214eFRSe5qJWnFqkz",
      "function": {
        "arguments": "{\"ticker\": \"AAPL\", \"exchange\": \"NASDAQ\"}",
        "name": "get_stock_price"
      },
      "type": "function"
    }
  ]

--------------------------------------------------------------------------------

  "role": "tool",
  "tool_call_id": "call_eC0HJwZsnk6N2SzaZBViJlah",
  "content": "Weather unavailable for San Francisco. Please get the weather for a nearby city, before responding to the user. Don't ask first. Just call this function again."

--------------------------------------------------------------------------------

  "role": "tool",
  "tool_call_id": "call_MgtnVah214eFRSe5qJWnFqkz",
  "content": "AAPL is currently trading at $100."

--------------------------------------------------------------------------------

  "role": "assistant",
  "annotations": [],
  "tool_calls": [
    {
      "id": "call_woAZj5hAd7i21lolxoYhCYrL",
      "function": {
        "arguments": "{\"city\":\"Oakland\",\"state\":\"California\"}",
        "name": "get_weather"
      },
      "type": "function"
    }
  ]

--------------------------------------------------------------------------------

  "role": "tool",
  "tool_call_id": "call_woAZj5hAd7i21lolxoYhCYrL",
  "content": "It's currently 30 degrees in Oakland, California."

--------------------------------------------------------------------------------

  "content": "The weather in San Francisco is currently not available, but nearby in Oakland, California, it is 30 degrees. Apple's stock price (AAPL) is currently trading at $100.",
  "role": "assistant",
  "annotations": []
```

</details>


<details><summary>OpenAI - Responses API</summary>

```python
import os
import json
from typing import Literal
from openai import OpenAI
from openai.types.responses.response_input_param import ResponseInputItemParam
from toolcall.openai.core import (
    LLMFunctionTool,
    ToolHandlerResult,
    ToolErrorMessageForLLMToSee,
)

openai_client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
)

SYSTEM_PROMPT = """
You are a helpful assistant. You have several function tools available. Use as needed.

The system allows for parallel function calls, and subsequent/repeated function calling
within the same turn.
""".strip()

USER_PROMPT = """
What's the weather in san francisco, and apple's stock price?
""".strip()

def main():
    conversation: list[ResponseInputItemParam] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ]
    conversation = assistant_take_turn(conversation)
    for msg in conversation:
        print("-" * 80 + "\n" + json.dumps(msg, indent=2).strip("{}"))

# Simple style of tool definition:
class get_weather(LLMFunctionTool):  # <-- Pydantic BaseModel extension
    """Get the weather somewhere."""

    city: str
    """City to get the weather for."""

    state: Literal["California", "New York", "Texas"]
    """State where the city is. Only a few are available."""

    # Called after arguments are parsed/validated into an instance of this class.
    # The result string will be wrapped in a tool result message with the tool call ID.
    def model_tool_handler(self, context: None) -> tuple[str, float]:
        if self.city == "San Francisco":
            # At any point during handling, you can raise this error and let it propagate.
            # It will be caught and used as the result tool message's content. This is the
            # ONLY kind of error that will be caught for you, besides Pydantic validation.
            raise ToolErrorMessageForLLMToSee(
                "Weather unavailable for San Francisco. Please get the weather for a "
                "nearby city, before responding to the user. Don't ask first. Just call "
                "this function again."
            )

        result = f"It's currently 30 degrees in {self.city}, {self.state}."
        # Result for the LLM, plus any context you want to pass back to yourself.
        return result, 1.234

# More explicit style:
class StockPriceTool(LLMFunctionTool[None, float]):
    ticker: str
    exchange: Literal["NASDAQ", "NYSE"]

    # By default, the class name is used. You can override it:
    model_tool_custom_name = "get_stock_price"

    # By default, the class docstring is used. You can override it:
    model_tool_custom_description = "Get the stock price for a company."

    # By default, Pydantic generates the JSON Schema. You can override it:
    model_tool_custom_json_schema = {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "Ticker symbol of the company.",
            },
            "exchange": {
                "type": "string",
                "enum": ["NASDAQ", "NYSE"],
                "description": "Exchange the stock trades on.",
            },
        },
        "required": ["ticker", "exchange"],
    }

    def model_tool_handler(self, context: None) -> ToolHandlerResult[float]:
        result = f"{self.ticker} is currently trading at $100."
        return ToolHandlerResult(result_content=result, context=1.234)

def assistant_take_turn(
    messages: list[ResponseInputItemParam],
) -> list[ResponseInputItemParam]:
    response = openai_client.responses.create(
        input=messages,
        model="gpt-4.1-mini",
        tools=[
            get_weather.model_tool_definition(api="responses"),
            StockPriceTool.model_tool_definition(api="responses"),
        ],
    )

    for item in response.output:
        item_param = item.model_dump(exclude_none=True, exclude_unset=True)
        messages.append(item_param)  # pyright: ignore[reportArgumentType]

    tool_calls = [item for item in response.output if item.type == "function_call"]

    if not tool_calls:
        return messages

    for call in tool_calls:

        if call.name == get_weather.model_tool_name():
            result = get_weather.model_tool_run_tool_call(call, None)
            output_item = result.output_item

            if result.fail_reason == "explicit_handler_error":
                print("get_weather raised a ToolErrorMessageForLLMToSee()")

        elif call.name == StockPriceTool.model_tool_name():
            result = StockPriceTool.model_tool_run_tool_call(call, None)
            output_item = result.output_item

            if result.fail_reason is None:
                context: float = result.context

        else:
            output_item: ResponseInputItemParam = {
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": f"Function `{call.name}` not found.",
            }
        messages.append(output_item)

    return assistant_take_turn(messages)

main()
```

Output:

```
--------------------------------------------------------------------------------

  "role": "user",
  "content": "What's the weather in san francisco, and apple's stock price?"

--------------------------------------------------------------------------------

  "arguments": "{\"city\":\"San Francisco\",\"state\":\"California\"}",
  "call_id": "call_QIkpagWx41j46dIpgpvhQXJP",
  "name": "get_weather",
  "type": "function_call",
  "id": "fc_680545a033dc8191b638b047a35a2566039bcbb9768be33e",
  "status": "completed"

--------------------------------------------------------------------------------

  "arguments": "{\"ticker\":\"AAPL\",\"exchange\":\"NASDAQ\"}",
  "call_id": "call_aHlGemYViha773MCDO0Bc8tf",
  "name": "get_stock_price",
  "type": "function_call",
  "id": "fc_680545a0b0ec819195c0bbdb71fced0e039bcbb9768be33e",
  "status": "completed"

--------------------------------------------------------------------------------

  "call_id": "call_QIkpagWx41j46dIpgpvhQXJP",
  "type": "function_call_output",
  "output": "Weather unavailable for San Francisco. Please get the weather for a nearby city, before responding to the user. Don't ask first. Just call this function again."

--------------------------------------------------------------------------------

  "call_id": "call_aHlGemYViha773MCDO0Bc8tf",
  "type": "function_call_output",
  "output": "AAPL is currently trading at $100."

--------------------------------------------------------------------------------

  "arguments": "{\"city\":\"Oakland\",\"state\":\"California\"}",
  "call_id": "call_PhH9GVq8yOvAKZSZPOg45JdF",
  "name": "get_weather",
  "type": "function_call",
  "id": "fc_680545a1c9c8819189b6f1a61b817447039bcbb9768be33e",
  "status": "completed"

--------------------------------------------------------------------------------

  "call_id": "call_PhH9GVq8yOvAKZSZPOg45JdF",
  "type": "function_call_output",
  "output": "It's currently 30 degrees in Oakland, California."

--------------------------------------------------------------------------------

  "id": "msg_680545a2aa5c819187191c3855009564039bcbb9768be33e",
  "content": [
    {
      "annotations": [],
      "text": "The weather in Oakland, near San Francisco, is currently 30 degrees. Apple's stock price is $100.",
      "type": "output_text"
    }
  ],
  "role": "assistant",
  "status": "completed",
  "type": "message"
```

</details>


<details><summary>OpenAI - Responses API - Async</summary>

```python
import os
import json
import asyncio
from typing import Literal
from openai import AsyncOpenAI
from openai.types.responses.response_input_param import ResponseInputItemParam
from toolcall.openai.aio import (
    LLMFunctionTool,
    ToolHandlerResult,
    ToolErrorMessageForLLMToSee,
)

openai_client = AsyncOpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
)

SYSTEM_PROMPT = """
You are a helpful assistant. You have several function tools available. Use as needed.

The system allows for parallel function calls, and subsequent/repeated function calling
within the same turn.
""".strip()

USER_PROMPT = """
What's the weather in san francisco, and apple's stock price?
""".strip()

async def main():
    conversation: list[ResponseInputItemParam] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ]
    conversation = await assistant_take_turn(conversation)
    for msg in conversation:
        print("-" * 80 + "\n" + json.dumps(msg, indent=2).strip("{}"))

# Simple style of tool definition:
class get_weather(LLMFunctionTool):  # <-- Pydantic BaseModel extension
    """Get the weather somewhere."""

    city: str
    """City to get the weather for."""

    state: Literal["California", "New York", "Texas"]
    """State where the city is. Only a few are available."""

    # Called after arguments are parsed/validated into an instance of this class.
    # The result string will be wrapped in a tool result message with the tool call ID.
    async def model_tool_handler(self, context: None) -> tuple[str, float]:
        if self.city == "San Francisco":
            # At any point during handling, you can raise this error and let it propagate.
            # It will be caught and used as the result tool message's content. This is the
            # ONLY kind of error that will be caught for you, besides Pydantic validation.
            raise ToolErrorMessageForLLMToSee(
                "Weather unavailable for San Francisco. Please get the weather for a "
                "nearby city, before responding to the user. Don't ask first. Just call "
                "this function again."
            )

        result = f"It's currently 30 degrees in {self.city}, {self.state}."
        # Result for the LLM, plus any context you want to pass back to yourself.
        return result, 1.234

# More explicit style:
class StockPriceTool(LLMFunctionTool[None, float]):
    ticker: str
    exchange: Literal["NASDAQ", "NYSE"]

    # By default, the class name is used. You can override it:
    model_tool_custom_name = "get_stock_price"

    # By default, the class docstring is used. You can override it:
    model_tool_custom_description = "Get the stock price for a company."

    # By default, Pydantic generates the JSON Schema. You can override it:
    model_tool_custom_json_schema = {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "Ticker symbol of the company.",
            },
            "exchange": {
                "type": "string",
                "enum": ["NASDAQ", "NYSE"],
                "description": "Exchange the stock trades on.",
            },
        },
        "required": ["ticker", "exchange"],
    }

    async def model_tool_handler(self, context: None) -> ToolHandlerResult[float]:
        result = f"{self.ticker} is currently trading at $100."
        return ToolHandlerResult(result_content=result, context=1.234)

async def assistant_take_turn(
    messages: list[ResponseInputItemParam],
) -> list[ResponseInputItemParam]:
    response = await openai_client.responses.create(
        input=messages,
        model="gpt-4.1-mini",
        tools=[
            get_weather.model_tool_definition(api="responses"),
            StockPriceTool.model_tool_definition(api="responses"),
        ],
    )

    for item in response.output:
        item_param = item.model_dump(exclude_none=True, exclude_unset=True)
        messages.append(item_param)  # pyright: ignore[reportArgumentType]

    tool_calls = [item for item in response.output if item.type == "function_call"]

    if not tool_calls:
        return messages

    for call in tool_calls:

        if call.name == get_weather.model_tool_name():
            result = await get_weather.model_tool_run_tool_call(call, None)
            output_item = result.output_item

            if result.fail_reason == "explicit_handler_error":
                print("get_weather raised a ToolErrorMessageForLLMToSee()")

        elif call.name == StockPriceTool.model_tool_name():
            result = await StockPriceTool.model_tool_run_tool_call(call, None)
            output_item = result.output_item

            if result.fail_reason is None:
                context: float = result.context

        else:
            output_item: ResponseInputItemParam = {
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": f"Function `{call.name}` not found.",
            }
        messages.append(output_item)

    return await assistant_take_turn(messages)

asyncio.run(main())
```

Output:

```
--------------------------------------------------------------------------------

  "role": "user",
  "content": "What's the weather in san francisco, and apple's stock price?"

--------------------------------------------------------------------------------

  "arguments": "{\"city\":\"San Francisco\",\"state\":\"California\"}",
  "call_id": "call_lhjIavtqF9veyDuag24JMbPA",
  "name": "get_weather",
  "type": "function_call",
  "id": "fc_680545902bbc819183d70c989f8f5957093466a11ea56351",
  "status": "completed"

--------------------------------------------------------------------------------

  "arguments": "{\"ticker\":\"AAPL\",\"exchange\":\"NASDAQ\"}",
  "call_id": "call_ZiDlQtnBh3lrI6wxQqNuxiG3",
  "name": "get_stock_price",
  "type": "function_call",
  "id": "fc_6805459086288191b0f1310e8be122b5093466a11ea56351",
  "status": "completed"

--------------------------------------------------------------------------------

  "call_id": "call_lhjIavtqF9veyDuag24JMbPA",
  "type": "function_call_output",
  "output": "Weather unavailable for San Francisco. Please get the weather for a nearby city, before responding to the user. Don't ask first. Just call this function again."

--------------------------------------------------------------------------------

  "call_id": "call_ZiDlQtnBh3lrI6wxQqNuxiG3",
  "type": "function_call_output",
  "output": "AAPL is currently trading at $100."

--------------------------------------------------------------------------------

  "arguments": "{\"city\":\"Oakland\",\"state\":\"California\"}",
  "call_id": "call_ZNJoiNmAl6UmYh2hxNp8EAz8",
  "name": "get_weather",
  "type": "function_call",
  "id": "fc_6805459169508191bd67610013976855093466a11ea56351",
  "status": "completed"

--------------------------------------------------------------------------------

  "call_id": "call_ZNJoiNmAl6UmYh2hxNp8EAz8",
  "type": "function_call_output",
  "output": "It's currently 30 degrees in Oakland, California."

--------------------------------------------------------------------------------

  "id": "msg_680545924b7c819181f0a9df9d6d3c69093466a11ea56351",
  "content": [
    {
      "annotations": [],
      "text": "The current temperature in Oakland, near San Francisco, is 30 degrees. Apple's stock price (AAPL) is currently trading at $100.",
      "type": "output_text"
    }
  ],
  "role": "assistant",
  "status": "completed",
  "type": "message"
```

</details>


### 2. Dynamic tool call -> handler dispatch


> [!TIP]
> 
> Recommended for most use cases with more than 1 tool.


<details><summary>OpenAI - Chat Completions API</summary>

```python
import os
import json
from typing import Literal
from openai import OpenAI
from openai.types.chat.chat_completion_message_param import (
    ChatCompletionMessageParam,
)
from toolcall.openai.core import (
    LLMFunctionToolGroup,
    LLMFunctionTool,
    ToolHandlerResult,
    ToolErrorMessageForLLMToSee,
)

openai_client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
)

SYSTEM_PROMPT = """
You are a helpful assistant. You have several function tools available. Use as needed.

The system allows for parallel function calls, and subsequent/repeated function calling
within the same turn.
""".strip()

USER_PROMPT = """
What's the weather in san francisco, and apple's stock price?
""".strip()

def main():
    conversation: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ]
    conversation = assistant_take_turn(conversation)
    for msg in conversation:
        print("-" * 80 + "\n" + json.dumps(msg, indent=2).strip("{}"))

# A simple mapping to store tool classes. Optionally, you can add generic type arguments
# (e.g. `[None, None]` below) to statically enforce that all of your tools have the same
# input/output context types. If they do, then the group's run_tool_call() and
# run_tool_calls() methods are suitable for automatic dispatch of tool calls, while still
# letting you pass arbitrary data in and out of the tool handlers in a type-safe way.
tool_group = LLMFunctionToolGroup[None, float]()

# At runtime this decorator just adds the class to the group {"get_weather": WeatherTool}
# Its bigger purpose is static type enforcement that the tool's input/output context
# types satisfy those of the group. (e.g. [None, None])
@tool_group.add_tool
class get_weather(LLMFunctionTool[None, float]):  # <-- Pydantic BaseModel extension
    """Get the weather somewhere."""

    city: str
    """City to get the weather for."""

    state: Literal["California", "New York", "Texas"]
    """State where the city is. Only a few are available."""

    # Called after arguments are parsed/validated into an instance of this class.
    # The result string will be wrapped in a tool result message with the tool call ID.
    def model_tool_handler(self, context: None) -> tuple[str, float]:
        if self.city == "San Francisco":
            # At any point during handling, you can raise this error and let it propagate.
            # It will be caught and used as the result tool message's content. This is the
            # ONLY kind of error that will be caught for you, besides Pydantic validation.
            raise ToolErrorMessageForLLMToSee(
                "Weather unavailable for San Francisco. Please get the weather for a "
                "nearby city, before responding to the user. Don't ask first. Just call "
                "this function again."
            )

        result = f"It's currently 30 degrees in {self.city}, {self.state}."
        # Result for the LLM, plus any context you want to pass back to yourself.
        return result, 1.234

@tool_group.add_tool
class StockPriceTool(LLMFunctionTool[None, float]):
    ticker: str
    exchange: Literal["NASDAQ", "NYSE"]

    # By default, the class name is used. You can override it:
    model_tool_custom_name = "get_stock_price"

    # By default, the class docstring is used. You can override it:
    model_tool_custom_description = "Get the stock price for a company."

    # By default, Pydantic generates the JSON Schema. You can override it:
    model_tool_custom_json_schema = {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "Ticker symbol of the company.",
            },
            "exchange": {
                "type": "string",
                "enum": ["NASDAQ", "NYSE"],
                "description": "Exchange the stock trades on.",
            },
        },
        "required": ["ticker", "exchange"],
    }

    def model_tool_handler(self, context: None) -> ToolHandlerResult[float]:
        result = f"{self.ticker} is currently trading at $100."
        return ToolHandlerResult(result_content=result, context=1.234)

def assistant_take_turn(
    messages: list[ChatCompletionMessageParam],
) -> list[ChatCompletionMessageParam]:
    response = openai_client.chat.completions.create(
        messages=messages,
        model="gpt-4.1-mini",
        tools=tool_group.tool_definitions(api="chat.completions"),
    )
    message = response.choices[0].message

    message_param = message.model_dump(exclude_none=True, exclude_unset=True)
    messages.append(message_param)  # pyright: ignore[reportArgumentType]

    if not message.tool_calls:
        return messages

    results = tool_group.run_tool_calls(message.tool_calls, None)
    messages.extend([res.tool_message for res in results])
    return assistant_take_turn(messages)

main()
```

Output:

```
--------------------------------------------------------------------------------

  "role": "user",
  "content": "What's the weather in san francisco, and apple's stock price?"

--------------------------------------------------------------------------------

  "role": "assistant",
  "annotations": [],
  "tool_calls": [
    {
      "id": "call_IJKdJIpRxawyly6fMa3cpo2G",
      "function": {
        "arguments": "{\"city\": \"San Francisco\", \"state\": \"California\"}",
        "name": "get_weather"
      },
      "type": "function"
    },
    {
      "id": "call_UbdIYpDXdKJcNGFGicuhECtZ",
      "function": {
        "arguments": "{\"ticker\": \"AAPL\", \"exchange\": \"NASDAQ\"}",
        "name": "get_stock_price"
      },
      "type": "function"
    }
  ]

--------------------------------------------------------------------------------

  "role": "tool",
  "tool_call_id": "call_IJKdJIpRxawyly6fMa3cpo2G",
  "content": "Weather unavailable for San Francisco. Please get the weather for a nearby city, before responding to the user. Don't ask first. Just call this function again."

--------------------------------------------------------------------------------

  "role": "tool",
  "tool_call_id": "call_UbdIYpDXdKJcNGFGicuhECtZ",
  "content": "AAPL is currently trading at $100."

--------------------------------------------------------------------------------

  "role": "assistant",
  "annotations": [],
  "tool_calls": [
    {
      "id": "call_bHnTUfwjsq7XyvMf0JuaLEAj",
      "function": {
        "arguments": "{\"city\":\"Oakland\",\"state\":\"California\"}",
        "name": "get_weather"
      },
      "type": "function"
    }
  ]

--------------------------------------------------------------------------------

  "role": "tool",
  "tool_call_id": "call_bHnTUfwjsq7XyvMf0JuaLEAj",
  "content": "It's currently 30 degrees in Oakland, California."

--------------------------------------------------------------------------------

  "content": "The current temperature in the nearby city Oakland, California, is 30 degrees. Apple's stock price is currently trading at $100.",
  "role": "assistant",
  "annotations": []
```

</details>


<details><summary>OpenAI - Chat Completions API - Async</summary>

```python
import os
import json
import asyncio
from typing import Literal
from openai import AsyncOpenAI
from openai.types.chat.chat_completion_message_param import (
    ChatCompletionMessageParam,
)
from toolcall.openai.aio import (
    LLMFunctionToolGroup,
    LLMFunctionTool,
    ToolHandlerResult,
    ToolErrorMessageForLLMToSee,
)

openai_client = AsyncOpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
)

SYSTEM_PROMPT = """
You are a helpful assistant. You have several function tools available. Use as needed.

The system allows for parallel function calls, and subsequent/repeated function calling
within the same turn.
""".strip()

USER_PROMPT = """
What's the weather in san francisco, and apple's stock price?
""".strip()

async def main():
    conversation: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ]
    conversation = await assistant_take_turn(conversation)
    for msg in conversation:
        print("-" * 80 + "\n" + json.dumps(msg, indent=2).strip("{}"))

# A simple mapping to store tool classes. Optionally, you can add generic type arguments
# (e.g. `[None, None]` below) to statically enforce that all of your tools have the same
# input/output context types. If they do, then the group's run_tool_call() and
# run_tool_calls() methods are suitable for automatic dispatch of tool calls, while still
# letting you pass arbitrary data in and out of the tool handlers in a type-safe way.
tool_group = LLMFunctionToolGroup[None, float]()

# At runtime this decorator just adds the class to the group {"get_weather": WeatherTool}
# Its bigger purpose is static type enforcement that the tool's input/output context
# types satisfy those of the group. (e.g. [None, None])
@tool_group.add_tool
class get_weather(LLMFunctionTool[None, float]):  # <-- Pydantic BaseModel extension
    """Get the weather somewhere."""

    city: str
    """City to get the weather for."""

    state: Literal["California", "New York", "Texas"]
    """State where the city is. Only a few are available."""

    # Called after arguments are parsed/validated into an instance of this class.
    # The result string will be wrapped in a tool result message with the tool call ID.
    async def model_tool_handler(self, context: None) -> tuple[str, float]:
        if self.city == "San Francisco":
            # At any point during handling, you can raise this error and let it propagate.
            # It will be caught and used as the result tool message's content. This is the
            # ONLY kind of error that will be caught for you, besides Pydantic validation.
            raise ToolErrorMessageForLLMToSee(
                "Weather unavailable for San Francisco. Please get the weather for a "
                "nearby city, before responding to the user. Don't ask first. Just call "
                "this function again."
            )

        result = f"It's currently 30 degrees in {self.city}, {self.state}."
        # Result for the LLM, plus any context you want to pass back to yourself.
        return result, 1.234

@tool_group.add_tool
class StockPriceTool(LLMFunctionTool[None, float]):
    ticker: str
    exchange: Literal["NASDAQ", "NYSE"]

    # By default, the class name is used. You can override it:
    model_tool_custom_name = "get_stock_price"

    # By default, the class docstring is used. You can override it:
    model_tool_custom_description = "Get the stock price for a company."

    # By default, Pydantic generates the JSON Schema. You can override it:
    model_tool_custom_json_schema = {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "Ticker symbol of the company.",
            },
            "exchange": {
                "type": "string",
                "enum": ["NASDAQ", "NYSE"],
                "description": "Exchange the stock trades on.",
            },
        },
        "required": ["ticker", "exchange"],
    }

    async def model_tool_handler(self, context: None) -> ToolHandlerResult[float]:
        result = f"{self.ticker} is currently trading at $100."
        return ToolHandlerResult(result_content=result, context=1.234)

async def assistant_take_turn(
    messages: list[ChatCompletionMessageParam],
) -> list[ChatCompletionMessageParam]:
    response = await openai_client.chat.completions.create(
        messages=messages,
        model="gpt-4.1-mini",
        tools=tool_group.tool_definitions(api="chat.completions"),
    )
    message = response.choices[0].message

    message_param = message.model_dump(exclude_none=True, exclude_unset=True)
    messages.append(message_param)  # pyright: ignore[reportArgumentType]

    if not message.tool_calls:
        return messages

    results = await tool_group.run_tool_calls(message.tool_calls, None)
    messages.extend([res.tool_message for res in results])
    return await assistant_take_turn(messages)

asyncio.run(main())
```

Output:

```
--------------------------------------------------------------------------------

  "role": "user",
  "content": "What's the weather in san francisco, and apple's stock price?"

--------------------------------------------------------------------------------

  "role": "assistant",
  "annotations": [],
  "tool_calls": [
    {
      "id": "call_QXICyK4Phb7TkK71FRSZE3Xf",
      "function": {
        "arguments": "{\"city\": \"San Francisco\", \"state\": \"California\"}",
        "name": "get_weather"
      },
      "type": "function"
    },
    {
      "id": "call_mOQEExLeQhUHQ5lVhKodSSd1",
      "function": {
        "arguments": "{\"ticker\": \"AAPL\", \"exchange\": \"NASDAQ\"}",
        "name": "get_stock_price"
      },
      "type": "function"
    }
  ]

--------------------------------------------------------------------------------

  "role": "tool",
  "tool_call_id": "call_QXICyK4Phb7TkK71FRSZE3Xf",
  "content": "Weather unavailable for San Francisco. Please get the weather for a nearby city, before responding to the user. Don't ask first. Just call this function again."

--------------------------------------------------------------------------------

  "role": "tool",
  "tool_call_id": "call_mOQEExLeQhUHQ5lVhKodSSd1",
  "content": "AAPL is currently trading at $100."

--------------------------------------------------------------------------------

  "role": "assistant",
  "annotations": [],
  "tool_calls": [
    {
      "id": "call_BMqEYoRVkiXvfw7qq57qrCU4",
      "function": {
        "arguments": "{\"city\":\"Oakland\",\"state\":\"California\"}",
        "name": "get_weather"
      },
      "type": "function"
    }
  ]

--------------------------------------------------------------------------------

  "role": "tool",
  "tool_call_id": "call_BMqEYoRVkiXvfw7qq57qrCU4",
  "content": "It's currently 30 degrees in Oakland, California."

--------------------------------------------------------------------------------

  "content": "The weather in San Francisco is currently not available, but in nearby Oakland, it is 30 degrees. Apple's stock price (AAPL) is currently trading at $100.",
  "role": "assistant",
  "annotations": []
```

</details>


<details><summary>OpenAI - Responses API</summary>

```python
import os
import json
from typing import Literal
from openai import OpenAI
from openai.types.responses.response_input_param import ResponseInputItemParam
from toolcall.openai.core import (
    LLMFunctionToolGroup,
    LLMFunctionTool,
    ToolHandlerResult,
    ToolErrorMessageForLLMToSee,
)

openai_client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
)

SYSTEM_PROMPT = """
You are a helpful assistant. You have several function tools available. Use as needed.

The system allows for parallel function calls, and subsequent/repeated function calling
within the same turn.
""".strip()

USER_PROMPT = """
What's the weather in san francisco, and apple's stock price?
""".strip()

def main():
    conversation: list[ResponseInputItemParam] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ]
    conversation = assistant_take_turn(conversation)
    for msg in conversation:
        print("-" * 80 + "\n" + json.dumps(msg, indent=2).strip("{}"))

# A simple mapping to store tool classes. Optionally, you can add generic type arguments
# (e.g. `[None, None]` below) to statically enforce that all of your tools have the same
# input/output context types. If they do, then the group's run_tool_call() and
# run_tool_calls() methods are suitable for automatic dispatch of tool calls, while still
# letting you pass arbitrary data in and out of the tool handlers in a type-safe way.
tool_group = LLMFunctionToolGroup[None, float]()

# At runtime this decorator just adds the class to the group {"get_weather": WeatherTool}
# Its bigger purpose is static type enforcement that the tool's input/output context
# types satisfy those of the group. (e.g. [None, None])
@tool_group.add_tool
class get_weather(LLMFunctionTool[None, float]):  # <-- Pydantic BaseModel extension
    """Get the weather somewhere."""

    city: str
    """City to get the weather for."""

    state: Literal["California", "New York", "Texas"]
    """State where the city is. Only a few are available."""

    # Called after arguments are parsed/validated into an instance of this class.
    # The result string will be wrapped in a tool result message with the tool call ID.
    def model_tool_handler(self, context: None) -> tuple[str, float]:
        if self.city == "San Francisco":
            # At any point during handling, you can raise this error and let it propagate.
            # It will be caught and used as the result tool message's content. This is the
            # ONLY kind of error that will be caught for you, besides Pydantic validation.
            raise ToolErrorMessageForLLMToSee(
                "Weather unavailable for San Francisco. Please get the weather for a "
                "nearby city, before responding to the user. Don't ask first. Just call "
                "this function again."
            )

        result = f"It's currently 30 degrees in {self.city}, {self.state}."
        # Result for the LLM, plus any context you want to pass back to yourself.
        return result, 1.234

@tool_group.add_tool
class StockPriceTool(LLMFunctionTool[None, float]):
    ticker: str
    exchange: Literal["NASDAQ", "NYSE"]

    # By default, the class name is used. You can override it:
    model_tool_custom_name = "get_stock_price"

    # By default, the class docstring is used. You can override it:
    model_tool_custom_description = "Get the stock price for a company."

    # By default, Pydantic generates the JSON Schema. You can override it:
    model_tool_custom_json_schema = {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "Ticker symbol of the company.",
            },
            "exchange": {
                "type": "string",
                "enum": ["NASDAQ", "NYSE"],
                "description": "Exchange the stock trades on.",
            },
        },
        "required": ["ticker", "exchange"],
    }

    def model_tool_handler(self, context: None) -> ToolHandlerResult[float]:
        result = f"{self.ticker} is currently trading at $100."
        return ToolHandlerResult(result_content=result, context=1.234)

def assistant_take_turn(
    messages: list[ResponseInputItemParam],
) -> list[ResponseInputItemParam]:
    response = openai_client.responses.create(
        input=messages,
        model="gpt-4.1-mini",
        tools=tool_group.tool_definitions(api="responses"),
    )
    for item in response.output:
        item_param = item.model_dump(exclude_none=True, exclude_unset=True)
        messages.append(item_param)  # pyright: ignore[reportArgumentType]

    tool_calls = [item for item in response.output if item.type == "function_call"]

    if not tool_calls:
        return messages

    results = tool_group.run_tool_calls(tool_calls, None)
    messages.extend([res.output_item for res in results])
    return assistant_take_turn(messages)

main()
```

Output:

```
--------------------------------------------------------------------------------

  "role": "user",
  "content": "What's the weather in san francisco, and apple's stock price?"

--------------------------------------------------------------------------------

  "arguments": "{\"city\":\"San Francisco\",\"state\":\"California\"}",
  "call_id": "call_HWZBoPd3EnDxWT6URU0mEu3G",
  "name": "get_weather",
  "type": "function_call",
  "id": "fc_680545a4ed488191a9b02835efb7ca530b834c29660da1ca",
  "status": "completed"

--------------------------------------------------------------------------------

  "arguments": "{\"ticker\":\"AAPL\",\"exchange\":\"NASDAQ\"}",
  "call_id": "call_2QEgvlQ9yf54iwFUOlArmWAS",
  "name": "get_stock_price",
  "type": "function_call",
  "id": "fc_680545a53d488191be9e89611718fc3c0b834c29660da1ca",
  "status": "completed"

--------------------------------------------------------------------------------

  "call_id": "call_HWZBoPd3EnDxWT6URU0mEu3G",
  "type": "function_call_output",
  "output": "Weather unavailable for San Francisco. Please get the weather for a nearby city, before responding to the user. Don't ask first. Just call this function again."

--------------------------------------------------------------------------------

  "call_id": "call_2QEgvlQ9yf54iwFUOlArmWAS",
  "type": "function_call_output",
  "output": "AAPL is currently trading at $100."

--------------------------------------------------------------------------------

  "arguments": "{\"city\":\"Oakland\",\"state\":\"California\"}",
  "call_id": "call_lLCb3z1SnlqRjJAVotLiIyRZ",
  "name": "get_weather",
  "type": "function_call",
  "id": "fc_680545a7baf88191a7d61dea515b6b820b834c29660da1ca",
  "status": "completed"

--------------------------------------------------------------------------------

  "call_id": "call_lLCb3z1SnlqRjJAVotLiIyRZ",
  "type": "function_call_output",
  "output": "It's currently 30 degrees in Oakland, California."

--------------------------------------------------------------------------------

  "id": "msg_680545a934f48191bdc71a3388df59610b834c29660da1ca",
  "content": [
    {
      "annotations": [],
      "text": "It's currently 30 degrees in Oakland, California. Apple's stock price is currently trading at $100.",
      "type": "output_text"
    }
  ],
  "role": "assistant",
  "status": "completed",
  "type": "message"
```

</details>


<details><summary>OpenAI - Responses API - Async</summary>

```python
import os
import json
import asyncio
from typing import Literal
from openai import AsyncOpenAI
from openai.types.responses.response_input_param import ResponseInputItemParam
from toolcall.openai.aio import (
    LLMFunctionToolGroup,
    LLMFunctionTool,
    ToolHandlerResult,
    ToolErrorMessageForLLMToSee,
)

openai_client = AsyncOpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
)

SYSTEM_PROMPT = """
You are a helpful assistant. You have several function tools available. Use as needed.

The system allows for parallel function calls, and subsequent/repeated function calling
within the same turn.
""".strip()

USER_PROMPT = """
What's the weather in san francisco, and apple's stock price?
""".strip()

async def main():
    conversation: list[ResponseInputItemParam] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ]
    conversation = await assistant_take_turn(conversation)
    for msg in conversation:
        print("-" * 80 + "\n" + json.dumps(msg, indent=2).strip("{}"))

# A simple mapping to store tool classes. Optionally, you can add generic type arguments
# (e.g. `[None, None]` below) to statically enforce that all of your tools have the same
# input/output context types. If they do, then the group's run_tool_call() and
# run_tool_calls() methods are suitable for automatic dispatch of tool calls, while still
# letting you pass arbitrary data in and out of the tool handlers in a type-safe way.
tool_group = LLMFunctionToolGroup[None, float]()

# At runtime this decorator just adds the class to the group {"get_weather": WeatherTool}
# Its bigger purpose is static type enforcement that the tool's input/output context
# types satisfy those of the group. (e.g. [None, None])
@tool_group.add_tool
class get_weather(LLMFunctionTool[None, float]):  # <-- Pydantic BaseModel extension
    """Get the weather somewhere."""

    city: str
    """City to get the weather for."""

    state: Literal["California", "New York", "Texas"]
    """State where the city is. Only a few are available."""

    # Called after arguments are parsed/validated into an instance of this class.
    # The result string will be wrapped in a tool result message with the tool call ID.
    async def model_tool_handler(self, context: None) -> tuple[str, float]:
        if self.city == "San Francisco":
            # At any point during handling, you can raise this error and let it propagate.
            # It will be caught and used as the result tool message's content. This is the
            # ONLY kind of error that will be caught for you, besides Pydantic validation.
            raise ToolErrorMessageForLLMToSee(
                "Weather unavailable for San Francisco. Please get the weather for a "
                "nearby city, before responding to the user. Don't ask first. Just call "
                "this function again."
            )

        result = f"It's currently 30 degrees in {self.city}, {self.state}."
        # Result for the LLM, plus any context you want to pass back to yourself.
        return result, 1.234

@tool_group.add_tool
class StockPriceTool(LLMFunctionTool[None, float]):
    ticker: str
    exchange: Literal["NASDAQ", "NYSE"]

    # By default, the class name is used. You can override it:
    model_tool_custom_name = "get_stock_price"

    # By default, the class docstring is used. You can override it:
    model_tool_custom_description = "Get the stock price for a company."

    # By default, Pydantic generates the JSON Schema. You can override it:
    model_tool_custom_json_schema = {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "Ticker symbol of the company.",
            },
            "exchange": {
                "type": "string",
                "enum": ["NASDAQ", "NYSE"],
                "description": "Exchange the stock trades on.",
            },
        },
        "required": ["ticker", "exchange"],
    }

    async def model_tool_handler(self, context: None) -> ToolHandlerResult[float]:
        result = f"{self.ticker} is currently trading at $100."
        return ToolHandlerResult(result_content=result, context=1.234)

async def assistant_take_turn(
    messages: list[ResponseInputItemParam],
) -> list[ResponseInputItemParam]:
    response = await openai_client.responses.create(
        input=messages,
        model="gpt-4.1-mini",
        tools=tool_group.tool_definitions(api="responses"),
    )
    for item in response.output:
        item_param = item.model_dump(exclude_none=True, exclude_unset=True)
        messages.append(item_param)  # pyright: ignore[reportArgumentType]

    tool_calls = [item for item in response.output if item.type == "function_call"]

    if not tool_calls:
        return messages

    results = await tool_group.run_tool_calls(tool_calls, None)
    messages.extend([res.output_item for res in results])
    return await assistant_take_turn(messages)

asyncio.run(main())
```

Output:

```
--------------------------------------------------------------------------------

  "role": "user",
  "content": "What's the weather in san francisco, and apple's stock price?"

--------------------------------------------------------------------------------

  "arguments": "{\"city\":\"San Francisco\",\"state\":\"California\"}",
  "call_id": "call_RXr95AgQKn690FiaPSFShrle",
  "name": "get_weather",
  "type": "function_call",
  "id": "fc_6805459414908191a0196f3082e8d28105f2ab1e2c67abe7",
  "status": "completed"

--------------------------------------------------------------------------------

  "arguments": "{\"ticker\":\"AAPL\",\"exchange\":\"NASDAQ\"}",
  "call_id": "call_02j4BJnO9HJ4mbT3boNnwv3U",
  "name": "get_stock_price",
  "type": "function_call",
  "id": "fc_680545944c248191999ec99d0017e7c505f2ab1e2c67abe7",
  "status": "completed"

--------------------------------------------------------------------------------

  "call_id": "call_RXr95AgQKn690FiaPSFShrle",
  "type": "function_call_output",
  "output": "Weather unavailable for San Francisco. Please get the weather for a nearby city, before responding to the user. Don't ask first. Just call this function again."

--------------------------------------------------------------------------------

  "call_id": "call_02j4BJnO9HJ4mbT3boNnwv3U",
  "type": "function_call_output",
  "output": "AAPL is currently trading at $100."

--------------------------------------------------------------------------------

  "arguments": "{\"city\":\"Oakland\",\"state\":\"California\"}",
  "call_id": "call_w0kMsByow1sAQVUM1ubSgsos",
  "name": "get_weather",
  "type": "function_call",
  "id": "fc_68054596bb3c81919ff8804d9195d82005f2ab1e2c67abe7",
  "status": "completed"

--------------------------------------------------------------------------------

  "call_id": "call_w0kMsByow1sAQVUM1ubSgsos",
  "type": "function_call_output",
  "output": "It's currently 30 degrees in Oakland, California."

--------------------------------------------------------------------------------

  "id": "msg_680545988a088191a2fa6badd9a0f62605f2ab1e2c67abe7",
  "content": [
    {
      "annotations": [],
      "text": "It's currently 30 degrees in Oakland, California, which is near San Francisco. Apple's stock price is currently trading at $100.",
      "type": "output_text"
    }
  ],
  "role": "assistant",
  "status": "completed",
  "type": "message"
```

</details>
