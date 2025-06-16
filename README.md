<!-- File generated from /README.template.md using /generate_readme.py -->

# toolcall

[![PyPI](https://img.shields.io/pypi/v/toolcall)](https://pypi.org/project/toolcall/)
[![Tests](https://github.com/ryayoung/toolcall/actions/workflows/tests.yml/badge.svg)](https://github.com/ryayoung/toolcall/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/ryayoung/toolcall/branch/main/graph/badge.svg)](https://codecov.io/gh/ryayoung/toolcall)
[![License](https://img.shields.io/github/license/ryayoung/toolcall)](https://github.com/ryayoung/toolcall/blob/main/LICENSE)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/toolcall.svg)](https://pypi.python.org/pypi/toolcall/)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Pyright](https://img.shields.io/badge/type%20checker-pyright-blue)](https://github.com/microsoft/pyright)

```
pip install toolcall
```

The agentic *not*-framework for Python. This <200-line **micro**-library for
structured/agentic LLM communication provides utility types and primitives for
developers who prefer a more manual approach to building agentic behavior on
top of `pydantic` and `openai`, without sacrificing control or code clarity to
a framework.

<details><summary>More details: <i>Who is it for, and why?</i></summary>

---

For developers who prefer a more *manual* approach to LLM workflow
orchestration and context management - using `pydantic` for validation and
schema generation, an API client (`openai`) for its type-safe interface and
request handling, and rolling everything else on your own - `toolcall` is the
little abstraction you were going to eventually end up building anyway, as your
project scales in complexity.

`toolcall` is like a great desk chair: Simple, unexciting, and completely
unambiguous in runtime behavior, but something upon which your code can always
sit, no matter the use case, and whose sole purpose is to solve a handful of
basic problems that everyone has, and do so without compromise.

---

</details>



---

### *Should you use it?*

- You want a framework to handle agent/tool orchestration and control-flow for you:
    - ⛔ Do **not** use `toolcall`.
- You define, dispatch, and handle function tool calls yourself:
    - ✅ You **should** be using `toolcall`.


<br>

# Learn-by-example Documentation

Below is a set of end-to-end tool-calling and structured output workflows,
for every API type.

> Note: Every line of code below comes straight from `examples/` where it is
> tested before being injected into this readme.

---

<details><summary><b>Synchronous API</b></summary>

<br>

<details><summary><code>common.py</code> - Setup code used by all examples below</summary>

```python
# examples/core/common.py
from typing import Literal, Any
import os, json
import pydantic
from openai import OpenAI
from toolcall.openai.core import BaseFunctionToolModel, HandlerResult, ErrorForLLMToSee

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

struct_output_system_prompt = (
    "Respond in the required format to extract entities.\n\n"
    "NOTE: We are doing API testing. Your *first* response should fail validation. "
    'Violate the schema by passing `100` instead of `"100"` in numbers.'
)
struct_output_user_prompt = (
    "Query: What's the weather in San Francisco? Is it above 100 there?"
)


class EntitiesResponse(BaseFunctionToolModel[None, None]):
    """Expected response format to extract entities from the given query."""

    people: list[str]
    places: list[str]
    numbers: list[str]


# Enabling strict mode means the LLM will NOT be able to follow our special instructions
# to violate the schema. It will give a valid response the first time.
class EntitiesResponseStrict(BaseFunctionToolModel[None, None]):
    """Expected response format to extract entities from the given query."""

    people: list[str]
    places: list[str]
    numbers: list[str]

    # Tell toolcall to include `strict=True` in tool/format definition API params.
    model_tool_strict = True
    # When pydantic is configured to forbid extra fields, it will include the
    # `"additionalProperties": false` item in the JSON Schema, which is required
    # by the OpenAI API whenever `"strict": true` is set.
    model_config = pydantic.ConfigDict(extra="forbid")


TOOLS_SYSTEM_PROMPT = """
You are a helpful assistant. You have several function tools available. Use as needed.

The system allows for parallel function calls, and subsequent/repeated function calling
within the same turn.
""".strip()


# Minimal function tool that...
#   1. Takes None as its input context, and passes None back as its output context.
#   2. Uses its class name as the function tool name.
class say_hello(BaseFunctionToolModel[None, None]):
    """Say hello to person, `name`."""

    name: str

    # Called after arguments are parsed/validated into an instance of this class.
    # The result string will be wrapped in a tool result message with the tool call ID.
    def model_tool_handler(self, _):
        return f"Message delivered to {self.name}.", None


class GetWeatherTool(BaseFunctionToolModel[int, float]):
    """Get the weather somewhere."""

    model_tool_custom_name = "get_weather"

    city: str
    """City to get the weather for."""

    state: Literal["California", "New York", "Texas"]
    """State where the city is. Only a few are available."""

    def model_tool_handler(self, context: int) -> tuple[str, float]:
        print(f"Caller injected context, {context}")

        if self.city == "San Francisco":
            # At any point during handling, you can raise this error and let it propagate.
            # It will be caught and used as the result tool message's content. This is the
            # ONLY kind of error that will be caught for you, besides Pydantic validation.
            raise ErrorForLLMToSee(
                "Weather unavailable for San Francisco. Please get the weather for a "
                "nearby city, before responding to the user. Don't ask first. Just call "
                "this function again."
            )

        result = f"It's currently 30 degrees in {self.city}, {self.state}."
        return result, 1.234


class StockPriceTool(BaseFunctionToolModel[int, float]):
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

    def model_tool_handler(self, context: int) -> tuple[str, float]:
        result = f"{self.ticker} is currently trading at $100."
        # HandlerResult (a named tuple) is just a more explicit alternative.
        return HandlerResult(result_content=result, context=1.234)


from toolcall.openai.core import FunctionToolGroup

# A simple mapping to store tool classes. Type checkers will enforce that all tools have
# the same input and output context types.
# That's why we cannot include `say_hello` here.
tool_group = FunctionToolGroup.from_list([GetWeatherTool, StockPriceTool])


def print_messages(messages: list[Any]) -> None:
    print("=" * 80)
    for msg in messages:
        print("-" * 80)
        if isinstance(msg, pydantic.BaseModel):
            print(f"\n{repr(msg)}\n")
        else:
            msg = {k: v for k, v in msg.items() if v}
            print(json.dumps(msg, indent=2).strip("{}"))
```

</details>

#### Chat Completions API

<details><summary>Structured Outputs</summary>

```python
# examples/core/chat_output.py
from typing import Any
import pydantic
from toolcall.openai.core import BaseFunctionToolModel
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from .common import openai_client, print_messages
from .common import EntitiesResponse, EntitiesResponseStrict
from .common import struct_output_system_prompt, struct_output_user_prompt


def main():
    def run(response_model: type[BaseFunctionToolModel[Any, Any]]):
        conversation: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": struct_output_system_prompt},
            {"role": "user", "content": struct_output_user_prompt},
        ]
        entities = assistant_debug_until_correct(response_model, conversation)
        print_messages(conversation[1:] + [entities])

    run(EntitiesResponse)
    run(EntitiesResponseStrict)


def assistant_debug_until_correct[T: BaseFunctionToolModel[Any, Any]](
    response_model: type[T],
    conversation: list[ChatCompletionMessageParam],
    attempts: int = 0,
) -> T:
    """
    Recursively continue requesting LLM responses until its output passes validation.
    """

    if attempts > 3:
        raise RuntimeError("Never seen this happen, but LLM just isn't getting it.")

    # 1. Request an LLM response, and append it to the conversation.

    format = response_model.model_tool_format(api="chat.completions")
    response = openai_client.chat.completions.create(
        messages=conversation,
        model="gpt-4.1",
        response_format=format,
    )
    message = response.choices[0].message
    conversation.append(message.model_dump())  # pyright: ignore[reportArgumentType]

    # 2. Try to parse the response content into a valid EntitiesResponse. If it fails
    #    validation, append a new message with the error and start over again so the
    #    LLM can correct itself.

    # (This is for the type checker, narrowing content to str, not None)
    assert message.content is not None, "Impossible since no tools given"

    try:
        return response_model.model_validate_json(message.content)
    except pydantic.ValidationError as e:
        conversation.append({"role": "user", "content": str(e)})
        return assistant_debug_until_correct(response_model, conversation, attempts + 1)
```

</details>

<details><summary>Single Function Tool</summary>

```python
# examples/core/chat_tool.py
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from .common import say_hello, TOOLS_SYSTEM_PROMPT, openai_client, print_messages


def main():
    user_prompt = "Can you please say hello to John and Kate for me?"
    conversation: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": TOOLS_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    assistant_take_turn(conversation)
    print_messages(conversation[1:])


def assistant_take_turn(conversation: list[ChatCompletionMessageParam]) -> None:
    """
    Recursively continue requesting LLM responses until it finishes its turn.
    """

    # 1. Request an LLM response, and append it to the conversation.

    response = openai_client.chat.completions.create(
        messages=conversation,
        model="gpt-4.1",
        tools=[say_hello.model_tool_definition(api="chat.completions")],
    )
    message = response.choices[0].message
    conversation.append(message.model_dump())  # pyright: ignore[reportArgumentType]

    # 2. If there weren't any tool calls: we're done, and can finish this turn.
    #    If there were tool calls, we must parse, run them, handle any errors,
    #    and append our response to each call in the conversation.

    if not message.tool_calls:
        return

    results = [say_hello.model_tool_run_tool_call(c, None) for c in message.tool_calls]
    conversation.extend([res.tool_message for res in results])

    # 3. Since there were tool calls, this turn isn't finished yet. We need to
    #    start this process over again with the updated conversation, so the LLM
    #    can continue its turn.

    assistant_take_turn(conversation)
```

</details>

<details><summary>Multiple Function Tools</summary>

```python
# examples/core/chat_group.py
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from .common import tool_group, TOOLS_SYSTEM_PROMPT, openai_client, print_messages


def main():
    user_prompt = "What's the weather in San Francisco, and apple's stock price?"
    conversation: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": TOOLS_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    assistant_take_turn(conversation)
    print_messages(conversation[1:])


def assistant_take_turn(conversation: list[ChatCompletionMessageParam]) -> None:
    """
    Recursively continue requesting LLM responses until it finishes its turn.
    """

    # 1. Request an LLM response, and append it to the conversation.

    response = openai_client.chat.completions.create(
        messages=conversation,
        model="gpt-4.1",
        tools=tool_group.tool_definitions(api="chat.completions"),
    )
    message = response.choices[0].message
    conversation.append(message.model_dump())  # pyright: ignore[reportArgumentType]

    # 2. If there weren't any tool calls: we're done, and can finish this turn.
    #    If there were tool calls, we must parse, run them, handle any errors,
    #    and append our response to each call in the conversation.

    if not message.tool_calls:
        return

    input_context = 100
    results = tool_group.run_tool_calls(message.tool_calls, input_context)
    conversation.extend([res.tool_message for res in results])

    # 3. Optionally, we can inspect the results. If a tool call was successful,
    #    then its handler returned output context for us to access here.

    for result in results:
        if result.fail_reason is None:
            context: float = result.context  # Type checker knows this is safe
            print(f"Call {result.call_id} was successful. We got {context=} back.")
        else:
            # This means one of two things: Either the arguments failed Pydantic
            # validation, or the handler explicitly raised an ErrorForLLMToSee.
            _: None = result.context  # Always null. Handler didn't finish.
            print(
                f"Call {result.call_id} failed because of {result.fail_reason}, but "
                f'we handled it smoothly, replying: "{result.result_content[:12]}..."'
            )

    # 4. Since there were tool calls, this turn isn't finished yet. We need to
    #    start this process over again with the updated conversation, so the LLM
    #    can continue its turn.

    assistant_take_turn(conversation)
```

</details>

#### Responses API

<details><summary>Structured Outputs</summary>

```python
# examples/core/resp_output.py
from typing import Any
import pydantic
from toolcall.openai.core import BaseFunctionToolModel
from openai.types.responses.response_input_param import ResponseInputItemParam
from .common import openai_client, print_messages
from .common import EntitiesResponse, EntitiesResponseStrict
from .common import struct_output_system_prompt, struct_output_user_prompt


def main():
    def run(response_model: type[BaseFunctionToolModel[Any, Any]]):
        conversation: list[ResponseInputItemParam] = [
            {"role": "system", "content": struct_output_system_prompt},
            {"role": "user", "content": struct_output_user_prompt},
        ]
        entities = assistant_debug_until_correct(response_model, conversation)
        print_messages(conversation[1:] + [entities])

    run(EntitiesResponse)
    run(EntitiesResponseStrict)


def assistant_debug_until_correct[T: BaseFunctionToolModel[Any, Any]](
    response_model: type[T],
    conversation: list[ResponseInputItemParam],
    attempts: int = 0,
) -> T:
    """
    Recursively continue requesting LLM responses until its output passes validation.
    """

    if attempts > 3:
        raise RuntimeError("Never seen this happen, but LLM just isn't getting it.")

    # 1. Request an LLM response, and append it to the conversation.

    format = response_model.model_tool_format(api="responses")
    response = openai_client.responses.create(
        input=conversation,
        model="gpt-4.1",
        text={"format": format},
    )
    for item in response.output:
        conversation.append(item.model_dump())  # pyright: ignore[reportArgumentType]

    # 2. Try to parse the response content into a valid EntitiesResponse. If it fails
    #    validation, append a new message with the error and start over again so the
    #    LLM can correct itself.

    try:
        return response_model.model_validate_json(response.output_text)
    except pydantic.ValidationError as e:
        conversation.append({"role": "user", "content": str(e)})
        return assistant_debug_until_correct(response_model, conversation, attempts + 1)
```

</details>

<details><summary>Single Function Tool</summary>

```python
# examples/core/resp_tool.py
from openai.types.responses.response_input_param import ResponseInputItemParam
from .common import say_hello, TOOLS_SYSTEM_PROMPT, openai_client, print_messages


def main():
    user_prompt = "Can you please say hello to John and Kate for me?"
    conversation: list[ResponseInputItemParam] = [
        {"role": "system", "content": TOOLS_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    assistant_take_turn(conversation)
    print_messages(conversation[1:])


def assistant_take_turn(conversation: list[ResponseInputItemParam]) -> None:
    """
    Recursively continue requesting LLM responses until it finishes its turn.
    """

    # 1. Request an LLM response, and append it to the conversation.

    response = openai_client.responses.create(
        input=conversation,
        model="gpt-4.1",
        tools=[say_hello.model_tool_definition(api="responses")],
    )
    for item in response.output:
        conversation.append(item.model_dump())  # pyright: ignore[reportArgumentType]

    # 2. If there weren't any tool calls: we're done, and can finish this turn.
    #    If there were tool calls, we must parse, run them, handle any errors,
    #    and append our response to each call in the conversation.

    tool_calls = [item for item in response.output if item.type == "function_call"]
    if not tool_calls:
        return

    results = [say_hello.model_tool_run_tool_call(c, None) for c in tool_calls]
    conversation.extend([res.output_item for res in results])

    # 3. Since there were tool calls, this turn isn't finished yet. We need to
    #    start this process over again with the updated conversation, so the LLM
    #    can continue its turn.

    assistant_take_turn(conversation)
```

</details>

<details><summary>Multiple Function Tools</summary>

```python
# examples/core/resp_group.py
from openai.types.responses.response_input_param import ResponseInputItemParam
from .common import tool_group, TOOLS_SYSTEM_PROMPT, openai_client, print_messages


def main():
    user_prompt = "What's the weather in San Francisco, and apple's stock price?"
    conversation: list[ResponseInputItemParam] = [
        {"role": "system", "content": TOOLS_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    assistant_take_turn(conversation)
    print_messages(conversation[1:])


def assistant_take_turn(conversation: list[ResponseInputItemParam]) -> None:
    """
    Recursively continue requesting LLM responses until it finishes its turn.
    """

    # 1. Request an LLM response, and append it to the conversation.

    response = openai_client.responses.create(
        input=conversation,
        model="gpt-4.1",
        tools=tool_group.tool_definitions(api="responses"),
    )
    for item in response.output:
        conversation.append(item.model_dump())  # pyright: ignore[reportArgumentType]

    # 2. If there weren't any tool calls: we're done, and can finish this turn.
    #    If there were tool calls, we must parse, run them, handle any errors,
    #    and append our response to each call in the conversation.

    tool_calls = [item for item in response.output if item.type == "function_call"]
    if not tool_calls:
        return

    input_context = 100
    results = tool_group.run_tool_calls(tool_calls, input_context)
    conversation.extend([res.output_item for res in results])

    # 3. Optionally, we can inspect the results. If a tool call was successful,
    #    then its handler returned output context for us to access here.

    for result in results:
        if result.fail_reason is None:
            context: float = result.context  # Type checker knows this is safe
            print(f"Call {result.call_id} was successful. We got {context=} back.")
        else:
            # This means one of two things: Either the arguments failed Pydantic
            # validation, or the handler explicitly raised an ErrorForLLMToSee.
            _: None = result.context  # Always null. Handler didn't finish.
            print(
                f"Call {result.call_id} failed because of {result.fail_reason}, but "
                f'we handled it smoothly, replying: "{result.result_content[:12]}..."'
            )

    # 4. Since there were tool calls, this turn isn't finished yet. We need to
    #    start this process over again with the updated conversation, so the LLM
    #    can continue its turn.

    assistant_take_turn(conversation)
```

</details>

</details>

---

<details><summary><b>Asynchronous (async/await) API</b></summary>

<br>

<details><summary><code>common.py</code> - Setup code used by all examples below</summary>

```python
# examples/aio/common.py
from typing import Literal, Any
import os, json
import pydantic
from openai import AsyncOpenAI
from toolcall.openai.aio import BaseFunctionToolModel, HandlerResult, ErrorForLLMToSee

openai_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

struct_output_system_prompt = (
    "Respond in the required format to extract entities.\n\n"
    "NOTE: We are doing API testing. Your *first* response should fail validation. "
    'Violate the schema by passing `100` instead of `"100"` in numbers.'
)
struct_output_user_prompt = (
    "Query: What's the weather in San Francisco? Is it above 100 there?"
)


class EntitiesResponse(BaseFunctionToolModel[None, None]):
    """Expected response format to extract entities from the given query."""

    people: list[str]
    places: list[str]
    numbers: list[str]


# Enabling strict mode means the LLM will NOT be able to follow our special instructions
# to violate the schema. It will give a valid response the first time.
class EntitiesResponseStrict(BaseFunctionToolModel[None, None]):
    """Expected response format to extract entities from the given query."""

    people: list[str]
    places: list[str]
    numbers: list[str]

    # Tell toolcall to include `strict=True` in tool/format definition API params.
    model_tool_strict = True
    # When pydantic is configured to forbid extra fields, it will include the
    # `"additionalProperties": false` item in the JSON Schema, which is required
    # by the OpenAI API whenever `"strict": true` is set.
    model_config = pydantic.ConfigDict(extra="forbid")


TOOLS_SYSTEM_PROMPT = """
You are a helpful assistant. You have several function tools available. Use as needed.

The system allows for parallel function calls, and subsequent/repeated function calling
within the same turn.
""".strip()


# Minimal function tool that...
#   1. Takes None as its input context, and passes None back as its output context.
#   2. Uses its class name as the function tool name.
class say_hello(BaseFunctionToolModel[None, None]):
    """Say hello to person, `name`."""

    name: str

    # Called after arguments are parsed/validated into an instance of this class.
    # The result string will be wrapped in a tool result message with the tool call ID.
    async def model_tool_handler(self, _):
        return f"Message delivered to {self.name}.", None


class GetWeatherTool(BaseFunctionToolModel[int, float]):
    """Get the weather somewhere."""

    model_tool_custom_name = "get_weather"

    city: str
    """City to get the weather for."""

    state: Literal["California", "New York", "Texas"]
    """State where the city is. Only a few are available."""

    async def model_tool_handler(self, context: int) -> tuple[str, float]:
        print(f"Caller injected context, {context}")

        if self.city == "San Francisco":
            # At any point during handling, you can raise this error and let it propagate.
            # It will be caught and used as the result tool message's content. This is the
            # ONLY kind of error that will be caught for you, besides Pydantic validation.
            raise ErrorForLLMToSee(
                "Weather unavailable for San Francisco. Please get the weather for a "
                "nearby city, before responding to the user. Don't ask first. Just call "
                "this function again."
            )

        result = f"It's currently 30 degrees in {self.city}, {self.state}."
        return result, 1.234


class StockPriceTool(BaseFunctionToolModel[int, float]):
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

    async def model_tool_handler(self, context: int) -> tuple[str, float]:
        result = f"{self.ticker} is currently trading at $100."
        # HandlerResult (a named tuple) is just a more explicit alternative.
        return HandlerResult(result_content=result, context=1.234)


from toolcall.openai.aio import FunctionToolGroup

# A simple mapping to store tool classes. Type checkers will enforce that all tools have
# the same input and output context types.
# That's why we cannot include `say_hello` here.
tool_group = FunctionToolGroup.from_list([GetWeatherTool, StockPriceTool])


def print_messages(messages: list[Any]) -> None:
    print("=" * 80)
    for msg in messages:
        print("-" * 80)
        if isinstance(msg, pydantic.BaseModel):
            print(f"\n{repr(msg)}\n")
        else:
            msg = {k: v for k, v in msg.items() if v}
            print(json.dumps(msg, indent=2).strip("{}"))
```

</details>

#### Chat Completions API

<details><summary>Structured Outputs</summary>

```python
# examples/aio/chat_output.py
from typing import Any
import pydantic
from toolcall.openai.aio import BaseFunctionToolModel
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from .common import openai_client, print_messages
from .common import EntitiesResponse, EntitiesResponseStrict
from .common import struct_output_system_prompt, struct_output_user_prompt


async def main():
    async def run(response_model: type[BaseFunctionToolModel[Any, Any]]):
        conversation: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": struct_output_system_prompt},
            {"role": "user", "content": struct_output_user_prompt},
        ]
        entities = await assistant_debug_until_correct(response_model, conversation)
        print_messages(conversation[1:] + [entities])

    await run(EntitiesResponse)
    await run(EntitiesResponseStrict)


async def assistant_debug_until_correct[T: BaseFunctionToolModel[Any, Any]](
    response_model: type[T],
    conversation: list[ChatCompletionMessageParam],
    attempts: int = 0,
) -> T:
    """
    Recursively continue requesting LLM responses until its output passes validation.
    """

    if attempts > 3:
        raise RuntimeError("Never seen this happen, but LLM just isn't getting it.")

    # 1. Request an LLM response, and append it to the conversation.

    format = response_model.model_tool_format(api="chat.completions")
    response = await openai_client.chat.completions.create(
        messages=conversation,
        model="gpt-4.1",
        response_format=format,
    )
    message = response.choices[0].message
    conversation.append(message.model_dump())  # pyright: ignore[reportArgumentType]

    # 2. Try to parse the response content into a valid EntitiesResponse. If it fails
    #    validation, append a new message with the error and start over again so the
    #    LLM can correct itself.

    # (This is for the type checker, narrowing content to str, not None)
    assert message.content is not None, "Impossible since no tools given"

    try:
        return response_model.model_validate_json(message.content)
    except pydantic.ValidationError as e:
        conversation.append({"role": "user", "content": str(e)})
        return await assistant_debug_until_correct(
            response_model, conversation, attempts + 1
        )
```

</details>

<details><summary>Single Function Tool</summary>

```python
# examples/aio/chat_tool.py
import asyncio
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from .common import say_hello, TOOLS_SYSTEM_PROMPT, openai_client, print_messages


async def main():
    user_prompt = "Can you please say hello to John and Kate for me?"
    conversation: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": TOOLS_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    await assistant_take_turn(conversation)
    print_messages(conversation[1:])


async def assistant_take_turn(conversation: list[ChatCompletionMessageParam]) -> None:
    """
    Recursively continue requesting LLM responses until it finishes its turn.
    """

    # 1. Request an LLM response, and append it to the conversation.

    response = await openai_client.chat.completions.create(
        messages=conversation,
        model="gpt-4.1",
        tools=[say_hello.model_tool_definition(api="chat.completions")],
    )
    message = response.choices[0].message
    conversation.append(message.model_dump())  # pyright: ignore[reportArgumentType]

    # 2. If there weren't any tool calls: we're done, and can finish this turn.
    #    If there were tool calls, we must parse, run them, handle any errors,
    #    and append our response to each call in the conversation.

    if not message.tool_calls:
        return

    results = await asyncio.gather(
        *[say_hello.model_tool_run_tool_call(c, None) for c in message.tool_calls]
    )
    conversation.extend([res.tool_message for res in results])

    # 3. Since there were tool calls, this turn isn't finished yet. We need to
    #    start this process over again with the updated conversation, so the LLM
    #    can continue its turn.

    await assistant_take_turn(conversation)
```

</details>

<details><summary>Multiple Function Tools</summary>

```python
# examples/aio/chat_group.py
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from .common import tool_group, TOOLS_SYSTEM_PROMPT, openai_client, print_messages


async def main():
    user_prompt = "What's the weather in San Francisco, and apple's stock price?"
    conversation: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": TOOLS_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    await assistant_take_turn(conversation)
    print_messages(conversation[1:])


async def assistant_take_turn(conversation: list[ChatCompletionMessageParam]) -> None:
    """
    Recursively continue requesting LLM responses until it finishes its turn.
    """

    # 1. Request an LLM response, and append it to the conversation.

    response = await openai_client.chat.completions.create(
        messages=conversation,
        model="gpt-4.1",
        tools=tool_group.tool_definitions(api="chat.completions"),
    )
    message = response.choices[0].message
    conversation.append(message.model_dump())  # pyright: ignore[reportArgumentType]

    # 2. If there weren't any tool calls: we're done, and can finish this turn.
    #    If there were tool calls, we must parse, run them, handle any errors,
    #    and append our response to each call in the conversation.

    if not message.tool_calls:
        return

    input_context = 100
    results = await tool_group.run_tool_calls(message.tool_calls, input_context)
    conversation.extend([res.tool_message for res in results])

    # 3. Optionally, we can inspect the results. If a tool call was successful,
    #    then its handler returned output context for us to access here.

    for result in results:
        if result.fail_reason is None:
            context: float = result.context  # Type checker knows this is safe
            print(f"Call {result.call_id} was successful. We got {context=} back.")
        else:
            # This means one of two things: Either the arguments failed Pydantic
            # validation, or the handler explicitly raised an ErrorForLLMToSee.
            _: None = result.context  # Always null. Handler didn't finish.
            print(
                f"Call {result.call_id} failed because of {result.fail_reason}, but "
                f'we handled it smoothly, replying: "{result.result_content[:12]}..."'
            )

    # 4. Since there were tool calls, this turn isn't finished yet. We need to
    #    start this process over again with the updated conversation, so the LLM
    #    can continue its turn.

    await assistant_take_turn(conversation)
```

</details>

#### Responses API

<details><summary>Structured Outputs</summary>

```python
# examples/aio/resp_output.py
from typing import Any
import pydantic
from toolcall.openai.aio import BaseFunctionToolModel
from openai.types.responses.response_input_param import ResponseInputItemParam
from .common import openai_client, print_messages
from .common import EntitiesResponse, EntitiesResponseStrict
from .common import struct_output_system_prompt, struct_output_user_prompt


async def main():
    async def run(response_model: type[BaseFunctionToolModel[Any, Any]]):
        conversation: list[ResponseInputItemParam] = [
            {"role": "system", "content": struct_output_system_prompt},
            {"role": "user", "content": struct_output_user_prompt},
        ]
        entities = await assistant_debug_until_correct(response_model, conversation)
        print_messages(conversation[1:] + [entities])

    await run(EntitiesResponse)
    await run(EntitiesResponseStrict)


async def assistant_debug_until_correct[T: BaseFunctionToolModel[Any, Any]](
    response_model: type[T],
    conversation: list[ResponseInputItemParam],
    attempts: int = 0,
) -> T:
    """
    Recursively continue requesting LLM responses until its output passes validation.
    """

    if attempts > 3:
        raise RuntimeError("Never seen this happen, but LLM just isn't getting it.")

    # 1. Request an LLM response, and append it to the conversation.

    format = response_model.model_tool_format(api="responses")
    response = await openai_client.responses.create(
        input=conversation,
        model="gpt-4.1",
        text={"format": format},
    )
    for item in response.output:
        conversation.append(item.model_dump())  # pyright: ignore[reportArgumentType]

    # 2. Try to parse the response content into a valid EntitiesResponse. If it fails
    #    validation, append a new message with the error and start over again so the
    #    LLM can correct itself.

    try:
        return response_model.model_validate_json(response.output_text)
    except pydantic.ValidationError as e:
        conversation.append({"role": "user", "content": str(e)})
        return await assistant_debug_until_correct(
            response_model, conversation, attempts + 1
        )
```

</details>

<details><summary>Single Function Tool</summary>

```python
# examples/aio/resp_tool.py
import asyncio
from openai.types.responses.response_input_param import ResponseInputItemParam
from .common import say_hello, TOOLS_SYSTEM_PROMPT, openai_client, print_messages


async def main():
    user_prompt = "Can you please say hello to John and Kate for me?"
    conversation: list[ResponseInputItemParam] = [
        {"role": "system", "content": TOOLS_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    await assistant_take_turn(conversation)
    print_messages(conversation[1:])


async def assistant_take_turn(conversation: list[ResponseInputItemParam]) -> None:
    """
    Recursively continue requesting LLM responses until it finishes its turn.
    """

    # 1. Request an LLM response, and append it to the conversation.

    response = await openai_client.responses.create(
        input=conversation,
        model="gpt-4.1",
        tools=[say_hello.model_tool_definition(api="responses")],
    )
    for item in response.output:
        conversation.append(item.model_dump())  # pyright: ignore[reportArgumentType]

    # 2. If there weren't any tool calls: we're done, and can finish this turn.
    #    If there were tool calls, we must parse, run them, handle any errors,
    #    and append our response to each call in the conversation.

    tool_calls = [item for item in response.output if item.type == "function_call"]
    if not tool_calls:
        return

    results = await asyncio.gather(
        *[say_hello.model_tool_run_tool_call(c, None) for c in tool_calls]
    )
    conversation.extend([res.output_item for res in results])

    # 3. Since there were tool calls, this turn isn't finished yet. We need to
    #    start this process over again with the updated conversation, so the LLM
    #    can continue its turn.

    await assistant_take_turn(conversation)
```

</details>

<details><summary>Multiple Function Tools</summary>

```python
# examples/aio/resp_group.py
from openai.types.responses.response_input_param import ResponseInputItemParam
from .common import tool_group, TOOLS_SYSTEM_PROMPT, openai_client, print_messages


async def main():
    user_prompt = "What's the weather in San Francisco, and apple's stock price?"
    conversation: list[ResponseInputItemParam] = [
        {"role": "system", "content": TOOLS_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    await assistant_take_turn(conversation)
    print_messages(conversation[1:])


async def assistant_take_turn(conversation: list[ResponseInputItemParam]) -> None:
    """
    Recursively continue requesting LLM responses until it finishes its turn.
    """

    # 1. Request an LLM response, and append it to the conversation.

    response = await openai_client.responses.create(
        input=conversation,
        model="gpt-4.1",
        tools=tool_group.tool_definitions(api="responses"),
    )
    for item in response.output:
        conversation.append(item.model_dump())  # pyright: ignore[reportArgumentType]

    # 2. If there weren't any tool calls: we're done, and can finish this turn.
    #    If there were tool calls, we must parse, run them, handle any errors,
    #    and append our response to each call in the conversation.

    tool_calls = [item for item in response.output if item.type == "function_call"]
    if not tool_calls:
        return

    input_context = 100
    results = await tool_group.run_tool_calls(tool_calls, input_context)
    conversation.extend([res.output_item for res in results])

    # 3. Optionally, we can inspect the results. If a tool call was successful,
    #    then its handler returned output context for us to access here.

    for result in results:
        if result.fail_reason is None:
            context: float = result.context  # Type checker knows this is safe
            print(f"Call {result.call_id} was successful. We got {context=} back.")
        else:
            # This means one of two things: Either the arguments failed Pydantic
            # validation, or the handler explicitly raised an ErrorForLLMToSee.
            _: None = result.context  # Always null. Handler didn't finish.
            print(
                f"Call {result.call_id} failed because of {result.fail_reason}, but "
                f'we handled it smoothly, replying: "{result.result_content[:12]}..."'
            )

    # 4. Since there were tool calls, this turn isn't finished yet. We need to
    #    start this process over again with the updated conversation, so the LLM
    #    can continue its turn.

    await assistant_take_turn(conversation)
```

</details>

</details>

---

<br>

# API Reference - `toolcall.openai`

The entire API is mirrored across two namespaces with the same members and
naming conventions: `core` (regular) and `aio` (async/await). If your code is
async, import everything from `toolcall.openai.aio`. Otherwise use
`toolcall.openai.core`.


### `class BaseFunctionToolModel[ContextIn, ContextOut](BaseModel)`

A [Pydantic BaseModel](https://docs.pydantic.dev/latest/) that represents a data structure
an LLM should provide. Its `model_tool_handler()` method (if you implement it)
can be the home for your logic for handling a tool call to that model and
producing a text response.

#### API

- **Generic type arguments**: `[ContextIn, ContextOut]`
    
    If set, these apply a type constraint on the *additional* data your orchestration code
    must pass in to the tool handler, and that your handler must return back, respectively.

    Their purpose is to enable type safety when defining a **group** of tools
    (`FunctionToolGroup[ContextIn, ContextOut]`) and using the group to dynamically dispatch
    calls.

- **Members you _need_ to know about**

    **Create a definition of your model to send to the API**

    - `model_tool_definition(cls, api: "chat.completions" | "responses")`
        - Item to go in the `tools` array param to the OpenAI API.
    - `model_tool_format(cls, api: "chat.completions" | "responses")`
        - Structured Outputs format definition for the `response_format` and `text.format` params
          in the Chat Completions and Responses APIs respectively.

    **Handle a tool call that the LLM sent you, and produce a response.**

    - `async` `model_tool_handler(self, context: ContextIn) -> tuple[str, ContextOut]`
        - Your subclass should implement this to define how to respond when the LLM
          passes valid arguments to the tool. Your model instance - `self` - represents
          those valid arguments.
        - Takes a single argument, `context: ContextIn` of any arbitrary type you want.
          This allows your orchestration to inject additional data into the handler.
        - Returns **two** things:
            1. Result Content: A `str` to use in the response message to the LLM.
            2. Output Context: `ContextOut`: Arbitrary data to send back to your
               orchestration logic that initiated the tool call handling.
    - `async` `model_tool_run_tool_call(cls, call, context: ContextIn) -> ToolCallResult[ContextOut]`
        - Takes a tool call from any API type, applies Pydantic's parsing/validation,
          executes your handler, and wraps the result.

- **Config: Class-variables**

    Optional class configurations are set using class variables. (**Do not** declare type
    annotations when setting these.)

    - `model_tool_strict` : bool, default False
    - `model_tool_custom_name` : str or None, default None
    - `model_tool_name_generator` : ((str) => str) or None, default None
        - Function to generate a name based on the class name.
    - `model_tool_custom_description` : str or None, default None
    - `model_tool_custom_json_schema` : dict or None, default None

- **Other Utility Methods**

    - `model_tool_name(cls) -> str`
    - `model_tool_json_schema(cls) -> dict`
    - `model_tool_pretty_definition(cls) -> str`

---

### `class FunctionToolGroup[ContextIn, ContextOut](...)`

- Parent: `dict[str, type[BaseFunctionToolModel[ContextIn, ContextOut]]]`

A simple container that supports statically type-safe dynamic dispatch of tools.

Documentation coming soon. See the **Learn-by-example Documentation** near the top of
this page, in the dynamic tool call dispatch section. There are examples of how a
tool group is used.