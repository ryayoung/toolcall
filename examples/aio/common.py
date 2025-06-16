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
