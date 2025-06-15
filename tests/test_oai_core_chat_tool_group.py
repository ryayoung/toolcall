# Test is formatted this way for easy pasting into readme.


def test_example():
    import os
    import json
    from typing import Literal
    from openai import OpenAI
    from openai.types.chat.chat_completion_message_param import (
        ChatCompletionMessageParam,
    )
    from toolcall.openai.core import (
        FunctionToolGroup,
        BaseFunctionToolModel,
        HandlerResult,
        ErrorForLLMToSee,
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
    tool_group = FunctionToolGroup[None, float]()

    # At runtime this decorator just adds the class to the group {"get_weather": WeatherTool}
    # Its bigger purpose is static type enforcement that the tool's input/output context
    # types satisfy those of the group. (e.g. [None, None])
    @tool_group.add_tool
    class get_weather(
        BaseFunctionToolModel[None, float]
    ):  # <-- Pydantic BaseModel extension
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
                raise ErrorForLLMToSee(
                    "Weather unavailable for San Francisco. Please get the weather for a "
                    "nearby city, before responding to the user. Don't ask first. Just call "
                    "this function again."
                )

            result = f"It's currently 30 degrees in {self.city}, {self.state}."
            # Result for the LLM, plus any context you want to pass back to yourself.
            return result, 1.234

    @tool_group.add_tool
    class StockPriceTool(BaseFunctionToolModel[None, float]):
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

        def model_tool_handler(self, context: None) -> HandlerResult[float]:
            result = f"{self.ticker} is currently trading at $100."
            return HandlerResult(result_content=result, context=1.234)

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
        # REMOVE FROM README -----
        result = results[0]
        if result.fail_reason is None:
            from typing_extensions import assert_type

            assert_type(result.context, float)
        # ----- END REMOVE FROM README
        return assistant_take_turn(messages)

    main()


if __name__ == "__main__":
    test_example()
