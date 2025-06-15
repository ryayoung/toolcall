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
        LLMFunctionTool,
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
                raise ErrorForLLMToSee(
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

        def model_tool_handler(self, context: None) -> HandlerResult[float]:
            result = f"{self.ticker} is currently trading at $100."
            return HandlerResult(result_content=result, context=1.234)

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
                    # REMOVE FROM README -----
                    from typing_extensions import assert_type

                    assert_type(result.context, None)
                    # ----- END REMOVE FROM README
                    print("get_weather raised a ErrorForLLMToSee()")

            elif call.function.name == StockPriceTool.model_tool_name():
                result = StockPriceTool.model_tool_run_tool_call(call, None)
                tool_message = result.tool_message

                if result.fail_reason is None:
                    # REMOVE FROM README -----
                    from typing_extensions import assert_type

                    assert_type(result.context, float)
                    # ----- END REMOVE FROM README
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


if __name__ == "__main__":
    test_example()
