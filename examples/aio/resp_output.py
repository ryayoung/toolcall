import json
from typing import Any
import pydantic
from toolcall.openai.aio import BaseFunctionToolModel
from openai.types.responses.response_input_param import ResponseInputItemParam
from .common import (
    openai_client,
    struct_output_system_prompt,
    struct_output_user_prompt,
    EntitiesResponse,
    EntitiesResponseStrict,
)


async def main():
    async def run(response_model: type[BaseFunctionToolModel[Any, Any]]):
        conversation: list[ResponseInputItemParam] = [
            {"role": "system", "content": struct_output_system_prompt},
            {"role": "user", "content": struct_output_user_prompt},
        ]

        entities = await assistant_debug_until_correct(response_model, conversation)

        name = response_model.__name__
        print(f"\n{'-' * 80}\n{'-' * 80}\n\nConversation for {name}:\n")
        for msg in conversation[2:]:
            print("-" * 80 + "\n" + json.dumps(msg, indent=2).strip("{}"))
        print(f"{'-' * 80}\n\n{name}:\n{entities}")

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

    format = response_model.model_tool_json_format_definition(api="responses")
    response = await openai_client.responses.create(
        input=conversation,
        model="gpt-4.1-mini",
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
