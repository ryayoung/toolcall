# File generated from its async equivalent, examples/aio/resp_output.py
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
