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
