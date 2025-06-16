# File generated from its async equivalent, examples/aio/resp_group.py
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
