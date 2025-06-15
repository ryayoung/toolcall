from examples.aio import (
    chat_group,
    chat_output,
    chat_tool,
    resp_group,
    resp_output,
    resp_tool,
)


def test_aio():
    def main():
        runs = [
            chat_group.main(),
            chat_output.main(),
            chat_tool.main(),
            resp_group.main(),
            resp_output.main(),
            resp_tool.main(),
        ]
        _ = runs

    main()
