from typing import NamedTuple
from pathlib import Path
import re


class Job(NamedTuple):
    source: str
    destination: str


jobs = (
    Job(
        "toolcall/openai/aio/__init__.py",
        "toolcall/openai/core/__init__.py",
    ),
    Job(
        "toolcall/openai/aio/_group.py",
        "toolcall/openai/core/_group.py",
    ),
    Job(
        "toolcall/openai/aio/_tool.py",
        "toolcall/openai/core/_tool.py",
    ),
    Job(
        "examples/aio/common.py",
        "examples/core/common.py",
    ),
)


def main():
    for source, destination in jobs:
        source_path = Path(source).absolute()
        destination_path = Path(destination).absolute()
        assert source_path.is_file()
        process_and_write(source_path, destination_path)


def process_and_write(source_file: Path, destination_file: Path):
    source = source_file.read_text(encoding="utf-8")
    source = convert_async_code_to_sync(source)
    destination_file.parent.mkdir(parents=True, exist_ok=True)
    destination_file.write_text(source, encoding="utf-8")


def convert_async_code_to_sync(code: str) -> str:
    code = (
        code.replace("import asyncio\n", "")
        .replace("async def ", "def ")
        .replace(" await ", " ")
        .replace("from toolcall.openai.aio import", "from toolcall.openai.core import")
        .replace("AsyncOpenAI", "OpenAI")
    )
    code = re.sub(r"asyncio\.run\(([^\n]+)\)", r"\1", code)
    # Replace `asyncio.gather(*...)` with `...`
    code = re.sub(r"asyncio\.gather\(\s*\*([^\n]+)\s*\)", r"\1", code)
    return code


if __name__ == "__main__":
    main()
