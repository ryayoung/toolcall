from typing import NamedTuple
from pathlib import Path
import re


class Job(NamedTuple):
    source: str
    destination: str


jobs = (
    Job("toolcall/openai/aio", "toolcall/openai/core"),
    Job("examples/aio", "examples/core"),
    Job("tests/test_oai_examples_aio.py", "tests/test_oai_examples_core.py"),
)


def main():
    for source, destination in jobs:
        source_path = Path(source).absolute()
        destination_path = Path(destination).absolute()
        if source_path.is_file():
            process_and_write(source_path, destination_path)
        else:
            for path in source_path.iterdir():
                if path.suffix == ".py":
                    process_and_write(path, destination_path / path.name)


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
