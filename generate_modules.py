"""
Generates synchronous (i.e. "core") modules based on their async/await (i.e. "aio")
equivalents.
This works using simple str.replace() and re.sub() to remove async/await syntax,
asyncio imports, and other custom patterns like changing '.aio' import paths to '.core'.
"""

from typing import NamedTuple
from pathlib import Path
import subprocess
import re
from ruff.__main__ import find_ruff_bin  # pyright: ignore[reportMissingTypeStubs]

CURR_DIR = Path(__file__).parent

README_TEMPLATE_NAME = "README.template.md"
README_SOURCE_CODE_DIRS: list[str] = [
    "examples",
    "toolcall",
]
FILE_COMMENT_HEADER_PREFIX = "# File generated from its async equivalent, "


class Job(NamedTuple):
    source: str
    destination: str


jobs = (
    Job("toolcall/openai/aio", "toolcall/openai/core"),
    Job("examples/aio", "examples/core"),
    Job("tests/test_oai_aio.py", "tests/test_oai_core.py"),
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
    source = get_header_comment(source_file) + source
    source = ruff_format(source)
    destination_file.parent.mkdir(parents=True, exist_ok=True)
    destination_file.write_text(source, encoding="utf-8")


def convert_async_code_to_sync(code: str) -> str:
    code = (
        code.replace("import asyncio\n", "")
        .replace("async def ", "def ")
        .replace(" await ", " ")
        .replace(".aio import ", ".core import ")
        .replace("AsyncOpenAI", "OpenAI")
    )
    # Replace `asyncio.run(some_func())` with `some_func()`
    code = re.sub(r"asyncio\.run\(([^\n]+)\)", r"\1", code)
    # Replace `asyncio.gather(*FOO)` with `FOO`
    code = re.sub(r"asyncio\.gather\(\s*\*([^\n]+)\s*\)", r"\1", code)
    return code


def get_header_comment(source_file: Path) -> str:
    rel_source_file = source_file.relative_to(CURR_DIR)
    return f"{FILE_COMMENT_HEADER_PREFIX}{rel_source_file}\n"


def ruff_format(source: str) -> str:
    result = subprocess.run(
        [find_ruff_bin(), "format", "-"],
        input=source,
        text=True,
        capture_output=True,
        check=True,
        cwd=CURR_DIR,
    )
    result.check_returncode()
    return result.stdout


if __name__ == "__main__":
    main()
