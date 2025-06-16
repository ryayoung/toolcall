"""
Generates the README from the README.template by hydrating it with source code
snippets from specified files that contain real, tested code.
"""

from typing import Any
import glob
import textwrap
from pathlib import Path
import jinja2 as j2
from generate_modules import FILE_COMMENT_HEADER_PREFIX


CURR_DIR = Path(__file__).parent
CURR_SCRIPT_NAME = Path(__file__).name
README_TEMPLATE_NAME = "README.template.md"
README_SOURCE_CODE_DIRS: list[str] = [
    "examples",
    # "toolcall",
    # "tests",
]


def main():
    template_path = CURR_DIR / README_TEMPLATE_NAME
    assert template_path.exists()
    template_source = template_path.read_text(encoding="utf-8")
    template_kwargs = get_readme_template_kwargs()
    readme_content = hydrate_template(template_source, **template_kwargs)
    readme_path = CURR_DIR / "README.md"
    header_line = f"<!-- File generated from /{README_TEMPLATE_NAME} using /{CURR_SCRIPT_NAME} -->"
    readme_content = header_line + "\n\n" + readme_content
    readme_path.write_text(readme_content, encoding="utf-8")


def source_file_to_markdown_snippet(source_file: Path) -> str:
    if not source_file.exists():
        raise FileNotFoundError(f"Source file {source_file} does not exist.")
    source = source_file.read_text(encoding="utf-8")
    source_lines = [
        line
        for line in source.splitlines()
        if not line.startswith(FILE_COMMENT_HEADER_PREFIX)
    ]
    source_lines = [
        f"# {source_file.relative_to(CURR_DIR)}",
        *source_lines,
    ]
    source = "\n".join(source_lines)
    return f"\n\n```python\n{source}\n```\n\n"


def hydrate_template(source: str, **kwargs: Any) -> str:
    env = j2.Environment(undefined=j2.StrictUndefined)

    def indent(text: Any, spaces: int) -> str:
        return textwrap.indent(text, " " * spaces).lstrip()

    env.filters["indent"] = indent  # pyright: ignore[reportArgumentType]
    return env.from_string(source).render(**kwargs)


def get_readme_template_kwargs() -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for source_dir in README_SOURCE_CODE_DIRS:
        template_data = result[source_dir] = {}
        source_dir_path = Path(source_dir).absolute()

        for path in glob.glob(f"{source_dir_path}/**/*.py", recursive=True):
            full_source_file_path = Path(path).absolute()
            rel_source_file_path = full_source_file_path.relative_to(source_dir_path)
            snippet = source_file_to_markdown_snippet(full_source_file_path)

            key = str(rel_source_file_path).removesuffix(".py").replace("/", ".")

            template_data[key] = snippet

    return result


if __name__ == "__main__":
    main()
