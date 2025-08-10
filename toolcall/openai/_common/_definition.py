from typing import Any, Literal, Annotated
from pydantic import BaseModel, Discriminator
from openai.types.chat.chat_completion_custom_tool_param import Custom
from openai.types.shared_params import ResponseFormatJSONSchema
from openai.types.chat import (
    ChatCompletionFunctionToolParam,
    ChatCompletionCustomToolParam,
)
from openai.types.responses import (
    CustomToolParam,
    FunctionToolParam,
    ResponseFormatTextJSONSchemaConfigParam,
)

__all__ = [
    "StandardFunctionToolDefinition",
    "StandardCustomToolDefinition",
    "StandardCustomToolFormat",
    "StandardCustomToolTextFormat",
    "StandardCustomToolGrammarFormat",
]


class StandardFunctionToolDefinition(BaseModel):
    """
    A common data structure from which we can derive function tool definitions
    and structured output format definitions for both the Chat Completions API
    and the Responses API.
    """

    type: Literal["function"] = "function"
    name: str
    description: str
    strict: bool
    json_schema: dict[str, Any]

    def tool_def_for_chat_completions_api(self) -> ChatCompletionFunctionToolParam:
        """
        Tool definition for the `tools` array in the Chat Completions API
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.json_schema,
                "strict": self.strict,
            },
        }

    def tool_def_for_responses_api(self) -> FunctionToolParam:
        """
        Tool definition for the `tools` array in the Responses API
        """
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.json_schema,
            "strict": self.strict,
        }

    def format_for_chat_completions_api(self) -> ResponseFormatJSONSchema:
        """
        JSON Schema format for `response_format` field in Chat Completions API.
        """
        return {
            "type": "json_schema",
            "json_schema": {
                "name": self.name,
                "description": self.description,
                "schema": self.json_schema,
                "strict": self.strict,
            },
        }

    def format_for_responses_api(
        self,
    ) -> ResponseFormatTextJSONSchemaConfigParam:
        """
        JSON Schema format def for `text.format` field in the Responses API.
        """
        return {
            "name": self.name,
            "schema": self.json_schema,
            "type": "json_schema",
            "description": self.description,
            "strict": self.strict,
        }


class StandardCustomToolTextFormat(BaseModel):
    type: Literal["text"] = "text"


class StandardCustomToolGrammarFormat(BaseModel):
    type: Literal["grammar"] = "grammar"
    syntax: Literal["lark", "regex"]
    definition: str


type StandardCustomToolFormat = Annotated[
    StandardCustomToolTextFormat | StandardCustomToolGrammarFormat,
    Discriminator("type"),
]


class StandardCustomToolDefinition(BaseModel):
    """
    A common data structure from which we can derive custom tool definitions
    for both the Chat Completions API and the Responses API.
    """

    type: Literal["custom"] = "custom"
    name: str
    description: str
    format: StandardCustomToolFormat | None

    def tool_def_for_chat_completions_api(self) -> ChatCompletionCustomToolParam:
        """
        Tool definition for the `tools` array in the Chat Completions API
        """
        custom: Custom = {
            "name": self.name,
            "description": self.description,
        }
        if (fmt := self.format) is not None:
            if fmt.type == "text":
                custom["format"] = {"type": "text"}
            else:
                custom["format"] = {
                    "type": "grammar",
                    "grammar": {
                        "syntax": fmt.syntax,
                        "definition": fmt.definition,
                    },
                }
        return {"type": "custom", "custom": custom}

    def tool_def_for_responses_api(self) -> CustomToolParam:
        """
        Tool definition for the `tools` array in the Responses API
        """
        custom: CustomToolParam = {
            "type": "custom",
            "name": self.name,
            "description": self.description,
        }
        if (fmt := self.format) is not None:
            if fmt.type == "text":
                custom["format"] = {"type": "text"}
            else:
                custom["format"] = {
                    "type": "grammar",
                    "syntax": fmt.syntax,
                    "definition": fmt.definition,
                }
        return custom
