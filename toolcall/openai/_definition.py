from typing import Any
from pydantic import BaseModel
from openai.types.shared_params import FunctionDefinition, ResponseFormatJSONSchema
from openai.types.chat import ChatCompletionToolParam
from openai.types.responses import (
    FunctionToolParam,
    ResponseFormatTextJSONSchemaConfigParam,
)
from openai.types.shared_params.response_format_json_schema import JSONSchema

__all__ = ["StandardToolDefinition"]


class StandardToolDefinition(BaseModel):
    """
    A common data structure from which we can derive tool definitions and
    structured output format definitions for both the Chat Completions API
    and the Responses API.
    """

    name: str
    description: str
    strict: bool
    json_schema: dict[str, Any]

    def tool_def_for_chat_completions_api(self) -> ChatCompletionToolParam:
        """
        Tool definition for the `tools` array in the Chat Completions API
        """
        function: FunctionDefinition = {
            "name": self.name,
            "description": self.description,
            "parameters": self.json_schema,
            "strict": self.strict,
        }
        return {"type": "function", "function": function}

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
        json_schema: JSONSchema = {
            "name": self.name,
            "description": self.description,
            "schema": self.json_schema,
            "strict": self.strict,
        }
        return {"type": "json_schema", "json_schema": json_schema}

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

    def to_pretty(self) -> str:
        """For debugging purposes, get a pretty representation of the data."""
        definition = self.model_dump_json(indent=4)

        # If available, use black, since it's nicer than default json indentation.
        try:
            import black

            definition = black.format_str(definition, mode=black.Mode()).strip()
        except:  # pragma: no cover
            pass

        return definition
