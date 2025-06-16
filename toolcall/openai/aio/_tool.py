from typing import Any, Callable, Literal, ClassVar, overload
from textwrap import dedent
import pydantic
from openai.types.chat import ChatCompletionToolParam
from openai.types.shared_params import ResponseFormatJSONSchema
from openai.types.responses import (
    ResponseFormatTextJSONSchemaConfigParam,
    FunctionToolParam,
)
from .._result import (
    ToolCallResult,
    ToolCallSuccess,
    ToolCallFailure,
    ErrorForLLMToSee,
)
from .._definition import StandardToolDefinition
from .._call import StandardToolCall, AnyToolCall


__all__ = ["BaseFunctionToolModel"]


class BaseFunctionToolModel[ContextIn, ContextOut](pydantic.BaseModel):
    """
    Pydantic BaseModel with utilities for structured communication with LLMs.

    Equally useful for function-tool calling and structured output.
    """

    model_config = pydantic.ConfigDict(use_attribute_docstrings=True)

    # Things for subclasses to customize/override
    # ----------------------------------------------------------------------------------

    model_tool_strict: ClassVar[bool] = False
    """
    Class config: Whether to enable `strict` mode in the tool/function definition.
    """

    model_tool_custom_name: ClassVar[str | None] = None
    """
    Class config: Use a custom name, instead of the class name, for the function
    definition. This takes precedence over the name generator, if one is set.
    """

    model_tool_name_generator: ClassVar[Callable[[str], str] | None] = None
    """
    Class config: Function to generate a function name from the class name.
    """

    model_tool_custom_description: ClassVar[str | None] = None
    """
    Class config: Custom description to use instead of the class docstring.
    """

    model_tool_custom_json_schema: ClassVar[dict[str, Any] | None] = None
    """
    Class config: Use a custom JSON schema instead of letting Pydantic generate one.
    """

    async def model_tool_handler(self, context: ContextIn, /) -> tuple[str, ContextOut]:
        """
        Subclasses should override this with the handling logic for the tool.
        """
        raise NotImplementedError(f"{type(self).__name__}.model_tool_handler()")

    @classmethod
    def model_tool_json_schema(cls) -> dict[str, Any]:
        """
        Get the JSON schema to be used in the function definition.
        """
        return cls.model_tool_custom_json_schema or cls.model_json_schema()

    # Things to be used, not overridden
    # ----------------------------------------------------------------------------------

    @classmethod
    def model_tool_name(cls) -> str:
        """
        Name of the function tool.
        Order of priority: Custom name, name generator, class name.
        """
        custom, generate = cls.model_tool_custom_name, cls.model_tool_name_generator
        return custom or (generate and generate(cls.__name__)) or cls.__name__

    @classmethod
    def model_tool_standard_definition(cls) -> StandardToolDefinition:
        """
        Get a standard definition of this model.
        """
        name, strict = cls.model_tool_name(), cls.model_tool_strict
        schema = cls.model_tool_json_schema()

        schema.pop("title", None)  # Because we have function name.
        description = schema.pop("description", "")  # Because we pass it separately.
        description = cls.model_tool_custom_description or description
        description = dedent(description).strip()

        return StandardToolDefinition(
            name=name, description=description, json_schema=schema, strict=strict
        )

    @classmethod
    async def model_tool_run_tool_call(
        cls, call: AnyToolCall, context: ContextIn
    ) -> ToolCallResult[ContextOut]:
        """
        Parse, validate, and handle a tool call.
        """
        call = StandardToolCall.from_any_call(call)
        try:
            self = cls.model_validate_json(call.arguments)
        except pydantic.ValidationError as e:
            return ToolCallFailure(
                call_id=call.id,
                result_content=str(e),
                fail_reason="invalid_arguments",
                exception=e,
            )

        # Run the subclass's handler and **only** catch errors they explicitly threw
        # with the intent of being caught here.
        try:
            result = await self.model_tool_handler(context)
        except ErrorForLLMToSee as e:
            return ToolCallFailure(
                call_id=call.id,
                result_content=str(e),
                fail_reason="explicit_handler_error",
                exception=e,
            )

        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError(f"Expected tuple of length 2 from {cls.__name__}'s handler")

        content, ctx = result
        return ToolCallSuccess(call_id=call.id, result_content=content, context=ctx)

    @overload
    @classmethod
    def model_tool_definition(cls, api: Literal["responses"]) -> FunctionToolParam: ...

    @overload
    @classmethod
    def model_tool_definition(
        cls, api: Literal["chat.completions"]
    ) -> ChatCompletionToolParam: ...

    @overload
    @classmethod
    def model_tool_definition(
        cls, api: Literal["chat.completions", "responses"]
    ) -> ChatCompletionToolParam | FunctionToolParam: ...

    @classmethod
    def model_tool_definition(
        cls, api: Literal["chat.completions", "responses"]
    ) -> ChatCompletionToolParam | FunctionToolParam:
        """
        Tool definition for the `tools` array parameter in the API.
        """
        std = cls.model_tool_standard_definition()
        if api == "responses":
            return std.tool_def_for_responses_api()
        return std.tool_def_for_chat_completions_api()

    @overload
    @classmethod
    def model_tool_format(
        cls, api: Literal["responses"]
    ) -> ResponseFormatTextJSONSchemaConfigParam: ...

    @overload
    @classmethod
    def model_tool_format(
        cls, api: Literal["chat.completions"]
    ) -> ResponseFormatJSONSchema: ...

    @overload
    @classmethod
    def model_tool_format(
        cls, api: Literal["chat.completions", "responses"]
    ) -> ResponseFormatTextJSONSchemaConfigParam | ResponseFormatJSONSchema: ...

    @classmethod
    def model_tool_format(
        cls, api: Literal["chat.completions", "responses"]
    ) -> ResponseFormatTextJSONSchemaConfigParam | ResponseFormatJSONSchema:
        """
        Structured Output format definition for the `response_format` and `text.format`
        parameters in the Chat Completions and Responses APIs, respectively.
        """
        std = cls.model_tool_standard_definition()
        if api == "responses":
            return std.format_for_responses_api()
        return std.format_for_chat_completions_api()

    @classmethod
    def model_tool_pretty_definition(cls) -> str:
        """
        FOR DEBUGGING ONLY, get a pretty representation of the tool definition.
        """
        return f"{cls.__name__}({cls.model_tool_standard_definition().to_pretty()})"
