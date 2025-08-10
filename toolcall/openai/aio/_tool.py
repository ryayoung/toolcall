from typing import Any, Callable, Literal, ClassVar, overload
from textwrap import dedent
import pydantic
from openai.types.chat import (
    ChatCompletionFunctionToolParam,
    ChatCompletionCustomToolParam,
)
from openai.types.shared_params import ResponseFormatJSONSchema, CustomToolInputFormat
from openai.types.responses import (
    ResponseFormatTextJSONSchemaConfigParam,
    FunctionToolParam,
    CustomToolParam,
)
from .._common import (
    StandardFunctionToolDefinition,
    StandardCustomToolDefinition,
    StandardCustomToolFormat,
    StandardCustomToolGrammarFormat,
    StandardCustomToolTextFormat,
    AnyFunctionToolCall,
    StandardToolCall,
    AnyCustomToolCall,
    ToolCallResult,
    ToolCallSuccess,
    ToolCallFailure,
    ErrorForLLMToSee,
)

__all__ = [
    "BaseFunctionToolModel",
    "BaseCustomToolModel",
    "ToolModelType",
]


type ToolModelType[CtxIn, CtxOut] = (
    BaseFunctionToolModel[CtxIn, CtxOut] | BaseCustomToolModel[CtxIn, CtxOut]
)
"""
Generic type alias for any tool model (function or custom).
"""


class BaseFunctionToolModel[ContextIn, ContextOut](pydantic.BaseModel):
    """
    Pydantic BaseModel for defining a function tool for the OpenAI API,
    and handling calls to it.
    It also supports defining a structured output format and parsing
    input from the model.

    If you want a tool that takes arbitrary text input instead of a JSON
    object, use a Custom Tool instead.
    """

    model_config = pydantic.ConfigDict(use_attribute_docstrings=True)

    model_tool_type: ClassVar[Literal["function"]] = "function"
    """
    Class-level utility discriminator for identifying this as a Function Tool.
    """

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
    def model_tool_standard_definition(cls) -> StandardFunctionToolDefinition:
        """
        Get a standard definition of this model.
        """
        name, strict = cls.model_tool_name(), cls.model_tool_strict
        schema = cls.model_tool_json_schema()

        schema.pop("title", None)  # Because we have function name.
        description = schema.pop("description", "")  # Because we pass it separately.
        description = cls.model_tool_custom_description or description
        description = dedent(description).strip()

        return StandardFunctionToolDefinition(
            name=name, description=description, json_schema=schema, strict=strict
        )

    @classmethod
    async def model_tool_run_tool_call(
        cls, call: AnyFunctionToolCall, context: ContextIn
    ) -> ToolCallResult[ContextOut]:
        """
        Parse, validate, and handle a tool call.
        """
        call = StandardToolCall.from_any_call(call)
        try:
            self = cls.model_validate_json(call.input)
        except pydantic.ValidationError as e:
            return ToolCallFailure(
                type=call.type,
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
                type=call.type,
                call_id=call.id,
                result_content=str(e),
                fail_reason="explicit_handler_error",
                exception=e,
            )

        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError(f"Expected tuple of length 2 from {cls.__name__}'s handler")

        content, ctx = result
        return ToolCallSuccess(
            type=call.type, call_id=call.id, result_content=content, context=ctx
        )

    @overload
    @classmethod
    def model_tool_definition(cls, api: Literal["responses"]) -> FunctionToolParam: ...

    @overload
    @classmethod
    def model_tool_definition(
        cls, api: Literal["chat.completions"]
    ) -> ChatCompletionFunctionToolParam: ...

    @overload
    @classmethod
    def model_tool_definition(
        cls, api: Literal["chat.completions", "responses"]
    ) -> ChatCompletionFunctionToolParam | FunctionToolParam: ...

    @classmethod
    def model_tool_definition(
        cls, api: Literal["chat.completions", "responses"]
    ) -> ChatCompletionFunctionToolParam | FunctionToolParam:
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


# =======================================================================================
# =======================================================================================


class BaseCustomToolModel[ContextIn, ContextOut](pydantic.BaseModel):
    """
    Pydantic BaseModel for defining a custom tool for the OpenAI API,
    and handling calls to it.

    Unlike Function Tools - which take a JSON object that can be directly parsed
    and validated by a BaseModel - Custom Tools take only a single string input. This
    class declares a single `input` field for that string, and will handle tool calls
    by passing `{"input": call.input}` to this class's Pydantic model validation.
    """

    model_config = pydantic.ConfigDict(use_attribute_docstrings=True)

    model_tool_type: ClassVar[Literal["custom"]] = "custom"
    """
    Class-level utility discriminator for identifying this as a Custom Tool.
    """

    # Model fields
    # ----------------------------------------------------------------------------------

    input: str | Any
    """
    The field that stores the input to the tool. By default, this will always be a string
    with no validation.
    However, subclasses may override this field with another type
    (e.g. `Literal["foo", "bar"]`) or define a custom field validator, using
    `@pydantic.field_validator("input")`, to add validation logic.
    """

    # Things for subclasses to customize/override
    # ----------------------------------------------------------------------------------

    model_tool_format: ClassVar[
        StandardCustomToolFormat | CustomToolInputFormat | None
    ] = None
    """
    Class config: The `format` parameter for the custom tool definition.
    """

    model_tool_custom_name: ClassVar[str | None] = None
    """
    Class config: Use a custom name, instead of the class name, for the tool
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

    async def model_tool_handler(self, context: ContextIn, /) -> tuple[str, ContextOut]:
        """
        Subclasses should override this with the handling logic for the tool.
        """
        raise NotImplementedError(f"{type(self).__name__}.model_tool_handler()")

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
    def model_tool_standard_definition(cls) -> StandardCustomToolDefinition:
        """
        Get a standard definition of this model.
        """
        name = cls.model_tool_name()
        description: str = cls.model_tool_custom_description or cls.__doc__ or ""
        description = dedent(description).strip()

        format: StandardCustomToolFormat | None = None
        if (fmt := cls.model_tool_format) is not None:
            if not isinstance(fmt, dict):
                format = fmt
            elif fmt["type"] == "text":
                format = StandardCustomToolTextFormat()
            elif fmt["type"] == "grammar":
                format = StandardCustomToolGrammarFormat(
                    syntax=fmt["syntax"],
                    definition=fmt["definition"],
                )
            else:
                raise ValueError(f"Unsupported custom tool format type: {fmt['type']}")

        return StandardCustomToolDefinition(
            name=name, description=description, format=format
        )

    @classmethod
    async def model_tool_run_tool_call(
        cls, call: AnyCustomToolCall, context: ContextIn
    ) -> ToolCallResult[ContextOut]:
        """
        Parse, validate, and handle a tool call.
        """
        if len(cls.model_fields) > 1:
            bad = [f for f in cls.model_fields if f != "input"]
            raise ValueError(
                f"Expected Custom Tool, '{cls.__name__}', to have a single model field, "
                f"'input' (which is already declared in the base class). Please remove "
                f"fields, {bad}. If other fields should be derived from the input, use "
                "Pydantic computed fields. Or, if this tool is intended to have multiple "
                "parameters, consider using a Function Tool instead."
            )

        call = StandardToolCall.from_any_call(call)
        try:
            self = cls.model_validate({"input": call.input})
        except pydantic.ValidationError as e:
            return ToolCallFailure(
                type=call.type,
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
                type=call.type,
                call_id=call.id,
                result_content=str(e),
                fail_reason="explicit_handler_error",
                exception=e,
            )

        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError(f"Expected tuple of length 2 from {cls.__name__}'s handler")

        content, ctx = result
        return ToolCallSuccess(
            type=call.type, call_id=call.id, result_content=content, context=ctx
        )

    @overload
    @classmethod
    def model_tool_definition(cls, api: Literal["responses"]) -> CustomToolParam: ...

    @overload
    @classmethod
    def model_tool_definition(
        cls, api: Literal["chat.completions"]
    ) -> ChatCompletionCustomToolParam: ...

    @overload
    @classmethod
    def model_tool_definition(
        cls, api: Literal["chat.completions", "responses"]
    ) -> ChatCompletionCustomToolParam | CustomToolParam: ...

    @classmethod
    def model_tool_definition(
        cls, api: Literal["chat.completions", "responses"]
    ) -> ChatCompletionCustomToolParam | CustomToolParam:
        """
        Tool definition for the `tools` array parameter in the API.
        """
        std = cls.model_tool_standard_definition()
        if api == "responses":
            return std.tool_def_for_responses_api()
        return std.tool_def_for_chat_completions_api()
