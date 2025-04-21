from typing import (
    Any,
    Callable,
    Literal,
    Self,
    ClassVar,
    overload,
)
from textwrap import dedent
import pydantic
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)
from openai.types.responses.function_tool_param import (
    FunctionToolParam,
)
from openai.types.responses.response_function_tool_call import (
    ResponseFunctionToolCall,
)

from ..common import (
    StandardToolCall,
    ToolErrorMessageForLLMToSee,
    ToolHandlerResult,
    ToolCallSuccess,
    ToolCallFailure,
    ToolCallResult,
    standardize_tool_call,
    tool_def_for_chat_completions_api,
    tool_def_for_responses_api,
)


class LLMFunctionTool[ContextIn, ContextOut](pydantic.BaseModel):
    """
    Base class for defining an OpenAI function to be called by the LLM.

    This is equally useful for tool-calling as well as structured output.
    """

    model_config = pydantic.ConfigDict(
        populate_by_name=True,
        coerce_numbers_to_str=True,
        use_attribute_docstrings=True,
    )

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

    model_tool_custom_json_schema: ClassVar[dict | None] = None
    """
    Class config: Use a custom JSON schema instead of letting Pydantic generate one.
    """

    def model_tool_handler(
        self, context: ContextIn
    ) -> ToolHandlerResult[ContextOut] | tuple[str, ContextOut]:
        """
        Subclasses should override this with the handling logic for the tool.
        """
        raise NotImplementedError(f"{type(self).__name__}.model_tool_handler()")

    @classmethod
    def model_tool_format_invalid_arguments_error(
        cls, err: pydantic.ValidationError
    ) -> str:
        """
        Format a pydantic validation error for the tool message to the LLM.
        """
        return str(err)

    @classmethod
    def model_tool_format_explicit_error(cls, err: ToolErrorMessageForLLMToSee) -> str:
        """
        Format an explicit error from the handler, for the tool message to the LLM.
        """
        return str(err)

    @classmethod
    def model_tool_validate_tool_call(cls, name: str, arguments: str) -> Self:
        """
        Validate tool call into an instance of this class.
        """
        assert name == cls.model_tool_name()
        return cls.model_validate_json(arguments)

    @classmethod
    def model_tool_generate_json_schema(cls) -> dict:
        """
        Used when custom json schema is unset.
        """
        return cls.model_json_schema()

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

    @overload
    @classmethod
    def model_tool_definition(  # pragma: no cover
        cls, api: Literal["responses"]
    ) -> FunctionToolParam: ...

    @overload
    @classmethod
    def model_tool_definition(  # pragma: no cover
        cls, api: Literal["chat.completions"]
    ) -> ChatCompletionToolParam: ...

    @classmethod
    def model_tool_definition(
        cls, api: Literal["chat.completions", "responses"]
    ) -> ChatCompletionToolParam | FunctionToolParam:
        """
        Tool definition for the `tools` array in the API request.
        """
        name, strict = cls.model_tool_name(), cls.model_tool_strict
        schema = cls.model_tool_json_schema()
        description = cls.model_tool_custom_description or schema.get("description", "")
        description = dedent(description.strip())
        if api == "chat.completions":
            return tool_def_for_chat_completions_api(name, description, strict, schema)
        return tool_def_for_responses_api(name, description, strict, schema)

    @classmethod
    def model_tool_run_tool_call(
        cls,
        call: (
            ChatCompletionMessageToolCall | ResponseFunctionToolCall | StandardToolCall
        ),
        context: ContextIn,
    ) -> ToolCallResult[ContextOut]:
        """
        Parse, validate, and handle a tool call.
        """
        call = standardize_tool_call(call)
        try:
            self = cls.model_tool_validate_tool_call(call.name, call.arguments)
        except pydantic.ValidationError as e:
            return ToolCallFailure(
                call_id=call.id,
                result_content=cls.model_tool_format_invalid_arguments_error(e),
                fail_reason="invalid_arguments",
            )

        try:
            result = self.model_tool_handler(context)
        except ToolErrorMessageForLLMToSee as e:
            return ToolCallFailure(
                call_id=call.id,
                result_content=cls.model_tool_format_explicit_error(e),
                fail_reason="explicit_handler_error",
            )

        result = self.model_tool_validate_handler_result(result)
        return ToolCallSuccess(
            call_id=call.id,
            result_content=result.result_content,
            context=result.context,
        )

    @classmethod
    def model_tool_validate_handler_result(
        cls, result: ToolHandlerResult[ContextOut] | tuple[str, ContextOut] | Any
    ) -> ToolHandlerResult[ContextOut]:
        """
        Standardize the handler's return value to a ToolHandlerResult.
        """
        if not isinstance(result, tuple) or len(result) != 2:
            expected = "ToolHandlerResult or a tuple of (str, ContextOut)"
            raise TypeError(f"{cls.__name__} handler must return {expected}")

        if not isinstance(result, ToolHandlerResult):
            result = ToolHandlerResult(result[0], result[1])

        return result

    @classmethod
    def model_tool_json_schema(cls) -> dict:
        """
        Get the JSON schema to be used in the function definition.
        If a schema needs to be generated by Pydantic, it will be cached.
        """
        return (
            cls.model_tool_custom_json_schema
            or cls._generated_tool_schemas.get(cls)
            or cls._generated_tool_schemas.setdefault(
                cls, cls.model_tool_generate_json_schema()
            )
        )

    _generated_tool_schemas: ClassVar[dict[type[Self], dict]] = {}

    @classmethod
    def model_tool_pretty_definition(
        cls, api: Literal["chat.completions", "responses"]
    ) -> str:
        """
        For development only, get a pretty representation of the tool definition.
        """
        import json

        definition = json.dumps(cls.model_tool_definition(api), indent=4)

        # If available, use black, since it's nicer than default json indentation.
        try:
            import black  # pyright: ignore[reportMissingImports]

            definition = black.format_str(definition, mode=black.Mode()).strip()
        except:  # pragma: no cover
            pass  # pragma: no cover

        return f"{cls.__name__}({definition})"
