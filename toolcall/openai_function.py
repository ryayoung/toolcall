from __future__ import annotations
import json
import inspect
import textwrap
import docstring_parser
from typing import (
    get_type_hints,
    Callable,
    Any,
    TypeVar,
    Generic,
    cast,
    overload,
    ClassVar,
    ParamSpec,
    TypedDict,
    Literal,
    NotRequired,
    Dict,
)
from abc import ABC
import pydantic
from pydantic import (
    BaseModel,
    create_model,
)

P = ParamSpec("P")
R = TypeVar("R", covariant=True)


class FunctionCallDict(TypedDict):
    name: str
    arguments: str


class FunctionToolCallDict(TypedDict):
    id: str
    type: Literal["function"]
    function: FunctionCallDict


class FunctionCallModel(BaseModel):
    name: str
    arguments: str


class FunctionToolCallModel(BaseModel):
    id: str
    type: Literal["function"]
    function: FunctionCallModel


class ToolCallResult(TypedDict):
    role: Literal["tool"]
    content: str
    tool_call_id: str


class FunctionDefinition(TypedDict):
    name: str
    description: NotRequired[str]
    parameters: dict[str, Any]


class FunctionTool(TypedDict):
    type: Literal["function"]
    function: FunctionDefinition


class OpenaiFunction(BaseModel, Generic[P, R], ABC):
    schema: ClassVar[FunctionTool]

    def __init__(self, *args: P.args, **kwargs: P.kwargs):
        raise NotImplementedError("OpenaiFunction is an abstract class.")

    def __call__(self) -> R:
        ...

    def execute(self) -> R:
        ...

    @property
    def function(self) -> Callable[P, R]:
        ...

    @classmethod
    def run_tool_call(
        cls,
        tool_call: dict | FunctionToolCallDict | FunctionToolCallModel,
        error_handler: bool | Callable[[Exception], str] = False,
    ) -> ToolCallResult:
        ...

    def __getattr__(self, name: str) -> Any:
        ...


@overload
def openai_function(
    __func: Callable[P, R],
    /, 
    *, 
    annotate_null: bool = False,
    schema_properties: dict | None = None,
    **kwargs,
) -> type[OpenaiFunction[P, R]]:
    ...


@overload
def openai_function(
    __func: None = None, 
    /, 
    *, 
    annotate_null: bool = False,
    schema_properties: dict | None = None,
    **kwargs,
) -> Callable[[Callable[P, R]], type[OpenaiFunction[P, R]]]:
    ...


def openai_function(
    __func: None | Callable[P, R] = None, 
    /, 
    *, 
    annotate_null: bool = False,
    schema_properties: dict | None = None,
    **kwargs,
) -> (
    type[OpenaiFunction[P, R]] | Callable[[Callable[P, R]], type[OpenaiFunction[P, R]]]
):
    """
    Decorator for creating OpenAI functions.

    The returned class is created by passing your function's parameters to `pydantic.create_model()`.

    So the following code ...

        @openai_function
        def add(first: int, second: int = Field(default=0)):
            ...

    ... defines a model of this structure:

        class add(BaseModel):
            first: int
            second: int = Field(default=0)

    There are two ways the openai function differs from traditional Pydantic model behavior.
        - Parameter type hints are NOT required. The resulting model will use `Any` for params without type hints.
        - If your function can take positional arguments, then your model can also take positional arguments.

    Parameters
    ----------
    **kwargs
        Keyword arguments passed to `pydantic.create_model()`, before the attributes.
    """

    def decorator(fn):
        function_name = fn.__name__

        # Param names and default values obtained from `inspect.signature()`.
        # `get_type_hints()` MUST be used for annotations since it evaluates forward references.

        inspected_params = list(inspect.signature(fn).parameters.values())
        param_names = [p.name for p in inspected_params]
        param_types = get_type_hints(fn)

        blacklisted_param_names = {"schema", "function", "execute", "run_tool_call"}
        for name in blacklisted_param_names:
            if name in param_names:
                raise ValueError(
                    f"Parameter name '{name}' is reserved for OpenAI functions."
                )

        param_defaults = {
            param.name: param.default if param.default is not param.empty else ...
            for param in inspected_params
        }

        model_fields: dict[str, Any] = {
            name: (param_types.get(name, Any), param_defaults[name])
            for name in param_names
        }

        # Passing model fields last, so kwargs can't interfere
        Base = create_model(function_name, **{**kwargs, **model_fields})
        func_schema = get_schema(fn, Base, annotate_null, schema_properties)
        tool_schema = FunctionTool(type="function", function=func_schema)

        # This might seem a bit 'extra', but it's a great convenience to see your resulting
        # function schema neatly printed, as this is something you'll certainly want to do.
        class BaseMeta(type(Base)):
            def __repr__(self):
                return f"OpenaiFunction({json.dumps(tool_schema, indent=4)})"

        class Model(Base, metaclass=BaseMeta):
            schema: ClassVar[FunctionTool] = tool_schema

            def __init__(self, *args, **kwargs):
                # Pydantic doesn't allow positionals, so we move args to kwargs.
                for i, arg in enumerate(args):
                    name = param_names[i]
                    if name not in kwargs:
                        kwargs[name] = arg

                super().__init__(**kwargs)

            def execute(self):
                ...  # Implemented conditionally, below

            def __call__(self):
                return self.execute()

            @property
            def function(self):
                return fn

            @classmethod
            def run_tool_call(
                cls, tool_call, error_handler: bool | Callable[[Exception], str] = False
            ):
                return handle_tool_call(cls, tool_call, error_handler)


        # If the function has positional-only params, `__call__` should handle them.
        # Notice we aren't adjusting `__init__`, because the whole point of this class is
        # to support arguments coming from OpenAI (JSON) which will be keyword arguments anyway.
        positional_only_params = [
            p.name
            for p in inspected_params
            if p.kind == inspect.Parameter.POSITIONAL_ONLY
        ]
        if positional_only_params:

            def execute(self):
                kwargs = self.model_dump()
                args = [kwargs.pop(name) for name in positional_only_params]
                return fn(*args, **kwargs)

        else:

            def execute(self):
                return fn(**self.model_dump())

        Model.execute = execute
        Model.__name__ = function_name
        result = cast(type[OpenaiFunction[P, R]], Model)
        return result

    if __func is None:
        return decorator

    return decorator(__func)


def get_schema(
    fn: Callable, 
    model: type[BaseModel], 
    annotate_null: bool, 
    schema_properties: dict | None,
) -> FunctionDefinition:
    parameters = model.model_json_schema()
    parameters = order_properties(parameters, ["type", "properties", "required"])
    parameters = remove_key_recursive(parameters, "title")

    if not annotate_null:
        for prop in parameters["properties"].values():
            remove_null_annotations(prop)

    if schema_properties is not None:
        for k, v in schema_properties.items():
            parameters["properties"][k] = v

    schema: FunctionDefinition = {
        "name": fn.__name__,
        "parameters": parameters,
    }

    raw_docstring = fn.__doc__
    if not raw_docstring:
        return schema

    docstring = textwrap.dedent(raw_docstring).strip()

    doc = docstring_parser.parse(docstring)

    # Add descriptions to schema params
    if doc.params:
        descriptions = {
            param.arg_name: param.description
            for param in doc.params
            if param.description
            and param.arg_name
            and param.arg_name in parameters["properties"]
        }
        for name, description in descriptions.items():
            prop_obj = schema["parameters"]["properties"][name]
            if "description" not in prop_obj:
                prop_obj["description"] = description

    # Re-order props within each parameter
    for key, prop in parameters["properties"].items():
        parameters["properties"][key] = order_properties(
            prop, start=["type", "anyOf", "description"], end=["default"]
        )

    docstring_without_params = remove_params_from_docstring(docstring).strip()
    if not docstring_without_params:
        return schema

    schema["description"] = docstring_without_params

    # Re-order keys so description is first. Better readability.
    return order_properties(schema, ["name", "description", "parameters"])


T = TypeVar("T", bound=Dict | TypedDict)


def order_properties(items: T, start: list[str], end: list[str] | None = None) -> T:
    start_dict = {k: items[k] for k in start if k in items}

    if end:
        end_dict = {k: items[k] for k in end if k in items}
        rest = {k: items[k] for k in items if k not in start and k not in end}
        result = {**start_dict, **rest, **end_dict}

    else:
        rest = {k: items[k] for k in items if k not in start}
        result = {**start_dict, **rest}

    return cast(T, result)


def remove_null_annotations(prop: dict) -> None:
    if prop.get("default", 69) is None:
        del prop["default"]

    if "anyOf" in prop:
        prop["anyOf"] = [
            tp for tp in prop["anyOf"]
            if not (isinstance(tp, dict) and tp.get("type") == "null")
        ]
        if len(prop["anyOf"]) == 1:
            prop["type"] = prop["anyOf"][0]["type"]
            del prop["anyOf"]
        

def remove_key_recursive(item, key_to_remove) -> Any:
    if isinstance(item, dict):
        return {
            key: remove_key_recursive(value, key_to_remove)
            for key, value in item.items()
            if key != key_to_remove
        }

    if isinstance(item, (list, tuple)):
        return [remove_key_recursive(value, key_to_remove) for value in item]

    return item


def remove_params_from_docstring(docstring: str) -> str:
    sections = {
        "Parameters",
        "Returns",
        "Examples",
        "Raises",
        "Notes",
        "References",
        "Yields",
    }
    sections |= {f"{s}:" for s in sections}
    lines = docstring.split("\n")
    params_start = None
    params_end = len(lines)

    for i, line in enumerate(lines):
        line = line.strip()
        if line == "Parameters":
            params_start = i
            continue

        if params_start is not None and line in sections:
            params_end = i
            break

    if params_start is None:
        return docstring

    new_lines = lines[:params_start] + lines[params_end:]
    return "\n".join(new_lines)


def handle_tool_call(
    openai_func: OpenaiFunction,
    tool_call: dict | FunctionToolCallDict | FunctionToolCallModel,
    error_handler: bool | Callable[[Exception], str] = False,
) -> ToolCallResult:
    id, _, arguments = unpack_tool_call(tool_call)

    if not error_handler:
        parsed = json.loads(arguments)
        result = openai_func(**parsed).execute()

    else:
        if not callable(error_handler):
            error_handler = default_error_handler

        try:
            parsed = json.loads(arguments)
            result = openai_func(**parsed).execute()
        except Exception as e:
            result = error_handler(e)

    return {
        "role": "tool",
        "tool_call_id": id,
        "content": str(result),
    }


def default_error_handler(e: Exception) -> str:
    if not isinstance(e, pydantic.ValidationError):
        return f"{type(e).__name__}: {e}"

    s = "Validation failed for the following parameters\n\n"
    for error in e.errors():
        s += f"'{error['loc'][0]}':\n"
        s += f"  Input: {repr(error['input'])}\n"
        s += f"  Error: {error['msg']}\n\n"

    return s.strip()


class OpenaiToolGroup(dict[str, type[OpenaiFunction]]):
    def __init__(self):
        return dict.__init__({})

    @property
    def tools(self) -> list[FunctionTool]:
        return [
            func.schema for func in self.values()
        ]

    def add(self, function: Callable | type[OpenaiFunction]):
        if not isinstance(function, type(BaseModel)):
            function = openai_function(function)

        key = function.schema["function"]["name"]
        self[key] = cast(type[OpenaiFunction], function)

    def run_tool_call(
        self,
        tool_call: dict | FunctionToolCallDict | FunctionToolCallModel,
        error_handler: bool | Callable[[Exception], str] = False,
    ) -> ToolCallResult:
        if isinstance(tool_call, dict):
            name = tool_call["function"]["name"]
        else:
            name = tool_call.function.name

        return self[name].run_tool_call(tool_call, error_handler)

    def __repr__(self):
        return f"OpenaiToolGroup({json.dumps(self.tools, indent=4)})"


def unpack_tool_call(
    tool_call: dict | FunctionToolCallDict | FunctionToolCallModel
) -> tuple[str, str, str]:
    "returns id, name, arguments"
    if isinstance(tool_call, dict):
        func = tool_call["function"]
        return tool_call["id"], func["name"], func["arguments"]

    func = tool_call.function
    return tool_call.id, func.name, func.arguments



def openai_tool_group(
    functions: list[Callable | type[OpenaiFunction]] | None = None,
) -> OpenaiToolGroup:
    group = OpenaiToolGroup()

    for func in functions or []:
        group.add(func)

    return group
