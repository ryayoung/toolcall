# toolcall

[![PyPI](https://img.shields.io/pypi/v/toolcall)](https://pypi.org/project/toolcall/)
[![Tests](https://github.com/ryayoung/toolcall/actions/workflows/tests.yml/badge.svg)](https://github.com/ryayoung/toolcall/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/ryayoung/toolcall/branch/main/graph/badge.svg)](https://codecov.io/gh/ryayoung/toolcall)
[![License](https://img.shields.io/github/license/ryayoung/toolcall)](https://github.com/ryayoung/toolcall/blob/main/LICENSE)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/toolcall.svg)](https://pypi.python.org/pypi/toolcall/)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Pyright](https://img.shields.io/badge/type%20checker-pyright-blue)](https://github.com/microsoft/pyright)

```
pip install toolcall
```

The agentic *not*-framework for Python. This <200-line **micro**-library for
structured/agentic LLM communication provides utility types and primitives for
developers who prefer a more manual approach to building agentic behavior on
top of `pydantic` and `openai`, without sacrificing control or code clarity to
a framework.

<details><summary>More details: <i>Who is it for, and why?</i></summary>

---

For developers who prefer a more *manual* approach to LLM workflow
orchestration and context management - using `pydantic` for validation and
schema generation, an API client (`openai`) for its type-safe interface and
request handling, and rolling everything else on your own - `toolcall` is the
little abstraction you were going to eventually end up building anyway, as your
project scales in complexity.

`toolcall` is like a great desk chair: Simple, unexciting, and completely
unambiguous in runtime behavior, but something upon which your code can always
sit, no matter the use case, and whose sole purpose is to solve a handful of
basic problems that everyone has, and do so without compromise.

---

</details>



---

### *Should you use it?*

- You want a framework to handle agent/tool orchestration and control-flow for you:
    - ⛔ Do **not** use `toolcall`.
- You define, dispatch, and handle function tool calls yourself:
    - ✅ You **should** be using `toolcall`.


<br>

# Learn-by-example Documentation

Below is a set of end-to-end tool-calling and structured output workflows,
for every API type.

> Note: Every line of code below comes straight from `examples/` where it is
> tested before being injected into this readme.

---

<details><summary><b>Synchronous API</b></summary>

<br>

<details><summary><code>common.py</code> - Setup code used by all examples below</summary>{{examples['core.common']}}</details>

#### Chat Completions API

<details><summary>Structured Outputs</summary>{{examples['core.chat_output']}}</details>

<details><summary>Single Function Tool</summary>{{examples['core.chat_tool']}}</details>

<details><summary>Multiple Function Tools</summary>{{examples['core.chat_group']}}</details>

#### Responses API

<details><summary>Structured Outputs</summary>{{examples['core.resp_output']}}</details>

<details><summary>Single Function Tool</summary>{{examples['core.resp_tool']}}</details>

<details><summary>Multiple Function Tools</summary>{{examples['core.resp_group']}}</details>

</details>

---

<details><summary><b>Asynchronous (async/await) API</b></summary>

<br>

<details><summary><code>common.py</code> - Setup code used by all examples below</summary>{{examples['aio.common']}}</details>

#### Chat Completions API

<details><summary>Structured Outputs</summary>{{examples['aio.chat_output']}}</details>

<details><summary>Single Function Tool</summary>{{examples['aio.chat_tool']}}</details>

<details><summary>Multiple Function Tools</summary>{{examples['aio.chat_group']}}</details>

#### Responses API

<details><summary>Structured Outputs</summary>{{examples['aio.resp_output']}}</details>

<details><summary>Single Function Tool</summary>{{examples['aio.resp_tool']}}</details>

<details><summary>Multiple Function Tools</summary>{{examples['aio.resp_group']}}</details>

</details>

---

<br>

# API Reference - `toolcall.openai`

The entire API is mirrored across two namespaces with the same members and
naming conventions: `core` (regular) and `aio` (async/await). If your code is
async, import everything from `toolcall.openai.aio`. Otherwise use
`toolcall.openai.core`.


### `class BaseFunctionToolModel[ContextIn, ContextOut](BaseModel)`

A [Pydantic BaseModel](https://docs.pydantic.dev/latest/) that represents a data structure
an LLM should provide. Its `model_tool_handler()` method (if you implement it)
can be the home for your logic for handling a tool call to that model and
producing a text response.

#### API

- **Generic type arguments**: `[ContextIn, ContextOut]`
    
    If set, these apply a type constraint on the *additional* data your orchestration code
    must pass in to the tool handler, and that your handler must return back, respectively.

    Their purpose is to enable type safety when defining a **group** of tools
    (`FunctionToolGroup[ContextIn, ContextOut]`) and using the group to dynamically dispatch
    calls.

- **Members you _need_ to know about**

    **Create a definition of your model to send to the API**

    - `model_tool_definition(cls, api: "chat.completions" | "responses")`
        - Item to go in the `tools` array param to the OpenAI API.
    - `model_tool_format(cls, api: "chat.completions" | "responses")`
        - Structured Outputs format definition for the `response_format` and `text.format` params
          in the Chat Completions and Responses APIs respectively.

    **Handle a tool call that the LLM sent you, and produce a response.**

    - `async` `model_tool_handler(self, context: ContextIn) -> tuple[str, ContextOut]`
        - Your subclass should implement this to define how to respond when the LLM
          passes valid arguments to the tool. Your model instance - `self` - represents
          those valid arguments.
        - Takes a single argument, `context: ContextIn` of any arbitrary type you want.
          This allows your orchestration to inject additional data into the handler.
        - Returns **two** things:
            1. Result Content: A `str` to use in the response message to the LLM.
            2. Output Context: `ContextOut`: Arbitrary data to send back to your
               orchestration logic that initiated the tool call handling.
    - `async` `model_tool_run_tool_call(cls, call, context: ContextIn) -> ToolCallResult[ContextOut]`
        - Takes a tool call from any API type, applies Pydantic's parsing/validation,
          executes your handler, and wraps the result.

- **Config: Class-variables**

    Optional class configurations are set using class variables. (**Do not** declare type
    annotations when setting these.)

    - `model_tool_strict` : bool, default False
    - `model_tool_custom_name` : str or None, default None
    - `model_tool_name_generator` : ((str) => str) or None, default None
        - Function to generate a name based on the class name.
    - `model_tool_custom_description` : str or None, default None
    - `model_tool_custom_json_schema` : dict or None, default None

- **Other Utility Methods**

    - `model_tool_name(cls) -> str`
    - `model_tool_json_schema(cls) -> dict`
    - `model_tool_pretty_definition(cls) -> str`

---

### `class FunctionToolGroup[ContextIn, ContextOut](...)`

- Parent: `dict[str, type[BaseFunctionToolModel[ContextIn, ContextOut]]]`

A simple container that supports statically type-safe dynamic dispatch of tools.

Documentation coming soon. See the **Learn-by-example Documentation** near the top of
this page, in the dynamic tool call dispatch section. There are examples of how a
tool group is used.
