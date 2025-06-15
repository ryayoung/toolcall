def test_config_aio():
    from toolcall.openai.aio import BaseFunctionToolModel

    class RegularTool(BaseFunctionToolModel[None, None]):
        """A docstring"""

        pass

    assert RegularTool.model_tool_name() == "RegularTool"
    schema = RegularTool.model_tool_json_schema()
    assert schema["properties"] == {}
    definition = RegularTool.model_tool_standard_definition()
    assert definition.description == "A docstring"

    class BaseTool(BaseFunctionToolModel[None, None]):
        model_tool_name_generator = lambda name: name.upper()

    assert BaseTool.model_tool_name() == "BASETOOL"

    class Tool(BaseTool):
        """A docstring"""

        model_tool_custom_name = "custom_tool_name"
        model_tool_custom_description = "custom description"
        model_tool_custom_json_schema = {"1": "1"}

    assert Tool.model_tool_name() == "custom_tool_name"
    assert Tool.model_tool_json_schema() == {"1": "1"}
    definition = Tool.model_tool_standard_definition()
    assert definition.description == "custom description"


def test_config_core():
    from toolcall.openai.core import BaseFunctionToolModel

    class RegularTool(BaseFunctionToolModel[None, None]):
        """A docstring"""

        pass

    assert RegularTool.model_tool_name() == "RegularTool"
    schema = RegularTool.model_tool_json_schema()
    assert schema["properties"] == {}
    definition = RegularTool.model_tool_standard_definition()
    assert definition.description == "A docstring"

    class BaseTool(BaseFunctionToolModel[None, None]):
        model_tool_name_generator = lambda name: name.upper()

    assert BaseTool.model_tool_name() == "BASETOOL"

    class Tool(BaseTool):
        """A docstring"""

        model_tool_custom_name = "custom_tool_name"
        model_tool_custom_description = "custom description"
        model_tool_custom_json_schema = {"1": "1"}

    assert Tool.model_tool_name() == "custom_tool_name"
    assert Tool.model_tool_json_schema() == {"1": "1"}
    definition = Tool.model_tool_standard_definition()
    assert definition.description == "custom description"
