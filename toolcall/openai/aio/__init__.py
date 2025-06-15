from .._result import *
from .._call import *
from .._definition import *
from ._tool import *
from ._group import *

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import deprecated

    @deprecated("Renamed BaseFunctionToolModel")
    class LLMFunctionTool[In, Out](BaseFunctionToolModel[In, Out]): ...

    @deprecated("Renamed FunctionToolGroup")
    class LLMFunctionToolGroup[In, Out](FunctionToolGroup[In, Out]): ...
else:
    LLMFunctionTool = BaseFunctionToolModel
    LLMFunctionToolGroup = FunctionToolGroup
