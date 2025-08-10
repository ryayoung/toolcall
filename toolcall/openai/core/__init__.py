# File generated from its async equivalent, toolcall/openai/aio/__init__.py
from .._common import *
from ._tool import *
from ._group import *

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import deprecated

    @deprecated("Renamed ToolGroup")
    class FunctionToolGroup[In, Out](ToolGroup[In, Out]): ...
else:
    FunctionToolGroup = ToolGroup
