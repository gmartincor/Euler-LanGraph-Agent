"""Mathematical tools for the ReAct Agent."""

from .base import BaseTool, ToolInput, ToolOutput
from .registry import ToolRegistry, tool_registry
from .initialization import initialize_tools, get_tool_registry

# Import tools individually for explicit access
try:
    from .integral_tool import IntegralTool
    from .plot_tool import PlotTool
    from .analysis_tool import AnalysisTool
except ImportError as e:
    # Tools will be available after dependencies are installed
    pass

__all__ = [
    # Base classes
    "BaseTool",
    "ToolInput", 
    "ToolOutput",
    # Registry
    "ToolRegistry",
    "tool_registry",
    # Initialization
    "initialize_tools",
    "get_tool_registry",
    # Tools (when available)
    "IntegralTool",
    "PlotTool",
    "AnalysisTool",
]
