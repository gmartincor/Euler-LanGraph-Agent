"""Initialize and register all mathematical tools - Simplified for BigTool integration."""

from ..core.logging import get_logger
from .analysis_tool import AnalysisTool
from .integral_tool import IntegralTool
from .plot_tool import PlotTool
from .registry import ToolRegistry, tool_registry

logger = get_logger(__name__)


def initialize_tools() -> ToolRegistry:
    """
    Initialize and register all mathematical tools.
    
    Simplified for BigTool integration - tools are registered for semantic
    filtering, not for direct LLM exposure.
    
    Returns:
        ToolRegistry: Configured tool registry for BigTool semantic filtering
    """
    logger.info("Initializing mathematical tools for BigTool semantic filtering")
    
    try:
        # Create tool instances
        integral_tool = IntegralTool()
        plot_tool = PlotTool()
        analysis_tool = AnalysisTool()
        
        # Register tools with enhanced semantic context for BigTool
        tool_registry.register_tool(
            integral_tool,
            categories=["calculus", "integration", "mathematics"],
            tags=[
                "integral", "definite", "indefinite", "symbolic", "numerical",
                "calculus", "mathematics", "area", "antiderivative", "integration",
                "compute", "calculate", "evaluate"  # Enhanced for semantic search
            ]
        )
        
        tool_registry.register_tool(
            plot_tool,
            categories=["visualization", "plotting", "graphics"],
            tags=[
                "plot", "graph", "visualization", "area", "curve", "function",
                "matplotlib", "plotly", "interactive", "chart", "diagram",
                "show", "display", "visualize", "draw"  # Enhanced for semantic search
            ]
        )
        
        tool_registry.register_tool(
            analysis_tool,
            categories=["calculus", "analysis", "mathematics"],
            tags=[
                "derivative", "limit", "critical", "asymptote", "continuity",
                "monotonicity", "concavity", "taylor", "analysis", "calculus",
                "analyze", "examine", "study", "investigate"  # Enhanced for semantic search
            ]
        )
        
        logger.info(f"Successfully registered {len(tool_registry)} mathematical tools for BigTool")
        logger.info(f"Registry statistics: {tool_registry.get_registry_stats()}")
        
        return tool_registry
        
    except Exception as e:
        logger.error(f"Failed to initialize tools: {e}", exc_info=True)
        raise


def get_tool_registry() -> ToolRegistry:
    """
    Get the global tool registry instance for BigTool integration.
    
    Returns:
        ToolRegistry: Global tool registry configured for semantic filtering
    """
    return tool_registry


# REMOVED: Functions that expose tools directly to LLM
# - search_tools_for_query() -> Use BigTool semantic filtering instead
# - get_tool_recommendations() -> Use BigTool semantic filtering instead


# Initialize tools when module is imported
try:
    initialize_tools()
    logger.info("Mathematical tools initialized successfully for BigTool")
except Exception as e:
    logger.error(f"Failed to initialize tools on import: {e}")
    # Don't raise here to allow the module to load, tools can be initialized later
