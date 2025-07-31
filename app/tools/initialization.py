"""Initialize and register all mathematical tools."""

from ..core.logging import get_logger
from .analysis_tool import AnalysisTool
from .integral_tool import IntegralTool
from .plot_tool import PlotTool
from .registry import ToolRegistry, tool_registry

logger = get_logger(__name__)


def initialize_tools() -> ToolRegistry:
    """
    Initialize and register all mathematical tools.
    
    Returns:
        ToolRegistry: Configured tool registry with all tools registered
    """
    logger.info("Initializing mathematical tools")
    
    try:
        # Create tool instances
        integral_tool = IntegralTool()
        plot_tool = PlotTool()
        analysis_tool = AnalysisTool()
        
        # Register integral tool
        tool_registry.register_tool(
            integral_tool,
            categories=["calculus", "integration", "mathematics"],
            tags=[
                "integral", "definite", "indefinite", "symbolic", "numerical",
                "calculus", "mathematics", "area", "antiderivative", "integration"
            ]
        )
        
        # Register plot tool
        tool_registry.register_tool(
            plot_tool,
            categories=["visualization", "plotting", "graphics"],
            tags=[
                "plot", "graph", "visualization", "area", "curve", "function",
                "matplotlib", "plotly", "interactive", "chart", "diagram"
            ]
        )
        
        # Register analysis tool
        tool_registry.register_tool(
            analysis_tool,
            categories=["calculus", "analysis", "mathematics"],
            tags=[
                "derivative", "limit", "critical", "asymptote", "continuity",
                "monotonicity", "concavity", "taylor", "analysis", "calculus"
            ]
        )
        
        logger.info(f"Successfully registered {len(tool_registry)} mathematical tools")
        logger.info(f"Registry statistics: {tool_registry.get_registry_stats()}")
        
        return tool_registry
        
    except Exception as e:
        logger.error(f"Failed to initialize tools: {e}", exc_info=True)
        raise


def get_tool_registry() -> ToolRegistry:
    """
    Get the global tool registry instance.
    
    Returns:
        ToolRegistry: Global tool registry
    """
    return tool_registry


def search_tools_for_query(query: str, limit: int = 5) -> list:
    """
    Search for tools relevant to a query.
    
    Args:
        query: Search query describing desired functionality
        limit: Maximum number of results
    
    Returns:
        list: List of relevant tools with scores
    """
    return tool_registry.search_tools(query, limit)


def get_tool_recommendations(context: dict, limit: int = 3) -> list:
    """
    Get tool recommendations based on context.
    
    Args:
        context: Context information
        limit: Maximum recommendations
    
    Returns:
        list: Recommended tools with explanations
    """
    return tool_registry.get_tool_recommendations(context, limit)


# Initialize tools when module is imported
try:
    initialize_tools()
    logger.info("Mathematical tools initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize tools on import: {e}")
    # Don't raise here to allow the module to load, tools can be initialized later
