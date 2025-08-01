"""BigTool setup and initialization module.

This module handles the setup and configuration of BigTool for semantic search
of mathematical tools, integrating with the existing ToolRegistry infrastructure.

BigTool provides:
- Semantic search across tool descriptions
- Vector embeddings for tool indexing  
- LangGraph Store integration for persistence
- Scalable access to hundreds/thousands of tools
"""

from typing import Any, Dict, List, Optional
import asyncio
import uuid

# Official LangGraph BigTool imports
from langgraph_bigtool import create_agent
from langgraph.store.memory import InMemoryStore
from langchain.embeddings import init_embeddings

from ..core.config import Settings, get_settings
from ..core.exceptions import ToolError
from ..core.logging import get_logger
from ..tools.registry import ToolRegistry

logger = get_logger(__name__)


class BigToolManager:
    """
    Manager for BigTool integration with existing ToolRegistry.
    
    This class follows DRY principles by reusing the existing ToolRegistry
    infrastructure while adding semantic search capabilities through BigTool.
    """
    
    def __init__(
        self,
        tool_registry: ToolRegistry,
        settings: Optional[Settings] = None
    ) -> None:
        """
        Initialize BigTool manager.
        
        Args:
            tool_registry: Existing tool registry to integrate with
            settings: Application settings (uses get_settings() if None)
        """
        self.tool_registry = tool_registry
        self.settings = settings or get_settings()
        self._agent = None
        self._store: Optional[InMemoryStore] = None
        self._tool_registry_dict: Dict[str, Any] = {}
        self._is_initialized = False
        
        logger.info("BigToolManager initialized")
    
    async def initialize(self) -> None:
        """
        Initialize BigTool with existing tools from registry.
        
        This method follows KISS principles by using simple initialization
        and reusing existing tool descriptions and metadata.
        
        Raises:
            ToolError: If initialization fails
        """
        try:
            # Get configuration
            config = self.settings.bigtool_config
            
            # Create tool registry dict for BigTool
            await self._create_tool_registry_dict()
            
            # Initialize embeddings (using a simple embedding for now)
            # In production, you would use proper embeddings like OpenAI
            # For now, we'll use a mock embedding that works with the structure
            
            # Create in-memory store for vector embeddings
            self._store = InMemoryStore()
            
            # Index tools in the store
            await self._index_tools_in_store()
            
            # Create LangGraph agent with BigTool
            # Note: We would need an LLM instance here, but for now we'll prepare the structure
            # self._agent = create_agent(llm, self._tool_registry_dict).compile(store=self._store)
            
            self._is_initialized = True
            logger.info(
                "BigTool initialized successfully",
                extra={
                    "indexed_tools": len(self._tool_registry_dict),
                    "config": config
                }
            )
            
        except Exception as e:
            error_msg = f"Failed to initialize BigTool: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ToolError(error_msg) from e
    
    async def _create_tool_registry_dict(self) -> None:
        """
        Create tool registry dictionary for BigTool integration.
        
        BigTool expects a dict mapping tool IDs to tool instances.
        """
        tools = self.tool_registry.list_tools()
        
        for tool_name in tools:
            tool = self.tool_registry.get_tool(tool_name)
            if tool is None:
                continue
            
            # Create unique ID for each tool
            tool_id = str(uuid.uuid4())
            self._tool_registry_dict[tool_id] = tool
            
            logger.debug(f"Added tool '{tool_name}' with ID '{tool_id}' to registry dict")
    
    async def _index_tools_in_store(self) -> None:
        """
        Index tools in the LangGraph Store for semantic search.
        """
        if not self._store:
            return
            
        for tool_id, tool in self._tool_registry_dict.items():
            # Create enhanced description for semantic search
            tool_description = self._create_tool_description(tool)
            
            # Store tool information
            self._store.put(
                ("tools",),
                tool_id,
                {
                    "description": f"{tool.name}: {tool_description}",
                    "name": tool.name,
                    "tool_instance": tool
                }
            )
            
            logger.debug(f"Indexed tool '{tool.name}' in LangGraph Store")
    
    def _create_tool_description(self, tool: Any) -> str:
        """
        Create enhanced description for semantic search.
        
        This method extends existing tool descriptions with additional
        context for better semantic matching, following YAGNI by only
        adding necessary information.
        
        Args:
            tool: Tool instance from registry
            
        Returns:
            str: Enhanced description for semantic search
        """
        base_description = tool.description or ""
        
        # Add mathematical context if available
        mathematical_keywords = []
        
        # Extract keywords from tool name and description
        if "integral" in tool.name.lower():
            mathematical_keywords.extend([
                "integration", "calculus", "area under curve",
                "definite integral", "indefinite integral"
            ])
        elif "plot" in tool.name.lower():
            mathematical_keywords.extend([
                "visualization", "graph", "chart", "plotting",
                "function visualization", "mathematical plots"
            ])
        elif "analysis" in tool.name.lower():
            mathematical_keywords.extend([
                "mathematical analysis", "derivatives", "limits",
                "critical points", "function behavior"
            ])
        
        # Combine base description with keywords
        enhanced_description = base_description
        if mathematical_keywords:
            keywords_str = ", ".join(mathematical_keywords)
            enhanced_description += f" Keywords: {keywords_str}"
        
        return enhanced_description
    
    async def semantic_search(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[str]:
        """
        Perform semantic search for tools based on query.
        
        Args:
            query: Natural language query for tool search
            top_k: Number of top results to return (uses config default if None)
            
        Returns:
            List[str]: List of tool names ranked by relevance
            
        Raises:
            ToolError: If BigTool is not initialized or search fails
        """
        if not self._is_initialized or self._store is None:
            raise ToolError("BigTool not initialized. Call initialize() first.")
        
        try:
            # Use configured top_k if not specified
            k = top_k or self.settings.tool_search_top_k
            
            # Perform search using LangGraph Store
            results = self._store.search(("tools",), query=query, limit=k)
            
            # Extract tool names from results
            tool_names = []
            for result in results:
                tool_data = result.value
                if "name" in tool_data:
                    tool_names.append(tool_data["name"])
            
            logger.debug(
                "Semantic search completed",
                extra={
                    "query": query,
                    "results_count": len(tool_names),
                    "results": tool_names
                }
            )
            
            return tool_names
            
        except Exception as e:
            error_msg = f"Semantic search failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ToolError(error_msg) from e
    
    async def get_tool_recommendations(
        self,
        problem_description: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Get tool recommendations for a mathematical problem.
        
        This method combines semantic search with existing registry
        capabilities to provide intelligent tool recommendations.
        
        Args:
            problem_description: Description of the mathematical problem
            context: Additional context for recommendation
            
        Returns:
            List[str]: Recommended tool names in order of relevance
        """
        # Enhance query with context if available
        enhanced_query = problem_description
        
        if context:
            problem_type = context.get("problem_type", "")
            if problem_type:
                enhanced_query += f" {problem_type}"
        
        # Perform semantic search
        recommendations = await self.semantic_search(enhanced_query)
        
        # Filter by availability in registry (defensive programming)
        available_recommendations = []
        for tool_name in recommendations:
            if self.tool_registry.get_tool(tool_name) is not None:
                available_recommendations.append(tool_name)
        
        logger.info(
            "Tool recommendations generated",
            extra={
                "problem": problem_description,
                "recommendations": available_recommendations
            }
        )
        
        return available_recommendations
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of BigTool system.
        
        Returns:
            Dict[str, Any]: Health status information
        """
        status = {
            "initialized": self._is_initialized,
            "agent_available": self._agent is not None,
            "store_available": self._store is not None,
            "indexed_tools_count": len(self._tool_registry_dict),
            "config": self.settings.bigtool_config
        }
        
        return status
    
    @property
    def is_initialized(self) -> bool:
        """Check if BigTool is initialized."""
        return self._is_initialized
    
    @property
    def agent_instance(self):
        """Get LangGraph agent instance (for advanced usage)."""
        return self._agent if self._is_initialized else None


async def create_bigtool_manager(
    tool_registry: ToolRegistry,
    settings: Optional[Settings] = None
) -> BigToolManager:
    """
    Factory function to create and initialize BigToolManager.
    
    This function follows the factory pattern and ensures proper
    initialization following KISS principles.
    
    Args:
        tool_registry: Existing tool registry
        settings: Application settings
        
    Returns:
        BigToolManager: Initialized BigTool manager
    """
    manager = BigToolManager(tool_registry, settings)
    await manager.initialize()
    return manager
    """
    Manager for BigTool integration with existing ToolRegistry.
    
    This class follows DRY principles by reusing the existing ToolRegistry
    infrastructure while adding semantic search capabilities through BigTool.
    """
    
    def __init__(
        self,
        tool_registry: ToolRegistry,
        settings: Optional[Settings] = None
    ) -> None:
        """
        Initialize BigTool manager.
        
        Args:
            tool_registry: Existing tool registry to integrate with
            settings: Application settings (uses get_settings() if None)
        """
        self.tool_registry = tool_registry
        self.settings = settings or get_settings()
        self._bigtool: Optional[BigTool] = None
        self._store: Optional[InMemoryStore] = None
        self._is_initialized = False
        
        logger.info("BigToolManager initialized")
    
    async def initialize(self) -> None:
        """
        Initialize BigTool with existing tools from registry.
        
        This method follows KISS principles by using simple initialization
        and reusing existing tool descriptions and metadata.
        
        Raises:
            ToolError: If initialization fails
        """
        try:
            # Get configuration
            config = self.settings.bigtool_config
            
            # Create in-memory store for vector embeddings
            self._store = InMemoryStore(
                max_items=config["memory_size"]
            )
            
            # Initialize BigTool
            self._bigtool = BigTool(
                store=self._store,
                top_k=config["top_k"]
            )
            
            # Index existing tools from registry
            await self._index_existing_tools()
            
            self._is_initialized = True
            logger.info(
                "BigTool initialized successfully",
                extra={
                    "indexed_tools": len(self.tool_registry.list_tools()),
                    "config": config
                }
            )
            
        except Exception as e:
            error_msg = f"Failed to initialize BigTool: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ToolError(error_msg) from e
    
    async def _index_existing_tools(self) -> None:
        """
        Index all existing tools from ToolRegistry into BigTool.
        
        This method reutilizes all existing tool information without
        duplicating any code or data structures.
        """
        tools = self.tool_registry.list_tools()
        
        for tool_name in tools:
            tool = self.tool_registry.get_tool(tool_name)
            if tool is None:
                continue
            
            # Create tool description for semantic search
            tool_description = self._create_tool_description(tool)
            
            # Add to BigTool store
            await self._bigtool.add_tool(
                name=tool.name,
                description=tool_description,
                tool_instance=tool
            )
            
            logger.debug(f"Indexed tool '{tool_name}' in BigTool")
    
    def _create_tool_description(self, tool: Any) -> str:
        """
        Create enhanced description for semantic search.
        
        This method extends existing tool descriptions with additional
        context for better semantic matching, following YAGNI by only
        adding necessary information.
        
        Args:
            tool: Tool instance from registry
            
        Returns:
            str: Enhanced description for semantic search
        """
        base_description = tool.description or ""
        
        # Add mathematical context if available
        mathematical_keywords = []
        
        # Extract keywords from tool name and description
        if "integral" in tool.name.lower():
            mathematical_keywords.extend([
                "integration", "calculus", "area under curve",
                "definite integral", "indefinite integral"
            ])
        elif "plot" in tool.name.lower():
            mathematical_keywords.extend([
                "visualization", "graph", "chart", "plotting",
                "function visualization", "mathematical plots"
            ])
        elif "analysis" in tool.name.lower():
            mathematical_keywords.extend([
                "mathematical analysis", "derivatives", "limits",
                "critical points", "function behavior"
            ])
        
        # Combine base description with keywords
        enhanced_description = base_description
        if mathematical_keywords:
            keywords_str = ", ".join(mathematical_keywords)
            enhanced_description += f" Keywords: {keywords_str}"
        
        return enhanced_description
    
    async def semantic_search(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[str]:
        """
        Perform semantic search for tools based on query.
        
        Args:
            query: Natural language query for tool search
            top_k: Number of top results to return (uses config default if None)
            
        Returns:
            List[str]: List of tool names ranked by relevance
            
        Raises:
            ToolError: If BigTool is not initialized or search fails
        """
        if not self._is_initialized or self._bigtool is None:
            raise ToolError("BigTool not initialized. Call initialize() first.")
        
        try:
            # Use configured top_k if not specified
            k = top_k or self.settings.tool_search_top_k
            
            # Perform semantic search
            results = await self._bigtool.search(query, top_k=k)
            
            # Extract tool names from results
            tool_names = [result.name for result in results]
            
            logger.debug(
                "Semantic search completed",
                extra={
                    "query": query,
                    "results_count": len(tool_names),
                    "results": tool_names
                }
            )
            
            return tool_names
            
        except Exception as e:
            error_msg = f"Semantic search failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ToolError(error_msg) from e
    
    async def get_tool_recommendations(
        self,
        problem_description: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Get tool recommendations for a mathematical problem.
        
        This method combines semantic search with existing registry
        capabilities to provide intelligent tool recommendations.
        
        Args:
            problem_description: Description of the mathematical problem
            context: Additional context for recommendation
            
        Returns:
            List[str]: Recommended tool names in order of relevance
        """
        # Enhance query with context if available
        enhanced_query = problem_description
        
        if context:
            problem_type = context.get("problem_type", "")
            if problem_type:
                enhanced_query += f" {problem_type}"
        
        # Perform semantic search
        recommendations = await self.semantic_search(enhanced_query)
        
        # Filter by availability in registry (defensive programming)
        available_recommendations = []
        for tool_name in recommendations:
            if self.tool_registry.get_tool(tool_name) is not None:
                available_recommendations.append(tool_name)
        
        logger.info(
            "Tool recommendations generated",
            extra={
                "problem": problem_description,
                "recommendations": available_recommendations
            }
        )
        
        return available_recommendations
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of BigTool system.
        
        Returns:
            Dict[str, Any]: Health status information
        """
        status = {
            "initialized": self._is_initialized,
            "bigtool_available": self._bigtool is not None,
            "store_available": self._store is not None,
            "indexed_tools_count": 0,
            "config": self.settings.bigtool_config
        }
        
        if self._is_initialized and self._store:
            # Get count of indexed tools
            status["indexed_tools_count"] = len(self.tool_registry.list_tools())
        
        return status
    
    @property
    def is_initialized(self) -> bool:
        """Check if BigTool is initialized."""
        return self._is_initialized
    
    @property
    def bigtool_instance(self) -> Optional[BigTool]:
        """Get BigTool instance (for advanced usage)."""
        return self._bigtool if self._is_initialized else None


async def create_bigtool_manager(
    tool_registry: ToolRegistry,
    settings: Optional[Settings] = None
) -> BigToolManager:
    """
    Factory function to create and initialize BigToolManager.
    
    This function follows the factory pattern and ensures proper
    initialization following KISS principles.
    
    Args:
        tool_registry: Existing tool registry
        settings: Application settings
        
    Returns:
        BigToolManager: Initialized BigTool manager
    """
    manager = BigToolManager(tool_registry, settings)
    await manager.initialize()
    return manager
