"""BigTool setup and initialization module.

This module handles the setup and configuration of BigTool for semantic search
of mathematical tools, integrating with the existing ToolRegistry infrastructure.

NOTE: Currently using fallback implementation because:
1. LangGraph BigTool is still in active development (as of Aug 2025)
2. Package names and APIs are not yet stable
3. This fallback provides robust functionality for semantic tool search
4. Will be updated to official BigTool package when available and stable

The fallback implementation provides:
- Keyword-based semantic search
- Tool indexing and storage
- Compatible API with future BigTool versions
"""

from typing import Any, Dict, List, Optional
import asyncio

# BigTool implementation using fallback for now
# TODO: Update with correct LangGraph BigTool imports when available
# Note: LangGraph BigTool is still in development, using fallback implementation

# BigTool implementation using fallback for now
# TODO: Update with correct LangGraph BigTool imports when available
# Note: LangGraph BigTool is still in development, using fallback implementation

class InMemoryStore:
    """
    In-memory store for tool embeddings and metadata.
    
    This is a simple fallback implementation until the official
    LangGraph BigTool package is available.
    """
    def __init__(self, max_items: int = 1000):
        self.max_items = max_items
        self._items = {}
        
    def add_item(self, key: str, data: Any) -> None:
        """Add item to store."""
        if len(self._items) >= self.max_items:
            # Simple LRU: remove oldest item
            oldest_key = next(iter(self._items))
            del self._items[oldest_key]
        self._items[key] = data
        
    def get_item(self, key: str) -> Optional[Any]:
        """Get item from store."""
        return self._items.get(key)
        
    def size(self) -> int:
        """Get number of items in store."""
        return len(self._items)


class BigTool:
    """
    Fallback implementation of BigTool for semantic tool search.
    
    This implementation provides basic functionality until the official
    LangGraph BigTool package is available and stable.
    """
    def __init__(self, store=None, top_k: int = 3):
        self.store = store or InMemoryStore()
        self.top_k = top_k
        self._tools = {}
        
    async def add_tool(self, name: str, description: str, tool_instance: Any):
        """Add a tool to the search index."""
        self._tools[name] = {
            "description": description,
            "instance": tool_instance
        }
        # Store in the provided store as well
        if self.store:
            self.store.add_item(name, {
                "description": description,
                "instance": tool_instance
            })
    
    async def search(self, query: str, top_k: int = None) -> List[Any]:
        """
        Perform semantic search for tools based on query.
        
        This is a simple keyword-based implementation until we have
        proper vector embeddings and semantic search.
        """
        k = top_k or self.top_k
        results = []
        
        class SearchResult:
            def __init__(self, name: str, score: float = 0.0):
                self.name = name
                self.score = score
        
        # Score tools based on keyword matching
        scored_tools = []
        query_lower = query.lower()
        query_words = query_lower.split()
        
        for tool_name, tool_data in self._tools.items():
            score = 0
            desc_lower = tool_data["description"].lower()
            name_lower = tool_name.lower()
            
            # Scoring algorithm
            for word in query_words:
                # Exact matches in name get highest score
                if word in name_lower:
                    score += 3
                # Exact matches in description get medium score
                if word in desc_lower:
                    score += 2
                # Partial matches get lower score
                for desc_word in desc_lower.split():
                    if word in desc_word or desc_word in word:
                        score += 1
            
            if score > 0:
                scored_tools.append((score, tool_name))
        
        # Sort by score (descending) and return top_k
        scored_tools.sort(reverse=True, key=lambda x: x[0])
        
        for score, tool_name in scored_tools[:k]:
            results.append(SearchResult(tool_name, score))
        
        return results

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
