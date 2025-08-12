from typing import Any, Dict, List, Optional
import uuid
from abc import ABC, abstractmethod

from langraph.store.memory import InMemoryStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from ..core.config import Settings, get_settings
from ..core.exceptions import ToolError
from ..core.logging import get_logger
from ..tools.registry import ToolRegistry

logger = get_logger(__name__)


class SemanticFilter(ABC):
    """Abstract base for semantic filtering implementations."""
    
    @abstractmethod
    async def filter_tools_for_query(
        self, 
        query: str, 
        context: Optional[Dict] = None
    ) -> List[str]:
        """Filter tools based on semantic relevance to query."""
        pass


class BigToolSemanticFilter(SemanticFilter):
    """
    BigTool-based semantic filter for pre-LLM tool filtering.
    """
    
    def __init__(
        self,
        tool_registry: ToolRegistry,
        settings: Optional[Settings] = None
    ) -> None:
        """Initialize semantic filter with tool registry."""
        self.tool_registry = tool_registry
        self.settings = settings or get_settings()
        self._store: Optional[InMemoryStore] = None
        self._embeddings = None
        self._tool_registry_dict: Dict[str, Any] = {}
        self._is_initialized = False
        
        logger.info("BigTool semantic filter initialized")
    
    async def initialize(self) -> None:
        """Initialize semantic filtering components."""
        try:
            # Step 1: Initialize embeddings
            config = self.settings.bigtool_config
            self._embeddings = GoogleGenerativeAIEmbeddings(
                model=config['embedding_model'],
                google_api_key=config["api_key"]
            )
            
            # Step 2: Create semantic store
            embedding_dims = config.get("embedding_dimensions", 768)
            self._store = InMemoryStore(
                index={
                    "embed": self._embeddings,
                    "dims": embedding_dims,
                    "fields": ["description"],
                }
            )
            
            # Step 3: Index tools for semantic search
            await self._index_tools_for_semantic_search()
            
            self._is_initialized = True
            logger.info("BigTool semantic filter initialized successfully")
            
        except Exception as e:
            error_msg = f"Failed to initialize semantic filter: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ToolError(error_msg) from e
    
    async def filter_tools_for_query(
        self, 
        query: str, 
        context: Optional[Dict] = None
    ) -> List[str]:
        """
        MAIN SEMANTIC FILTERING METHOD
        
        This is the core functionality that BigTool should provide:
        filtering tools by semantic relevance BEFORE the LLM sees them.
        
        Args:
            query: User query
            context: Additional context for filtering
        
        Returns:
            List[str]: Only semantically relevant tool names
        """
        if not self._is_initialized or self._store is None:
            raise ToolError("Semantic filter not initialized. Call initialize() first.")
        
        try:
            # Semantic search configuration
            max_tools = context.get('max_tools', 3) if context else 3
            relevance_threshold = context.get('relevance_threshold', 0.7) if context else 0.7
            
            # Perform semantic search using BigTool's store
            results = self._store.search(("tools",), query=query, limit=max_tools * 2)
            
            # Filter by relevance threshold
            relevant_tools = []
            
            for result in results:
                if result.score >= relevance_threshold:
                    tool_data = result.value
                    if "description" in tool_data:
                        description = tool_data["description"]
                        if ":" in description:
                            tool_name = description.split(":")[0].strip()
                            # Verify tool exists in registry
                            if self.tool_registry.get_tool(tool_name) is not None:
                                relevant_tools.append(tool_name)
            
            # Limit to max_tools
            relevant_tools = relevant_tools[:max_tools]
            
            logger.info(
                f"Semantic filtering: '{query}' → {relevant_tools}",
                extra={"query": query, "filtered_tools": relevant_tools, "threshold": relevance_threshold}
            )
            
            return relevant_tools
            
        except Exception as e:
            error_msg = f"Semantic filtering failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ToolError(error_msg) from e
    
    async def _index_tools_for_semantic_search(self) -> None:
        """Index tools in semantic store for efficient search."""
        if not self._store:
            return
        
        tools = self.tool_registry.get_all_tools()
        
        for tool in tools:
            tool_id = str(uuid.uuid4())
            
            # Create enhanced description for semantic search
            enhanced_description = self._create_enhanced_description(tool)
            
            # Store in BigTool pattern
            self._store.put(
                ("tools",),
                tool_id,
                {
                    "description": f"{tool.name}: {enhanced_description}",
                }
            )
            
            logger.debug(f"Indexed tool '{tool.name}' for semantic search")
        
        logger.info(f"Indexed {len(tools)} tools for semantic search")
    
    def _create_enhanced_description(self, tool: Any) -> str:
        """Create enhanced description for better semantic matching."""
        base_description = getattr(tool, 'description', '') or ""
        
        # Add mathematical context keywords for better semantic search
        mathematical_keywords = []
        tool_name_lower = tool.name.lower()
        
        if "integral" in tool_name_lower or "integrate" in tool_name_lower:
            mathematical_keywords.extend([
                "integration", "calculus", "area under curve",
                "definite integral", "indefinite integral", "antiderivative"
            ])
        elif "plot" in tool_name_lower or "graph" in tool_name_lower:
            mathematical_keywords.extend([
                "visualization", "graph", "chart", "plotting",
                "function visualization", "mathematical plots", "area visualization"
            ])
        elif "analysis" in tool_name_lower or "analyze" in tool_name_lower:
            mathematical_keywords.extend([
                "mathematical analysis", "derivatives", "limits",
                "critical points", "function behavior"
            ])
        
        # Combine base description with semantic context
        enhanced_description = base_description
        if mathematical_keywords:
            keywords_str = ", ".join(mathematical_keywords)
            enhanced_description += f" Mathematical context: {keywords_str}"
        
        return enhanced_description


# Factory function for clean interface
async def create_semantic_filter(
    tool_registry: ToolRegistry,
    settings: Optional[Settings] = None
) -> BigToolSemanticFilter:
    """Create and initialize semantic filter."""
    filter_instance = BigToolSemanticFilter(tool_registry, settings)
    await filter_instance.initialize()
    return filter_instance
