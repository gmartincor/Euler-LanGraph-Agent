from typing import Any, Dict, List, Optional
import uuid

# Official LangGraph BigTool imports
from langgraph_bigtool import create_agent
from langgraph.store.memory import InMemoryStore
from langchain.embeddings import init_embeddings
from langchain.chat_models import init_chat_model

# Google GenAI specific imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# LangChain tool creation
from langchain_core.tools import tool

from ..core.config import Settings, get_settings
from ..core.exceptions import ToolError
from ..core.logging import get_logger
from ..tools.registry import ToolRegistry

logger = get_logger(__name__)


class BigToolManager:
    """Manager for BigTool integration with existing ToolRegistry."""
    
    def __init__(
        self,
        tool_registry: ToolRegistry,
        settings: Optional[Settings] = None
    ) -> None:
        """Initialize BigTool manager with tool registry and settings."""
        self.tool_registry = tool_registry
        self.settings = settings or get_settings()
        self._agent = None
        self._store: Optional[InMemoryStore] = None
        self._tool_registry_dict: Dict[str, Any] = {}
        self._embeddings = None
        self._llm = None
        self._is_initialized = False
        
        logger.info("BigToolManager initialized")
    
    @property
    def is_enabled(self) -> bool:
        """Check if BigTool is enabled in settings."""
        return self.settings.bigtool_config.get("enabled", True)
    
    @property
    def is_initialized(self) -> bool:
        """Check if BigTool is initialized."""
        return self._is_initialized
    
    async def initialize(self) -> None:
        """Initialize BigTool with existing tools from registry."""
        try:
            config = self.settings.bigtool_config
            
            # Step 1: Create tool registry dict for BigTool
            self._create_tool_registry_dict()
            
            # Step 2: Initialize embeddings using Google GenAI directly
            try:
                # Use GoogleGenerativeAIEmbeddings directly (KISS principle)
                # This bypasses the init_embeddings provider issue
                self._embeddings = GoogleGenerativeAIEmbeddings(
                    model=config['embedding_model'],
                    google_api_key=config["api_key"]
                )
                logger.info(f"BigTool embeddings initialized: {config['embedding_model']}")
                
            except Exception as embedding_error:
                # Fail fast with clear error message
                error_msg = (
                    f"Failed to initialize Google GenAI embeddings with {config['embedding_model']}. "
                    f"Error: {embedding_error}. "
                    f"Please verify your Google API key and model configuration."
                )
                logger.error(error_msg)
                raise ToolError(error_msg) from embedding_error
            
            # Step 3: Create in-memory store with embeddings
            # Use dynamic dimensions based on the embedding model and configuration
            embedding_dims = config.get("embedding_dimensions", self._get_embedding_dimensions())
            
            self._store = InMemoryStore(
                index={
                    "embed": self._embeddings,
                    "dims": embedding_dims,
                    "fields": ["description"],
                }
            )
            logger.info(f"Created InMemoryStore with {embedding_dims} dimensions")
            
            # Step 4: Index tools in the store following BigTool pattern
            self._index_tools_in_store()
            
            # Step 5: Initialize LLM using Gemini directly (consistent with embeddings approach)
            gemini_config = self.settings.gemini_config
            
            self._llm = ChatGoogleGenerativeAI(
                model=gemini_config["model_name"],
                temperature=gemini_config["temperature"],
                max_output_tokens=gemini_config["max_output_tokens"],
                google_api_key=gemini_config["api_key"]
            )
            
            # Step 6: Create LangGraph agent with BigTool
            builder = create_agent(self._llm, self._tool_registry_dict)
            self._agent = builder.compile(store=self._store)
            
            self._is_initialized = True
            logger.info(
                "BigTool initialized successfully",
                extra={
                    "indexed_tools": len(self._tool_registry_dict),
                    "store_type": "InMemoryStore",
                    "agent_created": True
                }
            )
            
        except Exception as e:
            error_msg = f"Failed to initialize BigTool: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ToolError(error_msg) from e
    
    def _create_tool_registry_dict(self) -> None:
        """Create tool registry dict following BigTool pattern."""
        tools = self.tool_registry._list_tools_internal()
        logger.info(f"Creating tool registry dict for {len(tools)} tools: {tools}")
        
        for tool_name in tools:
            custom_tool = self.tool_registry.get_tool(tool_name)
            if custom_tool is None:
                logger.warning(f"Tool '{tool_name}' found in list but not accessible via get_tool()")
                continue
            
            # Create LangChain-compatible tool using the @tool decorator pattern
            langchain_tool = self._create_langchain_tool_adapter(custom_tool)
            
            # Create unique ID for each tool (BigTool pattern)
            tool_id = str(uuid.uuid4())
            self._tool_registry_dict[tool_id] = langchain_tool
            
            logger.info(f"Added tool '{tool_name}' with ID '{tool_id}' to BigTool registry")
    
    def _create_langchain_tool_adapter(self, custom_tool):
        """Create LangChain-compatible tool from custom tool."""
        from langchain_core.tools import tool
        
        # Create the LangChain tool with proper metadata using correct decorator syntax
        @tool(custom_tool.description)
        def tool_wrapper(**kwargs) -> str:
            """Adapter function to make custom tool compatible with LangChain."""
            try:
                result = custom_tool.execute(kwargs)
                
                # Extract the actual result from our tool output format
                if isinstance(result, dict):
                    if result.get('success'):
                        return str(result.get('result', 'Operation completed successfully'))
                    else:
                        return f"Error: {result.get('error_message', 'Unknown error')}"
                else:
                    return str(result)
                    
            except Exception as e:
                return f"Error executing {custom_tool.name}: {str(e)}"
        
        # Set the correct name for the tool
        tool_wrapper.name = custom_tool.name
        
        return tool_wrapper
    
    def _index_tools_in_store(self) -> None:
        """Index tools in LangGraph Store following BigTool pattern."""
        if not self._store:
            logger.warning("No store available for tool indexing")
            return
            
        logger.info(f"Indexing {len(self._tool_registry_dict)} tools in LangGraph Store")
        
        for tool_id, tool in self._tool_registry_dict.items():
            # Create enhanced description for semantic search
            tool_description = self._create_enhanced_description(tool)
            
            # Store following BigTool official pattern
            self._store.put(
                ("tools",),
                tool_id,
                {
                    "description": f"{tool.name}: {tool_description}",
                }
            )
            
            logger.info(f"Indexed tool '{tool.name}' (ID: {tool_id}) in LangGraph Store")
    
    def _create_enhanced_description(self, tool: Any) -> str:
        """Create enhanced description for semantic search."""
        base_description = getattr(tool, 'description', '') or ""
        
        # Add mathematical context for better semantic matching
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
                "function visualization", "mathematical plots"
            ])
        elif "analysis" in tool_name_lower or "analyze" in tool_name_lower:
            mathematical_keywords.extend([
                "mathematical analysis", "derivatives", "limits",
                "critical points", "function behavior"
            ])
        
        # Combine base description with context
        enhanced_description = base_description
        if mathematical_keywords:
            keywords_str = ", ".join(mathematical_keywords)
            enhanced_description += f" Mathematical context: {keywords_str}"
        
        return enhanced_description
    
    def _get_embedding_dimensions(self) -> int:
        """Get the correct embedding dimensions for the initialized model."""
        # All current Google embedding models use 768 dimensions
        # Could be made configurable in the future if needed
        return 768
    
    async def filter_tools_for_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        MAIN SEMANTIC FILTERING METHOD - BigTool's core purpose.
        
        Filters tools by semantic relevance BEFORE the LLM sees them.
        This replaces get_tool_recommendations with correct BigTool usage.
        
        Args:
            query: User query
            context: Additional context for filtering
            
        Returns:
            List[str]: Only semantically relevant tool names
        """
        if not self._is_initialized or self._store is None:
            raise ToolError("BigTool not initialized. Call initialize() first.")
        
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
    
    def health_check(self) -> Dict[str, Any]:
        """Check BigTool system health."""
        if not self.is_enabled:
            return {
                "status": "disabled",
                "is_enabled": False,
                "is_initialized": self._is_initialized
            }
        
        if not self._is_initialized:
            return {
                "status": "not_initialized", 
                "is_enabled": True,
                "is_initialized": False
            }
            
        return {
            "status": "healthy",
            "is_enabled": True,
            "is_initialized": True
        }

    @property  
    def agent(self):
        """Get BigTool agent instance (for advanced usage)."""
        return self._agent if self._is_initialized else None

    @property
    def store(self) -> Optional[InMemoryStore]:
        """Get LangGraph store instance (for advanced usage)."""
        return self._store if self._is_initialized else None


async def create_bigtool_manager(
    tool_registry: ToolRegistry,
    settings: Optional[Settings] = None
) -> BigToolManager:
    """Create and initialize BigToolManager."""
    manager = BigToolManager(tool_registry, settings)
    await manager.initialize()
    return manager
