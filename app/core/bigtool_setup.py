from typing import Any, Dict, List, Optional
import uuid

# Official LangGraph BigTool imports
from langgraph_bigtool import create_agent
from langgraph.store.memory import InMemoryStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

from ..core.config import Settings, get_settings
from ..core.exceptions import ToolError
from ..core.logging import get_logger
from ..tools.langchain_tools import get_langchain_tools

logger = get_logger(__name__)


class BigToolManager:
    """Refactored BigTool manager using native LangChain tools - DRY approach."""
    
    def __init__(self, settings: Optional[Settings] = None) -> None:
        """Initialize BigTool manager."""
        self.settings = settings or get_settings()
        self._agent = None
        self._store: Optional[InMemoryStore] = None
        self._tools: List[Any] = []
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
    
    @property
    def agent(self):
        """Get BigTool agent instance."""
        return self._agent if self._is_initialized else None
    
    async def initialize(self) -> None:
        """Initialize BigTool with LangChain tools."""
        try:
            config = self.settings.bigtool_config
            
            # Step 1: Get LangChain-compatible tools
            self._tools = get_langchain_tools()
            logger.info(f"Loaded {len(self._tools)} LangChain tools")
            
            # Step 2: Initialize embeddings
            self._embeddings = GoogleGenerativeAIEmbeddings(
                model=config['embedding_model'],
                google_api_key=config["api_key"]
            )
            logger.info(f"BigTool embeddings initialized: {config['embedding_model']}")
            
            # Step 3: Create tool registry for BigTool
            tool_registry = {
                str(uuid.uuid4()): tool 
                for tool in self._tools
            }
            
            # Step 4: Create store and index tools
            embedding_dims = config.get("embedding_dimensions", 768)
            self._store = InMemoryStore(
                index={
                    "embed": self._embeddings,
                    "dims": embedding_dims,
                    "fields": ["description"],
                }
            )
            logger.info(f"Created InMemoryStore with {embedding_dims} dimensions")
            
            # Index tools in store
            for tool_id, tool in tool_registry.items():
                self._store.put(
                    ("tools",),
                    tool_id,
                    {
                        "description": f"{tool.name}: {tool.description}",
                    }
                )
                logger.info(f"Indexed tool '{tool.name}' (ID: {tool_id}) in store")
            
            # Step 5: Initialize LLM with system prompt for mathematical capabilities
            gemini_config = self.settings.gemini_config
            self._llm = ChatGoogleGenerativeAI(
                model=gemini_config["model_name"],
                temperature=gemini_config["temperature"],
                max_output_tokens=gemini_config["max_output_tokens"],
                google_api_key=gemini_config["api_key"]
            )
            
            # Step 6: Create BigTool agent with enhanced tool-aware LLM
            builder = create_agent(self._llm, tool_registry)
            self._agent = builder.compile(store=self._store)
            
            self._is_initialized = True
            logger.info(
                "BigTool initialized successfully",
                extra={
                    "indexed_tools": len(tool_registry),
                    "store_type": "InMemoryStore",
                    "agent_created": True
                }
            )
            
        except Exception as e:
            error_msg = f"Failed to initialize BigTool: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ToolError(error_msg) from e
    
    async def process_query(self, query: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a query using BigTool agent - with clear instructions and recursion control.
        """
        if not self._is_initialized:
            raise ToolError("BigTool not initialized. Call initialize() first.")
        
        try:
            logger.info(f"Processing query with BigTool: {query[:100]}...")
            
            # Create system message with clear instructions
            system_message = {
                "role": "system", 
                "content": """You are a mathematical agent with specialized tools for calculations and visualizations.

AVAILABLE TOOLS:
- integral_calculator: Calculate definite/indefinite integrals (use expression, lower_bound, upper_bound)
- plot_generator: Create function plots with area highlighting (use expression, x_min, x_max, show_area, area_bounds)  
- function_analyzer: Analyze functions (derivatives, critical points, limits)

INSTRUCTIONS:
1. For integral problems: Use integral_calculator with correct expression format (e.g., "x**2" not "x²")
2. For visualizations: Use plot_generator with show_area=True for area under curve
3. Provide numerical and symbolic results
4. ALWAYS stop after using tools successfully - do not retry unnecessarily
5. If tools work, provide the final answer immediately

Example: "Calculate integral of x² from 0 to 3" → Use integral_calculator(expression="x**2", lower_bound=0, upper_bound=3)"""
            }
            
            # Configure execution with lower recursion limit
            stream_config = config or {"recursion_limit": 8}  # Reduced from 15 to 8
            
            # Process with BigTool agent - include system message
            result = await self._agent.ainvoke(
                {"messages": [system_message, {"role": "user", "content": query}]},
                config=stream_config
            )
            
            # Extract final message - handle AIMessage properly
            if "messages" in result and result["messages"]:
                final_message = result["messages"][-1]
                
                # Handle AIMessage object correctly
                if hasattr(final_message, 'content'):
                    content = final_message.content
                elif isinstance(final_message, dict):
                    content = final_message.get("content", str(final_message))
                else:
                    content = str(final_message)
                
                logger.info("BigTool processing completed successfully")
                return {
                    "success": True,
                    "result": content,
                    "tool_calls": self._extract_tool_calls(result)
                }
            else:
                return {
                    "success": False,
                    "error": "No response from BigTool agent"
                }
                
        except Exception as e:
            error_msg = f"BigTool query processing failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "success": False,
                "error": error_msg
            }
    
    def _extract_tool_calls(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract tool calls from BigTool result - fixed for AIMessage objects."""
        tool_calls = []
        known_tools = {"integral_calculator", "plot_generator", "function_analyzer"}
        
        try:
            if "messages" in result:
                for message in result["messages"]:
                    # Handle AIMessage objects correctly
                    if hasattr(message, 'tool_calls') and message.tool_calls:
                        for tool_call in message.tool_calls:
                            tool_name = getattr(tool_call, "name", "unknown")
                            # Only include known tools to avoid BigTool internal calls
                            if tool_name in known_tools:
                                tool_calls.append({
                                    "tool_name": tool_name,
                                    "arguments": getattr(tool_call, "args", {}),
                                    "call_id": getattr(tool_call, "id", "")
                                })
                    # Handle dict-based messages
                    elif isinstance(message, dict) and "tool_calls" in message:
                        for tool_call in message["tool_calls"]:
                            tool_name = tool_call.get("name", "unknown")
                            # Only include known tools
                            if tool_name in known_tools:
                                tool_calls.append({
                                    "tool_name": tool_name,
                                    "arguments": tool_call.get("args", {}),
                                    "call_id": tool_call.get("id", "")
                                })
        except Exception as e:
            logger.debug(f"Failed to extract tool calls: {e}")
        
        return tool_calls
    
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
            "is_initialized": True,
            "tools_count": len(self._tools)
        }


# BigToolManager Global Instance (Singleton Pattern for Performance)
_bigtool_manager_instance: Optional[BigToolManager] = None


async def create_bigtool_manager(settings: Optional[Settings] = None) -> BigToolManager:
    """
    Create and initialize BigToolManager.
    """
    global _bigtool_manager_instance
    
    # Always create fresh instance to avoid event loop conflicts
    # This fixes "Event loop is closed" errors between consecutive calls
    logger.info("Creating fresh BigToolManager instance to avoid event loop conflicts")
    manager = BigToolManager(settings)
    await manager.initialize()
    
    # Cache the instance for potential reuse within same event loop
    _bigtool_manager_instance = manager
    return manager
