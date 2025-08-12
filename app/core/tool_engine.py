"""Tool Engine using LangGraph BigTool."""
from typing import Any, Dict, Optional
import uuid

from langgraph_bigtool import create_agent
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from .config import Settings, get_settings
from .exceptions import ToolError
from .logging import get_logger
from ..tools.langchain_tools import get_langchain_tools

logger = get_logger(__name__)


class ToolEngine:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._agent = None
        self._initialized = False
        
    async def initialize(self) -> None:
        if self._initialized:
            return
            
        try:
            tools = get_langchain_tools()
            tool_registry = {str(uuid.uuid4()): tool for tool in tools}
            
            config = self.settings.bigtool_config
            embeddings = GoogleGenerativeAIEmbeddings(
                model=config['embedding_model'],
                google_api_key=config["api_key"]
            )
            
            store = InMemoryStore(
                index={
                    "embed": embeddings,
                    "dims": config.get("embedding_dimensions", 768),
                    "fields": ["description"],
                }
            )
            
            for tool_id, tool in tool_registry.items():
                store.put(
                    ("tools",), tool_id,
                    {"description": f"{tool.name}: {tool.description}"}
                )
            
            gemini_config = self.settings.gemini_config
            llm = ChatGoogleGenerativeAI(
                model=gemini_config["model_name"],
                temperature=gemini_config["temperature"],
                google_api_key=gemini_config["api_key"]
            )
            
            builder = create_agent(llm, tool_registry)
            self._agent = builder.compile(
                store=store,
                checkpointer=MemorySaver()
            )
            
            self._initialized = True
            logger.info(f"Tool engine initialized with {len(tools)} tools")
            
        except Exception as e:
            logger.error(f"Failed to initialize tool engine: {e}", exc_info=True)
            raise ToolError(f"Tool engine initialization failed: {e}") from e
    
    async def execute(self, query: str) -> Dict[str, Any]:
        if not self._initialized:
            await self.initialize()
        
        try:
            result = await self._agent.ainvoke(
                {"messages": [HumanMessage(content=query)]},
                config={"configurable": {"thread_id": str(uuid.uuid4())[:8]}}
            )
            
            final_message = result["messages"][-1] if result.get("messages") else None
            if not final_message:
                return {"success": False, "error": "No response generated"}
                
            return {
                "success": True,
                "result": getattr(final_message, 'content', str(final_message))
            }
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}


_engine = ToolEngine()


async def execute_tool_query(query: str, settings: Optional[Settings] = None) -> Dict[str, Any]:
    if settings:
        engine = ToolEngine(settings)
        await engine.initialize()
        return await engine.execute(query)
    return await _engine.execute(query)


def get_tool_engine_health() -> Dict[str, Any]:
    return {"initialized": _engine._initialized}
