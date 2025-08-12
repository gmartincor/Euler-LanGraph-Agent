"""
Legacy compatibility for semantic filter.
Now redirects to unified tool engine.
"""
from typing import Any, Dict, List, Optional
from .tool_engine import get_tool_engine_health
from ..core.config import Settings


class SemanticFilter:
    """Legacy compatibility base class."""
    
    async def filter_tools_for_query(self, query: str, context: Optional[Dict] = None) -> List[str]:
        """Legacy compatibility - returns empty list since tool selection is now internal."""
        return []


class BigToolSemanticFilter(SemanticFilter):
    """Legacy compatibility for BigTool semantic filter."""
    
    def __init__(self, tool_registry=None, settings: Optional[Settings] = None) -> None:
        self.tool_registry = tool_registry
        self.settings = settings
        self._is_initialized = False
        
    async def initialize(self) -> None:
        """Legacy compatibility - no-op since tool selection is now internal."""
        self._is_initialized = True
        
    async def filter_tools_for_query(self, query: str, context: Optional[Dict] = None) -> List[str]:
        """Legacy compatibility - returns empty list since tool selection is now internal."""
        return []


async def create_semantic_filter(tool_registry=None, settings: Optional[Settings] = None):
    """Legacy compatibility for semantic filter creation."""
    filter_instance = BigToolSemanticFilter(tool_registry, settings)
    await filter_instance.initialize()
    return filter_instance
