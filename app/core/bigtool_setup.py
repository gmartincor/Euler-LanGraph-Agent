"""
Compatibility layer for legacy BigTool imports.
Redirects to unified tool engine.
"""
from typing import Dict, Any, Optional
from .tool_engine import execute_tool_query, get_tool_engine_health
from .config import Settings


# Legacy BigTool compatibility
async def create_bigtool_manager(settings: Optional[Settings] = None):
    """Legacy compatibility for BigTool manager creation."""
    class LegacyBigToolManager:
        def __init__(self):
            self.settings = settings
            
        async def initialize(self):
            pass
            
        async def process_query(self, query: str, config: Optional[Dict] = None) -> Dict[str, Any]:
            return await execute_tool_query(query, self.settings)
            
        def health_check(self) -> Dict[str, Any]:
            return get_tool_engine_health()
    
    manager = LegacyBigToolManager()
    await manager.initialize()
    return manager
