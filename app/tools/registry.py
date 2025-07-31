"""Tool registry for managing and discovering mathematical tools."""

import asyncio
from typing import Any, Dict, List, Optional, Set, Union

from ..core.exceptions import ToolError
from ..core.logging import get_logger
from .base import BaseTool

logger = get_logger(__name__)


class ToolRegistry:
    """
    Central registry for managing and discovering mathematical tools.
    
    This class provides semantic search capabilities for tool selection
    and integrates with BigTool for intelligent tool management.
    """
    
    def __init__(self) -> None:
        """Initialize the tool registry."""
        self._tools: Dict[str, BaseTool] = {}
        self._categories: Dict[str, Set[str]] = {}
        self._tags: Dict[str, Set[str]] = {}
        self._usage_history: List[Dict[str, Any]] = []
    
    def register_tool(
        self,
        tool: BaseTool,
        categories: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """
        Register a tool in the registry.
        
        Args:
            tool: Tool instance to register
            categories: Categories this tool belongs to
            tags: Tags for semantic search
        """
        if tool.name in self._tools:
            logger.warning(f"Tool '{tool.name}' is already registered, replacing")
        
        self._tools[tool.name] = tool
        
        # Register categories
        if categories:
            for category in categories:
                if category not in self._categories:
                    self._categories[category] = set()
                self._categories[category].add(tool.name)
        
        # Register tags
        if tags:
            for tag in tags:
                if tag not in self._tags:
                    self._tags[tag] = set()
                self._tags[tag].add(tool.name)
        
        logger.info(f"Registered tool '{tool.name}' with categories: {categories}, tags: {tags}")
    
    def unregister_tool(self, tool_name: str) -> bool:
        """
        Unregister a tool from the registry.
        
        Args:
            tool_name: Name of the tool to unregister
        
        Returns:
            bool: True if tool was unregistered, False if not found
        """
        if tool_name not in self._tools:
            logger.warning(f"Tool '{tool_name}' not found in registry")
            return False
        
        # Remove from tools
        del self._tools[tool_name]
        
        # Remove from categories
        for category, tools in self._categories.items():
            tools.discard(tool_name)
        
        # Remove from tags
        for tag, tools in self._tags.items():
            tools.discard(tool_name)
        
        logger.info(f"Unregistered tool '{tool_name}'")
        return True
    
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """
        Get a tool by name.
        
        Args:
            tool_name: Name of the tool
        
        Returns:
            Optional[BaseTool]: Tool instance or None if not found
        """
        return self._tools.get(tool_name)
    
    def list_tools(self, category: Optional[str] = None) -> List[str]:
        """
        List all registered tools, optionally filtered by category.
        
        Args:
            category: Optional category filter
        
        Returns:
            List[str]: List of tool names
        """
        if category:
            return list(self._categories.get(category, set()))
        return list(self._tools.keys())
    
    def search_tools(
        self,
        query: str,
        limit: int = 5,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for tools using semantic matching.
        
        Args:
            query: Search query describing the desired functionality
            limit: Maximum number of results to return
            category: Optional category filter
        
        Returns:
            List[Dict[str, Any]]: Ranked list of matching tools with scores
        """
        query_lower = query.lower()
        candidates = []
        
        # Filter by category if specified
        tool_names = self.list_tools(category) if category else list(self._tools.keys())
        
        for tool_name in tool_names:
            tool = self._tools[tool_name]
            score = self._calculate_relevance_score(query_lower, tool)
            
            if score > 0:
                candidates.append({
                    "tool_name": tool_name,
                    "tool": tool,
                    "score": score,
                    "description": tool.description,
                    "usage_stats": tool.usage_stats,
                })
        
        # Sort by score (descending) and usage success rate
        candidates.sort(
            key=lambda x: (x["score"], x["usage_stats"]["success_rate"]),
            reverse=True
        )
        
        return candidates[:limit]
    
    def _calculate_relevance_score(self, query: str, tool: BaseTool) -> float:
        """
        Calculate relevance score between query and tool.
        
        Args:
            query: Lowercase search query
            tool: Tool to evaluate
        
        Returns:
            float: Relevance score (0.0 to 1.0)
        """
        score = 0.0
        
        # Check tool name
        if query in tool.name.lower():
            score += 0.3
        
        # Check tool description
        description_lower = tool.description.lower()
        query_words = query.split()
        
        for word in query_words:
            if word in description_lower:
                score += 0.2 / len(query_words)
        
        # Check categories
        for category, tools in self._categories.items():
            if tool.name in tools and query in category.lower():
                score += 0.3
        
        # Check tags
        for tag, tools in self._tags.items():
            if tool.name in tools and query in tag.lower():
                score += 0.2
        
        # Boost score based on success rate
        success_rate = tool.usage_stats["success_rate"]
        score *= (0.7 + 0.3 * success_rate)  # Scale between 0.7 and 1.0
        
        return min(score, 1.0)
    
    def get_tool_recommendations(
        self,
        context: Dict[str, Any],
        limit: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Get tool recommendations based on context.
        
        Args:
            context: Context information (previous tools used, current problem, etc.)
            limit: Maximum number of recommendations
        
        Returns:
            List[Dict[str, Any]]: Recommended tools with explanations
        """
        recommendations = []
        
        # Extract relevant information from context
        problem_type = context.get("problem_type", "")
        previous_tools = context.get("previous_tools", [])
        math_expressions = context.get("math_expressions", [])
        
        # Build recommendation query
        query_parts = [problem_type]
        if math_expressions:
            query_parts.extend(["integral", "calculus", "math"])
        
        query = " ".join(query_parts).strip()
        
        if query:
            # Search for relevant tools
            candidates = self.search_tools(query, limit * 2)
            
            # Filter out recently used tools (unless they're highly relevant)
            for candidate in candidates:
                tool_name = candidate["tool_name"]
                
                # If tool wasn't used recently or has very high relevance, recommend it
                if (tool_name not in previous_tools[-2:] or candidate["score"] > 0.8):
                    recommendations.append({
                        "tool_name": tool_name,
                        "tool": candidate["tool"],
                        "reason": self._generate_recommendation_reason(candidate, context),
                        "confidence": candidate["score"],
                    })
        
        return recommendations[:limit]
    
    def _generate_recommendation_reason(
        self,
        candidate: Dict[str, Any],
        context: Dict[str, Any],
    ) -> str:
        """
        Generate a human-readable reason for tool recommendation.
        
        Args:
            candidate: Tool candidate information
            context: Context information
        
        Returns:
            str: Explanation for why this tool is recommended
        """
        tool_name = candidate["tool_name"]
        score = candidate["score"]
        
        reasons = []
        
        if score > 0.8:
            reasons.append("highly relevant to your query")
        elif score > 0.6:
            reasons.append("relevant to your query")
        
        success_rate = candidate["usage_stats"]["success_rate"]
        if success_rate > 0.9:
            reasons.append("has excellent reliability")
        elif success_rate > 0.7:
            reasons.append("has good reliability")
        
        if not reasons:
            reasons.append("matches your request")
        
        return f"Recommended because it {' and '.join(reasons)}"
    
    def record_tool_usage(
        self,
        tool_name: str,
        success: bool,
        execution_time: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record tool usage for analytics and recommendations.
        
        Args:
            tool_name: Name of the tool used
            success: Whether the tool execution was successful
            execution_time: Time taken to execute
            context: Optional context information
        """
        usage_record = {
            "tool_name": tool_name,
            "success": success,
            "execution_time": execution_time,
            "timestamp": asyncio.get_event_loop().time(),
            "context": context or {},
        }
        
        self._usage_history.append(usage_record)
        
        # Keep only recent history (last 1000 records)
        if len(self._usage_history) > 1000:
            self._usage_history = self._usage_history[-1000:]
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive registry statistics.
        
        Returns:
            Dict[str, Any]: Registry statistics and analytics
        """
        total_tools = len(self._tools)
        total_categories = len(self._categories)
        total_tags = len(self._tags)
        total_usage = len(self._usage_history)
        
        # Calculate average success rate
        successful_uses = sum(1 for record in self._usage_history if record["success"])
        avg_success_rate = successful_uses / total_usage if total_usage > 0 else 0.0
        
        # Find most used tools
        tool_usage_count = {}
        for record in self._usage_history:
            tool_name = record["tool_name"]
            tool_usage_count[tool_name] = tool_usage_count.get(tool_name, 0) + 1
        
        most_used_tools = sorted(
            tool_usage_count.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            "total_tools": total_tools,
            "total_categories": total_categories,
            "total_tags": total_tags,
            "total_usage_records": total_usage,
            "average_success_rate": avg_success_rate,
            "most_used_tools": most_used_tools,
            "tools_by_category": {
                category: len(tools) for category, tools in self._categories.items()
            },
        }
    
    def reset_usage_history(self) -> None:
        """Reset usage history."""
        self._usage_history.clear()
        logger.info("Tool usage history reset")
    
    def __len__(self) -> int:
        """Return number of registered tools."""
        return len(self._tools)
    
    def __contains__(self, tool_name: str) -> bool:
        """Check if tool is registered."""
        return tool_name in self._tools


# Global tool registry instance
tool_registry = ToolRegistry()
