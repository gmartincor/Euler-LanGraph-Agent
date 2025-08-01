"""ReAct Mathematical Agents module.

This module provides the core ReAct (Reasoning and Acting) agent implementation
for mathematical problem solving, along with supporting components.

Key Components:
- ReactMathematicalAgent: Core ReAct agent for mathematical reasoning
- ChainFactory: Factory for creating reasoning chains
- Prompt templates: Specialized prompts for mathematical reasoning
- Integration with existing infrastructure (ToolRegistry, BigTool, etc.)
"""

from .react_agent import ReactMathematicalAgent, create_react_agent
from .chains import ChainFactory, create_chain_factory, create_all_chains
from .prompts import (
    MATHEMATICAL_REASONING_PROMPT,
    TOOL_SELECTION_PROMPT,
    REFLECTION_PROMPT,
    PROBLEM_ANALYSIS_PROMPT,
    ERROR_RECOVERY_PROMPT,
    get_prompt_template,
    build_tool_description,
    format_mathematical_context
)

__all__ = [
    # Core agent
    "ReactMathematicalAgent",
    "create_react_agent",
    
    # Chain factory
    "ChainFactory", 
    "create_chain_factory",
    "create_all_chains",
    
    # Prompts
    "MATHEMATICAL_REASONING_PROMPT",
    "TOOL_SELECTION_PROMPT", 
    "REFLECTION_PROMPT",
    "PROBLEM_ANALYSIS_PROMPT",
    "ERROR_RECOVERY_PROMPT",
    "get_prompt_template",
    "build_tool_description",
    "format_mathematical_context",
]
