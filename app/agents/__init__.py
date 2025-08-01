"""ReAct Mathematical Agents module.

This module provides the core ReAct (Reasoning and Acting) agent implementation
for mathematical problem solving, along with supporting components.

Key Components:
- ReactMathematicalAgent: Core ReAct agent for mathematical reasoning
- ChainFactory: Factory for creating reasoning chains
- Prompt templates: Specialized prompts for mathematical reasoning
- State management: Agent state and utilities (consolidated from /agent/)
- Integration with existing infrastructure (ToolRegistry, BigTool, etc.)
"""

# Import with error handling for optional dependencies
try:
    # State management components (moved from /agent/)
    from .state import MathAgentState, WorkflowStatus, WorkflowSteps
    from .state_utils import (
        create_initial_state,
        validate_state,
        serialize_state,
        deserialize_state,
        update_state_safely,
        get_state_summary
    )
    
    from .react_agent import ReactMathematicalAgent
    
    # Import factory function separately to handle dependency issues
    try:
        from .react_agent import create_react_agent
    except ImportError:
        create_react_agent = None
        
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
    
    REACT_AGENT_AVAILABLE = True
    
except ImportError as e:
    # Handle missing dependencies gracefully
    ReactMathematicalAgent = None
    create_react_agent = None
    ChainFactory = None
    create_chain_factory = None
    create_all_chains = None
    
    # Prompt templates should always be available
    try:
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
    except ImportError:
        # Fallback values
        MATHEMATICAL_REASONING_PROMPT = ""
        TOOL_SELECTION_PROMPT = ""
        REFLECTION_PROMPT = ""
        PROBLEM_ANALYSIS_PROMPT = ""
        ERROR_RECOVERY_PROMPT = ""
        get_prompt_template = lambda x: ""
        build_tool_description = lambda x: ""
        format_mathematical_context = lambda x: ""
    
    REACT_AGENT_AVAILABLE = False

__all__ = [
    # State management (moved from /agent/)
    "MathAgentState",
    "WorkflowStatus", 
    "WorkflowSteps",
    "create_initial_state",
    "validate_state",
    "serialize_state", 
    "deserialize_state",
    "update_state_safely",
    "get_state_summary",
    
    # Core agent (conditionally available)
    "ReactMathematicalAgent",
    "create_react_agent",
    
    # Chain factory (conditionally available)
    "ChainFactory", 
    "create_chain_factory",
    "create_all_chains",
    
    # Prompts (always available)
    "MATHEMATICAL_REASONING_PROMPT",
    "TOOL_SELECTION_PROMPT", 
    "REFLECTION_PROMPT",
    "PROBLEM_ANALYSIS_PROMPT",
    "ERROR_RECOVERY_PROMPT",
    "get_prompt_template",
    "build_tool_description",
    "format_mathematical_context",
    
    # Availability flag
    "REACT_AGENT_AVAILABLE",
]
