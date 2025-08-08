# Initialize availability flags
WORKFLOW_COMPONENTS_AVAILABLE = False
REACT_AGENT_AVAILABLE = False

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
        get_state_summary,
        format_agent_response
    )
    
    # Remove obsolete ReactMathematicalAgent (Phase 1 refactoring)
    ReactMathematicalAgent = None
    create_react_agent = None
    
    # Workflow components (Phase 3.4A - Extracted modular components)
    try:
        from .nodes import (
            analyze_problem_node,
            reasoning_node,
            tool_action_node,
            validation_node,
            final_response_node,
            error_recovery_node,
            NodeRegistry,
            create_all_node_wrappers
        )
        from .conditions import (
            should_use_tools,
            should_continue_reasoning,
            should_finalize,
            should_retry,
            ConditionRegistry,
            create_all_condition_wrappers,
            create_conditional_edges_config
        )
        # Phase 3.4B - Graph Orchestration
        from .graph import (
            MathematicalAgentGraph,
            create_mathematical_agent_graph,
            create_compiled_workflow
        )
        WORKFLOW_COMPONENTS_AVAILABLE = True
    except ImportError as e:
        WORKFLOW_COMPONENTS_AVAILABLE = False
        analyze_problem_node = None
        reasoning_node = None
        tool_action_node = None
        validation_node = None
        final_response_node = None
        error_recovery_node = None
        NodeRegistry = None
        create_all_node_wrappers = None
        should_use_tools = None
        should_continue_reasoning = None
        should_finalize = None
        should_retry = None
        ConditionRegistry = None
        create_all_condition_wrappers = None
        create_conditional_edges_config = None
        MathematicalAgentGraph = None
        create_mathematical_agent_graph = None
        create_compiled_workflow = None
        
    from .chains import ChainFactory, create_chain_factory, create_all_chains
    from .prompts import (
        # New centralized template system
        PromptTemplateRegistry,
        get_template_registry,
        get_prompt_template,
        format_prompt,
        build_tool_description,
        format_mathematical_context,
        
        # Legacy constants (backward compatibility)
        MATHEMATICAL_REASONING_PROMPT,
        PROBLEM_ANALYSIS_PROMPT,
        REFLECTION_PROMPT,
        ERROR_RECOVERY_PROMPT,
        TOOL_SELECTION_PROMPT,
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
            # New centralized template system
            PromptTemplateRegistry,
            get_template_registry,
            get_prompt_template,
            format_prompt,
            build_tool_description,
            format_mathematical_context,
            
            # Legacy constants (backward compatibility)
            MATHEMATICAL_REASONING_PROMPT,
            PROBLEM_ANALYSIS_PROMPT,
            REFLECTION_PROMPT,
            ERROR_RECOVERY_PROMPT,
            TOOL_SELECTION_PROMPT,
        )
    except ImportError:
        # Fallback values
        PromptTemplateRegistry = None
        get_template_registry = lambda: None
        get_prompt_template = lambda x: ""
        format_prompt = lambda x, **kwargs: ""
        build_tool_description = lambda x: ""
        format_mathematical_context = lambda x: ""
        MATHEMATICAL_REASONING_PROMPT = ""
        TOOL_SELECTION_PROMPT = ""
        REFLECTION_PROMPT = ""
        PROBLEM_ANALYSIS_PROMPT = ""
        ERROR_RECOVERY_PROMPT = ""
    
    REACT_AGENT_AVAILABLE = False

# Define __all__ first to avoid NameError
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
    "format_agent_response",
    
    # Core agent (conditionally available)
    "ReactMathematicalAgent",
    "create_react_agent",
    
    # Chain factory (conditionally available)
    "ChainFactory", 
    "create_chain_factory",
    "create_all_chains",
    
    # Prompts (centralized template system)
    "PromptTemplateRegistry",
    "get_template_registry", 
    "get_prompt_template",
    "format_prompt",
    "build_tool_description",
    "format_mathematical_context",
    
    # Legacy prompt constants (backward compatibility)
    "MATHEMATICAL_REASONING_PROMPT",
    "TOOL_SELECTION_PROMPT", 
    "REFLECTION_PROMPT",
    "PROBLEM_ANALYSIS_PROMPT",
    "ERROR_RECOVERY_PROMPT",
    
    # Availability flag
    "REACT_AGENT_AVAILABLE",
]

# Export workflow components if available
if WORKFLOW_COMPONENTS_AVAILABLE:
    __all__.extend([
        # Workflow nodes
        "analyze_problem_node",
        "reasoning_node", 
        "tool_action_node",
        "validation_node",
        "final_response_node",
        "error_recovery_node",
        "NodeRegistry",
        "create_all_node_wrappers",
        
        # Conditional logic
        "should_use_tools",
        "should_continue_reasoning",
        "should_finalize", 
        "should_retry",
        "ConditionRegistry",
        "create_all_condition_wrappers",
        "create_conditional_edges_config",
        
        # Graph orchestration (Phase 3.4B)
        "MathematicalAgentGraph",
        "create_mathematical_agent_graph", 
        "create_compiled_workflow",
        
        # Availability flag for workflow components
        "WORKFLOW_COMPONENTS_AVAILABLE"
    ])
