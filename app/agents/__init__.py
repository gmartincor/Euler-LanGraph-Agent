"""
Simplified agents module - BigTool only (KISS principle).

After refactoring, we use BigTool directly instead of complex workflow.
This module now only exports availability flags for backward compatibility.
"""

# Availability flags for backward compatibility
WORKFLOW_COMPONENTS_AVAILABLE = False
REACT_AGENT_AVAILABLE = False

# All components are now handled by BigTool
# No need for complex state management, workflows, or chains

__all__ = [
    # Availability flags only
    "WORKFLOW_COMPONENTS_AVAILABLE",
    "REACT_AGENT_AVAILABLE",
]
