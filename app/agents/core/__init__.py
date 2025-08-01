"""Core Agent Components - Professional Business Logic Modules.

This package contains the core business logic components for the mathematical
ReAct agent, following professional design patterns and SOLID principles.

Modules:
- mathematical_reasoner: Core mathematical reasoning engine
- tool_orchestrator: Professional tool selection and execution
- state_manager: Comprehensive state management with history

Design Philosophy:
- Single Responsibility: Each module has one clear purpose
- Dependency Injection: Clean interfaces without circular dependencies
- Zero Duplication: Eliminates scattered logic from original implementation
- Professional Quality: Production-ready code with comprehensive testing
"""

from .mathematical_reasoner import MathematicalReasoner
from .tool_orchestrator import ToolOrchestrator
from .state_manager import StateManager

__all__ = [
    'MathematicalReasoner',
    'ToolOrchestrator', 
    'StateManager'
]
