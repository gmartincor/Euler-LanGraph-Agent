"""
UI Module - Professional Streamlit Interface Architecture

This module implements a modular, reusable UI system following enterprise patterns:
- DRY: Reusable components and utilities
- KISS: Simple, focused components
- YAGNI: Only essential features
- Separation of concerns: UI, State, Business Logic
"""

from .components import *
from .pages import *
from .state import *
from .utils import *

__all__ = [
    # Components
    'ChatComponent',
    'SidebarComponent', 
    'MetricsComponent',
    'PlotComponent',
    'HistoryComponent',
    
    # Pages
    'MainChatPage',
    'AnalyticsPage',
    'SettingsPage',
    
    # State Management
    'SessionStateManager',
    'AgentStateManager',
    'UIStateManager',
    
    # Utilities
    'UIFormatters',
    'UIValidators',
    'StyleManager'
]
