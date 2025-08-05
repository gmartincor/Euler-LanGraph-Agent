"""
Mathematical Agent Streamlit Application - Clean Architecture Implementation

This module provides the main entry point for the mathematical agent application,
following professional software engineering principles: DRY, KISS, YAGNI, and modularization.

Key Architecture Principles:
- Single Responsibility: Each component has one clear purpose
- Dependency Injection: Clean separation of concerns
- Fail Fast: Early validation of dependencies
- Professional Error Handling: User-friendly error reporting
- Modular UI: Reusable, testable components
"""

import sys
import os
from typing import Dict, Any

import streamlit as st

# Fix import path for Docker environment
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    sys.path.insert(0, project_root)

from app.core import get_logger, get_settings, setup_logging
from app.core.exceptions import AgentError, DependencyError, ConfigurationError
from app.core.health_check import perform_startup_validation
from app.database import initialize_database
from app.tools.initialization import initialize_tools

# Professional UI Components - Clean Architecture
from app.ui.pages import MainChatPage
from app.ui.state import get_state_manager

# Initialize logging first
setup_logging()
logger = get_logger(__name__)


def setup_page_config() -> None:
    """Configure Streamlit page with professional settings."""
    st.set_page_config(
        page_title="🤖 Mathematical Agent",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': None,
            'Report a bug': None,
            'About': """
            # 🤖 Mathematical Agent - Professional Architecture
            
            An intelligent mathematical agent using **LangGraph + BigTool + Gemini AI** 
            to solve complex mathematical problems with professional-grade reasoning.
            
            **Architecture Features:**
            - 🏗️ Clean Architecture: Modular, testable, maintainable
            - 🧠 LangGraph Workflow: Sophisticated reasoning patterns
            - 🔧 BigTool Integration: Intelligent tool selection  
            - 🤖 Gemini AI: Advanced mathematical understanding
            - 📊 Interactive Visualizations: Rich mathematical graphics
            - 💾 Persistent State: Conversation continuity
            - ⚡ Professional Performance: Optimized for production
            """
        }
    )


def initialize_app() -> None:
    """
    Initialize application with fail-fast dependency validation.
    
    Follows professional initialization patterns:
    - Early validation of all dependencies
    - Clear error reporting with actionable messages
    - Proper resource management
    - State initialization
    """
    try:
        settings = get_settings()
        logger.info(f"Starting {settings.app_name} v{settings.app_version}")
        logger.info(f"Environment: {settings.environment}")
        
        # FAIL FAST: Validate all dependencies before proceeding
        perform_startup_validation(settings)
        logger.info("All dependencies validated successfully")
        
        # Initialize core services
        initialize_database()
        logger.info("Database initialized successfully")
        
        # Initialize mathematical tools
        tool_registry = initialize_tools()
        st.session_state["tool_registry"] = tool_registry
        logger.info(f"Initialized {len(tool_registry)} mathematical tools")
        
        # Initialize state manager with dependency injection
        state_manager = get_state_manager()
        state_manager.update_state(tool_registry=tool_registry)
        logger.info("UI state manager initialized")
        
        logger.info("Application initialized successfully")
        
    except (DependencyError, ConfigurationError) as e:
        logger.critical(f"Critical dependency error: {e}")
        st.error(f"❌ **Critical Error**: {e.message}")
        st.error("Please check your configuration and ensure all dependencies are properly installed.")
        st.stop()
    except Exception as e:
        logger.critical(f"Unexpected initialization error: {e}")
        st.error(f"❌ **Unexpected Error**: {str(e)}")
        st.error("Please check the logs and try restarting the application.")
        st.stop()


def main() -> None:
    """
    Main application entry point - Clean Architecture Implementation.
    
    Follows KISS principle: Simple, focused, single responsibility.
    Uses modular UI components for maintainability and testability.
    """
    try:
        # Configure page (must be called first)
        setup_page_config()
        
        # Initialize app if not already done (ensures single initialization)
        if 'app_initialized' not in st.session_state:
            initialize_app()
            st.session_state['app_initialized'] = True
        
        # Render main interface using modular architecture
        main_page = MainChatPage()
        main_page.render()
        
    except AgentError as e:
        logger.error(f"Agent error: {e}")
        st.error(f"🤖 **Agent Error**: {str(e)}")
        
        with st.expander("🔍 Error Details"):
            st.json({
                "error": str(e), 
                "type": "AgentError",
                "timestamp": str(st.session_state.get('timestamp', 'unknown'))
            })
            
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        st.error(f"❌ **Unexpected Error**: {e}")
        
        with st.expander("🔍 Technical Details"):
            st.code(str(e), language="text")


if __name__ == "__main__":
    main()
