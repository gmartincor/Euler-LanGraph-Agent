import sys
import os
from typing import Dict, Any

import streamlit as st

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    sys.path.insert(0, project_root)

from app.core import get_logger, get_settings, setup_logging
from app.core.exceptions import AgentError, DependencyError, ConfigurationError
from app.core.health_check import perform_startup_validation
from app.database import initialize_database
from app.tools.initialization import initialize_tools
from app.ui.pages import MainChatPage
from app.ui.state import get_state_manager

setup_logging()
logger = get_logger(__name__)


def setup_page_config() -> None:
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
    try:
        settings = get_settings()
        logger.info(f"Starting {settings.app_name} v{settings.app_version}")
        logger.info(f"Environment: {settings.environment}")
        perform_startup_validation(settings)
        logger.info("All dependencies validated successfully")
        initialize_database()
        logger.info("Database initialized successfully")
        tool_registry = initialize_tools()
        st.session_state["tool_registry"] = tool_registry
        logger.info(f"Initialized {len(tool_registry)} mathematical tools")
        
        # Initialize state manager after tools are ready
        state_manager = get_state_manager()
        # Only update state if initialization was successful
        if state_manager and hasattr(state_manager, 'state'):
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
    try:
        setup_page_config()
        if 'app_initialized' not in st.session_state:
            initialize_app()
            st.session_state['app_initialized'] = True
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
