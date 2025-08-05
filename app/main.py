"""Main entry point for the Mathematical Agent Streamlit application - Unified Architecture.

This module provides the Streamlit interface for the unified mathematical agent,
using clean architecture principles and professional UI components.
"""

import asyncio
import sys
import os
from typing import Optional, Dict, Any

import streamlit as st

# Fix import path for Docker environment
if __name__ == "__main__":
    # Add project root to path when running directly
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    sys.path.insert(0, project_root)

from app.core import get_logger, get_settings, setup_logging
from app.core.exceptions import AgentError, DependencyError, ConfigurationError
from app.core.health_check import perform_startup_validation
from app.database import initialize_database, shutdown_database
from app.tools.initialization import initialize_tools, get_tool_registry
from app.agents.interface import create_mathematical_agent

# Professional UI Components (NEW)
from app.ui.pages import MainChatPage
from app.ui.state import get_state_manager

# Initialize logging first
setup_logging()
logger = get_logger(__name__)


def initialize_app() -> None:
    """Initialize the application components with professional dependency validation."""
    try:
        # Load settings
        settings = get_settings()
        logger.info(f"Starting {settings.app_name} v{settings.app_version}")
        logger.info(f"Environment: {settings.environment}")
        
        # FAIL FAST: Validate all dependencies before proceeding
        perform_startup_validation(settings)
        logger.info("All dependencies validated successfully")
        
        # Initialize database
        initialize_database()
        logger.info("Database initialized successfully")
        
        # Initialize mathematical tools
        tool_registry = initialize_tools()
        st.session_state["tool_registry"] = tool_registry
        logger.info(f"Initialized {len(tool_registry)} mathematical tools")
        
        # Initialize state manager
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


def setup_page_config() -> None:
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="🤖 Mathematical Agent",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': None,
            'Report a bug': None,
            'About': """
            # 🤖 Mathematical Agent - Unified Architecture
            
            An intelligent agent using **LangGraph + BigTool + Gemini AI** to solve 
            mathematical problems with professional-grade reasoning.
            
            **Features:**
            - 🧠 Unified LangGraph workflow
            - 🔧 BigTool for automatic tool selection  
            - 🤖 Gemini AI for mathematical reasoning
            - 📊 Interactive visualizations
            - 💾 Persistent conversations
            - 🏗️ Clean, professional architecture
            """
        }
    )


def main() -> None:
    """Main application entry point with professional UI architecture."""
    # Configure page first (must be done only once)
    setup_page_config()
    
    # Initialize app if not already done
    if 'app_initialized' not in st.session_state:
        initialize_app()
        st.session_state['app_initialized'] = True
    
    # Render main chat page using modular components
    main_page = MainChatPage()
    main_page.render()


if __name__ == "__main__":
    main()


def show_header() -> None:
    """Display the application header."""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="text-align: center;">
            <h1>🤖 Mathematical Agent</h1>
            <p style="font-size: 18px; color: #666;">
                Intelligent mathematical reasoning with unified LangGraph + BigTool + Gemini AI
            </p>
        </div>
        """, unsafe_allow_html=True)


def show_sidebar() -> dict:
    """
    Display and handle sidebar configuration.
    
    Returns:
        dict: Sidebar configuration values
    """
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # Session configuration
        st.markdown("### Session")
        session_id = st.text_input(
            "Session ID",
            value=st.session_state.get("session_id", "default-session"),
            help="Unique identifier for your conversation session"
        )
        
        if st.button("🔄 New Session"):
            st.session_state.clear()
            st.rerun()
        
        st.divider()
        
        # Model configuration
        st.markdown("### Model Settings")
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.1,
            help="Controls randomness in AI responses"
        )
        
        max_tokens = st.number_input(
            "Max Tokens",
            min_value=256,
            max_value=2048,
            value=1024,
            step=128,
            help="Maximum tokens in AI response"
        )
        
        st.divider()
        
        # Tool preferences
        st.markdown("### Tool Preferences")
        
        show_step_by_step = st.checkbox(
            "Show step-by-step solutions",
            value=True,
            help="Display detailed solution steps"
        )
        
        show_plots = st.checkbox(
            "Show visualizations",
            value=True,
            help="Display graphs and plots"
        )
        
        plot_style = st.selectbox(
            "Plot Style",
            options=["matplotlib", "plotly"],
            index=1,
            help="Choose visualization library"
        )
        
        st.divider()
        
        # System information
        st.markdown("### System Info")
        
        with st.expander("Database Status"):
            try:
                from .database import check_database_health
                if check_database_health():
                    st.success("✅ Database connected")
                else:
                    st.error("❌ Database disconnected")
            except Exception as e:
                st.error(f"❌ Database error: {e}")
        
        with st.expander("Settings"):
            settings = get_settings()
            st.json({
                "app_version": settings.app_version,
                "environment": settings.environment,
                "debug": settings.debug,
                "gemini_model": settings.gemini_model_name,
            })
    
    return {
        "session_id": session_id,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "show_step_by_step": show_step_by_step,
        "show_plots": show_plots,
        "plot_style": plot_style,
    }


def show_welcome_message() -> None:
    """Display welcome message and instructions."""
    st.markdown("""
    ## 👋 Welcome to the Mathematical Agent!
    
    I'm an AI agent specialized in solving mathematical problems using a unified LangGraph architecture.
    I can help you with:
    
    - 🧮 **Mathematical Calculations**: Integrals, derivatives, equations
    - 📊 **Visualizations**: Function plots, area calculations, graphs
    - 🔍 **Step-by-step Solutions**: Detailed mathematical reasoning  
    - 💡 **Problem Analysis**: Understanding mathematical concepts
    - 🏗️ **Professional Architecture**: Clean, maintainable, scalable
    
    ### 🚀 Getting Started
    
    Try asking me something like:
    - "Calculate the integral of x² from 0 to 5"
    - "What's the area under sin(x) from 0 to π?"
    - "Integrate e^x * cos(x) dx"
    - "Show me the graph of x³ - 2x² + 1"
    - "Solve the equation x² + 2x - 3 = 0"
    
    ### 💬 Start a Conversation
    
    Type your mathematical question in the chat box below!
    """)


def show_chat_interface(config: dict) -> None:
    """
    Display the main chat interface.
    
    Args:
        config: Configuration from sidebar
    """
    # Store configuration in session state
    for key, value in config.items():
        st.session_state[f"config_{key}"] = value
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display tool calls if present
            if "tool_calls" in message:
                with st.expander("🔧 Tool Calls", expanded=False):
                    for tool_call in message["tool_calls"]:
                        st.code(f"Tool: {tool_call['name']}\nInput: {tool_call['input']}")
                        if "output" in tool_call:
                            st.code(f"Output: {tool_call['output']}")
    
    # Chat input
    if prompt := st.chat_input("Ask me about integrals and mathematical functions..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process the user's message using the unified mathematical agent
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Initialize agent if not already done
                    if "mathematical_agent" not in st.session_state:
                        st.session_state.mathematical_agent = create_mathematical_agent(
                            session_id=config["session_id"]
                        )
                    
                    # Process message with agent (using sync wrapper)
                    response = process_user_message_with_agent(prompt, config)
                    st.markdown(response["content"])
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response["content"],
                        "metadata": response.get("metadata", {})
                    })
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })


def process_user_message_with_agent(message: str, config: dict) -> Dict[str, Any]:
    """
    Process user message with the unified mathematical agent.
    
    Args:
        message: User's message
        config: Configuration from sidebar
    
    Returns:
        Dict: Agent's response with content and metadata
    """
    logger.info(f"Processing message with unified agent: {message[:100]}...")
    
    try:
        # Get agent from session state
        agent = st.session_state.get("mathematical_agent")
        if not agent:
            return {
                "content": "⚠️ Agent not initialized. Please refresh the page.",
                "metadata": {"error": "agent_not_initialized"}
            }
        
        # Use asyncio to run the async solve method
        import asyncio
        
        try:
            # Create event loop if needed
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run the agent solve method
            result = loop.run_until_complete(agent.solve(message))
            
            # Format the response for Streamlit
            formatted_content = f"""
## 🎯 **Mathematical Solution**

**Problem**: {message}

**Solution**: {result.get('answer', 'No answer calculated')}

**Confidence**: {result.get('confidence', 0.0):.1%}

**Method**: {result.get('explanation', 'Mathematical reasoning applied')}

### 📝 **Solution Steps**:
"""
            
            steps = result.get('steps', [])
            if steps:
                for i, step in enumerate(steps, 1):
                    formatted_content += f"\n{i}. {step}"
            else:
                formatted_content += "\nNo detailed steps available."
            
            formatted_content += f"""

### ⚙️ **Configuration Used**:
- **Session ID**: `{config['session_id']}`
- **Temperature**: `{config['temperature']}`
- **Max Tokens**: `{config['max_tokens']}`
- **Step-by-step**: `{'✅' if config['show_step_by_step'] else '❌'} {config['show_step_by_step']}`
- **Visualizations**: `{'✅' if config['show_plots'] else '❌'} {config['show_plots']}`
- **Plot Style**: `{config['plot_style']}`

### 🏗️ **Architecture Status**: 
✅ **Unified LangGraph Agent** - Successfully executed mathematical reasoning workflow!
            """
            
            return {
                "content": formatted_content,
                "metadata": {
                    "architecture": "unified_langgraph",
                    "result": result,
                    "config": config,
                    "execution_time": result.get('execution_time', 0),
                    "message_length": len(message)
                }
            }
            
        except Exception as async_error:
            logger.error(f"Error in async execution: {async_error}")
            return {
                "content": f"""
## ⚠️ **Execution Error**

Sorry, I encountered an error while processing your request:

**Error**: {str(async_error)}

**Problem**: {message}

**Next Steps**: 
1. Check if the problem format is correct
2. Try a simpler version of the problem
3. Refresh the page to reset the agent

The agent architecture is ready, but needs refinement for complex mathematical operations.
                """,
                "metadata": {"error": str(async_error)}
            }
        
    except Exception as e:
        logger.error(f"Error processing message with agent: {e}")
        return {
            "content": f"❌ Error processing your message: {str(e)}",
            "metadata": {"error": str(e)}
        }


def process_user_message(message: str, config: dict) -> str:
    """
    Legacy process user message function (kept for compatibility).
    
    Args:
        message: User's message
        config: Configuration from sidebar
    
    Returns:
        str: Agent's response
    """
    result = process_user_message_with_agent(message, config)
    return result["content"]


def main() -> None:
    """Main application entry point."""
    try:
        # Setup page configuration
        setup_page_config()
        
        # Initialize application
        if "app_initialized" not in st.session_state:
            initialize_app()
            st.session_state.app_initialized = True
        
        # Show header
        show_header()
        
        # Show sidebar and get configuration
        config = show_sidebar()
        
        # Add tool demonstration section
        show_tools_demo()
        
        # Main content area
        if not st.session_state.get("messages"):
            show_welcome_message()
        
        # Show chat interface
        show_chat_interface(config)
        
    except AgentError as e:
        logger.error(f"Agent error: {e}")
        st.error(f"Agent Error: {str(e)}")
        
        with st.expander("Error Details"):
            st.json({"error": str(e), "type": "AgentError"})
            
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        st.error(f"Unexpected error: {e}")


def show_tools_demo() -> None:
    """Show mathematical tools demonstration."""
    st.markdown("---")
    st.markdown("## 🔧 Mathematical Tools Demo")
    
    tool_registry = st.session_state.get("tool_registry")
    if not tool_registry:
        st.warning("Tools not initialized")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📊 Available Tools")
        for tool_name in tool_registry.list_tools():
            tool = tool_registry.get_tool(tool_name)
            if tool:
                with st.expander(f"🛠️ {tool.name}"):
                    st.write(f"**Description:** {tool.description}")
                    stats = tool.usage_stats
                    st.metric("Usage Count", stats["usage_count"])
                    st.metric("Success Rate", f"{stats['success_rate']:.1%}")
    
    with col2:
        st.markdown("### 🔍 Tool Search")
        search_query = st.text_input("Search for tools:", placeholder="e.g., integral, plot, derivative")
        
        if search_query:
            results = tool_registry.search_tools(search_query, limit=3)
            for result in results:
                with st.expander(f"📈 {result['tool_name']} (Score: {result['score']:.2f})"):
                    st.write(result['description'])
    
    with col3:
        st.markdown("### 📈 Registry Stats")
        stats = tool_registry.get_registry_stats()
        
        st.metric("Total Tools", stats["total_tools"])
        st.metric("Total Categories", stats["total_categories"])
        st.metric("Usage Records", stats["total_usage_records"])
        
        if stats["most_used_tools"]:
            st.markdown("**Most Used:**")
            for tool_name, count in stats["most_used_tools"][:3]:
                st.write(f"• {tool_name}: {count} uses")


if __name__ == "__main__":
    main()
