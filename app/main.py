"""Main entry point for the ReAct Agent Streamlit application."""

import asyncio
from typing import Optional

import streamlit as st

from .core import get_logger, get_settings, setup_logging
from .core.exceptions import ReactAgentError
from .database import initialize_database, shutdown_database
from .tools.initialization import initialize_tools, get_tool_registry

# Initialize logging first
setup_logging()
logger = get_logger(__name__)


def initialize_app() -> None:
    """Initialize the application components."""
    try:
        # Load settings
        settings = get_settings()
        logger.info(f"Starting {settings.app_name} v{settings.app_version}")
        logger.info(f"Environment: {settings.environment}")
        
        # Initialize database
        initialize_database()
        logger.info("Database initialized successfully")
        
        # Initialize mathematical tools
        tool_registry = initialize_tools()
        st.session_state["tool_registry"] = tool_registry
        logger.info(f"Initialized {len(tool_registry)} mathematical tools")
        
        logger.info("Application initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        st.error(f"Application initialization failed: {e}")
        st.stop()


def setup_page_config() -> None:
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="🤖 ReAct Integral Agent",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': None,
            'Report a bug': None,
            'About': """
            # 🤖 ReAct Agent for Integral Calculus
            
            An intelligent agent using **BigTool + Gemini AI** to solve mathematical 
            integrals and visualize the area under the curve.
            
            **Features:**
            - 🧠 ReAct Agent with intelligent reasoning
            - 🔧 BigTool for automatic tool selection  
            - 🤖 Gemini AI for mathematical reasoning
            - 📊 Interactive visualizations
            - 💾 Persistent conversations
            """
        }
    )


def show_header() -> None:
    """Display the application header."""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="text-align: center;">
            <h1>🤖 ReAct Integral Agent</h1>
            <p style="font-size: 18px; color: #666;">
                Intelligent mathematical reasoning with BigTool + Gemini AI
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
                "gemini_model": settings.gemini_model,
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
    ## 👋 Welcome to the ReAct Integral Agent!
    
    I'm an AI agent specialized in solving mathematical integrals and creating visualizations.
    I can help you with:
    
    - 🧮 **Integral Calculations**: Definite and indefinite integrals
    - 📊 **Visualizations**: Area under the curve, function plots
    - 🔍 **Step-by-step Solutions**: Detailed mathematical reasoning
    - 💡 **Problem Analysis**: Understanding mathematical concepts
    
    ### 🚀 Getting Started
    
    Try asking me something like:
    - "Calculate the integral of x² from 0 to 5"
    - "What's the area under sin(x) from 0 to π?"
    - "Integrate e^x * cos(x) dx"
    - "Show me the graph of x³ - 2x² + 1"
    
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
        
        # Process the user's message (placeholder for now)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # TODO: This will be implemented in Phase 3 with the actual agent
                response = process_user_message(prompt, config)
                st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})


def process_user_message(message: str, config: dict) -> str:
    """
    Process user message and return response.
    
    This is a placeholder implementation that will be replaced
    with the actual ReAct agent in Phase 3.
    
    Args:
        message: User's message
        config: Configuration from sidebar
    
    Returns:
        str: Agent's response
    """
    logger.info(f"Processing message: {message[:100]}...")
    
    # Placeholder response
    return f"""
    Thank you for your message: "{message}"
    
    🚧 **Development Notice**: 
    This is currently a placeholder response. The full ReAct agent with BigTool 
    integration will be implemented in Phase 3.
    
    **Your Configuration:**
    - Session ID: {config['session_id']}
    - Temperature: {config['temperature']}
    - Max Tokens: {config['max_tokens']}
    - Show Step-by-step: {config['show_step_by_step']}
    - Show Plots: {config['show_plots']}
    - Plot Style: {config['plot_style']}
    
    **Next Steps:**
    1. ✅ Phase 1: Architecture Base (Current)
    2. 🔄 Phase 2: Mathematical Tools Implementation
    3. 🔄 Phase 3: ReAct Agent with LangGraph
    4. 🔄 Phase 4: Full Streamlit Integration
    """


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
        
    except ReactAgentError as e:
        logger.error(f"Agent error: {e}")
        st.error(f"Agent Error: {e.message}")
        
        with st.expander("Error Details"):
            st.json(e.to_dict())
            
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
