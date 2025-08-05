"""
Chat Component - Professional chat interface with real-time agent interaction

Implements enterprise patterns:
- Async agent communication
- Real-time updates
- Error handling with recovery
- Message persistence
"""

from typing import List, Dict, Any, Optional
import streamlit as st
import asyncio
from datetime import datetime

from app.ui.state import get_state_manager, UIState
from app.ui.utils import UIFormatters, UIValidators, StyleManager


class ChatComponent:
    """Professional chat interface component with agent integration."""
    
    def __init__(self):
        self.state_manager = get_state_manager()
        self.formatters = UIFormatters()
        self.validators = UIValidators()
    
    def render(self) -> None:
        """Render the complete chat interface."""
        # Apply styling
        StyleManager.inject_global_styles()
        
        # Main chat container
        with st.container():
            self._render_header()
            self._render_messages()
            self._render_input_area()
            self._render_status_bar()
    
    def _render_header(self) -> None:
        """Render chat header with status and controls."""
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.title("🤖 Mathematical Agent")
            st.caption("ReAct Agent for Integral Calculus")
        
        with col2:
            # Status indicator
            ui_state = self.state_manager.state.ui_state
            status_text = ui_state.value.title()
            status_html = StyleManager.create_status_indicator(ui_state.value, status_text)
            st.markdown(status_html, unsafe_allow_html=True)
        
        with col3:
            # Controls
            if st.button("🗑️ Clear Chat", help="Clear conversation history"):
                self._clear_conversation()
            
            if st.button("⚙️ Settings", help="Open settings"):
                self.state_manager.update_state(show_settings=True)
    
    def _render_messages(self) -> None:
        """Render conversation messages with professional styling."""
        messages = self.state_manager.state.message_history
        
        if not messages:
            self._render_welcome_message()
            return
        
        # Messages container with scrolling
        message_container = st.container()
        
        with message_container:
            for i, message in enumerate(messages):
                self._render_single_message(message, i)
    
    def _render_single_message(self, message: Dict[str, Any], index: int) -> None:
        """Render a single message with metadata."""
        role = message.get('role', 'unknown')
        content = message.get('content', '')
        timestamp = message.get('timestamp', '')
        metadata = message.get('metadata', {})
        
        # Message container
        with st.container():
            # Role and timestamp header
            col1, col2 = st.columns([4, 1])
            
            with col1:
                if role == 'user':
                    st.markdown("**👤 You**")
                elif role == 'assistant':
                    st.markdown("**🤖 Agent**")
                else:
                    st.markdown(f"**{role.title()}**")
            
            with col2:
                if timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        formatted_time = self.formatters.format_timestamp(dt, "relative")
                        st.caption(formatted_time)
                    except:
                        st.caption(timestamp)
            
            # Message content
            if role == 'user':
                message_html = StyleManager.create_chat_message(content, 'user')
                st.markdown(message_html, unsafe_allow_html=True)
            else:
                message_html = StyleManager.create_chat_message(content, 'assistant')
                st.markdown(message_html, unsafe_allow_html=True)
                
                # Show metadata if available
                if metadata:
                    with st.expander("📊 Details", expanded=False):
                        self._render_message_metadata(metadata)
            
            st.markdown("---")
    
    def _render_message_metadata(self, metadata: Dict[str, Any]) -> None:
        """Render message metadata (tool usage, timing, etc.)."""
        # Tool usage
        if 'tools_used' in metadata:
            st.markdown("**🔧 Tools Used:**")
            for tool in metadata['tools_used']:
                tool_name = tool.get('name', 'Unknown')
                duration = tool.get('duration', 0)
                status = tool.get('status', 'unknown')
                
                duration_str = self.formatters.format_duration(duration)
                status_icon = "✅" if status == "success" else "❌" if status == "error" else "⏳"
                
                st.markdown(f"- {status_icon} **{tool_name}** ({duration_str})")
        
        # Execution time
        if 'execution_time' in metadata:
            duration = self.formatters.format_duration(metadata['execution_time'])
            st.markdown(f"**⏱️ Execution Time:** {duration}")
        
        # Token usage
        if 'token_usage' in metadata:
            tokens = metadata['token_usage']
            st.markdown(f"**🪙 Tokens:** {tokens.get('total', 0)} (prompt: {tokens.get('prompt', 0)}, completion: {tokens.get('completion', 0)})")
        
        # Raw metadata
        if st.checkbox("Show raw metadata", key=f"metadata_{id(metadata)}"):
            st.json(metadata)
    
    def _render_welcome_message(self) -> None:
        """Render welcome message for new conversations."""
        welcome_html = StyleManager.create_chat_message(
            """
            👋 **Welcome to the Mathematical Agent!**
            
            I can help you with:
            - **Calculate integrals** with step-by-step solutions
            - **Create visualizations** of functions and areas under curves
            - **Analyze mathematical results** and provide insights
            
            **Try asking me:**
            - "Calculate the integral of x² from 0 to 3"
            - "Show me the plot of sin(x) from 0 to π"
            - "What's the area under e^x from 1 to 2?"
            """,
            'assistant'
        )
        st.markdown(welcome_html, unsafe_allow_html=True)
    
    def _render_input_area(self) -> None:
        """Render message input area with validation."""
        with st.container():
            st.markdown("### 💬 Ask the Agent")
            
            # Input form
            with st.form(key="chat_input_form", clear_on_submit=True):
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    user_input = st.text_area(
                        "Your question:",
                        placeholder="e.g., Calculate the integral of x² from 0 to 5 and show the plot",
                        height=100,
                        help="Ask any question about integral calculus, plotting, or mathematical analysis"
                    )
                
                with col2:
                    st.markdown("<br>", unsafe_allow_html=True)  # Spacing
                    submit_button = st.form_submit_button(
                        "🚀 Send",
                        type="primary",
                        disabled=self.state_manager.state.processing
                    )
                
                # Quick action buttons
                st.markdown("**Quick Examples:**")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.form_submit_button("📊 Simple Integral"):
                        user_input = "Calculate the integral of x² from 0 to 3"
                        submit_button = True
                
                with col2:
                    if st.form_submit_button("📈 With Plot"):
                        user_input = "Calculate ∫sin(x)dx from 0 to π and show the plot"
                        submit_button = True
                
                with col3:
                    if st.form_submit_button("🔍 Complex Analysis"):
                        user_input = "Analyze the integral of e^(-x²) from -∞ to +∞"
                        submit_button = True
                
                # Process input
                if submit_button and user_input.strip():
                    self._process_user_input(user_input.strip())
    
    def _render_status_bar(self) -> None:
        """Render status bar with processing information."""
        if self.state_manager.state.processing:
            with st.container():
                loading_html = StyleManager.create_loading_indicator("Agent is thinking...")
                st.markdown(loading_html, unsafe_allow_html=True)
        
        # Error display
        if self.state_manager.state.last_error:
            st.error(self.formatters.format_error_message(self.state_manager.state.last_error))
            
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("🔄 Retry"):
                    self.state_manager.clear_error()
                    st.rerun()
            
            with col2:
                if st.button("❌ Dismiss"):
                    self.state_manager.clear_error()
                    st.rerun()
    
    def _process_user_input(self, user_input: str) -> None:
        """Process user input through the agent."""
        # Add user message
        self.state_manager.add_message('user', user_input)
        
        # Set processing state
        self.state_manager.set_processing(True)
        
        # Clear any previous errors
        self.state_manager.clear_error()
        
        # Trigger rerun to show processing state
        st.rerun()
    
    def _clear_conversation(self) -> None:
        """Clear conversation history."""
        self.state_manager.clear_messages()
        self.state_manager.update_state(conversation_id=None)
        st.success("Conversation cleared!")
        st.rerun()
    
    async def process_with_agent(self, user_input: str) -> None:
        """Process user input with the mathematical agent (async)."""
        try:
            # Get agent instance
            agent = self.state_manager.state.agent_instance
            if not agent:
                raise Exception("Agent not initialized")
            
            # Get conversation context
            context = self.state_manager.get_conversation_context()
            
            # Process with agent
            start_time = datetime.now()
            result = await agent.process_message(user_input, context)
            end_time = datetime.now()
            
            # Calculate metadata
            execution_time = (end_time - start_time).total_seconds()
            metadata = {
                'execution_time': execution_time,
                'timestamp': end_time.isoformat(),
                'tools_used': result.get('tools_used', []),
                'token_usage': result.get('token_usage', {})
            }
            
            # Add agent response
            response_content = result.get('response', 'No response generated')
            self.state_manager.add_message('assistant', response_content, metadata)
            
            # Update state
            self.state_manager.set_processing(False)
            
        except Exception as e:
            self.state_manager.set_error(f"Agent processing failed: {str(e)}")
            self.state_manager.set_processing(False)
