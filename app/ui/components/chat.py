"""
Professional Chat Component - Clean Architecture Implementation

This module provides a professional chat interface component for the mathematical agent,
following clean architecture principles and UI design patterns.

Key Design Principles Applied:
- Single Responsibility: Focus only on chat interface
- Separation of Concerns: UI logic separated from business logic
- DRY Principle: Reusable message rendering and input handling
- KISS Principle: Simple, intuitive user interface
- Professional Error Handling: User-friendly error messages
"""

import streamlit as st
from typing import Optional, Dict, Any, List
from datetime import datetime

from app.core.logging import get_logger
from app.core.exceptions import AgentError, ValidationError
from app.core.agent_controller import get_agent_controller
from app.ui.state import get_state_manager, UIState

logger = get_logger(__name__)


class ChatComponent:
    """
    Professional chat interface component.
    
    Provides a clean, user-friendly interface for interacting with the mathematical agent.
    Handles message display, user input, and error states professionally.
    """
    
    def __init__(self):
        self.state_manager = get_state_manager()
    
    def render(self) -> None:
        """Render the complete chat interface."""
        self._render_chat_header()
        self._render_chat_history()
        self._render_input_area()
        self._handle_processing_state()
    
    def _render_chat_header(self) -> None:
        """Render the chat header with status information."""
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.markdown("### 💬 Mathematical Conversation")
        
        with col2:
            # Show processing status
            if self.state_manager.state.ui_state == UIState.PROCESSING:
                st.markdown("🤖 **Thinking...**")
            elif self.state_manager.state.ui_state == UIState.READY:
                st.markdown("✅ **Ready**")
            elif self.state_manager.state.ui_state == UIState.ERROR:
                st.markdown("❌ **Error**")
        
        with col3:
            if st.button("🗑️ Clear", help="Clear conversation history"):
                self.clear_chat()
        
        # Show error if present
        if self.state_manager.state.ui_state == UIState.ERROR:
            if self.state_manager.state.error_message:
                st.error(f"❌ {self.state_manager.state.error_message}")
    
    def _render_chat_history(self) -> None:
        """Render the conversation history."""
        messages = self.state_manager.state.message_history
        
        if not messages:
            self._render_welcome_message()
            return
        
        # Create scrollable chat container
        chat_container = st.container()
        
        with chat_container:
            for message in messages:
                self._render_message(message)
    
    def _render_welcome_message(self) -> None:
        """Render welcome message for new conversations."""
        st.markdown("""
        <div class="welcome-message">
            <h4>🚀 Welcome to the Mathematical Agent!</h4>
            <p>I can help you with advanced mathematical problems using professional-grade AI reasoning:</p>
            <ul>
                <li><strong>📊 Calculus:</strong> Integrals, derivatives, limits, optimization</li>
                <li><strong>📈 Analysis:</strong> Function behavior, critical points, asymptotes</li>
                <li><strong>📉 Visualization:</strong> Interactive plots, graphs, and area calculations</li>
                <li><strong>🔢 Computation:</strong> Symbolic and numerical calculations</li>
                <li><strong>🏗️ Step-by-step Solutions:</strong> Detailed mathematical reasoning</li>
            </ul>
            <p><strong>🎯 Example:</strong> "Calculate the integral of x² from 0 to 3 and show me the area under the curve"</p>
            <p><em>Type your mathematical question below to get started!</em></p>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_message(self, message: Dict[str, Any]) -> None:
        """
        Render a single message in the chat.
        
        Args:
            message: Message dictionary with role, content, and metadata
        """
        role = message.get("role", "user")
        content = message.get("content", "")
        timestamp = message.get("timestamp", datetime.now())
        
        if role == "user":
            self._render_user_message(content, timestamp)
        elif role == "assistant":
            self._render_assistant_message(message)
        elif role == "system":
            self._render_system_message(content)
    
    def _render_user_message(self, content: str, timestamp: datetime) -> None:
        """Render a user message with professional styling."""
        timestamp_str = timestamp.strftime('%H:%M') if isinstance(timestamp, datetime) else str(timestamp)
        
        st.markdown(f"""
        <div style="display: flex; justify-content: flex-end; margin: 16px 0;">
            <div class="chat-message user">
                <div style="font-weight: 500; line-height: 1.5;">{content}</div>
                <div style="font-size: 0.8em; opacity: 0.9; text-align: right; margin-top: 8px;">
                    👤 You • {timestamp_str}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_assistant_message(self, message: Dict[str, Any]) -> None:
        """Render an assistant message with rich content and professional styling."""
        content = message.get("content", "")
        timestamp = message.get("timestamp", datetime.now())
        reasoning = message.get("reasoning", [])
        tools_used = message.get("tools_used", [])
        visualizations = message.get("visualizations", [])
        metadata = message.get("metadata", {})
        
        # Debug logging to see what content we're getting
        logger.info(f"Rendering assistant message with content: {repr(content)}")
        
        # Handle None or empty content
        if content is None:
            content = "No response received from the agent."
        elif not isinstance(content, str):
            content = str(content)
        
        # Escape HTML special characters to prevent breaking the markup
        import html
        escaped_content = html.escape(content)
        
        timestamp_str = timestamp.strftime('%H:%M') if isinstance(timestamp, datetime) else str(timestamp)
        
        # Main assistant response using Streamlit markdown directly (safer than HTML)
        st.markdown(f"**🤖 Mathematical Agent** • {timestamp_str}")
        
        # Display content in a clean way
        if escaped_content.strip():
            st.markdown(escaped_content)
        else:
            st.warning("⚠️ No response content received")
        
        # Show execution time if available
        if metadata.get("execution_time"):
            st.caption(f"⚡ Processed in {metadata.get('execution_time', 0):.2f}s")
        
        # Show additional content in expandable sections
        self._render_additional_content(reasoning, tools_used, visualizations, metadata)
    
    def _render_additional_content(self, reasoning: list, tools_used: list, 
                                 visualizations: list, metadata: dict) -> None:
        """Render additional content like reasoning steps, tools, and visualizations."""
        
        # Create columns for expandable content
        if reasoning or tools_used or visualizations:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if reasoning:
                    with st.expander("🧠 **Reasoning Steps**", expanded=False):
                        for i, step in enumerate(reasoning, 1):
                            st.markdown(f"**{i}.** {step}")
            
            with col2:
                if tools_used:
                    with st.expander("🔧 **Tools Used**", expanded=False):
                        for tool in tools_used:
                            if isinstance(tool, dict):
                                st.markdown(f"• **{tool.get('name', 'Unknown Tool')}**")
                                if tool.get('duration'):
                                    st.markdown(f"  ⏱️ {tool['duration']:.2f}s")
                                if tool.get('status'):
                                    status_icon = "✅" if tool['status'] == 'success' else "❌"
                                    st.markdown(f"  {status_icon} {tool['status']}")
                            else:
                                st.markdown(f"• {tool}")
            
            with col3:
                if metadata.get('token_usage'):
                    with st.expander("📊 **Usage Stats**", expanded=False):
                        usage = metadata['token_usage']
                        st.metric("Tokens", usage.get('total', 0))
                        st.metric("Prompt", usage.get('prompt', 0))
                        st.metric("Response", usage.get('completion', 0))
        
        # Render visualizations
        if visualizations:
            st.markdown("### 📈 **Visualizations**")
            for viz in visualizations:
                self._render_single_visualization(viz)
    
    def _render_single_visualization(self, viz: Dict[str, Any]) -> None:
        """Render a single visualization."""
        viz_type = viz.get("type", "unknown")
        
        if viz_type == "matplotlib":
            if "figure" in viz:
                st.pyplot(viz["figure"], use_container_width=True)
        elif viz_type == "plotly":
            if "figure" in viz:
                st.plotly_chart(viz["figure"], use_container_width=True)
        elif viz_type == "image":
            if "data" in viz:
                st.image(viz["data"], caption=viz.get("caption", ""), use_column_width=True)
        else:
            st.info(f"📊 Visualization of type '{viz_type}' not supported yet.")
    
    def _render_system_message(self, content: str) -> None:
        """Render a system message."""
        st.info(f"ℹ️ {content}")
    
    def _render_input_area(self) -> None:
        """Render the professional message input area - Single, clean interface."""
        # Create a container for the input area with professional styling
        st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
        
        # Show processing status if agent is working
        if self.state_manager.state.ui_state == UIState.PROCESSING:
            st.markdown("""
            <div style="text-align: center; padding: 20px; color: var(--primary-color);">
                <div class="loading-spinner"></div>
                🤖 Agent is processing your request...
                <div style="margin-top: 10px; font-size: 0.9em; color: var(--text-secondary);">
                    This may take a few moments for complex calculations.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Single, professional input form
            with st.form(key="mathematical_chat_input", clear_on_submit=True):
                # Main input area
                user_input = st.text_area(
                    label="🧮 **Ask me about mathematics:**",
                    placeholder="e.g., Calculate the integral of x³ + 2x² - 5x + 1 from 0 to 4 and visualize the area under the curve",
                    height=120,
                    disabled=self.state_manager.state.ui_state == UIState.PROCESSING,
                    help="Enter your mathematical question. I can handle calculus, algebra, geometry, and more!"
                )
                
                # Input controls row
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    # Quick examples
                    st.markdown("**Quick examples:** *integral*, *derivative*, *plot function*, *solve equation*")
                
                with col2:
                    # Character count
                    char_count = len(user_input) if user_input else 0
                    max_chars = 1000
                    color = "red" if char_count > max_chars else "green"
                    st.markdown(f'<small style="color: {color}">{char_count}/{max_chars} chars</small>', 
                               unsafe_allow_html=True)
                
                with col3:
                    # Send button
                    send_button = st.form_submit_button(
                        "🚀 **Solve**",
                        disabled=(self.state_manager.state.ui_state == UIState.PROCESSING or 
                                char_count > max_chars),
                        use_container_width=True,
                        type="primary"
                    )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Handle message submission
        if 'send_button' in locals() and send_button and user_input and user_input.strip():
            if len(user_input.strip()) > 1000:
                st.error("❌ Message too long. Please keep it under 1000 characters.")
            else:
                self._process_user_message(user_input.strip())
    
    def _process_user_message(self, message: str) -> None:
        """
        Process a user message through the agent.
        
        Args:
            message: User input message
        """
        try:
            # Update UI state
            self.state_manager.set_ui_state(UIState.PROCESSING)
            
            # Add user message to history
            self.state_manager.add_message({
                "role": "user",
                "content": message,
                "timestamp": datetime.now()
            })
            
            # Get agent controller and process message
            session_id = self.state_manager.state.session_id
            controller = get_agent_controller(session_id)
            
            # Process message
            result = controller.process_message(
                message, 
                context=self.state_manager.state.message_history[-10:]  # Last 10 messages as context
            )
            
            # Add assistant response
            self.state_manager.add_message({
                "role": "assistant",
                "content": result.get("response", "No response received"),
                "timestamp": datetime.now(),
                "reasoning": result.get("reasoning", []),
                "tools_used": result.get("tools_used", []),
                "visualizations": result.get("visualizations", []),
                "metadata": result.get("metadata", {})
            })
            
            # Update UI state
            self.state_manager.set_ui_state(UIState.READY)
            
            # Force rerun to show new message
            st.rerun()
            
        except Exception as e:
            error_msg = f"Agent error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Set error state
            self.state_manager.set_error(error_msg)
            self.state_manager.set_ui_state(UIState.READY)
            
            # Force rerun to show error
            st.rerun()
    
    def _handle_processing_state(self) -> None:
        """Handle the processing state display."""
        if self.state_manager.state.ui_state == UIState.PROCESSING:
            # Show progress indicator
            with st.empty():
                st.markdown("""
                <div style="text-align: center; padding: 20px;">
                    <div style="font-size: 1.2em; color: #007bff;">
                        🤖 Agent is processing your request...
                    </div>
                    <div style="margin-top: 10px; color: #666;">
                        This may take a few moments for complex calculations.
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    def clear_chat(self) -> None:
        """Clear the chat history."""
        self.state_manager.clear_messages()
        st.rerun()
