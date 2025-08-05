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
        <div style="padding: 20px; background-color: #f0f2f6; border-radius: 10px; margin: 10px 0;">
            <h4>🚀 Welcome to the Mathematical Agent!</h4>
            <p>I can help you with:</p>
            <ul>
                <li><strong>📊 Calculus:</strong> Integrals, derivatives, limits</li>
                <li><strong>📈 Analysis:</strong> Function behavior, critical points</li>
                <li><strong>📉 Visualization:</strong> Interactive plots and graphs</li>
                <li><strong>🔢 Computation:</strong> Symbolic and numerical calculations</li>
            </ul>
            <p><strong>Example:</strong> "Calculate the integral of x² from 0 to 3 and show me the area under the curve"</p>
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
        """Render a user message."""
        st.markdown(f"""
        <div style="display: flex; justify-content: flex-end; margin: 10px 0;">
            <div style="max-width: 80%; background-color: #007bff; color: white; 
                        padding: 10px 15px; border-radius: 15px 15px 5px 15px;">
                <div style="font-weight: 500;">{content}</div>
                <div style="font-size: 0.8em; opacity: 0.8; text-align: right; margin-top: 5px;">
                    {timestamp.strftime('%H:%M') if isinstance(timestamp, datetime) else timestamp}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_assistant_message(self, message: Dict[str, Any]) -> None:
        """Render an assistant message with rich content."""
        content = message.get("content", "")
        timestamp = message.get("timestamp", datetime.now())
        reasoning = message.get("reasoning", [])
        tools_used = message.get("tools_used", [])
        visualizations = message.get("visualizations", [])
        
        # Main response
        st.markdown(f"""
        <div style="display: flex; justify-content: flex-start; margin: 10px 0;">
            <div style="max-width: 80%; background-color: #f8f9fa; 
                        padding: 15px; border-radius: 15px 15px 15px 5px; 
                        border-left: 4px solid #28a745;">
                <div>{content}</div>
                <div style="font-size: 0.8em; color: #666; margin-top: 10px;">
                    🤖 Assistant • {timestamp.strftime('%H:%M') if isinstance(timestamp, datetime) else timestamp}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show reasoning steps if available
        if reasoning:
            self._render_reasoning_steps(reasoning)
        
        # Show tools used
        if tools_used:
            self._render_tools_used(tools_used)
        
        # Show visualizations
        if visualizations:
            self._render_visualizations(visualizations)
    
    def _render_reasoning_steps(self, reasoning: List[str]) -> None:
        """Render reasoning steps."""
        with st.expander("🧠 Reasoning Steps", expanded=False):
            for i, step in enumerate(reasoning, 1):
                st.markdown(f"**Step {i}:** {step}")
    
    def _render_tools_used(self, tools: List[str]) -> None:
        """Render tools used information."""
        if tools:
            st.markdown(f"**🔧 Tools Used:** {', '.join(tools)}")
    
    def _render_visualizations(self, visualizations: List[Dict[str, Any]]) -> None:
        """Render visualizations."""
        for viz in visualizations:
            viz_type = viz.get("type", "unknown")
            if viz_type == "matplotlib":
                if "figure" in viz:
                    st.pyplot(viz.get("figure"))
            elif viz_type == "plotly":
                if "figure" in viz:
                    st.plotly_chart(viz.get("figure"), use_container_width=True)
            elif viz_type == "image":
                if "data" in viz:
                    st.image(viz.get("data"), caption=viz.get("caption", ""))
    
    def _render_system_message(self, content: str) -> None:
        """Render a system message."""
        st.info(f"ℹ️ {content}")
    
    def _render_input_area(self) -> None:
        """Render the message input area."""
        # Create input form
        with st.form(key="chat_input", clear_on_submit=True):
            col1, col2 = st.columns([4, 1])
            
            with col1:
                user_input = st.text_area(
                    "Ask me anything about mathematics:",
                    placeholder="e.g., Calculate the integral of x² from 0 to 3",
                    height=100,
                    disabled=self.state_manager.state.ui_state == UIState.PROCESSING
                )
            
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)  # Spacing
                send_button = st.form_submit_button(
                    "Send 🚀",
                    disabled=self.state_manager.state.ui_state == UIState.PROCESSING,
                    use_container_width=True
                )
        
        # Handle message submission
        if send_button and user_input.strip():
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
