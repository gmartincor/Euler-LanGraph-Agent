"""
Main Chat Page - Primary application interface

Orchestrates the main chat interface with modular components.
"""

import streamlit as st
import asyncio
from typing import Optional

from app.ui.components import ChatComponent, SidebarComponent
from app.ui.state import get_state_manager, UIState
from app.ui.utils import StyleManager


class MainChatPage:
    """Main chat page orchestrator with professional layout."""
    
    def __init__(self):
        self.state_manager = get_state_manager()
        self.chat_component = ChatComponent()
        self.sidebar_component = SidebarComponent()
    
    def render(self) -> None:
        """Render the complete main chat page."""
        # Apply global styles
        StyleManager.inject_global_styles()
        
        # Initialize application state
        self._ensure_initialization()
        
        # Render sidebar
        self.sidebar_component.render()
        
        # Render main content
        self._render_main_content()
        
        # Handle async processing
        self._handle_async_processing()
    
    def _ensure_initialization(self) -> None:
        """Ensure application is properly initialized."""
        if self.state_manager.state.ui_state == UIState.INITIALIZING:
            # Check if all components are ready
            if (hasattr(st.session_state, 'tool_registry') and 
                st.session_state.tool_registry is not None):
                
                # Agent should be initialized
                if not self.state_manager.state.agent_instance:
                    try:
                        # TODO: Initialize agent instance
                        # agent = create_mathematical_agent(st.session_state.tool_registry)
                        # self.state_manager.update_state(agent_instance=agent)
                        pass
                    except Exception as e:
                        self.state_manager.set_error(f"Agent initialization failed: {e}")
                        return
                
                # Mark as ready
                self.state_manager.set_ui_state(UIState.READY)
            else:
                st.warning("⏳ Initializing application components...")
                st.stop()
    
    def _render_main_content(self) -> None:
        """Render main content area."""
        # Error state handling
        if self.state_manager.state.ui_state == UIState.ERROR:
            self._render_error_state()
            return
        
        # Main chat interface
        self.chat_component.render()
    
    def _render_error_state(self) -> None:
        """Render error state with recovery options."""
        st.error("🚨 Application Error")
        
        error_msg = self.state_manager.state.last_error
        if error_msg:
            st.code(error_msg, language="text")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("🔄 Retry"):
                self.state_manager.clear_error()
                self.state_manager.set_ui_state(UIState.INITIALIZING)
                st.rerun()
        
        with col2:
            if st.button("🏠 Reset"):
                # Clear all state and restart
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        with col3:
            st.info("💡 Try reloading the page if the error persists")
    
    def _handle_async_processing(self) -> None:
        """Handle async processing of user messages."""
        # Check if we have a pending user message to process
        messages = self.state_manager.state.message_history
        
        if (messages and 
            self.state_manager.state.processing and
            messages[-1].get('role') == 'user'):
            
            # Get the last user message
            user_message = messages[-1].get('content', '')
            
            # Process asynchronously (simulated for now)
            self._simulate_agent_processing(user_message)
    
    def _simulate_agent_processing(self, user_message: str) -> None:
        """Simulate agent processing (placeholder for real implementation)."""
        import time
        import random
        
        # Simulate processing delay
        time.sleep(1)
        
        # Generate mock response
        responses = [
            f"I understand you want to work with: '{user_message}'. Let me analyze this mathematical problem...",
            f"Processing your request: '{user_message}'. I'll use the appropriate mathematical tools...",
            f"Analyzing: '{user_message}'. This appears to be an integral calculus problem..."
        ]
        
        response = random.choice(responses)
        
        # Add mock tool usage metadata
        metadata = {
            'execution_time': random.uniform(0.5, 3.0),
            'tools_used': [
                {
                    'name': 'IntegralTool',
                    'duration': random.uniform(0.2, 1.0),
                    'status': 'success'
                }
            ],
            'token_usage': {
                'prompt': random.randint(50, 200),
                'completion': random.randint(100, 500),
                'total': random.randint(150, 700)
            }
        }
        
        # Add agent response
        self.state_manager.add_message('assistant', response, metadata)
        self.state_manager.set_processing(False)
        
        # Trigger rerun to show response
        st.rerun()
