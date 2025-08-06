import streamlit as st
from typing import Optional

from app.ui.components import ChatComponent, SidebarComponent
from app.ui.state import get_state_manager, UIState
from app.ui.utils import StyleManager
from app.core import get_logger

logger = get_logger(__name__)


class MainChatPage:
    def __init__(self):
        self.state_manager = get_state_manager()
        self.chat_component = ChatComponent()
        self.sidebar_component = SidebarComponent()
    
    def render(self) -> None:
        try:
            StyleManager.inject_global_styles()
            self._ensure_app_initialization()
            self._render_sidebar()
            self._render_main_content()
        except Exception as e:
            logger.error(f"Error rendering main chat page: {e}", exc_info=True)
            self._render_error_fallback(str(e))
    
    def _ensure_app_initialization(self) -> None:
        if self.state_manager.state.ui_state == UIState.INITIALIZING:
            if (hasattr(st.session_state, 'tool_registry') and 
                st.session_state.tool_registry is not None):
                self.state_manager.update_state(
                    tool_registry=st.session_state.tool_registry
                )
                self.state_manager.set_ui_state(UIState.READY)
                logger.info("Application initialized and marked as ready")
            else:
                self._render_initialization_state()
                st.stop()
    
    def _render_sidebar(self) -> None:
        try:
            self.sidebar_component.render()
        except Exception as e:
            logger.error(f"Error rendering sidebar: {e}")
            with st.sidebar:
                st.error("❌ Sidebar error")
                st.code(str(e), language="text")
    
    def _render_main_content(self) -> None:
        current_state = self.state_manager.state.ui_state
        if current_state == UIState.ERROR:
            self._render_error_state()
        elif current_state == UIState.INITIALIZING:
            self._render_initialization_state()
        else:
            self._render_chat_interface()
    
    def _render_chat_interface(self) -> None:
        try:
            self._render_page_header()
            self.chat_component.render()
        except Exception as e:
            logger.error(f"Error rendering chat interface: {e}")
            st.error(f"❌ Chat interface error: {str(e)}")
    
    def _render_page_header(self) -> None:
        st.markdown("""
        <div style="text-align: center; padding: 20px 0; margin-bottom: 30px;">
            <h1 style="color: var(--primary-color); margin: 0; font-size: 2.5rem;">
                🤖 Mathematical Agent
            </h1>
            <p style="color: var(--text-secondary); margin: 10px 0 0 0; font-size: 1.2rem;">
                Professional AI-powered mathematical reasoning and computation
            </p>
            <div style="margin-top: 15px;">
                <span style="background: linear-gradient(135deg, var(--primary-color), var(--secondary-color)); 
                           color: white; padding: 5px 15px; border-radius: 20px; font-size: 0.9rem;">
                    🏗️ Clean Architecture • 🧠 LangGraph • 🔧 BigTool • ⚡ Optimized
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_initialization_state(self) -> None:
        st.markdown("""
        <div style="text-align: center; padding: 60px 20px;">
            <div class="loading-spinner" style="margin: 0 auto 20px; width: 40px; height: 40px;"></div>
            <h2 style="color: var(--primary-color);">🔧 Initializing Mathematical Agent</h2>
            <p style="color: var(--text-secondary); margin: 20px 0;">
                Setting up professional-grade mathematical tools and AI models...
            </p>
            <div style="background: var(--bg-tertiary); border-radius: 10px; padding: 20px; margin: 20px 0; text-align: left; max-width: 500px; margin: 20px auto;">
                <h4 style="color: var(--primary-color); margin: 0 0 10px 0;">🏗️ Initialization Steps:</h4>
                <ul style="color: var(--text-primary); margin: 0; padding-left: 20px;">
                    <li>✅ Core application framework</li>
                    <li>✅ Database connection</li>
                    <li>🔄 Mathematical tool registry</li>
                    <li>⏳ AI model configuration</li>
                    <li>⏳ UI component initialization</li>
                </ul>
            </div>
            <p style="color: var(--text-secondary); font-size: 0.9rem;">
                This may take a few moments on first load...
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_error_state(self) -> None:
        error_msg = self.state_manager.state.error_message or "Unknown error occurred"
        st.markdown("""
        <div style="text-align: center; padding: 40px 20px;">
            <h2 style="color: var(--error-color);">🚨 Application Error</h2>
            <p style="color: var(--text-secondary);">
                The Mathematical Agent encountered an error and needs to recover.
            </p>
        </div>
        """, unsafe_allow_html=True)
        with st.expander("🔍 **Error Details**", expanded=True):
            st.code(error_msg, language="text")
            st.markdown("**🕐 Error Time:** " + str(self.state_manager.state.last_error_time or "Unknown"))
            st.markdown(f"**📊 Error Count:** {self.state_manager.state.error_count}")
        st.markdown("### 🔧 **Recovery Options**")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔄 **Retry**", use_container_width=True):
                self._retry_initialization()
        with col2:
            if st.button("🏠 **Reset App**", use_container_width=True):
                self._reset_application()
        with col3:
            if st.button("💬 **Clear Chat**", use_container_width=True):
                self.state_manager.clear_messages()
                self.state_manager.clear_error()
                st.rerun()
        st.info("""
        💡 **Troubleshooting Tips:**
        - Try the **Retry** option first
        - **Reset App** will clear all session data
        - **Clear Chat** removes conversation history only
        - If the problem persists, check the system logs
        """)
    
    def _render_error_fallback(self, error: str) -> None:
        st.error("🚨 **Critical Application Error**")
        st.code(error, language="text")
        if st.button("🔄 **Reload Page**"):
            st.rerun()
    
    def _retry_initialization(self) -> None:
        try:
            self.state_manager.clear_error()
            self.state_manager.set_ui_state(UIState.INITIALIZING)
            st.success("🔄 Retrying initialization...")
            st.rerun()
        except Exception as e:
            logger.error(f"Error during retry: {e}")
            st.error(f"❌ Retry failed: {str(e)}")
    
    def _reset_application(self) -> None:
        try:
            keys_to_keep = ['_state_manager']
            for key in list(st.session_state.keys()):
                if key not in keys_to_keep:
                    del st.session_state[key]
            st.success("🏠 Application reset successfully!")
            st.rerun()
        except Exception as e:
            logger.error(f"Error during reset: {e}")
            st.error(f"❌ Reset failed: {str(e)}")
