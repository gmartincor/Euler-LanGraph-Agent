from typing import Dict, Any, List, Optional
import streamlit as st
from datetime import datetime, timedelta

from app.ui.state import get_state_manager, UIState
from app.ui.utils import UIFormatters, StyleManager, ComponentBuilder


class SidebarComponent:
    """Professional sidebar component with metrics and configuration."""
    
    def __init__(self):
        self.state_manager = get_state_manager()
        self.formatters = UIFormatters()
    
    def render(self) -> None:
        with st.sidebar:
            self._render_header()
            self._render_agent_status()
            self._render_conversation_metrics() 
            self._render_agent_configuration()
            self._render_tool_insights()
            self._render_conversation_management()
            self._render_system_info()
    
    def _render_header(self) -> None:
        st.markdown("""
        <div style="text-align: center; padding: 20px 0; border-bottom: 1px solid var(--border-color);">
            <h2 style="color: var(--primary-color); margin: 0;">🤖 Agent Dashboard</h2>
            <p style="color: var(--text-secondary); margin: 5px 0 0 0;">Mathematical Intelligence</p>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_agent_status(self) -> None:
        st.markdown("### 🔋 **Agent Status**")
        agent_ready = bool(self.state_manager.state.agent_instance)
        agent_status = ComponentBuilder.create_status_badge(
            "ready" if agent_ready else "error",
            "🟢 Ready" if agent_ready else "🔴 Initializing"
        )
        st.markdown(f"**Agent:** {agent_status}", unsafe_allow_html=True)
        ui_state = self.state_manager.state.ui_state
        ui_status = ComponentBuilder.create_status_badge(
            ui_state.value,
            f"{self._get_status_icon(ui_state)} {ui_state.value.title()}"
        )
        st.markdown(f"**Status:** {ui_status}", unsafe_allow_html=True)
        tool_count = len(self.state_manager.state.tool_registry or {})
        st.markdown(f"**Tools:** 🔧 {tool_count} available")
        st.markdown("---")
    
    def _render_conversation_metrics(self) -> None:
        st.markdown("### 💬 **Conversation Metrics**")
        messages = self.state_manager.state.message_history
        total_messages = len(messages)
        user_messages = len([m for m in messages if m.get('role') == 'user'])
        agent_messages = len([m for m in messages if m.get('role') == 'assistant'])
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(ComponentBuilder.create_metric_card(
                "Total Messages", str(total_messages)
            ), unsafe_allow_html=True)
        with col2:
            st.markdown(ComponentBuilder.create_metric_card(
                "Exchanges", str(user_messages)
            ), unsafe_allow_html=True)
        if messages and len(messages) > 1:
            try:
                first_msg = messages[0]['timestamp']
                last_msg = messages[-1]['timestamp']
                if isinstance(first_msg, datetime) and isinstance(last_msg, datetime):
                    duration = (last_msg - first_msg).total_seconds()
                    duration_str = self.formatters.format_duration(duration)
                    st.markdown(f"**🕐 Duration:** {duration_str}")
            except:
                st.markdown("**🕐 Duration:** Active session")
        st.markdown("---")
    
    def _render_agent_configuration(self) -> None:
        st.markdown("### ⚙️ **Agent Configuration**")
        current_session = self.state_manager.state.session_id[:8] + "..."
        st.markdown(f"**Session ID:** `{current_session}`")
        if st.button("🔄 **New Session**", use_container_width=True):
            self._start_new_session()
        st.markdown("**Response Preferences:**")
        show_steps = st.checkbox(
            "📝 Show detailed steps",
            value=True,
            help="Display step-by-step mathematical reasoning"
        )
        show_viz = st.checkbox(
            "📊 Generate visualizations", 
            value=True,
            help="Create graphs and plots when applicable"
        )
        viz_style = st.selectbox(
            "🎨 Plot style",
            options=["plotly", "matplotlib"],
            index=0,
            help="Choose visualization library"
        )
        self.state_manager.update_state(
            show_detailed_steps=show_steps,
            show_visualizations=show_viz,
            visualization_style=viz_style
        )
        st.markdown("---")
    
    def _render_tool_insights(self) -> None:
        st.markdown("### 🔧 **Tool Insights**")
        tool_registry = self.state_manager.state.tool_registry
        if not tool_registry:
            st.info("🔄 Tools loading...")
            return
        try:
            if hasattr(tool_registry, 'get_registry_stats'):
                stats = tool_registry.get_registry_stats()
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("🛠️ Tools", stats.get("total_tools", 0))
                with col2:
                    st.metric("📂 Categories", stats.get("total_categories", 0))
                most_used = stats.get("most_used_tools", [])
                if most_used:
                    st.markdown("**🔥 Most Used:**")
                    for tool_name, count in most_used[:3]:
                        st.markdown(f"• {tool_name}: {count} uses")
            else:
                st.markdown("📊 Tool analytics available")
        except Exception as e:
            st.warning(f"Tool stats unavailable: {str(e)[:50]}...")
        st.markdown("---")
    
    def _render_conversation_management(self) -> None:
        st.markdown("### 💾 **Conversation**")
        messages = self.state_manager.state.message_history
        if messages:
            st.markdown(f"**Messages:** {len(messages)}")
            if st.button("📥 **Export Chat**", use_container_width=True):
                self._export_conversation()
            if st.button("🗑️ **Clear History**", use_container_width=True):
                self._clear_conversation()
        else:
            st.info("💬 No conversation yet. Start chatting!")
        st.markdown("---")
    
    def _render_system_info(self) -> None:
        with st.expander("ℹ️ **System Information**"):
            try:
                from app.core import get_settings
                settings = get_settings()
                st.json({
                    "🏗️ Architecture": "Clean + Modular",
                    "🤖 Agent": "LangGraph Mathematical",
                    "🔧 Tools": "BigTool Integration",
                    "🧠 LLM": settings.gemini_model_name,
                    "🌍 Environment": settings.environment,
                    "📦 Version": settings.app_version,
                    "🎯 UI": "Streamlit Professional"
                })
            except Exception as e:
                st.error(f"System info unavailable: {e}")
    
    def _get_status_icon(self, ui_state: UIState) -> str:
        icons = {
            UIState.INITIALIZING: "🔄",
            UIState.READY: "✅",
            UIState.PROCESSING: "⚡",
            UIState.ERROR: "❌",
            UIState.COMPLETE: "🎉"
        }
        return icons.get(ui_state, "❓")
    
    def _start_new_session(self) -> None:
        for key in list(st.session_state.keys()):
            if not key.startswith('_'):
                del st.session_state[key]
        st.success("🎉 New session started!")
        st.rerun()
    
    def _export_conversation(self) -> None:
        messages = self.state_manager.state.message_history
        export_content = []
        export_content.append("# Mathematical Agent Conversation Export")
        export_content.append(f"**Session ID:** {self.state_manager.state.session_id}")
        export_content.append(f"**Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        export_content.append("---\n")
        for i, msg in enumerate(messages, 1):
            role = msg.get('role', 'unknown').title()
            content = msg.get('content', '')
            timestamp = msg.get('timestamp', 'unknown')
            export_content.append(f"## Message {i} - {role}")
            export_content.append(f"**Time:** {timestamp}")
            export_content.append(f"**Content:**\n{content}\n")
            if msg.get('tools_used'):
                export_content.append(f"**Tools Used:** {', '.join(msg['tools_used'])}")
            export_content.append("---\n")
        export_text = "\n".join(export_content)
        st.download_button(
            label="📄 Download as Markdown",
            data=export_text.encode('utf-8'),
            file_name=f"mathematical_agent_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )
    
    def _clear_conversation(self) -> None:
        self.state_manager.clear_messages()
        st.success("🧹 Conversation cleared!")
        st.rerun()
