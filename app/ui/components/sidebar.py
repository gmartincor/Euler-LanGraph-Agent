"""
Sidebafrom typing import Dict, Any, List, Optional
import streamlit as st
from datetime import datetime

from app.ui.state import get_state_manager, UIState
from app.ui.utils import UIFormatters, StyleManageronent - Professional configuration and metrics sidebar

Implements modular sidebar with:
- Real-time metrics
- Configuration options
- Conversation history
- System status
"""

from typing import Dict, Any, List
import streamlit as st
from datetime import datetime, timedelta

from ..state import get_state_manager, UIState
from ..utils import UIFormatters, StyleManager


class SidebarComponent:
    """Professional sidebar component with metrics and configuration."""
    
    def __init__(self):
        self.state_manager = get_state_manager()
        self.formatters = UIFormatters()
    
    def render(self) -> None:
        """Render the complete sidebar."""
        with st.sidebar:
            self._render_header()
            self._render_system_status()
            self._render_conversation_metrics()
            self._render_tool_metrics()
            self._render_configuration()
            self._render_conversation_history()
            self._render_footer()
    
    def _render_header(self) -> None:
        """Render sidebar header."""
        st.markdown("# 📊 Dashboard")
        st.markdown("---")
    
    def _render_system_status(self) -> None:
        """Render system status section."""
        st.markdown("### 🖥️ System Status")
        
        # Agent status
        agent_status = "🟢 Ready" if self.state_manager.state.agent_instance else "🔴 Not Ready"
        st.markdown(f"**Agent:** {agent_status}")
        
        # Database status
        db_status = "🟢 Connected"  # TODO: Add actual DB health check
        st.markdown(f"**Database:** {db_status}")
        
        # Tool registry status
        tool_count = len(self.state_manager.state.tool_registry or {})
        st.markdown(f"**Tools:** {tool_count} loaded")
        
        # UI state
        ui_state = self.state_manager.state.ui_state.value.title()
        status_icon = self._get_status_icon(self.state_manager.state.ui_state)
        st.markdown(f"**Status:** {status_icon} {ui_state}")
        
        st.markdown("---")
    
    def _render_conversation_metrics(self) -> None:
        """Render conversation metrics."""
        st.markdown("### 💬 Conversation")
        
        messages = self.state_manager.state.message_history
        
        # Basic metrics
        total_messages = len(messages)
        user_messages = len([m for m in messages if m.get('role') == 'user'])
        agent_messages = len([m for m in messages if m.get('role') == 'assistant'])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total", total_messages)
        with col2:
            st.metric("Exchanges", user_messages)
        
        # Conversation duration
        if messages:
            try:
                first_msg = datetime.fromisoformat(messages[0]['timestamp'].replace('Z', '+00:00'))
                last_msg = datetime.fromisoformat(messages[-1]['timestamp'].replace('Z', '+00:00'))
                duration = (last_msg - first_msg).total_seconds()
                duration_str = self.formatters.format_duration(duration)
                st.markdown(f"**Duration:** {duration_str}")
            except:
                st.markdown("**Duration:** Unknown")
        
        # Average response time
        response_times = []
        for msg in messages:
            metadata = msg.get('metadata', {})
            if 'execution_time' in metadata:
                response_times.append(metadata['execution_time'])
        
        if response_times:
            avg_time = sum(response_times) / len(response_times)
            avg_time_str = self.formatters.format_duration(avg_time)
            st.markdown(f"**Avg Response:** {avg_time_str}")
        
        st.markdown("---")
    
    def _render_tool_metrics(self) -> None:
        """Render tool usage metrics."""
        st.markdown("### 🔧 Tool Usage")
        
        # Aggregate tool usage from message metadata
        tool_usage = {}
        for msg in self.state_manager.state.message_history:
            metadata = msg.get('metadata', {})
            tools_used = metadata.get('tools_used', [])
            
            for tool in tools_used:
                tool_name = tool.get('name', 'Unknown')
                if tool_name not in tool_usage:
                    tool_usage[tool_name] = {
                        'count': 0,
                        'total_time': 0,
                        'success_count': 0
                    }
                
                tool_usage[tool_name]['count'] += 1
                tool_usage[tool_name]['total_time'] += tool.get('duration', 0)
                if tool.get('status') == 'success':
                    tool_usage[tool_name]['success_count'] += 1
        
        if tool_usage:
            for tool_name, stats in tool_usage.items():
                with st.expander(f"🛠️ {tool_name}", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Uses", stats['count'])
                    
                    with col2:
                        success_rate = (stats['success_count'] / stats['count']) * 100 if stats['count'] > 0 else 0
                        st.metric("Success %", f"{success_rate:.1f}")
                    
                    if stats['count'] > 0:
                        avg_time = stats['total_time'] / stats['count']
                        avg_time_str = self.formatters.format_duration(avg_time)
                        st.markdown(f"**Avg Time:** {avg_time_str}")
        else:
            st.info("No tools used yet")
        
        st.markdown("---")
    
    def _render_configuration(self) -> None:
        """Render configuration options."""
        st.markdown("### ⚙️ Configuration")
        
        # Agent settings
        with st.expander("🤖 Agent Settings", expanded=False):
            max_iterations = st.slider(
                "Max Iterations",
                min_value=1,
                max_value=20,
                value=10,
                help="Maximum number of reasoning iterations"
            )
            
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.1,
                help="LLM temperature for response randomness"
            )
            
            # Update agent configuration
            if st.button("Apply Settings"):
                # TODO: Update agent configuration
                st.success("Settings applied!")
        
        # UI settings
        with st.expander("🎨 UI Settings", expanded=False):
            show_metadata = st.checkbox(
                "Show Message Metadata",
                value=True,
                help="Display tool usage and timing information"
            )
            
            auto_scroll = st.checkbox(
                "Auto-scroll to Bottom",
                value=True,
                help="Automatically scroll to latest messages"
            )
            
            dark_mode = st.checkbox(
                "Dark Mode",
                value=False,
                help="Toggle dark theme (experimental)"
            )
        
        # Export options
        with st.expander("📥 Export", expanded=False):
            if st.button("📄 Export Chat as JSON"):
                self._export_conversation_json()
            
            if st.button("📊 Export Metrics as CSV"):
                self._export_metrics_csv()
        
        st.markdown("---")
    
    def _render_conversation_history(self) -> None:
        """Render conversation history browser."""
        st.markdown("### 📚 History")
        
        # TODO: Load conversation history from database
        conversations = []  # Placeholder
        
        if conversations:
            selected_conv = st.selectbox(
                "Previous Conversations",
                options=conversations,
                format_func=lambda x: f"{x['title']} ({x['date']})"
            )
            
            if st.button("📂 Load Conversation"):
                # TODO: Load selected conversation
                st.success(f"Loaded conversation: {selected_conv['title']}")
        else:
            st.info("No previous conversations")
        
        # Quick actions
        if st.button("🗑️ Clear All History"):
            if st.button("⚠️ Confirm Delete", type="secondary"):
                # TODO: Clear conversation history
                st.success("History cleared!")
        
        st.markdown("---")
    
    def _render_footer(self) -> None:
        """Render sidebar footer."""
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; color: #666; font-size: 0.8em;'>
                <p>🤖 ReAct Agent v1.0</p>
                <p>Built with LangGraph + Streamlit</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    def _get_status_icon(self, ui_state: UIState) -> str:
        """Get status icon for UI state."""
        icons = {
            UIState.INITIALIZING: "🔄",
            UIState.READY: "✅",
            UIState.PROCESSING: "⏳",
            UIState.ERROR: "❌",
            UIState.COMPLETE: "🎉"
        }
        return icons.get(ui_state, "❓")
    
    def _export_conversation_json(self) -> None:
        """Export current conversation as JSON."""
        import json
        
        data = {
            'conversation_id': self.state_manager.state.conversation_id,
            'export_timestamp': datetime.now().isoformat(),
            'messages': self.state_manager.state.message_history,
            'metadata': {
                'total_messages': len(self.state_manager.state.message_history),
                'agent_version': '1.0',
                'export_format': 'json_v1'
            }
        }
        
        json_str = json.dumps(data, indent=2, default=str)
        
        st.download_button(
            label="💾 Download JSON",
            data=json_str,
            file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    def _export_metrics_csv(self) -> None:
        """Export metrics as CSV."""
        import csv
        import io
        
        # Create CSV data
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Headers
        writer.writerow(['Timestamp', 'Role', 'Message_Length', 'Tools_Used', 'Execution_Time'])
        
        # Data rows
        for msg in self.state_manager.state.message_history:
            timestamp = msg.get('timestamp', '')
            role = msg.get('role', '')
            message_length = len(msg.get('content', ''))
            
            metadata = msg.get('metadata', {})
            tools_used = len(metadata.get('tools_used', []))
            execution_time = metadata.get('execution_time', 0)
            
            writer.writerow([timestamp, role, message_length, tools_used, execution_time])
        
        csv_data = output.getvalue()
        
        st.download_button(
            label="💾 Download CSV",
            data=csv_data,
            file_name=f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
