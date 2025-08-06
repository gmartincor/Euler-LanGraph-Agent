import streamlit as st
from typing import Optional

class StyleManager:
    COLORS = {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e', 
        'success': '#2ca02c',
        'warning': '#ff9800',
        'error': '#d62728',
        'info': '#17a2b8',
        'light': '#f8f9fa',
        'dark': '#343a40',
        'muted': '#6c757d'
    }
    FONT_SIZES = {
        'xs': '0.75rem',
        'sm': '0.875rem',
        'base': '1rem',
        'lg': '1.125rem',
        'xl': '1.25rem',
        '2xl': '1.5rem',
        '3xl': '1.875rem'
    }
    @classmethod
    def inject_global_styles(cls) -> None:
        css = f"""
        <style>
        :root {{
            --primary-color: {cls.COLORS['primary']};
            --secondary-color: {cls.COLORS['secondary']};
            --success-color: {cls.COLORS['success']};
            --warning-color: {cls.COLORS['warning']};
            --error-color: {cls.COLORS['error']};
            --info-color: {cls.COLORS['info']};
            --light-color: {cls.COLORS['light']};
            --dark-color: {cls.COLORS['dark']};
            --muted-color: {cls.COLORS['muted']};
            --bg-primary: #0e1117;
            --bg-secondary: #262730;
            --bg-tertiary: #1e1e2e;
            --text-primary: #fafafa;
            --text-secondary: #b3b3b3;
            --border-color: #2f3349;
            --accent-color: #ff6b6b;
        }}
        .stApp {{
            background-color: var(--bg-primary);
            color: var(--text-primary);
        }}
        .main .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
            background-color: var(--bg-primary);
        }}
        .css-1d391kg {{
            background-color: var(--bg-secondary);
            padding-top: 2rem;
        }}
        .css-1d391kg .stMarkdown h1,
        .css-1d391kg .stMarkdown h2,
        .css-1d391kg .stMarkdown h3 {{
            color: var(--text-primary);
        }}
        .chat-container {{
            background-color: var(--bg-primary);
            border-radius: 12px;
            padding: 20px;
            margin: 10px 0;
            border: 1px solid var(--border-color);
        }}
        .chat-message {{
            padding: 16px 20px;
            border-radius: 16px;
            margin-bottom: 16px;
            word-wrap: break-word;
            max-width: 85%;
            position: relative;
        }}
        .chat-message.user {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            margin-left: auto;
            margin-right: 0;
            border-bottom-right-radius: 4px;
        }}
        .chat-message.assistant {{
            background-color: var(--bg-tertiary);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
            border-left: 4px solid var(--primary-color);
            margin-left: 0;
            margin-right: auto;
        }}
        .chat-message.error {{
            background-color: #2d1b1b;
            color: #ffcccb;
            border-left: 4px solid var(--error-color);
        }}
        .stTextArea textarea {{
            background-color: var(--bg-tertiary) !important;
            color: var(--text-primary) !important;
            border: 2px solid var(--border-color) !important;
            border-radius: 12px !important;
        }}
        .stTextArea textarea:focus {{
            border-color: var(--primary-color) !important;
            box-shadow: 0 0 0 3px rgba(31, 119, 180, 0.2) !important;
        }}
        .stTextInput input {{
            background-color: var(--bg-tertiary) !important;
            color: var(--text-primary) !important;
            border: 2px solid var(--border-color) !important;
            border-radius: 8px !important;
        }}
        .stTextInput input:focus {{
            border-color: var(--primary-color) !important;
            box-shadow: 0 0 0 3px rgba(31, 119, 180, 0.2) !important;
        }}
        .stButton > button {{
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            padding: 12px 24px;
            transition: all 0.3s ease;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(31, 119, 180, 0.3);
        }}
        .metric-card {{
            background: var(--bg-tertiary);
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border: 1px solid var(--border-color);
            transition: transform 0.2s ease;
        }}
        .metric-card:hover {{
            transform: translateY(-2px);
        }}
        .metric-card .metric-value {{
            font-size: {cls.FONT_SIZES['2xl']};
            font-weight: bold;
            color: var(--primary-color);
            margin-bottom: 4px;
        }}
        .metric-card .metric-label {{
            font-size: {cls.FONT_SIZES['sm']};
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .status-indicator {{
            display: inline-flex;
            align-items: center;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: {cls.FONT_SIZES['sm']};
            font-weight: 500;
            margin: 2px;
        }}
        .status-indicator.processing {{
            background-color: #fef3c7;
            color: #92400e;
        }}
        .status-indicator.ready {{
            background-color: #d1fae5;
            color: #065f46;
        }}
        .status-indicator.error {{
            background-color: #fee2e2;
            color: #991b1b;
        }}
        .code-block {{
            background-color: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 16px;
            font-family: 'Monaco', 'Consolas', 'Courier New', monospace;
            font-size: {cls.FONT_SIZES['sm']};
            color: var(--text-primary);
            overflow-x: auto;
        }}
        .loading-spinner {{
            border: 3px solid var(--border-color);
            border-top: 3px solid var(--primary-color);
            border-radius: 50%;
            width: 24px;
            height: 24px;
            animation: spin 1s linear infinite;
            display: inline-block;
            margin-right: 8px;
        }}
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        .math-expression {{
            font-family: 'KaTeX_Math', 'Times New Roman', serif;
            font-size: {cls.FONT_SIZES['lg']};
            text-align: center;
            padding: 16px;
            background-color: var(--bg-tertiary);
            border-radius: 8px;
            margin: 16px 0;
            border: 1px solid var(--border-color);
            color: var(--text-primary);
        }}
        .welcome-message {{
            background: linear-gradient(135deg, var(--bg-tertiary) 0%, var(--bg-secondary) 100%);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 24px;
            margin: 20px 0;
            color: var(--text-primary);
        }}
        .welcome-message h4 {{
            color: var(--primary-color);
            margin-bottom: 16px;
        }}
        .stSelectbox > div > div {{
            background-color: var(--bg-tertiary);
            color: var(--text-primary);
        }}
        .streamlit-expanderHeader {{
            background-color: var(--bg-tertiary);
            color: var(--text-primary);
        }}
        .chat-input-container {{
            background-color: var(--bg-secondary);
            border-radius: 16px;
            padding: 20px;
            margin-top: 20px;
            border: 1px solid var(--border-color);
        }}
        @media (max-width: 768px) {{
            .main .block-container {{
                padding-left: 1rem;
                padding-right: 1rem;
            }}
            .chat-message {{
                max-width: 95%;
            }}
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)

class UIFormatters:
    @staticmethod
    def format_duration(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    @staticmethod
    def format_number(value: float, precision: int = 2) -> str:
        if abs(value) >= 1_000_000:
            return f"{value/1_000_000:.{precision}f}M"
        elif abs(value) >= 1_000:
            return f"{value/1_000:.{precision}f}K"
        else:
            return f"{value:.{precision}f}"
    @staticmethod
    def format_percentage(value: float) -> str:
        return f"{value:.1%}"
    @staticmethod 
    def format_timestamp(timestamp) -> str:
        try:
            if hasattr(timestamp, 'strftime'):
                return timestamp.strftime('%H:%M:%S')
            else:
                return str(timestamp)
        except:
            return "Unknown"

class ComponentBuilder:
    @staticmethod
    def create_metric_card(title: str, value: str, change: Optional[str] = None) -> str:
        change_html = ""
        if change:
            color = "green" if change.startswith("+") else "red"
            change_html = f'<div style="color: {color}; font-size: 0.9em;">{change}</div>'
        return f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{title}</div>
            {change_html}
        </div>
        """
    @staticmethod
    def create_status_badge(status: str, text: str) -> str:
        return f'<span class="status-indicator {status}">{text}</span>'
