"""
Style Manager - Centralized CSS and styling utilities

Implements consistent styling across the application following design system principles.
"""

import streamlit as st
from typing import Dict, Any, Optional


class StyleManager:
    """Professional styling utilities for consistent UI appearance."""
    
    # Color palette following modern design principles
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
    
    # Typography scale
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
        """Inject global CSS styles."""
        css = f"""
        <style>
        /* Global Variables */
        :root {{
            --primary-color: {cls.COLORS['primary']};
            --secondary-color: {cls.COLORS['secondary']};
            --success-color: {cls.COLORS['success']};
            --warning-color: {cls.COLORS['warning']};
            --error-color: {cls.COLORS['error']};
            --info-color: {cls.COLORS['info']};
        }}
        
        /* Main content area improvements */
        .main .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }}
        
        /* Chat message styling */
        .chat-message {{
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            border-left: 4px solid var(--primary-color);
        }}
        
        .chat-message.user {{
            background-color: #f0f9ff;
            border-left-color: var(--info-color);
        }}
        
        .chat-message.assistant {{
            background-color: #f9fafb;
            border-left-color: var(--primary-color);
        }}
        
        .chat-message.error {{
            background-color: #fef2f2;
            border-left-color: var(--error-color);
        }}
        
        /* Metrics cards */
        .metric-card {{
            background: white;
            padding: 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border: 1px solid #e5e7eb;
        }}
        
        .metric-card .metric-value {{
            font-size: {cls.FONT_SIZES['2xl']};
            font-weight: bold;
            color: var(--primary-color);
        }}
        
        .metric-card .metric-label {{
            font-size: {cls.FONT_SIZES['sm']};
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        
        /* Status indicators */
        .status-indicator {{
            display: inline-flex;
            align-items: center;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: {cls.FONT_SIZES['sm']};
            font-weight: 500;
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
        
        /* Code blocks */
        .code-block {{
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 0.375rem;
            padding: 1rem;
            font-family: 'Monaco', 'Consolas', 'Courier New', monospace;
            font-size: {cls.FONT_SIZES['sm']};
        }}
        
        /* Loading animations */
        .loading-spinner {{
            border: 3px solid #f3f3f3;
            border-top: 3px solid var(--primary-color);
            border-radius: 50%;
            width: 24px;
            height: 24px;
            animation: spin 1s linear infinite;
            display: inline-block;
            margin-right: 0.5rem;
        }}
        
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        
        /* Sidebar improvements */
        .css-1d391kg {{
            padding-top: 2rem;
        }}
        
        /* Button improvements */
        .stButton > button {{
            border-radius: 0.375rem;
            border: none;
            background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
            color: white;
            font-weight: 500;
            transition: all 0.2s;
        }}
        
        .stButton > button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        
        /* Input improvements */
        .stTextInput > div > div > input {{
            border-radius: 0.375rem;
            border: 1px solid #d1d5db;
        }}
        
        .stTextInput > div > div > input:focus {{
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(31, 119, 180, 0.1);
        }}
        
        /* Mathematical expressions */
        .math-expression {{
            font-family: 'KaTeX_Math', 'Times New Roman', serif;
            font-size: {cls.FONT_SIZES['lg']};
            text-align: center;
            padding: 1rem;
            background-color: #fafafa;
            border-radius: 0.375rem;
            margin: 1rem 0;
        }}
        
        /* Plot containers */
        .plot-container {{
            background: white;
            border-radius: 0.5rem;
            padding: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin: 1rem 0;
        }}
        
        /* Hide Streamlit default elements */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {{
            width: 8px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: #f1f1f1;
            border-radius: 4px;
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: #c1c1c1;
            border-radius: 4px;
        }}
        
        ::-webkit-scrollbar-thumb:hover {{
            background: #a8a8a8;
        }}
        </style>
        """
        
        st.markdown(css, unsafe_allow_html=True)
    
    @classmethod
    def create_status_indicator(cls, status: str, text: str) -> str:
        """Create styled status indicator HTML."""
        return f"""
        <div class="status-indicator {status}">
            {text}
        </div>
        """
    
    @classmethod
    def create_metric_card(cls, value: str, label: str, delta: Optional[str] = None) -> str:
        """Create styled metric card HTML."""
        delta_html = ""
        if delta:
            delta_color = cls.COLORS['success'] if not delta.startswith('-') else cls.COLORS['error']
            delta_html = f'<div style="color: {delta_color}; font-size: {cls.FONT_SIZES["sm"]};">{delta}</div>'
        
        return f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
            {delta_html}
        </div>
        """
    
    @classmethod
    def create_loading_indicator(cls, text: str = "Processing...") -> str:
        """Create loading indicator HTML."""
        return f"""
        <div style="display: flex; align-items: center; justify-content: center; padding: 2rem;">
            <div class="loading-spinner"></div>
            <span>{text}</span>
        </div>
        """
    
    @classmethod
    def create_code_block(cls, code: str, language: str = "python") -> str:
        """Create styled code block HTML."""
        return f"""
        <div class="code-block">
            <pre><code class="language-{language}">{code}</code></pre>
        </div>
        """
    
    @classmethod
    def create_chat_message(cls, content: str, role: str = "assistant") -> str:
        """Create styled chat message HTML."""
        return f"""
        <div class="chat-message {role}">
            {content}
        </div>
        """
