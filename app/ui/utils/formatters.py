from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
import json
import re
from decimal import Decimal
import streamlit as st

def format_message(content: str) -> str:
    if not content:
        return ""
    content = content.replace("&", "&amp;")
    content = content.replace("<", "&lt;")
    content = content.replace(">", "&gt;")
    content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', content)
    content = re.sub(r'\*(.*?)\*', r'<em>\1</em>', content)
    content = content.replace('\n', '<br>')
    return content

def format_error(error: str) -> str:
    if not error:
        return "An unknown error occurred"
    if "Failed to solve problem" in error:
        return "I encountered an issue while solving your problem. Please try rephrasing your question."
    if "ModuleNotFoundError" in error:
        return "There's a system configuration issue. Please contact support."
    if "ValidationError" in error:
        return "Please check your input and try again."
    if "TimeoutError" in error:
        return "The calculation is taking too long. Please try a simpler problem."
    error = re.sub(r'Traceback.*?:', '', error, flags=re.DOTALL)
    error = error.strip()
    return error[:200] + "..." if len(error) > 200 else error

class UIFormatters:
    @staticmethod
    def format_number(value: Union[int, float, Decimal], decimals: int = 3) -> str:
        if isinstance(value, (int, float, Decimal)):
            if abs(value) < 0.001 and value != 0:
                return f"{value:.2e}"
            return f"{value:.{decimals}f}".rstrip('0').rstrip('.')
        return str(value)

    @staticmethod
    def format_duration(seconds: float) -> str:
        if seconds < 1:
            return f"{seconds*1000:.0f}ms"
        elif seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{minutes:.0f}m {secs:.0f}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours:.0f}h {minutes:.0f}m"

    @staticmethod
    def format_timestamp(dt: datetime, format_type: str = "relative") -> str:
        now = datetime.now()
        if format_type == "relative":
            diff = now - dt
            if diff < timedelta(seconds=60):
                return "Just now"
            elif diff < timedelta(hours=1):
                minutes = int(diff.total_seconds() / 60)
                return f"{minutes} min ago"
            elif diff < timedelta(days=1):
                hours = int(diff.total_seconds() / 3600)
                return f"{hours}h ago"
            elif diff < timedelta(days=7):
                days = int(diff.total_seconds() / 86400)
                return f"{days}d ago"
            else:
                return dt.strftime("%b %d, %Y")
        elif format_type == "short":
            return dt.strftime("%H:%M")
        elif format_type == "medium":
            return dt.strftime("%b %d, %H:%M")
        elif format_type == "full":
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        return str(dt)

    @staticmethod
    def format_mathematical_expression(expr: str) -> str:
        expr = expr.replace("**", "^")
        expr = expr.replace("sqrt(", "\\sqrt{")
        expr = expr.replace(")", "}")
        if any(char in expr for char in "^\\{}"):
            return f"${expr}$"
        return expr

    @staticmethod
    def format_integral_result(result: Dict[str, Any]) -> str:
        if not result:
            return "No result"
        value = result.get('value', 'Unknown')
        function = result.get('function', '')
        limits = result.get('limits', {})
        if isinstance(value, (int, float)):
            formatted_value = UIFormatters.format_number(value)
        else:
            formatted_value = str(value)
        if function and limits:
            lower = limits.get('lower', '')
            upper = limits.get('upper', '')
            func_latex = UIFormatters.format_mathematical_expression(function)
            return f"∫[{lower} to {upper}] {func_latex} dx = {formatted_value}"
        return f"Result: {formatted_value}"

    @staticmethod
    def format_error_message(error: str, context: Optional[str] = None) -> str:
        if context:
            return f"❌ **Error in {context}**: {error}"
        return f"❌ **Error**: {error}"

    @staticmethod
    def format_success_message(message: str) -> str:
        return f"✅ {message}"

    @staticmethod
    def format_info_message(message: str) -> str:
        return f"ℹ️ {message}"

    @staticmethod
    def format_warning_message(message: str) -> str:
        return f"⚠️ {message}"

    @staticmethod
    def format_json_pretty(data: Dict[str, Any], max_depth: int = 3) -> str:
        def truncate_deep(obj, depth=0):
            if depth >= max_depth:
                return "..."
            if isinstance(obj, dict):
                return {k: truncate_deep(v, depth + 1) for k, v in obj.items()}
            elif isinstance(obj, list):
                if len(obj) > 5:
                    return [truncate_deep(item, depth + 1) for item in obj[:5]] + ["..."]
                return [truncate_deep(item, depth + 1) for item in obj]
            return obj
        truncated = truncate_deep(data)
        return json.dumps(truncated, indent=2, default=str)

    @staticmethod
    def format_tool_usage_stats(stats: Dict[str, Any]) -> str:
        if not stats:
            return "No statistics available"
        lines = []
        for tool_name, tool_stats in stats.items():
            usage_count = tool_stats.get('usage_count', 0)
            avg_duration = tool_stats.get('avg_duration', 0)
            success_rate = tool_stats.get('success_rate', 0) * 100
            duration_str = UIFormatters.format_duration(avg_duration)
            lines.append(f"**{tool_name}**: {usage_count} uses, {duration_str} avg, {success_rate:.1f}% success")
        return "\n".join(lines)

    @staticmethod
    def format_memory_usage(bytes_used: int) -> str:
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_used < 1024:
                return f"{bytes_used:.1f} {unit}"
            bytes_used /= 1024
        return f"{bytes_used:.1f} TB"

class UIValidators:
    @staticmethod
    def validate_mathematical_expression(expr: str) -> tuple[bool, str]:
        if not expr or not expr.strip():
            return False, "Expression cannot be empty"
        allowed_chars = set("0123456789+-*/().xyzabcdefghijklmnopqrstuvwXYZ^_")
        if not all(c in allowed_chars or c.isspace() for c in expr):
            return False, "Expression contains invalid characters"
        if expr.count('(') != expr.count(')'):
            return False, "Unbalanced parentheses"
        return True, "Valid expression"

    @staticmethod
    def validate_numeric_input(value: str, min_val: Optional[float] = None, max_val: Optional[float] = None) -> tuple[bool, str]:
        try:
            num_val = float(value)
            if min_val is not None and num_val < min_val:
                return False, f"Value must be at least {min_val}"
            if max_val is not None and num_val > max_val:
                return False, f"Value must be at most {max_val}"
            return True, "Valid number"
        except ValueError:
            return False, "Invalid number format"

    @staticmethod
    def validate_integration_limits(lower: str, upper: str) -> tuple[bool, str]:
        lower_valid, lower_msg = UIValidators.validate_numeric_input(lower)
        if not lower_valid:
            return False, f"Lower limit: {lower_msg}"
        upper_valid, upper_msg = UIValidators.validate_numeric_input(upper)
        if not upper_valid:
            return False, f"Upper limit: {upper_msg}"
        try:
            if float(lower) >= float(upper):
                return False, "Lower limit must be less than upper limit"
        except ValueError:
            return False, "Invalid limit values"
        return True, "Valid limits"
