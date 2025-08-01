"""Unit tests for prompt utilities.

Simple, focused tests following KISS principle to improve coverage
without overcomplicating the test suite.
"""

import pytest
from typing import Dict, Any

from app.agents.prompts import (
    MATHEMATICAL_REASONING_PROMPT,
    TOOL_SELECTION_PROMPT,
    REFLECTION_PROMPT,
    PROBLEM_ANALYSIS_PROMPT,
    ERROR_RECOVERY_PROMPT,
    get_prompt_template,
    build_tool_description,
    format_mathematical_context,
    PROMPT_TEMPLATES
)


class TestPromptConstants:
    """Test prompt template constants."""
    
    def test_prompt_templates_exist(self):
        """Test that all prompt templates are defined and non-empty."""
        prompts = [
            MATHEMATICAL_REASONING_PROMPT,
            TOOL_SELECTION_PROMPT,
            REFLECTION_PROMPT,
            PROBLEM_ANALYSIS_PROMPT,
            ERROR_RECOVERY_PROMPT,
        ]
        
        for prompt in prompts:
            assert isinstance(prompt, str)
            assert len(prompt) > 50  # Should be substantial
            assert "problem" in prompt.lower()  # Should mention problem


class TestPromptRegistry:
    """Test prompt template registry functionality."""
    
    def test_prompt_templates_registry(self):
        """Test that PROMPT_TEMPLATES registry contains all templates."""
        expected_keys = [
            "mathematical_reasoning",
            "tool_selection", 
            "reflection",
            "problem_analysis",
            "error_recovery"
        ]
        
        for key in expected_keys:
            assert key in PROMPT_TEMPLATES
            assert isinstance(PROMPT_TEMPLATES[key], str)
            assert len(PROMPT_TEMPLATES[key]) > 50
    
    def test_get_prompt_template_valid(self):
        """Test getting valid prompt templates."""
        for template_name in PROMPT_TEMPLATES.keys():
            template = get_prompt_template(template_name)
            assert isinstance(template, str)
            assert len(template) > 0
            assert template == PROMPT_TEMPLATES[template_name]
    
    def test_get_prompt_template_invalid(self):
        """Test getting invalid prompt template raises KeyError."""
        with pytest.raises(KeyError) as exc_info:
            get_prompt_template("nonexistent_template")
        
        assert "not found" in str(exc_info.value)
        assert "nonexistent_template" in str(exc_info.value)
        assert "Available:" in str(exc_info.value)


class TestToolDescriptionBuilder:
    """Test tool description building functionality."""
    
    def test_build_tool_description_empty(self):
        """Test building description with empty tools."""
        result = build_tool_description({})
        assert isinstance(result, str)
        assert result == ""  # Function returns empty string for empty dict
    
    def test_build_tool_description_single_tool(self):
        """Test building description with single tool."""
        tools_info = {
            "integral_tool": {
                "description": "Calculate integrals",
                "capabilities": ["symbolic", "numerical"]
            }
        }
        
        result = build_tool_description(tools_info)
        assert isinstance(result, str)
        assert "integral_tool" in result
        assert "Calculate integrals" in result
        assert "Capabilities: symbolic, numerical" in result
    
    def test_build_tool_description_multiple_tools(self):
        """Test building description with multiple tools."""
        tools_info = {
            "integral_tool": {
                "description": "Calculate integrals",
                "capabilities": ["symbolic", "numerical"],
                "usage_stats": {"success_rate": 0.9}
            },
            "plot_tool": {
                "description": "Create plots",
                "capabilities": ["2D", "3D"]
            }
        }
        
        result = build_tool_description(tools_info)
        assert isinstance(result, str)
        assert "**integral_tool**:" in result
        assert "**plot_tool**:" in result
        assert "Calculate integrals" in result
        assert "Create plots" in result
        assert "Capabilities: symbolic, numerical" in result
        assert "Success Rate: 90.0%" in result
    
    def test_build_tool_description_with_usage_stats(self):
        """Test building description includes usage statistics."""
        tools_info = {
            "test_tool": {
                "description": "Test tool",
                "usage_stats": {
                    "success_rate": 0.85,
                    "usage_count": 42,
                    "average_execution_time": 1.5
                }
            }
        }
        
        result = build_tool_description(tools_info)
        assert "Success Rate: 85.0%" in result
        # Note: usage_count and average_execution_time may not be in the actual output
        # Only test what we know is included based on real behavior


class TestMathematicalContextFormatter:
    """Test mathematical context formatting functionality."""
    
    def test_format_mathematical_context_empty(self):
        """Test formatting empty context."""
        result = format_mathematical_context({})
        assert result == "No specific context"
    
    def test_format_mathematical_context_functions(self):
        """Test formatting context with functions."""
        context = {"functions": ["sin", "cos", "tan"]}
        result = format_mathematical_context(context)
        assert "Functions: sin, cos, tan" in result
    
    def test_format_mathematical_context_variables(self):
        """Test formatting context with variables."""
        context = {"variables": ["x", "y", "z"]}
        result = format_mathematical_context(context)
        assert "Variables: x, y, z" in result
    
    def test_format_mathematical_context_domain(self):
        """Test formatting context with domain."""
        context = {"domain": "[-5, 5]"}
        result = format_mathematical_context(context)
        assert "Domain: [-5, 5]" in result
    
    def test_format_mathematical_context_constraints(self):
        """Test formatting context with constraints."""
        context = {"constraints": "x > 0"}
        result = format_mathematical_context(context)
        assert "Constraints: x > 0" in result
    
    def test_format_mathematical_context_previous_results(self):
        """Test formatting context with previous results."""
        context = {"previous_results": "integral = x^2/2"}
        result = format_mathematical_context(context)
        assert "Previous Results: integral = x^2/2" in result
    
    def test_format_mathematical_context_complete(self):
        """Test formatting complete context with all fields."""
        context = {
            "functions": ["sin", "cos"],
            "variables": ["x", "t"],
            "domain": "[0, π]",
            "constraints": "x >= 0",
            "previous_results": "derivative = cos(x)"
        }
        
        result = format_mathematical_context(context)
        assert "Functions: sin, cos" in result
        assert "Variables: x, t" in result  
        assert "Domain: [0, π]" in result
        assert "Constraints: x >= 0" in result
        assert "Previous Results: derivative = cos(x)" in result
        assert "|" in result  # Should use separator
