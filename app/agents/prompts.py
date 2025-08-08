from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod

# === Professional Prompt Management System ===

class PromptTemplate(ABC):
    """
    Abstract base class for professional prompt templates.
    
    Enforces consistent interface and validation for all prompts.
    """
    
    @property
    @abstractmethod
    def template(self) -> str:
        """Get the template string."""
        pass
    
    @property 
    @abstractmethod
    def required_fields(self) -> List[str]:
        """Get list of required template fields."""
        pass
    
    @abstractmethod
    def format(self, **kwargs) -> str:
        """Format template with provided values."""
        pass
    
    def validate_inputs(self, **kwargs) -> None:
        """Validate that all required fields are provided."""
        missing_fields = [field for field in self.required_fields if field not in kwargs]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")


class MathematicalReasoningTemplate(PromptTemplate):
    """Professional mathematical reasoning prompt with strict JSON output."""
    
    @property
    def template(self) -> str:
        return """You are a mathematical reasoning expert. Analyze the given problem and provide a structured approach.

Your task is to:
1. Understand the mathematical problem
2. Determine the best approach to solve it
3. Identify the required tools and steps
4. Provide a confidence assessment

CRITICAL RULES FOR TOOL SELECTION:
- If the problem mentions "show", "visualize", "plot", "area under curve", "graph", or asks for visualization, ALWAYS include "plot_generator" in tools_needed.
- For integrals, derivatives, or mathematical calculations: use "integral_calculator" 
- For function analysis (critical points, asymptotes, etc.): use "function_analyzer"

Available tools:
{tool_descriptions}

RESPONSE FORMAT - Return VALID JSON only:
{{
    "approach": "detailed approach to solve the problem",
    "steps": ["step1", "step2", "step3"],
    "tools_needed": ["tool1", "tool2"],
    "confidence": 0.9
}}

EXAMPLE for integral with visualization:
{{
    "approach": "Calculate the definite integral using integration rules and visualize the area under the curve",
    "steps": ["Apply power rule to integrate x²", "Evaluate definite integral from 0 to 3", "Plot function and shade area under curve"],
    "tools_needed": ["integral_calculator", "plot_generator"],
    "confidence": 0.9
}}

IMPORTANT: 
- Return only valid JSON (no markdown, no extra text)
- For ANY visualization request, include "plot_generator"
- Use proper JSON syntax with double quotes
- End arrays and objects properly"""
    
    @property
    def required_fields(self) -> List[str]:
        return ["problem", "context", "tool_descriptions"]
    
    def format(self, **kwargs) -> str:
        self.validate_inputs(**kwargs)
        return self.template.format(
            tool_descriptions=kwargs["tool_descriptions"]
        )


class ProblemAnalysisTemplate(PromptTemplate):
    """Professional problem analysis prompt with structured output."""
    
    @property
    def template(self) -> str:
        return """You are a mathematical problem analyzer. Analyze the given problem and return a structured response.

Return VALID JSON only with exactly these fields:
{{
    "problem_type": "integral|derivative|algebra|analysis|etc",
    "complexity": "low|medium|high",
    "requires_tools": true|false,
    "description": "clear description of what needs to be solved",
    "approach": "recommended approach or strategy",
    "confidence": 0.9
}}

IMPORTANT: Return only valid JSON, no markdown, no extra text."""
    
    @property
    def required_fields(self) -> List[str]:
        return ["problem"]
    
    def format(self, **kwargs) -> str:
        self.validate_inputs(**kwargs)
        return self.template


class ValidationTemplate(PromptTemplate):
    """Professional solution validation prompt with scoring."""
    
    @property
    def template(self) -> str:
        return """You are a mathematical solution validator. Validate the given solution.

VALIDATION CRITERIA:
- If tools were executed successfully and produced results, the solution is generally valid
- If mathematical reasoning is sound, score highly
- If visualization/plotting was requested and tools were executed, score highly
- Only mark as invalid if there are clear mathematical errors

Return VALID JSON only with exactly these fields:
{{
    "is_valid": true|false,
    "score": 0.9,
    "issues": ["issue1", "issue2"],
    "suggestions": ["suggestion1", "suggestion2"]
}}

SCORING GUIDELINES:
- 0.8+ if tools executed successfully and reasoning is sound
- 0.9+ if all requested operations (calculation + visualization) completed
- 0.6-0.7 if partial completion but correct
- <0.6 only for mathematical errors

IMPORTANT: Return only valid JSON, no markdown, no extra text."""
    
    @property
    def required_fields(self) -> List[str]:
        return ["problem", "reasoning", "tool_results", "trace"]
    
    def format(self, **kwargs) -> str:
        self.validate_inputs(**kwargs)
        return self.template


class ErrorRecoveryTemplate(PromptTemplate):
    """Professional error recovery prompt with action planning."""
    
    @property
    def template(self) -> str:
        return """You are an error recovery specialist. Analyze the error and provide recovery strategies.

Return a JSON object with exactly these fields:
- action: string (recovery action to take)
- note: string (explanation of the recovery strategy)
- confidence: number (confidence in the recovery approach 0-1)

Example response:
{{
    "action": "retry_with_simplified_approach",
    "note": "The error suggests the approach was too complex. Retry with a more basic integration method.",
    "confidence": 0.8
}}"""
    
    @property
    def required_fields(self) -> List[str]:
        return ["problem", "error", "error_type", "retry_count"]
    
    def format(self, **kwargs) -> str:
        self.validate_inputs(**kwargs)
        return self.template


class ResponseFormattingTemplate(PromptTemplate):
    """Professional response formatting prompt with structured output."""
    
    @property
    def template(self) -> str:
        return """You are a response formatter. Create a clear, well-structured final response.

Return a JSON object with exactly these fields:
- answer: string (the final numerical or symbolic answer)
- steps: array (list of solution steps taken)
- explanation: string (clear explanation of the approach used)
- confidence: number (confidence score between 0 and 1)

Example response:
{{
    "answer": "9",
    "steps": ["Set up integral of x² from 0 to 3", "Apply power rule: ∫x²dx = x³/3", "Evaluate: [x³/3] from 0 to 3 = 27/3 - 0 = 9"],
    "explanation": "The definite integral represents the area under the curve x² from 0 to 3, which equals 9 square units.",
    "confidence": 0.95
}}"""
    
    @property
    def required_fields(self) -> List[str]:
        return ["problem", "reasoning", "tool_results", "validation", "trace"]
    
    def format(self, **kwargs) -> str:
        self.validate_inputs(**kwargs)
        return self.template


class ToolSelectionTemplate(PromptTemplate):
    """Professional tool selection prompt with reasoning."""
    
    @property
    def template(self) -> str:
        return """You are a tool selection expert. Based on the problem analysis, select appropriate tools.

Available tools:
{tool_descriptions}

Return VALID JSON only with exactly these fields:
{{
    "selected_tools": ["tool1", "tool2"],
    "reasoning": "explanation of tool selection"
}}

IMPORTANT: Return only valid JSON, no markdown, no extra text."""
    
    @property
    def required_fields(self) -> List[str]:
        return ["problem", "problem_type", "analysis", "tool_descriptions"]
    
    def format(self, **kwargs) -> str:
        self.validate_inputs(**kwargs)
        return self.template.format(
            tool_descriptions=kwargs["tool_descriptions"]
        )


# === Template Registry for Centralized Management ===

class PromptTemplateRegistry:
    """
    Centralized registry for all prompt templates.
    
    Implements the Registry pattern for clean template management.
    """
    
    def __init__(self):
        self._templates = {
            "mathematical_reasoning": MathematicalReasoningTemplate(),
            "problem_analysis": ProblemAnalysisTemplate(),
            "validation": ValidationTemplate(),
            "error_recovery": ErrorRecoveryTemplate(),
            "response_formatting": ResponseFormattingTemplate(),
            "tool_selection": ToolSelectionTemplate(),
        }
    
    def get_template(self, name: str) -> PromptTemplate:
        """Get template by name."""
        if name not in self._templates:
            raise KeyError(f"Template '{name}' not found. Available: {list(self._templates.keys())}")
        return self._templates[name]
    
    def list_templates(self) -> List[str]:
        """List all available template names."""
        return list(self._templates.keys())
    
    def format_prompt(self, template_name: str, **kwargs) -> str:
        """Format a prompt template with given parameters."""
        template = self.get_template(template_name)
        return template.format(**kwargs)


# === Global Registry Instance ===
_template_registry = PromptTemplateRegistry()


# === Professional Utility Functions ===

def build_tool_description(tools_info: Dict[str, Any]) -> str:
    """
    Build formatted tool descriptions for prompts.
    
    Professional implementation with error handling and formatting.
    
    Args:
        tools_info: Dictionary with tool information
        
    Returns:
        str: Formatted tool descriptions
    """
    if not tools_info:
        return "No tools available"
    
    descriptions = []
    
    for tool_name, tool_data in tools_info.items():
        if not isinstance(tool_data, dict):
            descriptions.append(f"- {tool_name}: {str(tool_data)}")
            continue
            
        description = f"- {tool_name}"
        if 'description' in tool_data:
            description += f": {tool_data['description']}"
        if 'capabilities' in tool_data and tool_data['capabilities']:
            description += f"\n  Capabilities: {', '.join(tool_data['capabilities'])}"
        if 'usage_stats' in tool_data:
            stats = tool_data['usage_stats']
            success_rate = stats.get('success_rate', 0) * 100
            description += f"\n  Success Rate: {success_rate:.1f}%"
        
        descriptions.append(description)
    
    return "\n".join(descriptions)


def format_mathematical_context(context: Dict[str, Any]) -> str:
    """
    Format mathematical context for prompts.
    
    Professional implementation with validation and formatting.
    
    Args:
        context: Mathematical context dictionary
        
    Returns:
        str: Formatted context string
    """
    if not context:
        return "No specific context"
    
    formatted_parts = []
    
    # Process standard context fields
    context_mappings = {
        'functions': 'Functions',
        'variables': 'Variables', 
        'domain': 'Domain',
        'constraints': 'Constraints',
        'previous_results': 'Previous Results'
    }
    
    for key, label in context_mappings.items():
        if key in context and context[key]:
            value = context[key]
            if isinstance(value, list):
                value = ', '.join(str(v) for v in value)
            formatted_parts.append(f"{label}: {value}")
    
    return " | ".join(formatted_parts) if formatted_parts else "No specific context"


# === Factory Functions ===

def get_prompt_template(template_name: str) -> str:
    """
    Get a prompt template by name using the centralized registry.
    
    Args:
        template_name: Name of the template
        
    Returns:
        str: Prompt template
        
    Raises:
        KeyError: If template name not found
    """
    return _template_registry.get_template(template_name).template


def format_prompt(template_name: str, **kwargs) -> str:
    """
    Format a prompt template with given parameters.
    
    Professional wrapper with validation and error handling.
    
    Args:
        template_name: Name of the template to format
        **kwargs: Template parameters
        
    Returns:
        str: Formatted prompt
        
    Raises:
        KeyError: If template not found
        ValueError: If required parameters missing
    """
    return _template_registry.format_prompt(template_name, **kwargs)


def get_template_registry() -> PromptTemplateRegistry:
    """
    Get the global template registry instance.
    
    Returns:
        PromptTemplateRegistry: The global registry
    """
    return _template_registry


# === Public Interface ===

__all__ = [
    # Template classes
    "PromptTemplate",
    "MathematicalReasoningTemplate", 
    "ProblemAnalysisTemplate",
    "ValidationTemplate",
    "ErrorRecoveryTemplate",
    "ResponseFormattingTemplate",
    "ToolSelectionTemplate",
    
    # Registry
    "PromptTemplateRegistry",
    "get_template_registry",
    
    # Utility functions
    "build_tool_description",
    "format_mathematical_context",
    "get_prompt_template",
    "format_prompt",
]
