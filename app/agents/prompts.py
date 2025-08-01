"""Specialized prompt templates for mathematical ReAct agent.

This module contains professionally crafted prompts for different stages
of the ReAct reasoning process, optimized for mathematical problem solving.

Following KISS principles, prompts are clear, structured, and focused.
"""

from typing import Dict, Any

# === Core ReAct Reasoning Prompts ===

MATHEMATICAL_REASONING_PROMPT = """You are an expert mathematical assistant using ReAct (Reasoning and Acting) methodology.

**Current Problem**: {problem}

**Mathematical Context**: {context}

**Available Tools**: {available_tools}

**Your task**: Solve this mathematical problem step by step using the following ReAct pattern:

1. **Thought**: Analyze the problem and determine what needs to be done
2. **Action**: Choose the appropriate tool and specify parameters
3. **Observation**: Analyze the tool's result
4. **Thought**: Decide on the next step based on the observation
5. **Action**: Continue with the next tool if needed
6. **Final Answer**: Provide the complete solution

**Mathematical Problem Types You Handle**:
- Calculus: integration, differentiation, limits
- Algebra: solving equations, simplification
- Analysis: function behavior, critical points
- Visualization: plotting functions, areas under curves

**Guidelines**:
- Be precise with mathematical notation
- Show all intermediate steps
- Validate results when possible
- Use appropriate tools for each subtask
- Explain your reasoning clearly

Begin your analysis now:"""


TOOL_SELECTION_PROMPT = """**Tool Selection for Mathematical Problem**

**Problem**: {problem}
**Problem Type**: {problem_type}
**Mathematical Context**: {mathematical_context}

**Available Tools**:
{available_tools_description}

**Previous Tool Results**: {previous_results}

**Your task**: Select the most appropriate tool(s) for the next step.

**Consider**:
- Problem requirements and current progress
- Tool capabilities and limitations  
- Efficiency and accuracy trade-offs
- Integration with previous results

**Response Format**:
Selected Tool: [tool_name]
Rationale: [why this tool is best]
Parameters: [specific parameters to use]
Expected Output: [what you expect to get]"""


REFLECTION_PROMPT = """**Solution Reflection and Validation**

**Original Problem**: {problem}
**Tools Used**: {tools_used}
**Results Obtained**: {results}
**Solution Steps**: {solution_steps}

**Reflection Tasks**:
1. **Correctness**: Are the results mathematically sound?
2. **Completeness**: Have all aspects of the problem been addressed?
3. **Efficiency**: Could this have been solved more efficiently?
4. **Clarity**: Is the solution clearly explained?

**Validation Checks**:
- Check dimensional consistency
- Verify mathematical relationships
- Test edge cases if applicable
- Cross-validate with alternative methods

**Response Format**:
Confidence Score: [0-1]
Validation Status: [VALID/INVALID/NEEDS_REVIEW]
Issues Found: [list any problems]
Recommendations: [improvements or corrections]
Final Answer: [validated final answer]"""


PROBLEM_ANALYSIS_PROMPT = """**Mathematical Problem Analysis**

**Problem Statement**: {problem}
**User Context**: {user_context}

**Analysis Framework**:

1. **Problem Classification**:
   - Domain: [Calculus/Algebra/Analysis/etc.]
   - Type: [Integration/Differentiation/Equation/etc.]
   - Complexity: [Basic/Intermediate/Advanced]

2. **Requirements Identification**:
   - Input variables and constraints
   - Expected output format
   - Precision requirements
   - Visualization needs

3. **Solution Strategy**:
   - Decompose into subtasks
   - Identify required tools
   - Plan execution sequence
   - Anticipate potential issues

4. **Resource Planning**:
   - Computational complexity
   - Memory requirements
   - Visualization resources

**Response Format**:
Problem Type: [specific classification]
Difficulty: [1-5 scale]
Required Tools: [list of tools]
Solution Plan: [step-by-step approach]
Estimated Time: [complexity assessment]"""


ERROR_RECOVERY_PROMPT = """**Error Recovery and Problem Resolution**

**Error Details**:
- Error Type: {error_type}
- Error Message: {error_message}
- Failed Action: {failed_action}
- Context: {error_context}

**Current State**:
- Problem: {current_problem}
- Progress: {current_progress}
- Previous Results: {previous_results}

**Recovery Strategy**:

1. **Error Analysis**:
   - Identify root cause
   - Assess impact on solution
   - Determine recovery options

2. **Alternative Approaches**:
   - Can we use a different tool?
   - Can we modify parameters?
   - Can we break down the problem?

3. **Recovery Actions**:
   - Immediate fixes
   - Alternative solution paths
   - Simplification strategies

**Response Format**:
Error Cause: [analysis of what went wrong]
Recovery Plan: [specific steps to take]
Alternative Tools: [backup options]
Modified Approach: [adjusted strategy]
Continue/Restart: [recommendation]"""


# === Prompt Building Utilities ===

def build_tool_description(tools_info: Dict[str, Any]) -> str:
    """
    Build formatted tool descriptions for prompts.
    
    Args:
        tools_info: Dictionary with tool information
        
    Returns:
        str: Formatted tool descriptions
    """
    descriptions = []
    
    for tool_name, tool_data in tools_info.items():
        description = f"**{tool_name}**"
        if 'description' in tool_data:
            description += f": {tool_data['description']}"
        if 'capabilities' in tool_data:
            description += f"\n  Capabilities: {', '.join(tool_data['capabilities'])}"
        if 'usage_stats' in tool_data:
            stats = tool_data['usage_stats']
            success_rate = stats.get('success_rate', 0) * 100
            description += f"\n  Success Rate: {success_rate:.1f}%"
        
        descriptions.append(description)
    
    return "\n\n".join(descriptions)


def format_mathematical_context(context: Dict[str, Any]) -> str:
    """
    Format mathematical context for prompts.
    
    Args:
        context: Mathematical context dictionary
        
    Returns:
        str: Formatted context string
    """
    formatted_parts = []
    
    if 'functions' in context:
        formatted_parts.append(f"Functions: {', '.join(context['functions'])}")
    
    if 'variables' in context:
        formatted_parts.append(f"Variables: {', '.join(context['variables'])}")
    
    if 'domain' in context:
        formatted_parts.append(f"Domain: {context['domain']}")
    
    if 'constraints' in context:
        formatted_parts.append(f"Constraints: {context['constraints']}")
    
    if 'previous_results' in context:
        formatted_parts.append(f"Previous Results: {context['previous_results']}")
    
    return " | ".join(formatted_parts) if formatted_parts else "No specific context"


# === Prompt Template Registry ===

PROMPT_TEMPLATES = {
    "mathematical_reasoning": MATHEMATICAL_REASONING_PROMPT,
    "tool_selection": TOOL_SELECTION_PROMPT,
    "reflection": REFLECTION_PROMPT,
    "problem_analysis": PROBLEM_ANALYSIS_PROMPT,
    "error_recovery": ERROR_RECOVERY_PROMPT,
}


def get_prompt_template(template_name: str) -> str:
    """
    Get a prompt template by name.
    
    Args:
        template_name: Name of the template
        
    Returns:
        str: Prompt template
        
    Raises:
        KeyError: If template name not found
    """
    if template_name not in PROMPT_TEMPLATES:
        raise KeyError(f"Prompt template '{template_name}' not found. Available: {list(PROMPT_TEMPLATES.keys())}")
    
    return PROMPT_TEMPLATES[template_name]
