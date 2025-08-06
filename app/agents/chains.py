from typing import Any, Dict, List, Optional, Callable, Union
from functools import lru_cache
import logging

# Required imports - no fallbacks in production code
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableLambda

from ..core.config import Settings
from ..core.logging import get_logger, log_function_call
from ..core.exceptions import ConfigurationError, DependencyError
from ..tools.registry import ToolRegistry
from ..utils.robust_parser import RobustJsonOutputParser

logger = get_logger(__name__)


class LLMProvider:
    """
    Professional LLM provider with proper error handling.
    
    Separates LLM creation concerns from chain factory.
    """
    
    @staticmethod
    def create_gemini_llm(settings: Settings) -> ChatGoogleGenerativeAI:
        """
        Create Google Gemini 2.5 Flash LLM instance with validation and optimization.
        
        Args:
            settings: Application settings
            
        Returns:
            ChatGoogleGenerativeAI: Configured LLM instance for Gemini 2.5 Flash
            
        Raises:
            ConfigurationError: If configuration is invalid
            DependencyError: If LLM cannot be created
        """
        try:
            gemini_config = settings.gemini_config
            
            # Validate required configuration
            required_fields = ["model_name", "api_key"]
            for field in required_fields:
                if not gemini_config.get(field):
                    raise ConfigurationError(f"Missing required Gemini config: {field}")
            
            # Get safety settings in correct format
            safety_settings = gemini_config.get("safety_settings", [])
            
            # Enhanced configuration for Gemini 2.5 Flash
            llm = ChatGoogleGenerativeAI(
                model=gemini_config["model_name"],  # gemini-2.5-flash
                temperature=gemini_config.get("temperature", 0.1),
                max_output_tokens=gemini_config.get("max_output_tokens", 8192),  # Flash's max tokens
                top_p=gemini_config.get("top_p", 0.9),
                top_k=gemini_config.get("top_k", 40),
                api_key=gemini_config["api_key"],
                google_api_key=gemini_config["api_key"],  # For compatibility
                # safety_settings=safety_settings,  # Commented out to avoid validation error
                convert_system_message_to_human=True,  # Better system message handling
                # Removed streaming parameter as it's causing warnings
            )
            
            logger.info(f"Gemini 2.5 Flash LLM created successfully: {gemini_config['model_name']}")
            return llm
            
        except Exception as e:
            logger.error(f"Failed to create Gemini 2.5 Flash LLM: {e}")
            raise DependencyError(f"Could not initialize LLM: {str(e)}") from e


class ChainFactory:
    """
    Professional factory class for creating mathematical reasoning chains.
    
    Clean, single-responsibility factory without fallbacks or anti-patterns.
    Uses proper dependency injection and fails fast on configuration issues.
    """
    
    def __init__(
        self,
        settings: Settings,
        tool_registry: ToolRegistry,
        llm: Optional[ChatGoogleGenerativeAI] = None
    ):
        """
        Initialize chain factory with dependencies.
        
        Args:
            settings: Application settings
            tool_registry: Tool registry instance
            llm: Optional pre-configured LLM (for dependency injection in tests)
            
        Raises:
            DependencyError: If required dependencies cannot be initialized
        """
        self.settings = settings
        self.tool_registry = tool_registry
        
        # Create or use injected LLM
        if llm is not None:
            self._llm = llm
            logger.info("ChainFactory initialized with injected LLM")
        else:
            self._llm = LLMProvider.create_gemini_llm(settings)
            logger.info("ChainFactory initialized with new LLM")
    
    @property
    def llm(self) -> ChatGoogleGenerativeAI:
        """Get the configured LLM instance."""
        return self._llm
    
    def _get_tool_descriptions(self) -> str:
        """Get formatted tool descriptions for prompts."""
        try:
            tools = self.tool_registry.get_all_tools()
            descriptions = []
            
            for tool in tools:
                if hasattr(tool, 'name') and hasattr(tool, 'description'):
                    descriptions.append(f"- {tool.name}: {tool.description}")
                else:
                    descriptions.append(f"- {str(tool)}")
                    
            return "\n".join(descriptions) if descriptions else "No tools available"
            
        except Exception as e:
            logger.warning(f"Could not get tool descriptions: {e}")
            return "integral_calculator, plot_generator, function_analyzer"
    
    def create_reasoning_chain(self) -> RunnableSequence:
        """
        Create mathematical reasoning chain.
        
        Returns:
            RunnableSequence: LangChain reasoning chain
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a mathematical reasoning expert. Analyze the given problem and provide a structured approach.

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
- End arrays and objects properly"""),
            ("human", """Problem: {problem}

Context: {context}

Analyze this problem and return the JSON structure.""")
        ])
        
        # Use robust parser to handle JSON formatting errors
        chain = (
            RunnablePassthrough.assign(
                tool_descriptions=lambda x: self._get_tool_descriptions()
            )
            | prompt
            | self._llm
            | RobustJsonOutputParser()
        )
        
        logger.info("Mathematical reasoning chain created with robust JSON parser")
        return chain
    
    def create_analysis_chain(self) -> RunnableSequence:
        """
        Create problem analysis chain.
        
        Returns:
            RunnableSequence: Problem analysis chain
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a mathematical problem analyzer. Analyze the given problem and return a structured response.

Return VALID JSON only with exactly these fields:
{{
    "problem_type": "integral|derivative|algebra|analysis|etc",
    "complexity": "low|medium|high",
    "requires_tools": true|false,
    "description": "clear description of what needs to be solved",
    "approach": "recommended approach or strategy",
    "confidence": 0.9
}}

IMPORTANT: Return only valid JSON, no markdown, no extra text."""),
            ("human", "Analyze this mathematical problem: {problem}")
        ])
        
        chain = prompt | self._llm | RobustJsonOutputParser()
        
        logger.info("Problem analysis chain created with robust parser")
        return chain
    
    def create_validation_chain(self) -> RunnableSequence:
        """
        Create solution validation chain.
        
        Returns:
            RunnableSequence: Validation chain
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a mathematical solution validator. Validate the given solution.

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

IMPORTANT: Return only valid JSON, no markdown, no extra text."""),
            ("human", """Problem: {problem}
Reasoning: {reasoning}
Tool Results: {tool_results}
Trace: {trace}

Validate this mathematical solution based on the reasoning and tool results.""")
        ])
        
        chain = prompt | self._llm | RobustJsonOutputParser()
        
        logger.info("Solution validation chain created with robust parser")
        return chain
    
    def create_tool_selection_chain(self) -> RunnableSequence:
        """
        Create tool selection chain.
        
        Returns:
            RunnableSequence: Tool selection chain
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a tool selection expert. Based on the problem analysis, select appropriate tools.

Available tools:
{tool_descriptions}

Return VALID JSON only with exactly these fields:
{{
    "selected_tools": ["tool1", "tool2"],
    "reasoning": "explanation of tool selection"
}}

IMPORTANT: Return only valid JSON, no markdown, no extra text."""),
            ("human", """Problem: {problem}
Problem Type: {problem_type}
Analysis: {analysis}

Select the best tools for this problem.""")
        ])
        
        chain = (
            RunnablePassthrough.assign(
                tool_descriptions=lambda x: self._get_tool_descriptions()
            )
            | prompt
            | self._llm
            | RobustJsonOutputParser()
        )
        
        logger.info("Tool selection chain created with robust parser")
        return chain
    
    def create_error_recovery_chain(self) -> RunnableSequence:
        """
        Create error recovery chain.
        
        Returns:
            RunnableSequence: Error recovery chain
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an error recovery specialist. Analyze the error and provide recovery strategies.

Return a JSON object with exactly these fields:
- action: string (recovery action to take)
- note: string (explanation of the recovery strategy)
- confidence: number (confidence in the recovery approach 0-1)

Example response:
{{
    "action": "retry_with_simplified_approach",
    "note": "The error suggests the approach was too complex. Retry with a more basic integration method.",
    "confidence": 0.8
}}"""),
            ("human", """Problem: {problem}
Error: {error}
Error Type: {error_type}
Retry Count: {retry_count}

Provide recovery strategies for this error.""")
        ])
        
        chain = prompt | self._llm | RobustJsonOutputParser()
        
        logger.info("Error recovery chain created with robust parser")
        return chain
    
    def create_response_chain(self) -> RunnableSequence:
        """
        Create response formatting chain.
        
        Returns:
            RunnableSequence: Response formatting chain
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a response formatter. Create a clear, well-structured final response.

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
}}"""),
            ("human", """Problem: {problem}
Reasoning: {reasoning}
Tool Results: {tool_results}
Validation: {validation}
Trace: {trace}

Format this into a clear final response JSON.""")
        ])
        
        chain = prompt | self._llm | RobustJsonOutputParser()
        
        logger.info("Response formatting chain created with robust parser")
        return chain


# === Factory Functions ===

@log_function_call(logger)
def create_chain_factory(
    settings: Optional[Settings] = None,
    tool_registry: Optional[ToolRegistry] = None,
    llm: Optional[ChatGoogleGenerativeAI] = None
) -> ChainFactory:
    """
    Factory function to create ChainFactory instance.
    
    Args:
        settings: Application settings (auto-detected if None)
        tool_registry: Tool registry instance (auto-created if None)
        llm: Optional pre-configured LLM (for dependency injection in tests)
        
    Returns:
        ChainFactory: Initialized chain factory
        
    Raises:
        DependencyError: If factory cannot be created
    """
    # Professional pattern: Auto-detect dependencies for testing flexibility
    if settings is None:
        from ..core.config import get_settings
        settings = get_settings()
    
    if tool_registry is None:
        tool_registry = ToolRegistry()
    
    return ChainFactory(settings, tool_registry, llm)


def create_all_chains(chain_factory: ChainFactory) -> Dict[str, RunnableSequence]:
    """
    Create all standard chains for the ReAct agent.
    
    Args:
        chain_factory: Initialized chain factory
        
    Returns:
        Dict[str, RunnableSequence]: Dictionary of all chains
    """
    chains = {
        "reasoning": chain_factory.create_reasoning_chain(),
        "analysis": chain_factory.create_analysis_chain(),
        "validation": chain_factory.create_validation_chain(),
        "tool_selection": chain_factory.create_tool_selection_chain(),
        "error_recovery": chain_factory.create_error_recovery_chain(),
        "response": chain_factory.create_response_chain(),
    }
    
    logger.info(f"Created {len(chains)} chains for ReAct agent")
    return chains
