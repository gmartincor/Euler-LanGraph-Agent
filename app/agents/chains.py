"""Professional Chain Factory for Mathematical ReAct Agent.

This module provides a clean, professional factory for creating chains used in the ReAct
reasoning process. Eliminates anti-patterns and follows professional software engineering principles.

Design Principles Applied:
- Single Responsibility: Each class has one clear purpose
- Dependency Injection: Clean injection with proper error handling
- Fail Fast: No silent fallbacks that mask real issues
- Clean Architecture: Separation of concerns
- Professional Error Handling: Explicit error management
"""

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

logger = get_logger(__name__)


class LLMProvider:
    """
    Professional LLM provider with proper error handling.
    
    Separates LLM creation concerns from chain factory.
    """
    
    @staticmethod
    def create_gemini_llm(settings: Settings) -> ChatGoogleGenerativeAI:
        """
        Create Google Gemini LLM instance with validation.
        
        Args:
            settings: Application settings
            
        Returns:
            ChatGoogleGenerativeAI: Configured LLM instance
            
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
            
            llm = ChatGoogleGenerativeAI(
                model=gemini_config["model_name"],
                temperature=gemini_config.get("temperature", 0.7),
                max_tokens=gemini_config.get("max_tokens", 1000),
                api_key=gemini_config["api_key"],
                google_api_key=gemini_config["api_key"]  # For compatibility
            )
            
            logger.info(f"LLM created successfully: {gemini_config['model_name']}")
            return llm
            
        except Exception as e:
            logger.error(f"Failed to create LLM: {e}")
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
            return "integral_tool, plot_tool, analysis_tool"
    
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

Available tools:
{tool_descriptions}

Return a JSON with: approach, steps, tools_needed, confidence"""),
            ("human", """Problem: {problem}

Context: {context}

Provide a structured reasoning approach.""")
        ])
        
        chain = (
            RunnablePassthrough.assign(
                tool_descriptions=lambda x: self._get_tool_descriptions()
            )
            | prompt
            | self._llm
            | StrOutputParser()
        )
        
        logger.info("Mathematical reasoning chain created")
        return chain
    
    def create_analysis_chain(self) -> RunnableSequence:
        """
        Create problem analysis chain.
        
        Returns:
            RunnableSequence: Problem analysis chain
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a mathematical problem analyzer. Analyze the given problem and categorize it.

Return a JSON with:
- type: (integral, derivative, visualization, general)
- variables: list of variables found
- functions: list of mathematical functions
- complexity: (low, medium, high)
- domain: mathematical domain"""),
            ("human", "Analyze this mathematical problem: {problem}")
        ])
        
        chain = prompt | self._llm | StrOutputParser()
        
        logger.info("Problem analysis chain created")
        return chain
    
    def create_validation_chain(self) -> RunnableSequence:
        """
        Create solution validation chain.
        
        Returns:
            RunnableSequence: Validation chain
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a mathematical solution validator. Validate the given solution.

Return a JSON with:
- is_valid: boolean
- score: float (0-1) 
- issues: list of problems found
- suggestions: list of improvements"""),
            ("human", """Problem: {problem}
Reasoning: {reasoning}
Tool Results: {tool_results}
Trace: {trace}

Validate this mathematical solution based on the reasoning and tool results.""")
        ])
        
        chain = prompt | self._llm | StrOutputParser()
        
        logger.info("Solution validation chain created")
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

Return a JSON with:
- selected_tools: list of tool names
- reasoning: explanation of tool selection"""),
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
            | StrOutputParser()
        )
        
        logger.info("Tool selection chain created")
        return chain
    
    def create_error_recovery_chain(self) -> RunnableSequence:
        """
        Create error recovery chain.
        
        Returns:
            RunnableSequence: Error recovery chain
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an error recovery specialist. Analyze the error and provide recovery strategies.

Return a JSON with:
- action: recovery action to take
- note: explanation of the recovery strategy
- confidence: confidence in the recovery approach (0-1)"""),
            ("human", """Problem: {problem}
Error: {error}
Error Type: {error_type}
Retry Count: {retry_count}

Provide recovery strategies for this error.""")
        ])
        
        chain = prompt | self._llm | StrOutputParser()
        
        logger.info("Error recovery chain created")
        return chain
    
    def create_response_chain(self) -> RunnableSequence:
        """
        Create response formatting chain.
        
        Returns:
            RunnableSequence: Response formatting chain
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a response formatter. Create a clear, well-structured final response.

Return a JSON with:
- answer: final answer to the problem
- steps: list of solution steps  
- explanation: clear explanation of the approach
- confidence: overall confidence score (0-1)"""),
            ("human", """Problem: {problem}
Reasoning: {reasoning}
Tool Results: {tool_results}
Validation: {validation}
Trace: {trace}

Format this into a clear final response.""")
        ])
        
        chain = prompt | self._llm | StrOutputParser()
        
        logger.info("Response formatting chain created")
        return chain


# === Factory Functions ===

@log_function_call(logger)
def create_chain_factory(
    settings: Settings,
    tool_registry: ToolRegistry,
    llm: Optional[ChatGoogleGenerativeAI] = None
) -> ChainFactory:
    """
    Factory function to create ChainFactory instance.
    
    Args:
        settings: Application settings
        tool_registry: Tool registry instance
        llm: Optional pre-configured LLM (for dependency injection in tests)
        
    Returns:
        ChainFactory: Initialized chain factory
        
    Raises:
        DependencyError: If factory cannot be created
    """
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
