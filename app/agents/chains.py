"""Chain factories for mathematical ReAct agent.

This module provides factory functions for creating different types of chains
used in the ReAct reasoning process. Following the Factory pattern and DRY
principles, it reuses existing infrastructure while providing specialized
mathematical reasoning capabilities.

Key features:
- Factory pattern for chain creation
- Integration with existing ToolRegistry
- Reuse of configuration and logging systems
- Mathematical problem specialization
"""

from typing import Any, Dict, List, Optional, Callable
from langchain_core.runnables import RunnableSequence, RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from ..core.config import Settings
from ..core.logging import get_logger, log_function_call
from ..tools.registry import ToolRegistry
from ..models.agent_state import AgentMemory
from .prompts import (
    MATHEMATICAL_REASONING_PROMPT,
    TOOL_SELECTION_PROMPT,
    REFLECTION_PROMPT,
    PROBLEM_ANALYSIS_PROMPT,
    ERROR_RECOVERY_PROMPT,
    build_tool_description,
    format_mathematical_context
)

logger = get_logger(__name__)


class ChainFactory:
    """
    Factory class for creating specialized chains for mathematical ReAct agent.
    
    This factory integrates with existing infrastructure:
    - Uses existing Settings configuration
    - Integrates with ToolRegistry
    - Reuses logging system
    - Follows established patterns
    """
    
    def __init__(
        self,
        settings: Settings,
        tool_registry: ToolRegistry,
        llm: Optional[ChatGoogleGenerativeAI] = None
    ):
        """
        Initialize chain factory.
        
        Args:
            settings: Application settings
            tool_registry: Existing tool registry
            llm: Optional pre-configured LLM (creates new if None)
        """
        self.settings = settings
        self.tool_registry = tool_registry
        self.llm = llm or self._create_llm()
        
        logger.info("ChainFactory initialized with existing infrastructure")
    
    def _create_llm(self) -> ChatGoogleGenerativeAI:
        """
        Create LLM using existing configuration.
        
        Returns:
            ChatGoogleGenerativeAI: Configured Gemini LLM
        """
        gemini_config = self.settings.gemini_config
        
        return ChatGoogleGenerativeAI(
            model=gemini_config["model_name"],
            api_key=gemini_config["api_key"],
            temperature=gemini_config["temperature"],
            max_output_tokens=gemini_config["max_tokens"],
            convert_system_message_to_human=True,  # Gemini requirement
        )
    
    @log_function_call(logger)
    def create_reasoning_chain(self) -> RunnableSequence:
        """
        Create mathematical reasoning chain.
        
        This chain handles the core ReAct reasoning process for mathematical
        problems, integrating with existing tool infrastructure.
        
        Returns:
            RunnableSequence: Complete reasoning chain
        """
        # Create dynamic tool description function
        def get_tools_description(_):
            tools_info = {}
            for tool_name in self.tool_registry.list_tools():
                tool = self.tool_registry.get_tool(tool_name)
                if tool:
                    tools_info[tool_name] = {
                        'description': tool.description,
                        'usage_stats': tool.usage_stats
                    }
            return build_tool_description(tools_info)
        
        # Create the reasoning prompt
        reasoning_prompt = PromptTemplate.from_template(MATHEMATICAL_REASONING_PROMPT)
        
        # Build the chain
        chain = (
            {
                "problem": lambda x: x["problem"],
                "context": lambda x: format_mathematical_context(x.get("context", {})),
                "available_tools": RunnableLambda(get_tools_description)
            }
            | reasoning_prompt
            | self.llm
            | StrOutputParser()
        )
        
        logger.info("Mathematical reasoning chain created")
        return chain
    
    @log_function_call(logger)
    def create_tool_selection_chain(self) -> RunnableSequence:
        """
        Create intelligent tool selection chain.
        
        This chain uses existing ToolRegistry infrastructure to provide
        intelligent tool recommendations based on problem analysis.
        
        Returns:
            RunnableSequence: Tool selection chain
        """
        def get_available_tools_info(input_data):
            """Get enhanced tool information for selection."""
            problem_type = input_data.get("problem_type", "general")
            
            # Use existing search functionality
            relevant_tools = self.tool_registry.search_tools(
                query=problem_type,
                limit=self.settings.tool_search_top_k
            )
            
            tools_info = {}
            for result in relevant_tools:
                tool_name = result["tool_name"]
                tool = self.tool_registry.get_tool(tool_name)
                if tool:
                    tools_info[tool_name] = {
                        'description': tool.description,
                        'capabilities': [problem_type],  # Could be enhanced
                        'usage_stats': tool.usage_stats,
                        'relevance_score': result["score"]
                    }
            
            return build_tool_description(tools_info)
        
        # Create the tool selection prompt
        selection_prompt = PromptTemplate.from_template(TOOL_SELECTION_PROMPT)
        
        # Build the chain
        chain = (
            {
                "problem": lambda x: x["problem"],
                "problem_type": lambda x: x.get("problem_type", "general"),
                "mathematical_context": lambda x: format_mathematical_context(x.get("context", {})),
                "available_tools_description": RunnableLambda(get_available_tools_info),
                "previous_results": lambda x: x.get("previous_results", "None")
            }
            | selection_prompt
            | self.llm
            | StrOutputParser()
        )
        
        logger.info("Tool selection chain created")
        return chain
    
    @log_function_call(logger)
    def create_validation_chain(self) -> RunnableSequence:
        """
        Create result validation and reflection chain.
        
        This chain validates mathematical results and provides reflection
        on the solution quality and correctness.
        
        Returns:
            RunnableSequence: Validation chain
        """
        reflection_prompt = PromptTemplate.from_template(REFLECTION_PROMPT)
        
        # Build the chain
        chain = (
            {
                "problem": lambda x: x["problem"],
                "tools_used": lambda x: x.get("tools_used", []),
                "results": lambda x: x.get("results", "No results"),
                "solution_steps": lambda x: x.get("solution_steps", [])
            }
            | reflection_prompt
            | self.llm
            | StrOutputParser()
        )
        
        logger.info("Validation chain created")
        return chain
    
    @log_function_call(logger)
    def create_analysis_chain(self) -> RunnableSequence:
        """
        Create problem analysis chain.
        
        This chain analyzes incoming problems to determine type,
        complexity, and solution strategy.
        
        Returns:
            RunnableSequence: Problem analysis chain
        """
        analysis_prompt = PromptTemplate.from_template(PROBLEM_ANALYSIS_PROMPT)
        
        # Build the chain
        chain = (
            {
                "problem": lambda x: x["problem"],
                "user_context": lambda x: x.get("user_context", "No additional context")
            }
            | analysis_prompt
            | self.llm
            | StrOutputParser()
        )
        
        logger.info("Problem analysis chain created")
        return chain
    
    @log_function_call(logger)
    def create_error_recovery_chain(self) -> RunnableSequence:
        """
        Create error recovery chain.
        
        This chain handles errors during the ReAct process and provides
        recovery strategies and alternative approaches.
        
        Returns:
            RunnableSequence: Error recovery chain
        """
        recovery_prompt = PromptTemplate.from_template(ERROR_RECOVERY_PROMPT)
        
        # Build the chain
        chain = (
            {
                "error_type": lambda x: x.get("error_type", "Unknown"),
                "error_message": lambda x: x.get("error_message", "No message"),
                "failed_action": lambda x: x.get("failed_action", "Unknown action"),
                "error_context": lambda x: x.get("error_context", "No context"),
                "current_problem": lambda x: x.get("current_problem", "No problem"),
                "current_progress": lambda x: x.get("current_progress", "No progress"),
                "previous_results": lambda x: x.get("previous_results", "No results")
            }
            | recovery_prompt
            | self.llm
            | StrOutputParser()
        )
        
        logger.info("Error recovery chain created")
        return chain
    
    def create_response_chain(self) -> RunnableSequence:
        """
        Create final response formatting chain.
        
        This chain formats the final response for the user, ensuring
        clarity and completeness of the mathematical solution.
        
        Returns:
            RunnableSequence: Response formatting chain
        """
        response_template = """**Mathematical Solution**

**Problem**: {problem}

**Solution Process**:
{solution_steps}

**Final Answer**: {final_answer}

**Confidence**: {confidence_score}/1.0

**Tools Used**: {tools_used}

**Verification**: {verification_status}

{additional_notes}"""
        
        response_prompt = PromptTemplate.from_template(response_template)
        
        # Simple formatting chain
        chain = (
            {
                "problem": lambda x: x["problem"],
                "solution_steps": lambda x: "\n".join(x.get("solution_steps", [])),
                "final_answer": lambda x: x.get("final_answer", "No answer provided"),
                "confidence_score": lambda x: x.get("confidence_score", 0.0),
                "tools_used": lambda x: ", ".join(x.get("tools_used", [])),
                "verification_status": lambda x: x.get("verification_status", "Not verified"),
                "additional_notes": lambda x: x.get("additional_notes", "")
            }
            | response_prompt
            | StrOutputParser()
        )
        
        logger.info("Response formatting chain created")
        return chain


# === Factory Functions (following functional approach) ===

@log_function_call(logger)
def create_chain_factory(
    settings: Settings,
    tool_registry: ToolRegistry,
    llm: Optional[ChatGoogleGenerativeAI] = None
) -> ChainFactory:
    """
    Factory function to create ChainFactory instance.
    
    This function follows the factory pattern and ensures proper
    initialization with existing infrastructure.
    
    Args:
        settings: Application settings
        tool_registry: Existing tool registry
        llm: Optional pre-configured LLM
        
    Returns:
        ChainFactory: Initialized chain factory
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
        "tool_selection": chain_factory.create_tool_selection_chain(),
        "validation": chain_factory.create_validation_chain(),
        "analysis": chain_factory.create_analysis_chain(),
        "error_recovery": chain_factory.create_error_recovery_chain(),
        "response": chain_factory.create_response_chain(),
    }
    
    logger.info(f"Created {len(chains)} chains for ReAct agent")
    return chains


# === Utility Functions ===

def create_custom_chain(
    prompt_template: str,
    llm: ChatGoogleGenerativeAI,
    input_mapper: Optional[Callable] = None,
    output_parser: Optional[Callable] = None
) -> RunnableSequence:
    """
    Create a custom chain with specified components.
    
    Args:
        prompt_template: Template string for the prompt
        llm: Language model instance
        input_mapper: Optional function to map inputs
        output_parser: Optional output parser
        
    Returns:
        RunnableSequence: Custom chain
    """
    prompt = PromptTemplate.from_template(prompt_template)
    parser = output_parser or StrOutputParser()
    
    if input_mapper:
        chain = input_mapper | prompt | llm | parser
    else:
        chain = prompt | llm | parser
    
    return chain
