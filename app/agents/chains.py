from typing import Any, Dict, List, Optional, Callable, Union
from functools import lru_cache
import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableLambda

from ..core.config import Settings
from ..core.logging import get_logger, log_function_call
from ..core.exceptions import ConfigurationError, DependencyError
from ..tools.registry import ToolRegistry
from ..utils.robust_parser import RobustJsonOutputParser
from .prompts import get_template_registry, build_tool_description

logger = get_logger(__name__)


class LLMProvider:
    """LLM provider with error handling."""
    
    @staticmethod
    def create_gemini_llm(settings: Settings) -> ChatGoogleGenerativeAI:
        """Create Google Gemini LLM instance."""
        try:
            gemini_config = settings.gemini_config
            
            required_fields = ["model_name", "api_key"]
            for field in required_fields:
                if not gemini_config.get(field):
                    raise ConfigurationError(f"Missing required Gemini config: {field}")
            
            safety_settings = gemini_config.get("safety_settings", [])
            
            llm = ChatGoogleGenerativeAI(
                model=gemini_config["model_name"],  
                temperature=gemini_config.get("temperature", 0.1),
                max_output_tokens=gemini_config.get("max_output_tokens", 8192),  
                top_p=gemini_config.get("top_p", 0.9),
                top_k=gemini_config.get("top_k", 40),
                api_key=gemini_config["api_key"],
                google_api_key=gemini_config["api_key"],  
                convert_system_message_to_human=True,  
            )
            
            logger.info(f"Gemini 2.5 Flash LLM created successfully: {gemini_config['model_name']}")
            return llm
            
        except Exception as e:
            logger.error(f"Failed to create Gemini 2.5 Flash LLM: {e}")
            raise DependencyError(f"Could not initialize LLM: {str(e)}") from e


class ChainFactory:
    """
    Factory for creating mathematical reasoning chains with centralized templates.
    """
    
    def __init__(
        self,
        settings: Settings,
        tool_registry: ToolRegistry,
        llm: Optional[ChatGoogleGenerativeAI] = None
    ):
        """Initialize chain factory."""
        self.settings = settings
        self.tool_registry = tool_registry
        self.prompt_registry = get_template_registry()
        
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
        """Get formatted tool descriptions."""
        try:
            tools = self.tool_registry.get_all_tools()
            tool_info = {}
            
            for tool in tools:
                if hasattr(tool, 'name') and hasattr(tool, 'description'):
                    tool_info[tool.name] = {"description": tool.description}
                else:
                    tool_info[str(tool)] = {"description": "Mathematical tool"}
                    
            return build_tool_description(tool_info)
            
        except Exception as e:
            logger.warning(f"Could not get tool descriptions: {e}")
            return "integral_calculator, plot_generator, function_analyzer"
    
    def _create_base_prompt_chain(self, template_name: str, human_message: str) -> RunnableSequence:
        """Create base prompt chain using centralized templates."""
        template = self.prompt_registry.get_template(template_name)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", template.template),
            ("human", human_message)
        ])
        
        return prompt | self._llm | RobustJsonOutputParser()
    
    def create_reasoning_chain(self) -> RunnableSequence:
        """Create mathematical reasoning chain."""
        chain = (
            RunnablePassthrough.assign(
                tool_descriptions=lambda x: self._get_tool_descriptions()
            )
            | self._create_base_prompt_chain(
                "mathematical_reasoning",
                """Problem: {problem}

Context: {context}

Analyze this problem and return the JSON structure."""
            )
        )
        
        logger.info("Mathematical reasoning chain created")
        return chain
    
    def create_analysis_chain(self) -> RunnableSequence:
        """Create problem analysis chain."""
        chain = self._create_base_prompt_chain(
            "problem_analysis",
            "Analyze this mathematical problem: {problem}"
        )
        
        logger.info("Problem analysis chain created")
        return chain
    
    def create_validation_chain(self) -> RunnableSequence:
        """Create solution validation chain."""
        chain = self._create_base_prompt_chain(
            "validation",
            """Problem: {problem}
Reasoning: {reasoning}
Tool Results: {tool_results}
Trace: {trace}

Validate this mathematical solution based on the reasoning and tool results."""
        )
        
        logger.info("Solution validation chain created")
        return chain
    
    def create_tool_selection_chain(self) -> RunnableSequence:
        """Create tool selection chain."""
        chain = (
            RunnablePassthrough.assign(
                tool_descriptions=lambda x: self._get_tool_descriptions()
            )
            | self._create_base_prompt_chain(
                "tool_selection",
                """Problem: {problem}
Problem Type: {problem_type}
Analysis: {analysis}

Select the best tools for this problem."""
            )
        )
        
        logger.info("Tool selection chain created")
        return chain
    
    def create_error_recovery_chain(self) -> RunnableSequence:
        """Create error recovery chain."""
        chain = self._create_base_prompt_chain(
            "error_recovery",
            """Problem: {problem}
Error: {error}
Error Type: {error_type}
Retry Count: {retry_count}

Provide recovery strategies for this error."""
        )
        
        logger.info("Error recovery chain created")
        return chain
    
    def create_response_chain(self) -> RunnableSequence:
        """Create response formatting chain."""
        chain = self._create_base_prompt_chain(
            "response_formatting",
            """Problem: {problem}
Reasoning: {reasoning}
Tool Results: {tool_results}
Validation: {validation}
Trace: {trace}

Format this into a clear final response JSON."""
        )
        
        logger.info("Response formatting chain created")
        return chain


# === Factory Functions ===

@log_function_call(logger)
def create_chain_factory(
    settings: Optional[Settings] = None,
    tool_registry: Optional[ToolRegistry] = None,
    llm: Optional[ChatGoogleGenerativeAI] = None
) -> ChainFactory:
    """Create ChainFactory instance."""
    # Auto-detect dependencies for testing flexibility
    if settings is None:
        from ..core.config import get_settings
        settings = get_settings()
    
    if tool_registry is None:
        tool_registry = ToolRegistry()
    
    return ChainFactory(settings, tool_registry, llm)


def create_all_chains(chain_factory: ChainFactory) -> Dict[str, RunnableSequence]:
    """Create all standard chains for the ReAct agent."""
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
