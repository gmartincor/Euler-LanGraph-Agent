"""Mathematical Reasoning Engine - Core Business Logic.

This module contains the pure mathematical reasoning logic extracted from 
ReactMathematicalAgent, following professional design patterns and eliminating
circular dependencies.

Key Design Principles Applied:
- Single Responsibility: Only mathematical reasoning logic
- Dependency Injection: Clean interfaces without circular refs
- Pure Business Logic: No workflow orchestration concerns
- Professional Error Handling: Comprehensive exception management
- Zero Duplication: Consolidates scattered reasoning logic

Architecture Benefits:
- Testable: Standalone reasoning without agent dependencies
- Reusable: Can be integrated with different workflow engines
- Maintainable: Clear separation of concerns
- Professional: Follows SOLID principles
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import asyncio

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from ...core.config import Settings
from ...core.logging import get_logger, log_function_call
from ...core.exceptions import AgentError, ValidationError
from ...tools.registry import ToolRegistry
from ...core.bigtool_setup import BigToolManager
from ..state import MathAgentState
from ..chains import ChainFactory
from ..prompts import get_prompt_template, build_tool_description, format_mathematical_context

logger = get_logger(__name__)


class MathematicalReasoner:
    """
    Pure mathematical reasoning engine without workflow dependencies.
    
    This class implements the core mathematical reasoning logic that was
    previously scattered across ReactMathematicalAgent. It follows the
    Single Responsibility Principle by focusing only on mathematical
    problem analysis and reasoning.
    
    Key Features:
    - Problem classification and complexity assessment
    - Mathematical context extraction and formatting
    - Step-by-step reasoning chain execution
    - Result validation and confidence scoring
    - Error analysis and recovery suggestions
    
    Design Philosophy:
    - PURE LOGIC: No workflow orchestration or state management
    - INJECTABLE: Dependencies provided via constructor
    - TESTABLE: All methods can be tested independently
    - REUSABLE: Can be used with different workflow engines
    """
    
    def __init__(
        self,
        llm: ChatGoogleGenerativeAI,
        chain_factory: ChainFactory,
        settings: Settings
    ):
        """
        Initialize the mathematical reasoning engine.
        
        Args:
            llm: Configured language model for reasoning
            chain_factory: Factory for creating reasoning chains
            settings: Application settings for configuration
        """
        self.llm = llm
        self.chain_factory = chain_factory
        self.settings = settings
        
        # Create reasoning chains (dependency injection pattern)
        self._chains = self._initialize_chains()
        
        logger.info("MathematicalReasoner initialized with configured LLM")
    
    def _initialize_chains(self) -> Dict[str, Any]:
        """Initialize all reasoning chains using the factory."""
        try:
            return {
                'analysis': self.chain_factory.create_analysis_chain(),
                'reasoning': self.chain_factory.create_reasoning_chain(),
                'validation': self.chain_factory.create_validation_chain(),
                'error_recovery': self.chain_factory.create_error_recovery_chain()
            }
        except Exception as e:
            logger.error(f"Failed to initialize reasoning chains: {e}")
            raise AgentError(f"Reasoning engine initialization failed: {e}") from e
    
    @log_function_call(logger)
    async def analyze_problem(
        self, 
        problem: str, 
        mathematical_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze mathematical problem and classify its characteristics.
        
        This method implements comprehensive problem analysis including:
        - Problem type classification (integral, derivative, algebraic, etc.)
        - Complexity assessment (basic, intermediate, advanced)
        - Required tools identification
        - Solution strategy recommendation
        
        Args:
            problem: Mathematical problem statement
            mathematical_context: Optional context from previous reasoning
            
        Returns:
            Dict containing problem analysis results:
            - problem_type: Classification of the mathematical problem
            - complexity: Assessed difficulty level
            - required_tools: List of recommended tools
            - strategy: Suggested solution approach
            - confidence: Analysis confidence score (0-1)
            
        Raises:
            ValidationError: If problem cannot be parsed
            AgentError: If analysis process fails
        """
        if not problem or not problem.strip():
            raise ValidationError("Problem statement cannot be empty")
        
        try:
            # Format input for analysis chain
            analysis_input = {
                'problem': problem.strip(),
                'context': format_mathematical_context(mathematical_context or {}),
                'available_capabilities': self._get_available_capabilities()
            }
            
            # Execute analysis chain
            analysis_result = await self._chains['analysis'].ainvoke(analysis_input)
            
            # Parse and validate results
            parsed_analysis = self._parse_analysis_result(analysis_result)
            
            logger.info(
                f"Problem analysis completed: {parsed_analysis.get('problem_type', 'unknown')} "
                f"(confidence: {parsed_analysis.get('confidence', 0):.2f})"
            )
            
            return parsed_analysis
            
        except Exception as e:
            logger.error(f"Problem analysis failed: {e}", exc_info=True)
            raise AgentError(f"Mathematical problem analysis failed: {str(e)}") from e
    
    @log_function_call(logger)
    async def perform_reasoning(
        self,
        problem: str,
        analysis_results: Dict[str, Any],
        available_tools: List[str],
        previous_attempts: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Perform step-by-step mathematical reasoning.
        
        This method implements the core ReAct reasoning process:
        - Step-by-step problem decomposition
        - Tool selection and usage planning
        - Mathematical strategy formulation
        - Intermediate result validation
        
        Args:
            problem: Original mathematical problem
            analysis_results: Results from problem analysis
            available_tools: List of available tool names
            previous_attempts: Optional previous reasoning attempts
            
        Returns:
            Dict containing reasoning results:
            - reasoning_steps: List of reasoning steps
            - tool_plan: Planned tool usage sequence
            - expected_outcome: Predicted result characteristics
            - confidence: Reasoning confidence score (0-1)
            - next_action: Recommended next step
            
        Raises:
            AgentError: If reasoning process fails
        """
        try:
            # Prepare reasoning context
            reasoning_input = {
                'problem': problem,
                'analysis': analysis_results,
                'available_tools': available_tools,
                'tool_descriptions': build_tool_description(available_tools),
                'previous_attempts': previous_attempts or [],
                'reasoning_depth': self.settings.agent_max_iterations
            }
            
            # Execute reasoning chain
            reasoning_result = await self._chains['reasoning'].ainvoke(reasoning_input)
            
            # Parse and structure results
            structured_reasoning = self._parse_reasoning_result(reasoning_result)
            
            logger.info(
                f"Reasoning completed with {len(structured_reasoning.get('reasoning_steps', []))} steps "
                f"(confidence: {structured_reasoning.get('confidence', 0):.2f})"
            )
            
            return structured_reasoning
            
        except Exception as e:
            logger.error(f"Mathematical reasoning failed: {e}", exc_info=True)
            raise AgentError(f"Reasoning process failed: {str(e)}") from e
    
    @log_function_call(logger)
    async def validate_results(
        self,
        problem: str,
        reasoning_steps: List[Dict[str, Any]],
        tool_results: Dict[str, Any],
        expected_outcome: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate mathematical results and assess solution quality.
        
        This method implements comprehensive result validation:
        - Mathematical correctness verification
        - Consistency checking across tool results
        - Completeness assessment
        - Confidence scoring
        - Error identification
        
        Args:
            problem: Original mathematical problem
            reasoning_steps: Executed reasoning steps
            tool_results: Results from mathematical tools
            expected_outcome: Expected result characteristics
            
        Returns:
            Dict containing validation results:
            - is_valid: Boolean indicating if results are valid
            - confidence: Validation confidence score (0-1)
            - completeness: Solution completeness score (0-1)
            - identified_errors: List of identified issues
            - recommendations: Suggestions for improvement
            
        Raises:
            ValidationError: If validation process fails
        """
        try:
            # Structure validation input
            validation_input = {
                'problem': problem,
                'reasoning_steps': reasoning_steps,
                'tool_results': tool_results,
                'expected_outcome': expected_outcome,
                'validation_criteria': self._get_validation_criteria()
            }
            
            # Execute validation chain
            validation_result = await self._chains['validation'].ainvoke(validation_input)
            
            # Parse validation results
            structured_validation = self._parse_validation_result(validation_result)
            
            # Log validation outcome
            is_valid = structured_validation.get('is_valid', False)
            confidence = structured_validation.get('confidence', 0)
            logger.info(
                f"Result validation: {'PASSED' if is_valid else 'FAILED'} "
                f"(confidence: {confidence:.2f})"
            )
            
            return structured_validation
            
        except Exception as e:
            logger.error(f"Result validation failed: {e}", exc_info=True)
            raise ValidationError(f"Validation process failed: {str(e)}") from e
    
    @log_function_call(logger)
    async def generate_error_recovery(
        self,
        problem: str,
        failed_attempts: List[Dict[str, Any]],
        error_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate error recovery strategy for failed reasoning attempts.
        
        Args:
            problem: Original mathematical problem
            failed_attempts: List of failed reasoning attempts
            error_analysis: Analysis of what went wrong
            
        Returns:
            Dict containing recovery strategy:
            - alternative_approach: Suggested alternative method
            - modified_strategy: Adjusted reasoning strategy
            - tool_alternatives: Alternative tool suggestions
            - confidence: Recovery strategy confidence (0-1)
            
        Raises:
            AgentError: If recovery generation fails
        """
        try:
            recovery_input = {
                'problem': problem,
                'failed_attempts': failed_attempts,
                'error_analysis': error_analysis,
                'available_alternatives': self._get_alternative_strategies()
            }
            
            recovery_result = await self._chains['error_recovery'].ainvoke(recovery_input)
            
            structured_recovery = self._parse_recovery_result(recovery_result)
            
            logger.info(
                f"Error recovery generated: {structured_recovery.get('alternative_approach', 'unknown')}"
            )
            
            return structured_recovery
            
        except Exception as e:
            logger.error(f"Error recovery generation failed: {e}", exc_info=True)
            raise AgentError(f"Recovery generation failed: {str(e)}") from e
    
    # === Private Helper Methods ===
    
    def _get_available_capabilities(self) -> List[str]:
        """Get list of available mathematical capabilities."""
        return [
            'symbolic_integration',
            'numerical_integration', 
            'function_analysis',
            'plotting_visualization',
            'algebraic_manipulation',
            'equation_solving'
        ]
    
    def _get_validation_criteria(self) -> Dict[str, Any]:
        """Get validation criteria for mathematical results."""
        return {
            'numerical_accuracy': 0.01,
            'symbolic_consistency': True,
            'dimensional_analysis': True,
            'boundary_conditions': True,
            'mathematical_properties': True
        }
    
    def _get_alternative_strategies(self) -> List[str]:
        """Get list of alternative solution strategies."""
        return [
            'analytical_approach',
            'numerical_approximation',
            'graphical_analysis',
            'series_expansion',
            'substitution_method',
            'integration_by_parts'
        ]
    
    def _parse_analysis_result(self, result: Any) -> Dict[str, Any]:
        """Parse and structure problem analysis result."""
        # Implementation for parsing LLM analysis output
        # This would include robust parsing logic
        if isinstance(result, dict):
            return result
        
        # Fallback parsing for string results
        return {
            'problem_type': 'unknown',
            'complexity': 'medium',
            'required_tools': [],
            'strategy': 'analytical',
            'confidence': 0.5
        }
    
    def _parse_reasoning_result(self, result: Any) -> Dict[str, Any]:
        """Parse and structure reasoning chain result."""
        if isinstance(result, dict):
            return result
            
        return {
            'reasoning_steps': [],
            'tool_plan': [],
            'expected_outcome': {},
            'confidence': 0.5,
            'next_action': 'use_tools'
        }
    
    def _parse_validation_result(self, result: Any) -> Dict[str, Any]:
        """Parse and structure validation result."""
        if isinstance(result, dict):
            return result
            
        return {
            'is_valid': False,
            'confidence': 0.5,
            'completeness': 0.5,
            'identified_errors': [],
            'recommendations': []
        }
    
    def _parse_recovery_result(self, result: Any) -> Dict[str, Any]:
        """Parse and structure error recovery result."""
        if isinstance(result, dict):
            return result
            
        return {
            'alternative_approach': 'retry_with_different_method',
            'modified_strategy': {},
            'tool_alternatives': [],
            'confidence': 0.5
        }
