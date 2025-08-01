"""Tool Orchestration Engine - Professional Tool Management.

This module contains the pure tool orchestration logic extracted from 
ReactMathematicalAgent, following professional design patterns and eliminating
circular dependencies.

Key Design Principles Applied:
- Single Responsibility: Only tool selection and execution logic
- Dependency Injection: Clean interfaces for tool registry and BigTool
- Strategy Pattern: Different orchestration strategies for different problem types
- Professional Error Handling: Comprehensive tool execution management
- Zero Duplication: Consolidates scattered tool logic

Architecture Benefits:
- Testable: Standalone tool orchestration without agent dependencies
- Reusable: Can be integrated with different reasoning engines
- Maintainable: Clear separation between reasoning and tool execution
- Professional: Follows SOLID principles and async patterns
"""

from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime
import asyncio

from ...core.logging import get_logger, log_function_call
from ...core.exceptions import AgentError, ToolError, ValidationError
from ...tools.registry import ToolRegistry
from ...tools.base import BaseTool
from ...core.bigtool_setup import BigToolManager
from ..state import MathAgentState

logger = get_logger(__name__)


class ToolOrchestrator:
    """
    Professional tool orchestration engine for mathematical problem solving.
    
    This class implements intelligent tool selection, execution, and result
    management without coupling to specific workflow engines or reasoning
    components. It provides a clean interface for tool orchestration.
    
    Key Features:
    - Semantic tool selection using BigTool integration
    - Parallel tool execution for performance optimization
    - Tool result validation and quality assessment
    - Error handling and retry mechanisms
    - Usage statistics and performance monitoring
    
    Design Philosophy:
    - STRATEGY PATTERN: Different orchestration strategies for different problems
    - ASYNC OPTIMIZED: Non-blocking tool execution with proper concurrency
    - ERROR RESILIENT: Comprehensive error handling and recovery
    - PERFORMANCE FOCUSED: Caching and optimization for tool operations
    """
    
    def __init__(
        self,
        tool_registry: ToolRegistry,
        bigtool_manager: Optional[BigToolManager] = None,
        max_concurrent_tools: int = 3,
        tool_timeout_seconds: int = 30
    ):
        """
        Initialize the tool orchestration engine.
        
        Args:
            tool_registry: Registry containing available mathematical tools
            bigtool_manager: Optional BigTool manager for semantic search
            max_concurrent_tools: Maximum tools to execute concurrently
            tool_timeout_seconds: Timeout for individual tool execution
        """
        self.tool_registry = tool_registry
        self.bigtool_manager = bigtool_manager
        self.max_concurrent_tools = max_concurrent_tools
        self.tool_timeout_seconds = tool_timeout_seconds
        
        # Performance tracking
        self._execution_stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'average_execution_time': 0.0
        }
        
        logger.info(
            f"ToolOrchestrator initialized with {len(tool_registry.list_tools())} tools "
            f"(BigTool: {'enabled' if bigtool_manager else 'disabled'})"
        )
    
    @log_function_call(logger)
    async def select_optimal_tools(
        self,
        problem_type: str,
        problem_description: str,
        required_capabilities: List[str],
        previous_tool_results: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Select optimal tools for the given mathematical problem.
        
        This method implements intelligent tool selection using both
        rule-based selection and semantic search via BigTool integration.
        
        Args:
            problem_type: Type of mathematical problem (integral, derivative, etc.)
            problem_description: Detailed problem description
            required_capabilities: List of required mathematical capabilities
            previous_tool_results: Results from previously executed tools
            
        Returns:
            List[str]: Ordered list of recommended tool names
            
        Raises:
            ValidationError: If selection criteria are invalid
            AgentError: If tool selection process fails
        """
        if not problem_type or not problem_description:
            raise ValidationError("Problem type and description are required for tool selection")
        
        try:
            # Get base tool recommendations using rule-based selection
            rule_based_tools = self._get_rule_based_tools(problem_type, required_capabilities)
            
            # Enhance with semantic search if BigTool is available
            semantic_tools = []
            if self.bigtool_manager and await self.bigtool_manager.health_check():
                semantic_tools = await self._get_semantic_tools(
                    problem_description, 
                    required_capabilities
                )
            
            # Combine and rank tools
            combined_tools = self._combine_and_rank_tools(
                rule_based_tools,
                semantic_tools,
                previous_tool_results
            )
            
            # Filter for availability and capabilities
            available_tools = self._filter_available_tools(combined_tools)
            
            logger.info(
                f"Tool selection completed: {len(available_tools)} tools selected "
                f"for {problem_type} problem"
            )
            
            return available_tools[:5]  # Limit to top 5 tools
            
        except Exception as e:
            logger.error(f"Tool selection failed: {e}", exc_info=True)
            raise AgentError(f"Tool selection process failed: {str(e)}") from e
    
    @log_function_call(logger)
    async def execute_tools_parallel(
        self,
        tool_names: List[str],
        tool_parameters: Dict[str, Dict[str, Any]],
        execution_strategy: str = "parallel"
    ) -> Dict[str, Any]:
        """
        Execute multiple tools with the specified strategy.
        
        This method implements optimized tool execution with proper
        concurrency control, error handling, and result aggregation.
        
        Args:
            tool_names: List of tool names to execute
            tool_parameters: Parameters for each tool
            execution_strategy: "parallel", "sequential", or "adaptive"
            
        Returns:
            Dict containing execution results:
            - results: Dict mapping tool names to their results
            - metadata: Execution metadata (timing, errors, etc.)
            - summary: Summary of execution outcomes
            
        Raises:
            ToolError: If tool execution fails critically
            AgentError: If orchestration process fails
        """
        if not tool_names:
            return {
                'results': {},
                'metadata': {'execution_time': 0, 'tools_executed': 0},
                'summary': 'No tools specified for execution'
            }
        
        start_time = datetime.now()
        
        try:
            # Choose execution strategy
            if execution_strategy == "parallel":
                results = await self._execute_parallel(tool_names, tool_parameters)
            elif execution_strategy == "sequential":
                results = await self._execute_sequential(tool_names, tool_parameters)
            else:  # adaptive
                results = await self._execute_adaptive(tool_names, tool_parameters)
            
            # Calculate execution metadata
            execution_time = (datetime.now() - start_time).total_seconds()
            metadata = self._calculate_execution_metadata(results, execution_time)
            
            # Generate execution summary
            summary = self._generate_execution_summary(results, metadata)
            
            # Update statistics
            self._update_execution_stats(results, execution_time)
            
            logger.info(
                f"Tool execution completed: {len(results['results'])} tools "
                f"in {execution_time:.2f}s (strategy: {execution_strategy})"
            )
            
            return {
                'results': results['results'],
                'metadata': metadata,
                'summary': summary
            }
            
        except Exception as e:
            logger.error(f"Tool execution failed: {e}", exc_info=True)
            raise AgentError(f"Tool orchestration failed: {str(e)}") from e
    
    @log_function_call(logger)
    async def validate_tool_results(
        self,
        tool_results: Dict[str, Any],
        expected_types: Dict[str, type],
        validation_rules: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validate tool execution results for quality and consistency.
        
        Args:
            tool_results: Results from tool execution
            expected_types: Expected result types for each tool
            validation_rules: Optional custom validation rules
            
        Returns:
            Dict containing validation results:
            - is_valid: Boolean indicating overall validity
            - tool_validations: Per-tool validation results  
            - quality_score: Overall quality score (0-1)
            - issues: List of identified issues
            
        Raises:
            ValidationError: If validation process fails
        """
        try:
            validation_results = {
                'is_valid': True,
                'tool_validations': {},
                'quality_score': 1.0,
                'issues': []
            }
            
            for tool_name, result in tool_results.items():
                # Validate individual tool result
                tool_validation = await self._validate_single_tool_result(
                    tool_name,
                    result,
                    expected_types.get(tool_name),
                    validation_rules
                )
                
                validation_results['tool_validations'][tool_name] = tool_validation
                
                # Update overall validation status
                if not tool_validation['is_valid']:
                    validation_results['is_valid'] = False
                    validation_results['issues'].extend(tool_validation['issues'])
                
                # Update quality score (weighted average)
                validation_results['quality_score'] *= tool_validation['quality_score']
            
            logger.info(
                f"Tool result validation: {'PASSED' if validation_results['is_valid'] else 'FAILED'} "
                f"(quality: {validation_results['quality_score']:.2f})"
            )
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Tool result validation failed: {e}", exc_info=True)
            raise ValidationError(f"Result validation failed: {str(e)}") from e
    
    def get_orchestration_stats(self) -> Dict[str, Any]:
        """Get current orchestration statistics."""
        return {
            **self._execution_stats,
            'registered_tools': len(self.tool_registry.list_tools()),
            'bigtool_enabled': self.bigtool_manager is not None,
            'max_concurrent_tools': self.max_concurrent_tools
        }
    
    # === Private Implementation Methods ===
    
    def _get_rule_based_tools(
        self, 
        problem_type: str, 
        required_capabilities: List[str]
    ) -> List[Tuple[str, float]]:
        """Get tools using rule-based selection with confidence scores."""
        tool_scores = []
        
        # Define problem-to-tool mappings
        tool_mappings = {
            'integral': ['integral_tool', 'plot_tool', 'analysis_tool'],
            'derivative': ['analysis_tool', 'plot_tool'],
            'algebraic': ['analysis_tool'],
            'visualization': ['plot_tool', 'analysis_tool'],
            'general': ['analysis_tool', 'integral_tool', 'plot_tool']
        }
        
        recommended_tools = tool_mappings.get(problem_type, tool_mappings['general'])
        
        for tool_name in recommended_tools:
            if self.tool_registry.has_tool(tool_name):
                # Calculate confidence based on capability match
                confidence = self._calculate_capability_match(
                    tool_name, 
                    required_capabilities
                )
                tool_scores.append((tool_name, confidence))
        
        # Sort by confidence score
        tool_scores.sort(key=lambda x: x[1], reverse=True)
        return tool_scores
    
    async def _get_semantic_tools(
        self, 
        problem_description: str, 
        required_capabilities: List[str]
    ) -> List[Tuple[str, float]]:
        """Get tools using BigTool semantic search."""
        try:
            # Create search query from problem description and capabilities
            search_query = f"{problem_description} {' '.join(required_capabilities)}"
            
            # Perform semantic search
            tool_recommendations = await self.bigtool_manager.get_tool_recommendations(
                search_query,
                top_k=5
            )
            
            # Convert to (name, score) tuples
            semantic_tools = []
            for recommendation in tool_recommendations:
                tool_name = recommendation.get('name', '')
                confidence = recommendation.get('confidence', 0.5)
                semantic_tools.append((tool_name, confidence))
            
            return semantic_tools
            
        except Exception as e:
            logger.warning(f"Semantic tool search failed: {e}")
            return []
    
    def _combine_and_rank_tools(
        self,
        rule_based_tools: List[Tuple[str, float]],
        semantic_tools: List[Tuple[str, float]],
        previous_results: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Combine and rank tools from different selection methods."""
        tool_scores = {}
        
        # Add rule-based scores (weight: 0.6)
        for tool_name, score in rule_based_tools:
            tool_scores[tool_name] = tool_scores.get(tool_name, 0) + (score * 0.6)
        
        # Add semantic scores (weight: 0.4)
        for tool_name, score in semantic_tools:
            tool_scores[tool_name] = tool_scores.get(tool_name, 0) + (score * 0.4)
        
        # Boost tools that complement previous results
        if previous_results:
            for tool_name in tool_scores:
                if self._complements_previous_results(tool_name, previous_results):
                    tool_scores[tool_name] *= 1.2
        
        # Sort by combined score
        ranked_tools = sorted(tool_scores.items(), key=lambda x: x[1], reverse=True)
        return [tool_name for tool_name, _ in ranked_tools]
    
    def _filter_available_tools(self, tool_names: List[str]) -> List[str]:
        """Filter tools for availability and readiness."""
        available_tools = []
        
        for tool_name in tool_names:
            if self.tool_registry.has_tool(tool_name):
                try:
                    tool = self.tool_registry.get_tool(tool_name)
                    if self._is_tool_ready(tool):
                        available_tools.append(tool_name)
                except Exception as e:
                    logger.warning(f"Tool {tool_name} unavailable: {e}")
        
        return available_tools
    
    async def _execute_parallel(
        self, 
        tool_names: List[str], 
        tool_parameters: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute tools in parallel with concurrency control."""
        semaphore = asyncio.Semaphore(self.max_concurrent_tools)
        
        async def execute_single_tool(tool_name: str) -> Tuple[str, Any]:
            async with semaphore:
                try:
                    tool = self.tool_registry.get_tool(tool_name)
                    params = tool_parameters.get(tool_name, {})
                    
                    # Execute with timeout
                    result = await asyncio.wait_for(
                        tool.aexecute(**params),
                        timeout=self.tool_timeout_seconds
                    )
                    
                    return tool_name, {'success': True, 'result': result, 'error': None}
                    
                except asyncio.TimeoutError:
                    return tool_name, {'success': False, 'result': None, 'error': 'Timeout'}
                except Exception as e:
                    return tool_name, {'success': False, 'result': None, 'error': str(e)}
        
        # Execute all tools concurrently
        tasks = [execute_single_tool(tool_name) for tool_name in tool_names]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        execution_results = {'results': {}, 'errors': {}}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Tool execution exception: {result}")
                continue
                
            tool_name, tool_result = result
            if tool_result['success']:
                execution_results['results'][tool_name] = tool_result['result']
            else:
                execution_results['errors'][tool_name] = tool_result['error']
        
        return execution_results
    
    async def _execute_sequential(
        self, 
        tool_names: List[str], 
        tool_parameters: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute tools sequentially."""
        execution_results = {'results': {}, 'errors': {}}
        
        for tool_name in tool_names:
            try:
                tool = self.tool_registry.get_tool(tool_name)
                params = tool_parameters.get(tool_name, {})
                
                result = await asyncio.wait_for(
                    tool.aexecute(**params),
                    timeout=self.tool_timeout_seconds
                )
                
                execution_results['results'][tool_name] = result
                
            except Exception as e:
                logger.error(f"Tool {tool_name} execution failed: {e}")
                execution_results['errors'][tool_name] = str(e)
        
        return execution_results
    
    async def _execute_adaptive(
        self, 
        tool_names: List[str], 
        tool_parameters: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute tools using adaptive strategy based on dependencies."""
        # For now, use parallel execution
        # In future, implement dependency analysis for smart ordering
        return await self._execute_parallel(tool_names, tool_parameters)
    
    def _calculate_capability_match(
        self, 
        tool_name: str, 
        required_capabilities: List[str]
    ) -> float:
        """Calculate how well a tool matches required capabilities."""
        try:
            tool = self.tool_registry.get_tool(tool_name)
            
            # Get tool capabilities (this would be defined in tool metadata)
            tool_capabilities = getattr(tool, 'capabilities', [])
            
            if not required_capabilities:
                return 0.5  # Neutral score if no requirements
            
            # Calculate intersection ratio
            matches = len(set(tool_capabilities) & set(required_capabilities))
            return matches / len(required_capabilities)
            
        except Exception:
            return 0.1  # Low score if tool info unavailable
    
    def _complements_previous_results(
        self, 
        tool_name: str, 
        previous_results: Dict[str, Any]
    ) -> bool:
        """Check if tool complements previous results."""
        # Simple heuristic: if we have numerical results, boost visualization tools
        has_numerical = any('result' in str(result).lower() for result in previous_results.values())
        is_visualization = 'plot' in tool_name.lower() or 'visual' in tool_name.lower()
        
        return has_numerical and is_visualization
    
    def _is_tool_ready(self, tool: BaseTool) -> bool:
        """Check if tool is ready for execution."""
        # Basic readiness check
        return hasattr(tool, 'execute') and callable(tool.execute)
    
    async def _validate_single_tool_result(
        self,
        tool_name: str,
        result: Any,
        expected_type: Optional[type],
        validation_rules: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate a single tool's result."""
        validation = {
            'is_valid': True,
            'quality_score': 1.0,
            'issues': []
        }
        
        # Type validation
        if expected_type and not isinstance(result, expected_type):
            validation['is_valid'] = False
            validation['issues'].append(f"Expected {expected_type}, got {type(result)}")
            validation['quality_score'] *= 0.5
        
        # Null result check
        if result is None:
            validation['is_valid'] = False
            validation['issues'].append("Tool returned null result")
            validation['quality_score'] = 0.0
        
        # Custom validation rules
        if validation_rules and tool_name in validation_rules:
            rules = validation_rules[tool_name]
            for rule_name, rule_function in rules.items():
                try:
                    if not rule_function(result):
                        validation['is_valid'] = False
                        validation['issues'].append(f"Failed custom rule: {rule_name}")
                        validation['quality_score'] *= 0.8
                except Exception as e:
                    logger.warning(f"Validation rule {rule_name} failed: {e}")
        
        return validation
    
    def _calculate_execution_metadata(
        self, 
        results: Dict[str, Any], 
        execution_time: float
    ) -> Dict[str, Any]:
        """Calculate execution metadata."""
        successful_tools = len(results.get('results', {}))
        failed_tools = len(results.get('errors', {}))
        
        return {
            'execution_time': execution_time,
            'tools_executed': successful_tools + failed_tools,
            'successful_tools': successful_tools,
            'failed_tools': failed_tools,
            'success_rate': successful_tools / (successful_tools + failed_tools) if (successful_tools + failed_tools) > 0 else 0
        }
    
    def _generate_execution_summary(
        self, 
        results: Dict[str, Any], 
        metadata: Dict[str, Any]
    ) -> str:
        """Generate human-readable execution summary."""
        successful = metadata['successful_tools']
        failed = metadata['failed_tools']
        time = metadata['execution_time']
        
        if failed == 0:
            return f"All {successful} tools executed successfully in {time:.2f}s"
        elif successful == 0:
            return f"All {failed} tools failed to execute"
        else:
            return f"{successful} tools succeeded, {failed} failed in {time:.2f}s"
    
    def _update_execution_stats(
        self, 
        results: Dict[str, Any], 
        execution_time: float
    ) -> None:
        """Update internal execution statistics."""
        successful = len(results.get('results', {}))
        failed = len(results.get('errors', {}))
        
        self._execution_stats['total_executions'] += 1
        self._execution_stats['successful_executions'] += successful
        self._execution_stats['failed_executions'] += failed
        
        # Update average execution time (exponential moving average)
        alpha = 0.1
        self._execution_stats['average_execution_time'] = (
            alpha * execution_time + 
            (1 - alpha) * self._execution_stats['average_execution_time']
        )
