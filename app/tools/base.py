"""Base class for all mathematical tools in the ReAct Agent system."""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from ..core.exceptions import ToolError, ValidationError
from ..core.logging import correlation_context, get_logger, log_function_call

logger = get_logger(__name__)


class ToolInput(BaseModel):
    """Base class for tool input validation."""
    
    class Config:
        """Pydantic configuration."""
        extra = "forbid"  # Prevent extra fields


class ToolOutput(BaseModel):
    """Base class for tool output formatting."""
    
    success: bool = Field(..., description="Whether the tool execution was successful")
    result: Optional[Any] = Field(None, description="Tool execution result")
    error: Optional[str] = Field(None, description="Error message if execution failed")
    execution_time: float = Field(..., description="Time taken to execute in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        """Pydantic configuration."""
        extra = "allow"  # Allow additional fields for specific tools


class BaseTool(ABC):
    """
    Abstract base class for all mathematical tools.
    
    This class provides a common interface and standard functionality for all tools,
    including error handling, logging, input validation, and execution timing.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        """
        Initialize the base tool.
        
        Args:
            name: Unique name for the tool
            description: Human-readable description of what the tool does
            timeout: Maximum execution time in seconds
            max_retries: Maximum number of retry attempts on failure
        """
        self.name = name
        self.description = description
        self.timeout = timeout
        self.max_retries = max_retries
        self._usage_count = 0
        self._total_execution_time = 0.0
        self._error_count = 0
    
    @property
    def usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for this tool."""
        return {
            "usage_count": self._usage_count,
            "error_count": self._error_count,
            "success_rate": (
                (self._usage_count - self._error_count) / self._usage_count
                if self._usage_count > 0 else 0.0
            ),
            "average_execution_time": (
                self._total_execution_time / self._usage_count
                if self._usage_count > 0 else 0.0
            ),
            "total_execution_time": self._total_execution_time,
        }
    
    @abstractmethod
    def _validate_input(self, input_data: Dict[str, Any]) -> ToolInput:
        """
        Validate and parse input data.
        
        Args:
            input_data: Raw input data to validate
        
        Returns:
            ToolInput: Validated input object
        
        Raises:
            ValidationError: If input validation fails
        """
        pass
    
    @abstractmethod
    def _execute_tool(self, validated_input: ToolInput) -> Any:
        """
        Execute the tool's main functionality.
        
        Args:
            validated_input: Validated input object
        
        Returns:
            Any: Tool execution result
        
        Raises:
            ToolError: If tool execution fails
        """
        pass
    
    def _format_output(
        self,
        result: Any,
        execution_time: float,
        error: Optional[str] = None,
        **metadata: Any,
    ) -> ToolOutput:
        """
        Format tool output in a standard way.
        
        Args:
            result: Tool execution result
            execution_time: Time taken to execute
            error: Error message if execution failed
            **metadata: Additional metadata
        
        Returns:
            ToolOutput: Formatted output object
        """
        return ToolOutput(
            success=error is None,
            result=result if error is None else None,
            error=error,
            execution_time=execution_time,
            metadata=metadata,
        )
    
    @log_function_call(logger)
    def execute(self, input_data: Dict[str, Any]) -> ToolOutput:
        """
        Execute the tool with proper error handling and logging.
        
        Args:
            input_data: Input data for the tool
        
        Returns:
            ToolOutput: Tool execution result
        """
        start_time = time.time()
        correlation_id = f"{self.name}_{int(start_time * 1000) % 10000}"
        
        with correlation_context(correlation_id):
            logger.info(f"Executing tool '{self.name}'", extra={
                "tool_name": self.name,
                "input_keys": list(input_data.keys()),
            })
            
            try:
                # Validate input
                validated_input = self._validate_input(input_data)
                logger.debug(f"Input validation successful for '{self.name}'")
                
                # Execute tool with retries
                result = self._execute_with_retries(validated_input)
                
                # Calculate execution time
                execution_time = time.time() - start_time
                
                # Update statistics
                self._usage_count += 1
                self._total_execution_time += execution_time
                
                logger.info(f"Tool '{self.name}' executed successfully", extra={
                    "execution_time": execution_time,
                    "usage_count": self._usage_count,
                })
                
                return self._format_output(result, execution_time)
                
            except ValidationError as e:
                execution_time = time.time() - start_time
                self._usage_count += 1
                self._error_count += 1
                
                logger.error(f"Input validation failed for '{self.name}': {e}", extra={
                    "tool_name": self.name,
                    "error_type": "ValidationError",
                })
                
                return self._format_output(
                    None, execution_time, f"Input validation error: {e}"
                )
                
            except ToolError as e:
                execution_time = time.time() - start_time
                self._usage_count += 1
                self._error_count += 1
                
                logger.error(f"Tool execution failed for '{self.name}': {e}", extra={
                    "tool_name": self.name,
                    "error_type": "ToolError",
                })
                
                return self._format_output(
                    None, execution_time, f"Tool execution error: {e}"
                )
                
            except Exception as e:
                execution_time = time.time() - start_time
                self._usage_count += 1
                self._error_count += 1
                
                logger.error(f"Unexpected error in '{self.name}': {e}", extra={
                    "tool_name": self.name,
                    "error_type": type(e).__name__,
                }, exc_info=True)
                
                return self._format_output(
                    None, execution_time, f"Unexpected error: {e}"
                )
    
    def _execute_with_retries(self, validated_input: ToolInput) -> Any:
        """
        Execute tool with retry logic.
        
        Args:
            validated_input: Validated input object
        
        Returns:
            Any: Tool execution result
        
        Raises:
            ToolError: If all retry attempts fail
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Executing '{self.name}', attempt {attempt + 1}/{self.max_retries}")
                return self._execute_tool(validated_input)
                
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed for '{self.name}': {e}")
                    time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error(f"All {self.max_retries} attempts failed for '{self.name}'")
        
        raise ToolError(
            f"Tool '{self.name}' failed after {self.max_retries} attempts",
            tool_name=self.name,
            tool_input=validated_input.dict() if hasattr(validated_input, 'dict') else str(validated_input),
        ) from last_error
    
    def get_schema(self) -> Dict[str, Any]:
        """
        Get tool schema for BigTool integration.
        
        Returns:
            Dict[str, Any]: Tool schema for semantic search and selection
        """
        return {
            "name": self.name,
            "description": self.description,
            "usage_stats": self.usage_stats,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }
    
    def reset_stats(self) -> None:
        """Reset usage statistics."""
        self._usage_count = 0
        self._total_execution_time = 0.0
        self._error_count = 0
        logger.info(f"Statistics reset for tool '{self.name}'")
    
    def to_langchain_tool(self) -> 'Tool':
        """
        Convert tool to LangChain Tool format for LangGraph integration.
        
        This method creates a LangChain-compatible Tool while maintaining
        all existing functionality (logging, statistics, error handling).
        Follows DRY by reusing the existing execute method.
        
        Returns:
            Tool: LangChain Tool instance
        """
        try:
            from langchain_core.tools import Tool
        except ImportError as e:
            raise ToolError(
                "LangChain not available. Install langchain-core for LangGraph integration.",
                tool_name=self.name
            ) from e
        
        def langchain_wrapper(*args, **kwargs) -> str:
            """
            Wrapper function that maintains existing BaseTool functionality.
            
            This wrapper ensures that all logging, statistics, and error handling
            continue to work when the tool is used through LangChain.
            """
            try:
                # Use existing execute method to maintain all functionality
                result = self.execute(*args, **kwargs)
                
                # Convert ToolOutput to string for LangChain compatibility
                if isinstance(result, ToolOutput):
                    if result.success:
                        return str(result.result) if result.result is not None else "Success"
                    else:
                        return f"Error: {result.error}"
                else:
                    return str(result)
                    
            except Exception as e:
                logger.error(f"LangChain wrapper error for '{self.name}': {e}")
                return f"Tool execution failed: {str(e)}"
        
        # Create LangChain Tool with existing metadata
        return Tool(
            name=self.name,
            description=self.description,
            func=langchain_wrapper,
            # Use input schema if available
            args_schema=getattr(self, 'input_schema', None)
        )
    
    def __repr__(self) -> str:
        """String representation of the tool."""
        return f"{self.__class__.__name__}(name='{self.name}')"
