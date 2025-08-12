"""Base class for all mathematical tools in the ReAct Agent system."""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, ConfigDict

from ..core.base_classes import BaseExecutor
from ..core.exceptions import ToolError, ValidationError
from ..core.logging import correlation_context, get_logger, log_function_call

logger = get_logger(__name__)


class ToolInput(BaseModel):
    """Base class for tool input validation."""
    
    model_config = ConfigDict(extra="allow")  # Allow additional fields for specific tools


class ToolOutput(BaseModel):
    """Base class for tool output."""
    
    success: bool = Field(..., description="Whether the tool execution was successful")
    result: Optional[Any] = Field(None, description="Tool execution result")
    error: Optional[str] = Field(None, description="Error message if execution failed")
    execution_time: float = Field(..., description="Time taken to execute in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class BaseTool(BaseExecutor):
    """
    Abstract base class for all mathematical tools.
    
    This class provides a common interface and standard functionality for all tools,
    including error handling, logging, input validation, and execution timing.
    Inherits from BaseExecutor for standardized execution patterns.
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
        super().__init__(name, description)
        self.timeout = timeout
        self.max_retries = max_retries
    
    @property
    def usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for this tool (inherited from BaseExecutor)."""
        base_stats = self.get_stats()
        # Add tool-specific metadata
        base_stats.update({
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        })
        return base_stats
    
    def _validate_input(self, input_data: Dict[str, Any]) -> None:
        """
        Validate input data using the tool's specific validation.
        
        Args:
            input_data: Raw input data to validate
        
        Raises:
            ValidationError: If input validation fails
        """
        # Delegate to tool-specific validation that returns ToolInput
        self._validate_tool_input(input_data)
    
    @abstractmethod
    def _validate_tool_input(self, input_data: Dict[str, Any]) -> ToolInput:
        """
        Validate and parse input data for the specific tool.
        
        Args:
            input_data: Raw input data to validate
        
        Returns:
            ToolInput: Validated input object
        
        Raises:
            ValidationError: If input validation fails
        """
        pass
    
    def _execute_core(self, input_data: Dict[str, Any]) -> Any:
        """
        Execute the tool's core logic with retries.
        
        Args:
            input_data: Validated input data
        
        Returns:
            Any: Tool execution result
        """
        # First validate input to get ToolInput object
        validated_input = self._validate_tool_input(input_data)
        
        # Then execute with retries
        return self._execute_with_retries(validated_input)
    
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
    
    def _format_tool_output(
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
    
    def execute_tool(self, input_data: Dict[str, Any]) -> ToolOutput:
        """
        Execute the tool and return ToolOutput format.
        Uses BaseExecutor.execute() internally but converts to ToolOutput.
        
        Args:
            input_data: Input data for the tool
        
        Returns:
            ToolOutput: Tool execution result in ToolOutput format
        """
        # Use inherited execute method from BaseExecutor
        result = self.execute(input_data)
        
        # Convert BaseExecutor output to ToolOutput format
        return ToolOutput(
            success=result["success"],
            result=result.get("result"),
            error=result.get("error"),
            execution_time=result["execution_time"],
            metadata={
                "timestamp": result["timestamp"],
                "executor": result["executor"]
            }
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
            tool_input=validated_input.model_dump() if hasattr(validated_input, 'model_dump') else str(validated_input),
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
        """Reset usage statistics (inherited from BaseExecutor)."""
        super().reset_stats()
        logger.info(f"Statistics reset for tool '{self.name}'")
    
    def to_langchain_tool(self) -> Any:
        """
        Convert tool to LangChain Tool format for LangGraph integration.
        
        This method creates a LangChain-compatible Tool while maintaining
        all existing functionality (logging, statistics, error handling).
        Follows DRY by reusing the existing execute method.
        
        Returns:
            Any: LangChain Tool instance
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
                # Use existing execute_tool method to maintain all functionality
                result = self.execute_tool(*args, **kwargs)
                
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
    
    async def arun(self, input_data: Union[str, Dict[str, Any]]) -> Any:
        """
        Async wrapper for tool execution (LangChain compatibility).
        
        Args:
            input_data: Input data (string or dict)
        
        Returns:
            Any: Tool execution result (compatible with LangChain)
        """
        # Convert string input to dict if necessary
        if isinstance(input_data, str):
            input_dict = {"expression": input_data}
        else:
            input_dict = input_data
        
        # Execute synchronously (tools are not inherently async)
        output = self.execute_tool(input_dict)
        
        # Return result for LangChain compatibility
        if output.success:
            return output.result
        else:
            raise ToolError(f"Tool {self.name} failed: {output.error}")
    
    def run(self, input_data: Union[str, Dict[str, Any]]) -> Any:
        """
        Sync wrapper for tool execution (LangChain compatibility).
        
        Args:
            input_data: Input data (string or dict)
        
        Returns:
            Any: Tool execution result (compatible with LangChain)
        """
        # Convert string input to dict if necessary
        if isinstance(input_data, str):
            input_dict = {"expression": input_data}
        else:
            input_dict = input_data
        
        # Execute and return result
        output = self.execute_tool(input_dict)
        
        if output.success:
            return output.result
        else:
            raise ToolError(f"Tool {self.name} failed: {output.error}")
