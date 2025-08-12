from typing import Any, Dict, Optional


class ReactAgentError(Exception):
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__.upper()
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
        }


class ConfigurationError(ReactAgentError):
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        if config_key:
            self.details["config_key"] = config_key


class DependencyError(ReactAgentError):
    
    def __init__(
        self,
        message: str,
        dependency_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        if dependency_name:
            self.details["dependency_name"] = dependency_name


class DatabaseError(ReactAgentError):
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        table: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        if operation:
            self.details["operation"] = operation
        if table:
            self.details["table"] = table


class ToolError(ReactAgentError):
    
    def __init__(
        self,
        message: str,
        tool_name: Optional[str] = None,
        tool_input: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        if tool_name:
            self.details["tool_name"] = tool_name
        if tool_input:
            self.details["tool_input"] = tool_input


class ValidationError(ReactAgentError):
    
    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        invalid_value: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        if field_name:
            self.details["field_name"] = field_name
        if invalid_value is not None:
            self.details["invalid_value"] = invalid_value


class IntegrationError(ReactAgentError):
    
    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        status_code: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        if service_name:
            self.details["service_name"] = service_name
        if status_code:
            self.details["status_code"] = status_code


class AgentStateError(ReactAgentError):
    
    def __init__(
        self,
        message: str,
        session_id: Optional[str] = None,
        state_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        if session_id:
            self.details["session_id"] = session_id
        if state_key:
            self.details["state_key"] = state_key


class MathematicalError(ReactAgentError):
    
    def __init__(
        self,
        message: str,
        expression: Optional[str] = None,
        computation_type: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        if expression:
            self.details["expression"] = expression
        if computation_type:
            self.details["computation_type"] = computation_type


# Aliases for backward compatibility and consistency  
StateError = AgentStateError
AgentError = ReactAgentError
