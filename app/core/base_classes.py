from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import time

from pydantic import BaseModel, Field

from .logging import get_logger, log_function_call
from .exceptions import ValidationError, ToolError

logger = get_logger(__name__)


class BaseExecutor(ABC):
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._usage_count = 0
        self._success_count = 0
        self._error_count = 0
        self._total_execution_time = 0.0
    
    @log_function_call(logger)
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            self._validate_input(input_data)
            result = self._execute_core(input_data)
            execution_time = time.time() - start_time
            self._record_success(execution_time)
            return self._format_output(result, execution_time)
        except ValidationError as e:
            execution_time = time.time() - start_time
            self._record_error()
            logger.error(f"Validation error in '{self.name}': {e}")
            return self._format_error_output(str(e), execution_time, "validation_error")
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._record_error()
            logger.error(f"Unexpected error in '{self.name}': {e}", exc_info=True)
            return self._format_error_output(str(e), execution_time, "execution_error")
    
    @abstractmethod
    def _validate_input(self, input_data: Dict[str, Any]) -> None:
        pass
    
    @abstractmethod
    def _execute_core(self, input_data: Dict[str, Any]) -> Any:
        pass
    
    def _record_success(self, execution_time: float) -> None:
        self._usage_count += 1
        self._success_count += 1
        self._total_execution_time += execution_time
    
    def _record_error(self) -> None:
        self._usage_count += 1
        self._error_count += 1
    
    def _format_output(self, result: Any, execution_time: float) -> Dict[str, Any]:
        return {
            "success": True,
            "result": result,
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat(),
            "executor": self.name
        }
    
    def _format_error_output(self, error: str, execution_time: float, error_type: str) -> Dict[str, Any]:
        return {
            "success": False,
            "error": error,
            "error_type": error_type,
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat(),
            "executor": self.name
        }
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "usage_count": self._usage_count,
            "success_count": self._success_count,
            "error_count": self._error_count,
            "success_rate": self._success_count / max(self._usage_count, 1),
            "average_execution_time": self._total_execution_time / max(self._success_count, 1),
            "total_execution_time": self._total_execution_time
        }
    
    def reset_stats(self) -> None:
        self._usage_count = 0
        self._success_count = 0
        self._error_count = 0
        self._total_execution_time = 0.0


class BaseUIComponent(ABC):
    
    def __init__(self, component_name: str):
        self.component_name = component_name
        self.logger = get_logger(f"ui.{component_name}")
    
    @abstractmethod
    def render(self) -> None:
        pass
    
    def format_duration(self, seconds: float) -> str:
        if seconds < 1:
            return f"{seconds*1000:.0f}ms"
        elif seconds < 60:
            return f"{seconds:.1f}s"
        else:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.1f}s"
    
    def format_timestamp(self, timestamp: str) -> str:
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return dt.strftime("%H:%M:%S")
        except:
            return timestamp
    
    def get_status_icon(self, status: str) -> str:
        icons = {
            "success": "✅",
            "error": "❌", 
            "warning": "⚠️",
            "info": "ℹ️",
            "processing": "⏳",
            "ready": "🟢",
            "not_ready": "🔴"
        }
        return icons.get(status.lower(), "❓")


class BaseStateManager(ABC):
    
    def __init__(self, namespace: str):
        self.namespace = namespace
        self.logger = get_logger(f"state.{namespace}")
    
    def get_namespaced_key(self, key: str) -> str:
        return f"{self.namespace}.{key}"
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def update_state(self, **kwargs) -> None:
        pass
    
    def validate_state_key(self, key: str) -> bool:
        return key and isinstance(key, str) and len(key) > 0
    
    def log_state_change(self, key: str, old_value: Any, new_value: Any) -> None:
        self.logger.debug(f"State change - {key}: {old_value} -> {new_value}")


class MetricsCollector:
    
    def __init__(self, prefix: str = ""):
        self.prefix = prefix
        self._metrics = {}
    
    def record_metric(self, name: str, value: Union[int, float], tags: Optional[Dict[str, str]] = None) -> None:
        full_name = f"{self.prefix}.{name}" if self.prefix else name
        timestamp = datetime.now().isoformat()
        
        if full_name not in self._metrics:
            self._metrics[full_name] = []
        
        self._metrics[full_name].append({
            "value": value,
            "timestamp": timestamp,
            "tags": tags or {}
        })
    
    def get_metric_summary(self, name: str) -> Dict[str, Any]:
        full_name = f"{self.prefix}.{name}" if self.prefix else name
        
        if full_name not in self._metrics:
            return {"count": 0}
        
        values = [m["value"] for m in self._metrics[full_name]]
        
        return {
            "count": len(values),
            "sum": sum(values),
            "average": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "latest": values[-1] if values else None
        }
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        return {
            name: self.get_metric_summary(name.replace(f"{self.prefix}.", "") if self.prefix else name)
            for name in self._metrics.keys()
        }
