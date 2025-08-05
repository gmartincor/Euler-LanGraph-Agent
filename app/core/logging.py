"""Structured logging configuration for the ReAct Agent application."""

import asyncio
import functools
import json
import logging
import logging.config
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Generator, Optional, Callable

from .config import get_settings


class CorrelationFilter(logging.Filter):
    """Add correlation ID to log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation ID to the log record."""
        if not hasattr(record, "correlation_id"):
            record.correlation_id = getattr(
                self, "_correlation_id", str(uuid.uuid4())[:8]
            )
        return True


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add correlation ID if available
        if hasattr(record, "correlation_id"):
            log_entry["correlation_id"] = record.correlation_id
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "getMessage",
                "correlation_id", "message", "timestamp", "level", "logger",
                "function", "line", "function_args", "function_kwargs",
            }:
                log_entry[key] = value
        
        return json.dumps(log_entry, ensure_ascii=False)


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for development."""
    
    COLORS = {
        "DEBUG": "\033[36m",      # Cyan
        "INFO": "\033[32m",       # Green
        "WARNING": "\033[33m",    # Yellow
        "ERROR": "\033[31m",      # Red
        "CRITICAL": "\033[35m",   # Magenta
    }
    RESET = "\033[0m"
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        
        # Add correlation ID if available
        correlation_part = ""
        if hasattr(record, "correlation_id"):
            correlation_part = f" [{record.correlation_id}]"
        
        formatted = super().format(record)
        return f"{formatted}{correlation_part}"


def setup_logging() -> None:
    """Configure logging for the application."""
    settings = get_settings()
    
    # Determine formatter based on environment
    if settings.log_format.lower() == "json":
        formatter_class = JSONFormatter
        formatter_kwargs = {}
    else:
        formatter_class = ColoredFormatter
        formatter_kwargs = {
            "fmt": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        }
    
    # Configure handlers
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter_class(**formatter_kwargs))
    console_handler.addFilter(CorrelationFilter())
    handlers.append(console_handler)
    
    # File handler if specified
    if settings.log_file:
        file_handler = logging.FileHandler(settings.log_file)
        file_handler.setFormatter(JSONFormatter())
        file_handler.addFilter(CorrelationFilter())
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        handlers=handlers,
        force=True,
    )
    
    # Configure specific loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(
        logging.INFO if settings.debug else logging.WARNING
    )


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (defaults to calling module name)
    
    Returns:
        logging.Logger: Configured logger instance
    """
    if name is None:
        # Get caller's module name
        import inspect
        frame = inspect.currentframe()
        if frame and frame.f_back:
            name = frame.f_back.f_globals.get("__name__", "unknown")
        else:
            name = "unknown"
    
    return logging.getLogger(name)


@contextmanager
def correlation_context(correlation_id: Optional[str] = None) -> Generator[str, None, None]:
    """
    Context manager to set correlation ID for all logs in the context.
    
    Args:
        correlation_id: Optional correlation ID (generates one if not provided)
    
    Yields:
        str: The correlation ID being used
    """
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())[:8]
    
    # Store original correlation ID
    correlation_filter = CorrelationFilter()
    original_correlation_id = getattr(correlation_filter, "_correlation_id", None)
    
    try:
        # Set new correlation ID
        correlation_filter._correlation_id = correlation_id
        yield correlation_id
    finally:
        # Restore original correlation ID
        if original_correlation_id is not None:
            correlation_filter._correlation_id = original_correlation_id
        else:
            delattr(correlation_filter, "_correlation_id")


def log_function_call(logger: logging.Logger, level: int = logging.INFO) -> Callable:
    """
    Professional decorator that logs function calls with async support.
    
    Properly handles both sync and async functions following professional standards.
    
    Args:
        logger: Logger instance to use
        level: Log level for function calls
        
    Returns:
        Decorator function
    """
    def decorator(func: Any) -> Any:
        if asyncio.iscoroutinefunction(func):
            # Handle async functions
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                func_name = f"{func.__module__}.{func.__qualname__}"
                
                # Log function entry
                logger.log(
                    level,
                    f"Entering {func_name}",
                    extra={
                        "function": func_name,
                        "function_args": str(args)[:200],  # Truncate long args
                        "function_kwargs": {k: str(v)[:100] for k, v in kwargs.items()},
                    },
                )
                
                try:
                    result = await func(*args, **kwargs)
                    
                    # Log successful exit
                    logger.log(
                        level,
                        f"Exiting {func_name}",
                        extra={
                            "function": func_name,
                            "result_type": type(result).__name__,
                        },
                    )
                    
                    return result
                except Exception as e:
                    # Log exception
                    logger.error(
                        f"Exception in {func_name}: {e}",
                        extra={
                            "function": func_name,
                            "exception_type": type(e).__name__,
                        },
                        exc_info=True,
                    )
                    raise
            
            return async_wrapper
        else:
            # Handle sync functions
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                func_name = f"{func.__module__}.{func.__qualname__}"
                
                # Log function entry
                logger.log(
                    level,
                    f"Entering {func_name}",
                    extra={
                        "function": func_name,
                        "function_args": str(args)[:200],  # Truncate long args
                        "function_kwargs": {k: str(v)[:100] for k, v in kwargs.items()},
                    },
                )
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Log successful exit
                    logger.log(
                        level,
                        f"Exiting {func_name}",
                        extra={
                            "function": func_name,
                            "result_type": type(result).__name__,
                        },
                    )
                    
                    return result
                except Exception as e:
                    # Log exception
                    logger.error(
                        f"Exception in {func_name}: {e}",
                        extra={
                            "function": func_name,
                            "exception_type": type(e).__name__,
                        },
                        exc_info=True,
                    )
                    raise
            
            return sync_wrapper
    
    return decorator
