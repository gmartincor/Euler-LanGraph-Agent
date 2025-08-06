import re
from typing import Any, Dict, List, Optional, Tuple, Union

import sympy as sp
from pydantic import field_validator

from ..core.exceptions import ValidationError
from ..core.logging import get_logger

logger = get_logger(__name__)


class MathExpressionValidator:
    ALLOWED_FUNCTIONS = {
        'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2',
        'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh',
        'exp', 'log', 'ln', 'sqrt', 'abs', 'floor', 'ceil',
        'factorial', 'gamma', 'pi', 'e', 'oo', 'inf',
        'integrate', 'diff', 'limit', 'series', 'expand',
        'simplify', 'factor', 'solve', 'Eq', 'Symbol',
        'Integer', 'Float', 'Rational', 'Add', 'Mul', 'Pow',
        'Number', 'Basic', 'Expr', 'AtomicExpr',
    }
    DANGEROUS_PATTERNS = [
        r'__\w+__',
        r'import\s+',
        r'exec\s*\(',
        r'eval\s*\(',
        r'open\s*\(',
        r'os\.',
        r'sys\.',
        r'subprocess',
    ]
    
    @classmethod
    def validate_expression(cls, expression: str) -> Tuple[bool, Optional[str], Optional[sp.Expr]]:
        if not expression or not expression.strip():
            return False, "Expression cannot be empty", None
        expression = expression.strip()
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, expression, re.IGNORECASE):
                return False, f"Expression contains dangerous pattern: {pattern}", None
        try:
            parsed_expr = sp.sympify(expression, evaluate=False)
            functions_used = set()
            for atom in parsed_expr.atoms():
                if hasattr(atom, 'func') and hasattr(atom.func, '__name__'):
                    functions_used.add(atom.func.__name__)
            disallowed = functions_used - cls.ALLOWED_FUNCTIONS
            if disallowed:
                return False, f"Disallowed functions found: {disallowed}", None
            return True, None, parsed_expr
        except (sp.SympifyError, ValueError, TypeError) as e:
            logger.warning(f"Failed to parse expression '{expression}': {e}")
            return False, f"Invalid mathematical expression: {e}", None
        except Exception as e:
            logger.error(f"Unexpected error validating expression '{expression}': {e}")
            return False, f"Unexpected validation error: {e}", None
    
    @classmethod
    def validate_integral_bounds(
        cls, 
        lower_bound: Union[str, float, int], 
        upper_bound: Union[str, float, int]
    ) -> Tuple[bool, Optional[str], Optional[Tuple[sp.Expr, sp.Expr]]]:
        try:
            if isinstance(lower_bound, (int, float)):
                parsed_lower = sp.Float(lower_bound)
            else:
                parsed_lower = sp.sympify(str(lower_bound))
            if isinstance(upper_bound, (int, float)):
                parsed_upper = sp.Float(upper_bound)
            else:
                parsed_upper = sp.sympify(str(upper_bound))
            try:
                lower_val = float(parsed_lower)
                upper_val = float(parsed_upper)
                if lower_val >= upper_val:
                    return False, "Lower bound must be less than upper bound", None
                if abs(lower_val) > 1e6 or abs(upper_val) > 1e6:
                    return False, "Bounds are too large (>1e6)", None
            except (ValueError, TypeError):
                pass
            return True, None, (parsed_lower, parsed_upper)
        except Exception as e:
            logger.error(f"Error validating bounds {lower_bound}, {upper_bound}: {e}")
            return False, f"Invalid bounds: {e}", None
    
    @classmethod
    def validate_variable(cls, variable: str) -> Tuple[bool, Optional[str], Optional[sp.Symbol]]:
        if not variable or not variable.strip():
            return False, "Variable name cannot be empty", None
        variable = variable.strip()
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', variable):
            return False, "Variable name must start with a letter and contain only letters, numbers, and underscores", None
        reserved_words = {'and', 'or', 'not', 'if', 'else', 'for', 'while', 'def', 'class', 'import', 'from'}
        if variable.lower() in reserved_words:
            return False, f"'{variable}' is a reserved word", None
        try:
            symbol = sp.Symbol(variable)
            return True, None, symbol
        except Exception as e:
            return False, f"Invalid variable name: {e}", None


class SessionIdValidator:
    SESSION_ID_PATTERN = re.compile(r'^[a-zA-Z0-9\-_]{8,64}$')
    
    @classmethod
    def validate_session_id(cls, session_id: str) -> Tuple[bool, Optional[str]]:
        if not session_id or not session_id.strip():
            return False, "Session ID cannot be empty"
        session_id = session_id.strip()
        if not cls.SESSION_ID_PATTERN.match(session_id):
            return False, "Session ID must be 8-64 characters long and contain only letters, numbers, hyphens, and underscores"
        return True, None


class NumericValidator:
    @classmethod
    def validate_number(
        cls, 
        value: Union[str, int, float], 
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        allow_infinity: bool = False,
    ) -> Tuple[bool, Optional[str], Optional[float]]:
        try:
            if isinstance(value, str):
                value = value.strip()
                if not value:
                    return False, "Value cannot be empty", None
                if value.lower() in ['inf', 'infinity', '+inf', '+infinity']:
                    if allow_infinity:
                        return True, None, float('inf')
                    else:
                        return False, "Infinity is not allowed", None
                if value.lower() in ['-inf', '-infinity']:
                    if allow_infinity:
                        return True, None, float('-inf')
                    else:
                        return False, "Negative infinity is not allowed", None
                parsed_value = float(value)
            else:
                parsed_value = float(value)
            if min_value is not None and parsed_value < min_value:
                return False, f"Value must be at least {min_value}", None
            if max_value is not None and parsed_value > max_value:
                return False, f"Value must be at most {max_value}", None
            return True, None, parsed_value
        except (ValueError, TypeError) as e:
            return False, f"Invalid numeric value: {e}", None


def validate_math_expression(expression: str) -> sp.Expr:
    is_valid, error_msg, parsed_expr = MathExpressionValidator.validate_expression(expression)
    if not is_valid:
        raise ValidationError(
            f"Invalid mathematical expression: {error_msg}",
            field_name="expression",
            invalid_value=expression,
        )
    return parsed_expr


def validate_session_id(session_id: str) -> str:
    is_valid, error_msg = SessionIdValidator.validate_session_id(session_id)
    if not is_valid:
        raise ValidationError(
            f"Invalid session ID: {error_msg}",
            field_name="session_id", 
            invalid_value=session_id,
        )
    return session_id.strip()


def validate_integral_bounds(
    lower_bound: Union[str, float, int],
    upper_bound: Union[str, float, int],
) -> Tuple[sp.Expr, sp.Expr]:
    is_valid, error_msg, bounds = MathExpressionValidator.validate_integral_bounds(
        lower_bound, upper_bound
    )
    if not is_valid:
        raise ValidationError(
            f"Invalid integral bounds: {error_msg}",
            field_name="bounds",
            invalid_value={"lower": lower_bound, "upper": upper_bound},
        )
    return bounds


def validate_number(
    value: Union[str, int, float],
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    allow_infinity: bool = False,
) -> float:
    is_valid, error_msg, parsed_value = NumericValidator.validate_number(
        value, min_value, max_value, allow_infinity
    )
    if not is_valid:
        raise ValidationError(
            f"Invalid numeric value: {error_msg}",
            field_name="value",
            invalid_value=value,
        )
    return parsed_value
