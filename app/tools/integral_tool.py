"""Integral calculation tool for symbolic and numerical integration."""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import sympy as sp
from pydantic import Field, field_validator
from scipy import integrate

from ..core.exceptions import MathematicalError, ToolError
from ..core.logging import get_logger
from ..utils.validators import MathExpressionValidator
from .base import BaseTool, ToolInput, ToolOutput

logger = get_logger(__name__)


class IntegralInput(ToolInput):
    """Input validation for integral calculations."""
    
    expression: str = Field(..., description="Mathematical expression to integrate")
    variable: str = Field(default="x", description="Variable of integration")
    lower_bound: Optional[Union[str, float]] = Field(None, description="Lower integration bound")
    upper_bound: Optional[Union[str, float]] = Field(None, description="Upper integration bound")
    method: str = Field(default="auto", description="Integration method: auto, symbolic, numerical")
    numerical_tolerance: float = Field(default=1e-8, description="Tolerance for numerical integration")
    max_subdivisions: int = Field(default=50, description="Maximum subdivisions for numerical integration")
    
    @field_validator("expression")
    @classmethod
    def validate_expression(cls, v: str) -> str:
        """Validate mathematical expression."""
        is_valid, error_msg, _ = MathExpressionValidator.validate_expression(v)
        if not is_valid:
            raise ValueError(f"Invalid expression: {error_msg}")
        return v.strip()
    
    @field_validator("variable")
    @classmethod
    def validate_variable(cls, v: str): 
        """Validate integration variable."""
        if not v.isalpha() or len(v) != 1:
            # Allow common multi-character variables
            if v not in ["theta", "phi", "alpha", "beta", "gamma"]:
                raise ValueError("Variable must be a single letter or common Greek letter name")
        return v
    
    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str): 
        """Validate integration method."""
        valid_methods = ["auto", "symbolic", "numerical"]
        if v.lower() not in valid_methods:
            raise ValueError(f"Method must be one of: {valid_methods}")
        return v.lower()
    
    @field_validator("numerical_tolerance")
    @classmethod
    def validate_tolerance(cls, v: float): 
        """Validate numerical tolerance."""
        if not 1e-15 <= v <= 1e-3:
            raise ValueError("Tolerance must be between 1e-15 and 1e-3")
        return v
    
    @field_validator("max_subdivisions")
    @classmethod
    def validate_subdivisions(cls, v: int) -> int:
        """Validate max subdivisions."""
        if not 10 <= v <= 1000:
            raise ValueError("Max subdivisions must be between 10 and 1000")
        return v


class IntegralOutput(ToolOutput):
    """Output format for integral calculations."""
    
    symbolic_result: Optional[str] = Field(None, description="Symbolic integration result")
    numerical_result: Optional[float] = Field(None, description="Numerical integration result")
    is_definite: bool = Field(..., description="Whether this is a definite integral")
    method_used: str = Field(..., description="Method used for integration")
    convergence_info: Optional[Dict[str, Any]] = Field(None, description="Convergence information")
    integral_properties: Dict[str, Any] = Field(default_factory=dict, description="Properties of the integral")


class IntegralTool(BaseTool):
    """
    Tool for calculating integrals using symbolic and numerical methods.
    
    This tool can handle both definite and indefinite integrals using SymPy
    for symbolic computation and SciPy for numerical integration with automatic
    fallback between methods.
    """
    
    def __init__(self) -> None:
        """Initialize the integral tool."""
        super().__init__(
            name="integral_calculator",
            description=(
                "Calculate definite and indefinite integrals using symbolic and numerical methods. "
                "Supports complex mathematical expressions, automatic method selection, and "
                "provides detailed analysis of integral properties."
            ),
            timeout=30.0,
            max_retries=2,
        )
    
    def _validate_input(self, input_data: Dict[str, Any]) -> IntegralInput:
        """Validate integral calculation input."""
        try:
            return IntegralInput(**input_data)
        except Exception as e:
            raise ToolError(f"Input validation failed: {e}", tool_name=self.name)
    
    def _execute_tool(self, validated_input: IntegralInput) -> IntegralOutput:
        """Execute integral calculation."""
        try:
            # Parse the mathematical expression
            expr = sp.sympify(validated_input.expression)
            var = sp.Symbol(validated_input.variable)
            
            # Determine if this is a definite or indefinite integral
            is_definite = (
                validated_input.lower_bound is not None and 
                validated_input.upper_bound is not None
            )
            
            # Choose integration method
            if validated_input.method == "auto":
                method = self._choose_optimal_method(expr, var, is_definite)
            else:
                method = validated_input.method
            
            logger.info(f"Using {method} method for integration", extra={
                "expression": str(expr),
                "variable": str(var),
                "is_definite": is_definite,
            })
            
            # Perform integration
            if method == "symbolic":
                result = self._symbolic_integration(
                    expr, var, validated_input.lower_bound, validated_input.upper_bound
                )
            else:  # numerical
                if not is_definite:
                    raise MathematicalError(
                        "Numerical integration requires definite bounds",
                        expression=validated_input.expression,
                        computation_type="numerical_integration",
                    )
                result = self._numerical_integration(
                    expr, var, validated_input.lower_bound, validated_input.upper_bound,
                    validated_input.numerical_tolerance, validated_input.max_subdivisions
                )
            
            # Analyze integral properties
            properties = self._analyze_integral_properties(
                expr, var, validated_input.lower_bound, validated_input.upper_bound, result
            )
            
            return IntegralOutput(
                success=True,
                result=result,
                execution_time=0.0,  # Will be set by base class
                symbolic_result=result.get("symbolic"),
                numerical_result=result.get("numerical"),
                is_definite=is_definite,
                method_used=method,
                convergence_info=result.get("convergence_info"),
                integral_properties=properties,
            )
            
        except Exception as e:
            logger.error(f"Integration failed: {e}", exc_info=True)
            raise MathematicalError(
                f"Integration calculation failed: {e}",
                expression=validated_input.expression,
                computation_type="integration",
            )
    
    def _choose_optimal_method(
        self, 
        expr: sp.Expr, 
        var: sp.Symbol, 
        is_definite: bool
    ) -> str:
        """
        Choose the optimal integration method based on expression complexity.
        
        Args:
            expr: SymPy expression
            var: Integration variable
            is_definite: Whether bounds are provided
        
        Returns:
            str: Chosen method ('symbolic' or 'numerical')
        """
        try:
            # Try symbolic first for simple expressions
            if not is_definite:
                return "symbolic"  # Always try symbolic for indefinite integrals
            
            # For definite integrals, check expression complexity
            atoms = expr.atoms()
            
            # If expression contains transcendental functions or is complex, prefer numerical
            has_transcendental = any(
                atom.func in [sp.exp, sp.log, sp.sin, sp.cos, sp.tan, 
                             sp.asin, sp.acos, sp.atan, sp.sinh, sp.cosh, sp.tanh]
                for atom in atoms if hasattr(atom, 'func')
            )
            
            # Check for special functions that are hard to integrate symbolically
            has_special = any(
                str(atom) in ["erf", "erfc", "gamma", "beta", "bessel"]
                for atom in atoms
            )
            
            if has_special or (has_transcendental and len(str(expr)) > 50):
                return "numerical"
            
            # Try a quick symbolic integration to see if it's feasible
            try:
                quick_result = sp.integrate(expr, var, risch=False, conds='none')
                if quick_result.has(sp.Integral):
                    # If result still contains unevaluated integrals, use numerical
                    return "numerical" if is_definite else "symbolic"
                return "symbolic"
            except:
                return "numerical" if is_definite else "symbolic"
                
        except Exception:
            # Default fallback
            return "numerical" if is_definite else "symbolic"
    
    def _symbolic_integration(
        self,
        expr: sp.Expr,
        var: sp.Symbol,
        lower_bound: Optional[Union[str, float]],
        upper_bound: Optional[Union[str, float]],
    ) -> Dict[str, Any]:
        """
        Perform symbolic integration using SymPy.
        
        Args:
            expr: Expression to integrate
            var: Integration variable
            lower_bound: Lower bound (if definite)
            upper_bound: Upper bound (if definite)
        
        Returns:
            Dict[str, Any]: Integration result with metadata
        """
        try:
            if lower_bound is not None and upper_bound is not None:
                # Parse bounds
                lower = self._parse_bound(lower_bound)
                upper = self._parse_bound(upper_bound)
                
                # Definite integral
                result = sp.integrate(expr, (var, lower, upper))
                
                # Try to evaluate numerically if result is symbolic
                numerical_value = None
                try:
                    numerical_value = float(result.evalf())
                except:
                    pass
                
                return {
                    "symbolic": str(result),
                    "numerical": numerical_value,
                    "bounds": {"lower": str(lower), "upper": str(upper)},
                }
            else:
                # Indefinite integral
                result = sp.integrate(expr, var)
                return {
                    "symbolic": str(result) + " + C",
                    "antiderivative": str(result),
                }
                
        except sp.PolynomialError as e:
            raise MathematicalError(f"Polynomial integration error: {e}")
        except NotImplementedError as e:
            raise MathematicalError(f"Symbolic integration not implemented for this expression: {e}")
        except Exception as e:
            raise MathematicalError(f"Symbolic integration failed: {e}")
    
    def _numerical_integration(
        self,
        expr: sp.Expr,
        var: sp.Symbol,
        lower_bound: Union[str, float],
        upper_bound: Union[str, float],
        tolerance: float,
        max_subdivisions: int,
    ) -> Dict[str, Any]:
        """
        Perform numerical integration using SciPy.
        
        Args:
            expr: Expression to integrate
            var: Integration variable
            lower_bound: Lower integration bound
            upper_bound: Upper integration bound
            tolerance: Integration tolerance
            max_subdivisions: Maximum subdivisions
        
        Returns:
            Dict[str, Any]: Integration result with convergence info
        """
        try:
            # Convert SymPy expression to numpy function
            func = sp.lambdify(var, expr, "numpy")
            
            # Parse bounds
            lower = self._parse_bound(lower_bound)
            upper = self._parse_bound(upper_bound)
            
            # Handle infinite bounds
            if lower == -sp.oo:
                lower = -np.inf
            elif lower == sp.oo:
                lower = np.inf
            else:
                lower = float(lower)
            
            if upper == -sp.oo:
                upper = -np.inf
            elif upper == sp.oo:
                upper = np.inf
            else:
                upper = float(upper)
            
            # Choose appropriate integration method
            if np.isinf(lower) or np.isinf(upper):
                # Use quad for infinite integrals
                result, error = integrate.quad(
                    func, lower, upper, 
                    epsabs=tolerance, epsrel=tolerance,
                    limit=max_subdivisions
                )
            else:
                # Use quad for finite integrals
                result, error = integrate.quad(
                    func, lower, upper,
                    epsabs=tolerance, epsrel=tolerance,
                    limit=max_subdivisions
                )
            
            return {
                "numerical": result,
                "bounds": {"lower": str(lower_bound), "upper": str(upper_bound)},
                "convergence_info": {
                    "estimated_error": error,
                    "tolerance": tolerance,
                    "max_subdivisions": max_subdivisions,
                    "converged": error < tolerance,
                },
            }
            
        except Exception as e:
            raise MathematicalError(f"Numerical integration failed: {e}")
    
    def _parse_bound(self, bound: Union[str, float]) -> Union[sp.Expr, float]:
        """
        Parse integration bound (could be number or symbolic).
        
        Args:
            bound: Bound value as string or number
        
        Returns:
            Union[sp.Expr, float]: Parsed bound
        """
        if isinstance(bound, (int, float)):
            return bound
        
        # Handle common symbolic bounds
        bound_str = str(bound).strip().lower()
        if bound_str in ["inf", "infinity", "+inf", "+infinity"]:
            return sp.oo
        elif bound_str in ["-inf", "-infinity"]:
            return -sp.oo
        elif bound_str == "pi":
            return sp.pi
        elif bound_str == "e":
            return sp.E
        else:
            # Try to parse as symbolic expression
            try:
                return sp.sympify(bound)
            except:
                # Fallback to float conversion
                return float(bound)
    
    def _analyze_integral_properties(
        self,
        expr: sp.Expr,
        var: sp.Symbol,
        lower_bound: Optional[Union[str, float]],
        upper_bound: Optional[Union[str, float]],
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Analyze properties of the integral and function.
        
        Args:
            expr: Original expression
            var: Integration variable
            lower_bound: Lower bound (if definite)
            upper_bound: Upper bound (if definite) 
            result: Integration result
        
        Returns:
            Dict[str, Any]: Properties analysis
        """
        properties = {}
        
        try:
            # Function continuity analysis
            discontinuities = sp.solve(sp.denom(expr), var)
            properties["discontinuities"] = [str(d) for d in discontinuities]
            
            # Symmetry analysis (for definite integrals with symmetric bounds)
            if lower_bound is not None and upper_bound is not None:
                try:
                    lower = self._parse_bound(lower_bound)
                    upper = self._parse_bound(upper_bound)
                    
                    # Check for even/odd symmetry around origin
                    if lower == -upper:
                        # Test if function is even or odd
                        expr_neg = expr.subs(var, -var)
                        if sp.simplify(expr - expr_neg) == 0:
                            properties["symmetry"] = "even"
                        elif sp.simplify(expr + expr_neg) == 0:
                            properties["symmetry"] = "odd"
                            if "numerical" in result:
                                properties["symmetry_note"] = "Integral of odd function over symmetric interval should be 0"
                    
                    # Calculate average value for definite integrals
                    if "numerical" in result and isinstance(lower, (int, float)) and isinstance(upper, (int, float)):
                        interval_length = upper - lower
                        if interval_length > 0:
                            properties["average_value"] = result["numerical"] / interval_length
                            
                except Exception:
                    pass  # Skip property analysis if it fails
            
            # Integration method recommendation
            if "symbolic" in result and "numerical" in result:
                symbolic_str = result.get("symbolic", "")
                if len(symbolic_str) > 100 or "Integral" in symbolic_str:
                    properties["method_recommendation"] = "Consider numerical method for this complex expression"
                else:
                    properties["method_recommendation"] = "Symbolic result available and clean"
            
        except Exception as e:
            logger.debug(f"Property analysis failed: {e}")
            properties["analysis_note"] = "Property analysis skipped due to complexity"
        
        return properties
