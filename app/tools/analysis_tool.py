"""Mathematical analysis tool for function properties and calculus operations."""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import sympy as sp
from pydantic import Field, field_validator

from ..core.exceptions import MathematicalError, ToolError
from ..core.logging import get_logger
from ..utils.validators import MathExpressionValidator
from .base import BaseTool, ToolInput, ToolOutput

logger = get_logger(__name__)


class AnalysisInput(ToolInput):
    """Input validation for mathematical analysis operations."""
    
    expression: str = Field(..., description="Mathematical expression to analyze")
    variable: str = Field(default="x", description="Variable of analysis")
    analysis_type: str = Field(
        default="comprehensive",
        description="Type of analysis: comprehensive, derivative, limits, critical_points, asymptotes"
    )
    point_of_interest: Optional[float] = Field(None, description="Specific point for local analysis")
    domain_range: Tuple[float, float] = Field(default=(-10, 10), description="Domain range for analysis")
    
    @field_validator("expression")
    @classmethod
    def validate_expression(cls, v: str) -> str:
        """Validate mathematical expression."""
        is_valid, error_msg, _ = MathExpressionValidator.validate_expression(v)
        if not is_valid:
            raise ValueError(f"Invalid expression: {error_msg}")
        return v.strip()
    
    @field_validator("analysis_type")
    @classmethod
    def validate_analysis_type(cls, v: str) -> str:
        """Validate analysis type."""
        valid_types = [
            "comprehensive", "derivative", "limits", "critical_points", 
            "asymptotes", "continuity", "monotonicity", "concavity"
        ]
        if v.lower() not in valid_types:
            raise ValueError(f"analysis_type must be one of: {valid_types}")
        return v.lower()
    
    @field_validator("domain_range")
    @classmethod
    def validate_domain_range(cls, v: Tuple[float, float]) -> Tuple[float, float]:
        """Validate domain range."""
        if len(v) != 2 or v[0] >= v[1]:
            raise ValueError("domain_range must be a tuple (min, max) with min < max")
        return v


class AnalysisOutput(ToolOutput):
    """Output format for mathematical analysis operations."""
    
    function_info: Dict[str, Any] = Field(default_factory=dict, description="Basic function information")
    derivatives: Dict[str, Any] = Field(default_factory=dict, description="Derivative analysis")
    critical_points: List[Dict[str, Any]] = Field(default_factory=list, description="Critical points analysis")
    limits: Dict[str, Any] = Field(default_factory=dict, description="Limit analysis")
    asymptotes: Dict[str, Any] = Field(default_factory=dict, description="Asymptote analysis")
    continuity: Dict[str, Any] = Field(default_factory=dict, description="Continuity analysis")
    monotonicity: Dict[str, Any] = Field(default_factory=dict, description="Monotonicity analysis")
    concavity: Dict[str, Any] = Field(default_factory=dict, description="Concavity analysis")
    taylor_series: Optional[Dict[str, Any]] = Field(None, description="Taylor series expansion")


class AnalysisTool(BaseTool):
    """
    Tool for comprehensive mathematical function analysis.
    
    This tool performs calculus operations including derivatives, limits, critical points,
    asymptotes, continuity, monotonicity, and concavity analysis.
    """
    
    def __init__(self) -> None:
        """Initialize the analysis tool."""
        super().__init__(
            name="function_analyzer",
            description=(
                "Perform comprehensive mathematical analysis of functions including "
                "derivatives, limits, critical points, asymptotes, continuity, "
                "monotonicity, concavity, and Taylor series expansion."
            ),
            timeout=25.0,
            max_retries=2,
        )
    
    def _validate_tool_input(self, input_data: Dict[str, Any]) -> AnalysisInput:
        """Validate analysis input."""
        try:
            return AnalysisInput(**input_data)
        except Exception as e:
            raise ToolError(f"Input validation failed: {e}", tool_name=self.name)
    
    def _execute_tool(self, validated_input: AnalysisInput) -> AnalysisOutput:
        """Execute mathematical analysis."""
        try:
            # Parse the mathematical expression
            expr = sp.sympify(validated_input.expression)
            var = sp.Symbol(validated_input.variable)
            
            logger.info(f"Performing {validated_input.analysis_type} analysis", extra={
                "expression": str(expr),
                "variable": str(var),
            })
            
            # Perform analysis based on type
            result = AnalysisOutput(
                success=True,
                result={},
                execution_time=0.0,  # Will be set by base class
            )
            
            if validated_input.analysis_type == "comprehensive":
                self._comprehensive_analysis(expr, var, validated_input, result)
            elif validated_input.analysis_type == "derivative":
                result.derivatives = self._derivative_analysis(expr, var)
            elif validated_input.analysis_type == "limits":
                result.limits = self._limit_analysis(expr, var, validated_input.domain_range)
            elif validated_input.analysis_type == "critical_points":
                result.critical_points = self._critical_points_analysis(expr, var, validated_input.domain_range)
            elif validated_input.analysis_type == "asymptotes":
                result.asymptotes = self._asymptote_analysis(expr, var)
            elif validated_input.analysis_type == "continuity":
                result.continuity = self._continuity_analysis(expr, var, validated_input.domain_range)
            elif validated_input.analysis_type == "monotonicity":
                result.monotonicity = self._monotonicity_analysis(expr, var, validated_input.domain_range)
            elif validated_input.analysis_type == "concavity":
                result.concavity = self._concavity_analysis(expr, var, validated_input.domain_range)
            
            # Add Taylor series if point of interest is specified
            if validated_input.point_of_interest is not None:
                result.taylor_series = self._taylor_series_analysis(
                    expr, var, validated_input.point_of_interest
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Mathematical analysis failed: {e}", exc_info=True)
            raise MathematicalError(
                f"Mathematical analysis failed: {e}",
                expression=validated_input.expression,
                computation_type="analysis",
            )
    
    def _comprehensive_analysis(
        self,
        expr: sp.Expr,
        var: sp.Symbol,
        input_params: AnalysisInput,
        result: AnalysisOutput,
    ) -> None:
        """
        Perform comprehensive function analysis.
        
        Args:
            expr: SymPy expression
            var: Variable symbol
            input_params: Input parameters
            result: Analysis output to populate
        """
        try:
            # Basic function information
            result.function_info = self._basic_function_info(expr, var)
            
            # Derivative analysis
            result.derivatives = self._derivative_analysis(expr, var)
            
            # Critical points
            result.critical_points = self._critical_points_analysis(expr, var, input_params.domain_range)
            
            # Limits
            result.limits = self._limit_analysis(expr, var, input_params.domain_range)
            
            # Asymptotes
            result.asymptotes = self._asymptote_analysis(expr, var)
            
            # Continuity
            result.continuity = self._continuity_analysis(expr, var, input_params.domain_range)
            
            # Monotonicity
            result.monotonicity = self._monotonicity_analysis(expr, var, input_params.domain_range)
            
            # Concavity
            result.concavity = self._concavity_analysis(expr, var, input_params.domain_range)
            
        except Exception as e:
            logger.warning(f"Some analysis components failed: {e}")
    
    def _basic_function_info(self, expr: sp.Expr, var: sp.Symbol) -> Dict[str, Any]:
        """Get basic function information."""
        info = {
            "expression": str(expr),
            "variable": str(var),
            "domain_restrictions": [],
            "function_type": self._classify_function(expr),
        }
        
        try:
            # Check for domain restrictions
            atoms = expr.atoms()
            
            # Logarithm restrictions
            for atom in atoms:
                if isinstance(atom, sp.log):
                    info["domain_restrictions"].append("Logarithm requires positive arguments")
                    break
            
            # Square root restrictions
            for atom in atoms:
                if isinstance(atom, sp.Pow) and atom.exp == sp.Rational(1, 2):
                    info["domain_restrictions"].append("Square root requires non-negative arguments")
                    break
            
            # Rational function restrictions (division by zero)
            if expr.as_numer_denom()[1] != 1:
                denom = expr.as_numer_denom()[1]
                try:
                    zeros = sp.solve(denom, var)
                    if zeros:
                        info["domain_restrictions"].append(f"Undefined at: {[str(z) for z in zeros]}")
                except:
                    info["domain_restrictions"].append("May have undefined points")
            
        except Exception as e:
            logger.debug(f"Domain analysis failed: {e}")
        
        return info
    
    def _classify_function(self, expr: sp.Expr) -> str:
        """Classify the function type."""
        if expr.is_polynomial():
            degree = sp.degree(expr)
            if degree == 1:
                return "linear"
            elif degree == 2:
                return "quadratic"
            elif degree == 3:
                return "cubic"
            else:
                return f"polynomial_degree_{degree}"
        elif expr.is_rational_function():
            return "rational"
        elif any(isinstance(atom, (sp.sin, sp.cos, sp.tan)) for atom in expr.atoms()):
            return "trigonometric"
        elif any(isinstance(atom, sp.exp) for atom in expr.atoms()):
            return "exponential"
        elif any(isinstance(atom, sp.log) for atom in expr.atoms()):
            return "logarithmic"
        else:
            return "general"
    
    def _derivative_analysis(self, expr: sp.Expr, var: sp.Symbol) -> Dict[str, Any]:
        """Analyze derivatives of the function."""
        derivatives = {}
        
        try:
            # First derivative
            first_deriv = sp.diff(expr, var)
            derivatives["first"] = {
                "expression": str(first_deriv),
                "simplified": str(sp.simplify(first_deriv)),
            }
            
            # Second derivative
            second_deriv = sp.diff(first_deriv, var)
            derivatives["second"] = {
                "expression": str(second_deriv),
                "simplified": str(sp.simplify(second_deriv)),
            }
            
            # Third derivative if not too complex
            if len(str(second_deriv)) < 100:
                third_deriv = sp.diff(second_deriv, var)
                derivatives["third"] = {
                    "expression": str(third_deriv),
                    "simplified": str(sp.simplify(third_deriv)),
                }
            
        except Exception as e:
            logger.warning(f"Derivative calculation failed: {e}")
            derivatives["error"] = str(e)
        
        return derivatives
    
    def _critical_points_analysis(
        self, 
        expr: sp.Expr, 
        var: sp.Symbol, 
        domain_range: Tuple[float, float]
    ) -> List[Dict[str, Any]]:
        """Find and analyze critical points."""
        critical_points = []
        
        try:
            # Find critical points (where derivative = 0 or undefined)
            first_deriv = sp.diff(expr, var)
            
            # Solve f'(x) = 0
            critical_candidates = sp.solve(first_deriv, var)
            
            # Analyze each critical point
            for candidate in critical_candidates:
                try:
                    if candidate.is_real:
                        x_val = float(candidate.evalf())
                        
                        # Check if point is in domain range
                        if domain_range[0] <= x_val <= domain_range[1]:
                            y_val = float(expr.subs(var, candidate).evalf())
                            
                            # Classify critical point using second derivative test
                            point_type = self._classify_critical_point(expr, var, candidate)
                            
                            critical_points.append({
                                "x": x_val,
                                "y": y_val,
                                "type": point_type,
                                "derivative_value": 0.0,
                            })
                            
                except Exception as e:
                    logger.debug(f"Failed to analyze critical point {candidate}: {e}")
            
            # Check for points where derivative is undefined
            # This would require more complex analysis of the derivative's domain
            
        except Exception as e:
            logger.warning(f"Critical points analysis failed: {e}")
            return [{"error": str(e)}]
        
        return critical_points
    
    def _classify_critical_point(
        self, 
        expr: sp.Expr, 
        var: sp.Symbol, 
        point: sp.Expr
    ) -> str:
        """Classify a critical point using the second derivative test."""
        try:
            second_deriv = sp.diff(expr, var, 2)
            second_deriv_at_point = second_deriv.subs(var, point)
            
            if second_deriv_at_point > 0:
                return "local_minimum"
            elif second_deriv_at_point < 0:
                return "local_maximum"
            else:
                # Inconclusive - could be inflection point or higher-order analysis needed
                return "inconclusive"
                
        except Exception:
            return "unknown"
    
    def _limit_analysis(
        self, 
        expr: sp.Expr, 
        var: sp.Symbol, 
        domain_range: Tuple[float, float]
    ) -> Dict[str, Any]:
        """Analyze limits of the function."""
        limits = {}
        
        try:
            # Limit as x approaches infinity
            try:
                limit_inf = sp.limit(expr, var, sp.oo)
                limits["as_x_approaches_infinity"] = str(limit_inf)
            except:
                limits["as_x_approaches_infinity"] = "undefined"
            
            # Limit as x approaches negative infinity
            try:
                limit_neg_inf = sp.limit(expr, var, -sp.oo)
                limits["as_x_approaches_negative_infinity"] = str(limit_neg_inf)
            except:
                limits["as_x_approaches_negative_infinity"] = "undefined"
            
            # Find points where function might be discontinuous
            discontinuous_points = []
            try:
                denom = expr.as_numer_denom()[1]
                if denom != 1:
                    zeros = sp.solve(denom, var)
                    for zero in zeros:
                        if zero.is_real:
                            x_val = float(zero.evalf())
                            if domain_range[0] <= x_val <= domain_range[1]:
                                # Calculate left and right limits
                                try:
                                    left_limit = sp.limit(expr, var, zero, '-')
                                    right_limit = sp.limit(expr, var, zero, '+')
                                    discontinuous_points.append({
                                        "x": x_val,
                                        "left_limit": str(left_limit),
                                        "right_limit": str(right_limit),
                                        "type": "removable" if left_limit == right_limit else "jump_or_infinite"
                                    })
                                except:
                                    discontinuous_points.append({
                                        "x": x_val,
                                        "type": "unknown"
                                    })
            except:
                pass
            
            limits["discontinuous_points"] = discontinuous_points
            
        except Exception as e:
            logger.warning(f"Limit analysis failed: {e}")
            limits["error"] = str(e)
        
        return limits
    
    def _asymptote_analysis(self, expr: sp.Expr, var: sp.Symbol) -> Dict[str, Any]:
        """Analyze asymptotes of the function."""
        asymptotes = {
            "vertical": [],
            "horizontal": [],
            "oblique": None,
        }
        
        try:
            # Vertical asymptotes (where denominator = 0)
            denom = expr.as_numer_denom()[1]
            if denom != 1:
                zeros = sp.solve(denom, var)
                for zero in zeros:
                    if zero.is_real:
                        # Check if it's actually a vertical asymptote
                        try:
                            left_limit = sp.limit(expr, var, zero, '-')
                            right_limit = sp.limit(expr, var, zero, '+')
                            if left_limit.is_infinite or right_limit.is_infinite:
                                asymptotes["vertical"].append(float(zero.evalf()))
                        except:
                            asymptotes["vertical"].append(float(zero.evalf()))
            
            # Horizontal asymptotes
            limit_inf = sp.limit(expr, var, sp.oo)
            limit_neg_inf = sp.limit(expr, var, -sp.oo)
            
            if limit_inf.is_finite:
                asymptotes["horizontal"].append(float(limit_inf))
            if limit_neg_inf.is_finite and limit_neg_inf != limit_inf:
                asymptotes["horizontal"].append(float(limit_neg_inf))
            
            # Oblique asymptotes (if no horizontal asymptotes)
            if not asymptotes["horizontal"] and expr.is_rational_function():
                numer, denom = expr.as_numer_denom()
                if sp.degree(numer) == sp.degree(denom) + 1:
                    # Oblique asymptote exists
                    oblique = sp.div(numer, denom)[0]
                    asymptotes["oblique"] = str(oblique)
            
        except Exception as e:
            logger.warning(f"Asymptote analysis failed: {e}")
            asymptotes["error"] = str(e)
        
        return asymptotes
    
    def _continuity_analysis(
        self, 
        expr: sp.Expr, 
        var: sp.Symbol, 
        domain_range: Tuple[float, float]
    ) -> Dict[str, Any]:
        """Analyze continuity of the function."""
        continuity = {
            "is_continuous_on_domain": True,
            "discontinuities": [],
            "continuity_type": "continuous",
        }
        
        try:
            # Check for obvious discontinuities
            denom = expr.as_numer_denom()[1]
            if denom != 1:
                zeros = sp.solve(denom, var)
                for zero in zeros:
                    if zero.is_real:
                        x_val = float(zero.evalf())
                        if domain_range[0] <= x_val <= domain_range[1]:
                            continuity["is_continuous_on_domain"] = False
                            continuity["discontinuities"].append({
                                "x": x_val,
                                "type": "infinite_discontinuity"
                            })
            
            # Check for domain restrictions that might cause discontinuities
            atoms = expr.atoms()
            for atom in atoms:
                if isinstance(atom, sp.log):
                    # Logarithm discontinuities
                    arg = atom.args[0]
                    if arg == var:
                        if domain_range[0] <= 0 <= domain_range[1]:
                            continuity["is_continuous_on_domain"] = False
                            continuity["discontinuities"].append({
                                "x": 0,
                                "type": "logarithmic_discontinuity"
                            })
            
            if continuity["discontinuities"]:
                continuity["continuity_type"] = "piecewise_continuous"
            else:
                continuity["continuity_type"] = "continuous"
                
        except Exception as e:
            logger.warning(f"Continuity analysis failed: {e}")
            continuity["error"] = str(e)
        
        return continuity
    
    def _monotonicity_analysis(
        self, 
        expr: sp.Expr, 
        var: sp.Symbol, 
        domain_range: Tuple[float, float]
    ) -> Dict[str, Any]:
        """Analyze monotonicity of the function."""
        monotonicity = {
            "intervals": [],
            "overall_behavior": "varies",
        }
        
        try:
            first_deriv = sp.diff(expr, var)
            
            # Find where derivative is zero or undefined
            critical_points = sp.solve(first_deriv, var)
            real_critical_points = []
            
            for cp in critical_points:
                if cp.is_real:
                    x_val = float(cp.evalf())
                    if domain_range[0] <= x_val <= domain_range[1]:
                        real_critical_points.append(x_val)
            
            real_critical_points.sort()
            
            # Test intervals between critical points
            test_points = []
            if not real_critical_points:
                test_points = [(domain_range[0] + domain_range[1]) / 2]
            else:
                # Add test points before first, between, and after last critical point
                if real_critical_points[0] > domain_range[0]:
                    test_points.append(domain_range[0] + 0.1)
                
                for i in range(len(real_critical_points) - 1):
                    test_points.append((real_critical_points[i] + real_critical_points[i + 1]) / 2)
                
                if real_critical_points[-1] < domain_range[1]:
                    test_points.append(domain_range[1] - 0.1)
            
            # Evaluate derivative at test points
            for i, test_point in enumerate(test_points):
                try:
                    deriv_value = first_deriv.subs(var, test_point).evalf()
                    
                    if i == 0:
                        start = domain_range[0]
                    else:
                        start = real_critical_points[i - 1]
                    
                    if i == len(test_points) - 1:
                        end = domain_range[1]
                    else:
                        end = real_critical_points[i]
                    
                    if deriv_value > 0:
                        monotonicity["intervals"].append({
                            "start": start,
                            "end": end,
                            "behavior": "increasing"
                        })
                    elif deriv_value < 0:
                        monotonicity["intervals"].append({
                            "start": start,
                            "end": end,
                            "behavior": "decreasing"
                        })
                    else:
                        monotonicity["intervals"].append({
                            "start": start,
                            "end": end,
                            "behavior": "constant"
                        })
                        
                except Exception:
                    pass
            
            # Determine overall behavior
            if len(monotonicity["intervals"]) == 1:
                monotonicity["overall_behavior"] = monotonicity["intervals"][0]["behavior"]
            elif all(interval["behavior"] == "increasing" for interval in monotonicity["intervals"]):
                monotonicity["overall_behavior"] = "strictly_increasing"
            elif all(interval["behavior"] == "decreasing" for interval in monotonicity["intervals"]):
                monotonicity["overall_behavior"] = "strictly_decreasing"
            else:
                monotonicity["overall_behavior"] = "non_monotonic"
                
        except Exception as e:
            logger.warning(f"Monotonicity analysis failed: {e}")
            monotonicity["error"] = str(e)
        
        return monotonicity
    
    def _concavity_analysis(
        self, 
        expr: sp.Expr, 
        var: sp.Symbol, 
        domain_range: Tuple[float, float]
    ) -> Dict[str, Any]:
        """Analyze concavity of the function."""
        concavity = {
            "intervals": [],
            "inflection_points": [],
        }
        
        try:
            second_deriv = sp.diff(expr, var, 2)
            
            # Find inflection points (where second derivative = 0)
            inflection_candidates = sp.solve(second_deriv, var)
            
            real_inflection_points = []
            for ip in inflection_candidates:
                if ip.is_real:
                    x_val = float(ip.evalf())
                    if domain_range[0] <= x_val <= domain_range[1]:
                        y_val = float(expr.subs(var, ip).evalf())
                        real_inflection_points.append(x_val)
                        concavity["inflection_points"].append({
                            "x": x_val,
                            "y": y_val
                        })
            
            real_inflection_points.sort()
            
            # Test intervals between inflection points
            test_points = []
            if not real_inflection_points:
                test_points = [(domain_range[0] + domain_range[1]) / 2]
            else:
                if real_inflection_points[0] > domain_range[0]:
                    test_points.append(domain_range[0] + 0.1)
                
                for i in range(len(real_inflection_points) - 1):
                    test_points.append((real_inflection_points[i] + real_inflection_points[i + 1]) / 2)
                
                if real_inflection_points[-1] < domain_range[1]:
                    test_points.append(domain_range[1] - 0.1)
            
            # Evaluate second derivative at test points
            for i, test_point in enumerate(test_points):
                try:
                    second_deriv_value = second_deriv.subs(var, test_point).evalf()
                    
                    if i == 0:
                        start = domain_range[0]
                    else:
                        start = real_inflection_points[i - 1]
                    
                    if i == len(test_points) - 1:
                        end = domain_range[1]
                    else:
                        end = real_inflection_points[i]
                    
                    if second_deriv_value > 0:
                        concavity["intervals"].append({
                            "start": start,
                            "end": end,
                            "concavity": "concave_up"
                        })
                    elif second_deriv_value < 0:
                        concavity["intervals"].append({
                            "start": start,
                            "end": end,
                            "concavity": "concave_down"
                        })
                        
                except Exception:
                    pass
                    
        except Exception as e:
            logger.warning(f"Concavity analysis failed: {e}")
            concavity["error"] = str(e)
        
        return concavity
    
    def _taylor_series_analysis(
        self, 
        expr: sp.Expr, 
        var: sp.Symbol, 
        point: float,
        order: int = 5
    ) -> Dict[str, Any]:
        """Calculate Taylor series expansion."""
        taylor = {}
        
        try:
            taylor_series = sp.series(expr, var, point, n=order + 1).removeO()
            
            taylor = {
                "expansion_point": point,
                "order": order,
                "series": str(taylor_series),
                "coefficients": [],
            }
            
            # Extract coefficients
            for i in range(order + 1):
                try:
                    coeff = sp.diff(expr, var, i).subs(var, point) / sp.factorial(i)
                    taylor["coefficients"].append({
                        "order": i,
                        "coefficient": float(coeff.evalf()),
                        "term": str(coeff * (var - point)**i)
                    })
                except:
                    break
                    
        except Exception as e:
            logger.warning(f"Taylor series analysis failed: {e}")
            taylor["error"] = str(e)
        
        return taylor
