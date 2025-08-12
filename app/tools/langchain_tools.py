"""LangChain-compatible tools for BigTool integration."""

from typing import Any, Dict, Optional, Union
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
import io
import base64

from ..core.logging import get_logger

logger = get_logger(__name__)


class IntegralInput(BaseModel):
    """Input schema for integral calculation."""
    expression: str = Field(description="Mathematical expression to integrate (e.g., 'x**2', 'sin(x)', 'exp(x)')")
    variable: str = Field(default="x", description="Variable of integration")
    lower_bound: Optional[Union[str, float]] = Field(None, description="Lower integration bound (number or 'inf')")
    upper_bound: Optional[Union[str, float]] = Field(None, description="Upper integration bound (number or 'inf')")


class PlotInput(BaseModel):
    """Input schema for plot generation."""
    expression: str = Field(description="Mathematical expression to plot (e.g., 'x**2', 'sin(x)')")
    variable: str = Field(default="x", description="Variable to plot")
    x_min: float = Field(default=-10, description="Minimum x value")
    x_max: float = Field(default=10, description="Maximum x value")
    show_area: bool = Field(default=False, description="Whether to highlight area under curve")
    area_bounds: Optional[tuple] = Field(None, description="Bounds for area highlighting (a, b)")


class AnalysisInput(BaseModel):
    """Input schema for function analysis."""
    expression: str = Field(description="Mathematical expression to analyze")
    variable: str = Field(default="x", description="Variable to analyze")
    analysis_type: str = Field(default="full", description="Type of analysis: 'derivative', 'critical_points', 'full'")


class LangChainIntegralTool(BaseTool):
    """LangChain-compatible integral calculation tool."""
    
    name: str = "integral_calculator"
    description: str = (
        "Calculate definite and indefinite integrals. "
        "For definite integrals, provide lower_bound and upper_bound. "
        "Example: expression='x**2', lower_bound=0, upper_bound=3"
    )
    args_schema: type[BaseModel] = IntegralInput
    
    def _run(self, expression: str, variable: str = "x", 
             lower_bound: Optional[Union[str, float]] = None, 
             upper_bound: Optional[Union[str, float]] = None) -> str:
        """Execute integral calculation."""
        try:
            # Parse expression
            expr = sp.sympify(expression)
            var = sp.Symbol(variable)
            
            logger.info(f"Calculating integral of {expression}")
            
            # Determine if definite or indefinite
            if lower_bound is not None and upper_bound is not None:
                # Definite integral
                lower = self._parse_bound(lower_bound)
                upper = self._parse_bound(upper_bound)
                
                # Calculate symbolic result
                symbolic_result = sp.integrate(expr, (var, lower, upper))
                
                # Try to get numerical value
                try:
                    numerical_result = float(symbolic_result.evalf())
                    result = (
                        f"Definite integral of {expression} from {lower_bound} to {upper_bound}:\n"
                        f"Symbolic result: {symbolic_result}\n"
                        f"Numerical result: {numerical_result:.6f}"
                    )
                except:
                    result = (
                        f"Definite integral of {expression} from {lower_bound} to {upper_bound}:\n"
                        f"Result: {symbolic_result}"
                    )
            else:
                # Indefinite integral
                symbolic_result = sp.integrate(expr, var)
                result = (
                    f"Indefinite integral of {expression}:\n"
                    f"Result: {symbolic_result} + C"
                )
            
            logger.info("Integral calculation completed successfully")
            return result
            
        except Exception as e:
            error_msg = f"Integral calculation failed: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def _parse_bound(self, bound: Union[str, float]):
        """Parse integration bound."""
        if isinstance(bound, (int, float)):
            return bound
        
        bound_str = str(bound).strip().lower()
        if bound_str in ["inf", "infinity", "+inf"]:
            return sp.oo
        elif bound_str in ["-inf", "-infinity"]:
            return -sp.oo
        elif bound_str == "pi":
            return sp.pi
        else:
            try:
                return sp.sympify(bound)
            except:
                return float(bound)


class LangChainPlotTool(BaseTool):
    """LangChain-compatible plotting tool."""
    
    name: str = "plot_generator"
    description: str = (
        "Generate mathematical function plots with optional area highlighting. "
        "Use show_area=True and area_bounds=(a,b) to highlight area under curve. "
        "Example: expression='x**2', x_min=0, x_max=3, show_area=True, area_bounds=(0,3)"
    )
    args_schema: type[BaseModel] = PlotInput
    
    def _run(self, expression: str, variable: str = "x", 
             x_min: float = -10, x_max: float = 10,
             show_area: bool = False, area_bounds: Optional[tuple] = None) -> str:
        """Generate function plot."""
        try:
            # Parse expression
            expr = sp.sympify(expression)
            var = sp.Symbol(variable)
            
            logger.info(f"Generating plot for {expression}")
            
            # Create function
            func = sp.lambdify(var, expr, "numpy")
            
            # Generate x values
            x_vals = np.linspace(x_min, x_max, 1000)
            y_vals = func(x_vals)
            
            # Create plot
            fig = go.Figure()
            
            # Add main function
            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals,
                mode='lines',
                name=f'f({variable}) = {expression}',
                line=dict(color='blue', width=2)
            ))
            
            # Add area highlighting if requested
            if show_area and area_bounds:
                a, b = area_bounds
                if a >= x_min and b <= x_max:
                    # Generate points for area
                    x_area = np.linspace(a, b, 200)
                    y_area = func(x_area)
                    
                    # Add area fill
                    fig.add_trace(go.Scatter(
                        x=np.concatenate([x_area, [b, a]]),
                        y=np.concatenate([y_area, [0, 0]]),
                        fill='toself',
                        fillcolor='rgba(255, 0, 0, 0.3)',
                        line=dict(color='red'),
                        name=f'Area from {a} to {b}'
                    ))
            
            # Update layout
            fig.update_layout(
                title=f'Plot of f({variable}) = {expression}',
                xaxis_title=variable,
                yaxis_title=f'f({variable})',
                showlegend=True,
                template='plotly_white'
            )
            
            # Store the plot in a global variable for ChatComponent to access
            import sys
            if not hasattr(sys.modules[__name__], '_generated_plots'):
                sys.modules[__name__]._generated_plots = []
            
            sys.modules[__name__]._generated_plots.append({
                'figure': fig,
                'expression': expression,
                'variable': variable,
                'show_area': show_area,
                'area_bounds': area_bounds,
                'type': 'plotly'
            })
            
            result = f"Plot generated for f({variable}) = {expression}"
            if show_area and area_bounds:
                result += f" with highlighted area from {area_bounds[0]} to {area_bounds[1]}"
            
            logger.info("Plot generation completed successfully")
            return result
            
        except Exception as e:
            error_msg = f"Plot generation failed: {str(e)}"
            logger.error(error_msg)
            return error_msg


class LangChainAnalysisTool(BaseTool):
    """LangChain-compatible function analysis tool."""
    
    name: str = "function_analyzer"
    description: str = (
        "Analyze mathematical functions: derivatives, critical points, behavior. "
        "Use analysis_type='derivative' for derivatives only, 'critical_points' for critical points, "
        "'full' for comprehensive analysis. Example: expression='x**3-3*x**2+2', analysis_type='full'"
    )
    args_schema: type[BaseModel] = AnalysisInput
    
    def _run(self, expression: str, variable: str = "x", 
             analysis_type: str = "full") -> str:
        """Analyze mathematical function."""
        try:
            # Parse expression
            expr = sp.sympify(expression)
            var = sp.Symbol(variable)
            
            logger.info(f"Analyzing function {expression}, type: {analysis_type}")
            
            results = []
            
            # Derivative analysis
            if analysis_type in ["derivative", "full"]:
                derivative = sp.diff(expr, var)
                results.append(f"First derivative: {derivative}")
                
                second_derivative = sp.diff(derivative, var)
                results.append(f"Second derivative: {second_derivative}")
            
            # Critical points
            if analysis_type in ["critical_points", "full"]:
                derivative = sp.diff(expr, var)
                critical_points = sp.solve(derivative, var)
                if critical_points:
                    real_critical_points = [cp for cp in critical_points if cp.is_real]
                    results.append(f"Critical points: {real_critical_points}")
                else:
                    results.append("No critical points found")
            
            # Full analysis
            if analysis_type == "full":
                # Limits at infinity
                try:
                    limit_pos_inf = sp.limit(expr, var, sp.oo)
                    limit_neg_inf = sp.limit(expr, var, -sp.oo)
                    results.append(f"Limit as {variable} → +∞: {limit_pos_inf}")
                    results.append(f"Limit as {variable} → -∞: {limit_neg_inf}")
                except:
                    results.append("Limit analysis failed")
                
                # Domain analysis
                try:
                    domain = sp.calculus.util.continuous_domain(expr, var, sp.Reals)
                    results.append(f"Domain: {domain}")
                except:
                    results.append("Domain analysis failed")
            
            result = f"Analysis of f({variable}) = {expression}:\n" + "\n".join(results)
            
            logger.info("Function analysis completed successfully")
            return result
            
        except Exception as e:
            error_msg = f"Function analysis failed: {str(e)}"
            logger.error(error_msg)
            return error_msg


# Export tools for BigTool registration
def get_langchain_tools():
    """Get all LangChain-compatible tools for BigTool."""
    return [
        LangChainIntegralTool(),
        LangChainPlotTool(),
        LangChainAnalysisTool()
    ]


def get_generated_plots():
    """Get and clear generated plots from tools."""
    import sys
    if hasattr(sys.modules[__name__], '_generated_plots'):
        plots = sys.modules[__name__]._generated_plots.copy()
        sys.modules[__name__]._generated_plots.clear()
        return plots
    return []
