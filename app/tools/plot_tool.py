"""Plotting tool for mathematical function visualization and area under curve."""

import base64
import io
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sympy as sp
from pydantic import Field, field_validator

from ..core.exceptions import MathematicalError, ToolError
from ..core.logging import get_logger
from ..utils.validators import MathExpressionValidator
from .base import BaseTool, ToolInput, ToolOutput

logger = get_logger(__name__)


class PlotInput(ToolInput):
    """Input validation for plotting operations."""
    
    expression: str = Field(..., description="Mathematical expression to plot")
    variable: str = Field(default="x", description="Independent variable")
    x_range: Tuple[float, float] = Field(default=(-10, 10), description="X-axis range")
    y_range: Optional[Tuple[float, float]] = Field(None, description="Y-axis range (auto if None)")
    plot_type: str = Field(default="function", description="Plot type: function, area, comparison")
    fill_area: bool = Field(default=False, description="Fill area under curve")
    integral_bounds: Optional[Tuple[Union[str, float], Union[str, float]]] = Field(
        None, description="Bounds for area highlighting"
    )
    style: str = Field(default="matplotlib", description="Plot style: matplotlib, plotly")
    theme: str = Field(default="default", description="Plot theme")
    interactive: bool = Field(default=False, description="Make plot interactive")
    resolution: int = Field(default=1000, description="Number of plot points")
    
    @field_validator("expression")
    @classmethod
    def validate_expression(cls, v: str): 
        """Validate mathematical expression."""
        is_valid, error_msg, _ = MathExpressionValidator.validate_expression(v)
        if not is_valid:
            raise ValueError(f"Invalid expression: {error_msg}")
        return v.strip()
    
    @field_validator("x_range")
    @classmethod
    def validate_x_range(cls, v: Tuple[float, float]): 
        """Validate x-range."""
        if len(v) != 2 or v[0] >= v[1]:
            raise ValueError("x_range must be a tuple (min, max) with min < max")
        if abs(v[1] - v[0]) > 1000:
            raise ValueError("x_range is too large (max span: 1000)")
        return v
    
    @field_validator("y_range")
    @classmethod
    def validate_y_range(cls, v: Optional[Tuple[float, float]]): 
        """Validate y-range."""
        if v is not None:
            if len(v) != 2 or v[0] >= v[1]:
                raise ValueError("y_range must be a tuple (min, max) with min < max")
            if abs(v[1] - v[0]) > 10000:
                raise ValueError("y_range is too large (max span: 10000)")
        return v
    
    @field_validator("plot_type")
    @classmethod
    def validate_plot_type(cls, v: str): 
        """Validate plot type."""
        valid_types = ["function", "area", "comparison", "derivative", "integral"]
        if v.lower() not in valid_types:
            raise ValueError(f"plot_type must be one of: {valid_types}")
        return v.lower()
    
    @field_validator("style")
    @classmethod
    def validate_style(cls, v: str): 
        """Validate plot style."""
        valid_styles = ["matplotlib", "plotly"]
        if v.lower() not in valid_styles:
            raise ValueError(f"style must be one of: {valid_styles}")
        return v.lower()
    
    @field_validator("resolution")
    @classmethod
    def validate_resolution(cls, v: int): 
        """Validate plot resolution."""
        if not 100 <= v <= 10000:
            raise ValueError("resolution must be between 100 and 10000")
        return v


class PlotOutput(ToolOutput):
    """Output format for plotting operations."""
    
    plot_data: Optional[Dict[str, Any]] = Field(None, description="Plot data and configuration")
    image_base64: Optional[str] = Field(None, description="Base64 encoded image (for matplotlib)")
    plotly_json: Optional[Dict[str, Any]] = Field(None, description="Plotly figure JSON")
    plot_info: Dict[str, Any] = Field(default_factory=dict, description="Plot information and statistics")
    area_calculation: Optional[Dict[str, Any]] = Field(None, description="Area calculation results")


class PlotTool(BaseTool):
    """
    Tool for creating mathematical function visualizations.
    
    This tool can generate static and interactive plots, highlight areas under curves,
    and provide visual analysis of mathematical functions with both Matplotlib and Plotly.
    """
    
    def __init__(self) -> None:
        """Initialize the plot tool."""
        super().__init__(
            name="plot_generator",
            description=(
                "Generate mathematical function plots with area visualization. "
                "Supports static and interactive plots, area under curve highlighting, "
                "derivative and integral visualization, and multiple plot styles."
            ),
            timeout=20.0,
            max_retries=2,
        )
        
        # Configure matplotlib for better output
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
    
    def _validate_tool_input(self, input_data: Dict[str, Any]) -> PlotInput:
        """Validate plot generation input."""
        try:
            return PlotInput(**input_data)
        except Exception as e:
            raise ToolError(f"Input validation failed: {e}", tool_name=self.name)
    
    def _execute_tool(self, validated_input: PlotInput) -> PlotOutput:
        """Execute plot generation."""
        try:
            # Parse the mathematical expression
            expr = sp.sympify(validated_input.expression)
            var = sp.Symbol(validated_input.variable)
            
            logger.info(f"Generating {validated_input.plot_type} plot", extra={
                "expression": str(expr),
                "style": validated_input.style,
                "interactive": validated_input.interactive,
            })
            
            # Generate plot data
            x_data, y_data = self._generate_plot_data(
                expr, var, validated_input.x_range, validated_input.resolution
            )
            
            # Calculate plot statistics
            plot_info = self._calculate_plot_info(x_data, y_data, expr, var)
            
            # Calculate area if requested
            area_calculation = None
            if validated_input.fill_area or validated_input.integral_bounds:
                area_calculation = self._calculate_area(
                    expr, var, validated_input.integral_bounds or validated_input.x_range,
                    x_data, y_data
                )
            
            # Generate plot based on style
            if validated_input.style == "matplotlib":
                plot_result = self._create_matplotlib_plot(
                    x_data, y_data, expr, validated_input, area_calculation
                )
            else:  # plotly
                plot_result = self._create_plotly_plot(
                    x_data, y_data, expr, validated_input, area_calculation
                )
            
            return PlotOutput(
                success=True,
                result=plot_result,
                execution_time=0.0,  # Will be set by base class
                plot_data={"x": x_data.tolist(), "y": y_data.tolist()},
                image_base64=plot_result.get("image_base64"),
                plotly_json=plot_result.get("plotly_json"),
                plot_info=plot_info,
                area_calculation=area_calculation,
            )
            
        except Exception as e:
            logger.error(f"Plot generation failed: {e}", exc_info=True)
            raise MathematicalError(
                f"Plot generation failed: {e}",
                expression=validated_input.expression,
                computation_type="visualization",
            )
    
    def _generate_plot_data(
        self,
        expr: sp.Expr,
        var: sp.Symbol,
        x_range: Tuple[float, float],
        resolution: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate x and y data for plotting.
        
        Args:
            expr: SymPy expression
            var: Variable symbol
            x_range: X-axis range
            resolution: Number of points
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: X and Y data arrays
        """
        try:
            # Create x values
            x_data = np.linspace(x_range[0], x_range[1], resolution)
            
            # Convert SymPy expression to numpy function
            func = sp.lambdify(var, expr, "numpy")
            
            # Calculate y values with error handling
            y_data = np.full_like(x_data, np.nan)
            
            for i, x_val in enumerate(x_data):
                try:
                    y_val = func(x_val)
                    if np.isfinite(y_val):
                        y_data[i] = y_val
                except:
                    pass  # Keep NaN for problematic points
            
            return x_data, y_data
            
        except Exception as e:
            raise MathematicalError(f"Failed to generate plot data: {e}")
    
    def _calculate_plot_info(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        expr: sp.Expr,
        var: sp.Symbol,
    ) -> Dict[str, Any]:
        """
        Calculate plot information and statistics.
        
        Args:
            x_data: X data array
            y_data: Y data array  
            expr: Original expression
            var: Variable symbol
        
        Returns:
            Dict[str, Any]: Plot information
        """
        info = {}
        
        try:
            # Basic statistics
            valid_y = y_data[np.isfinite(y_data)]
            if len(valid_y) > 0:
                info["y_min"] = float(np.min(valid_y))
                info["y_max"] = float(np.max(valid_y))
                info["y_mean"] = float(np.mean(valid_y))
                info["y_std"] = float(np.std(valid_y))
            
            # Find approximate zeros (x-intercepts)
            zeros = []
            for i in range(len(y_data) - 1):
                if (np.isfinite(y_data[i]) and np.isfinite(y_data[i + 1]) and
                    y_data[i] * y_data[i + 1] <= 0):
                    # Linear interpolation for better zero approximation
                    x_zero = x_data[i] - y_data[i] * (x_data[i + 1] - x_data[i]) / (y_data[i + 1] - y_data[i])
                    zeros.append(float(x_zero))
            
            info["approximate_zeros"] = zeros[:10]  # Limit to first 10 zeros
            
            # Function properties
            try:
                # Check for discontinuities
                discontinuities = sp.solve(sp.denom(expr), var)
                info["discontinuities"] = [float(d) for d in discontinuities if d.is_real][:5]
            except:
                info["discontinuities"] = []
            
            # Domain analysis
            domain_issues = []
            if sp.log in expr.atoms():
                domain_issues.append("Contains logarithm - ensure positive arguments")
            if sp.sqrt in expr.atoms():
                domain_issues.append("Contains square root - ensure non-negative arguments")
            
            info["domain_notes"] = domain_issues
            
        except Exception as e:
            logger.debug(f"Plot info calculation failed: {e}")
            info["calculation_note"] = "Some statistics could not be calculated"
        
        return info
    
    def _calculate_area(
        self,
        expr: sp.Expr,
        var: sp.Symbol,
        bounds: Tuple[Union[str, float], Union[str, float]],
        x_data: np.ndarray,
        y_data: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Calculate area under the curve.
        
        Args:
            expr: SymPy expression
            var: Variable symbol
            bounds: Integration bounds
            x_data: X data array
            y_data: Y data array
        
        Returns:
            Dict[str, Any]: Area calculation results
        """
        try:
            # Parse bounds
            lower_bound = float(bounds[0]) if isinstance(bounds[0], (int, float)) else bounds[0]
            upper_bound = float(bounds[1]) if isinstance(bounds[1], (int, float)) else bounds[1]
            
            # Numerical integration using trapezoidal rule
            mask = (x_data >= lower_bound) & (x_data <= upper_bound) & np.isfinite(y_data)
            x_region = x_data[mask]
            y_region = y_data[mask]
            
            if len(x_region) > 1:
                numerical_area = np.trapz(y_region, x_region)
                positive_area = np.trapz(np.maximum(y_region, 0), x_region)
                negative_area = np.trapz(np.minimum(y_region, 0), x_region)
                
                return {
                    "total_area": float(numerical_area),
                    "positive_area": float(positive_area),
                    "negative_area": float(abs(negative_area)),
                    "net_area": float(numerical_area),
                    "bounds": {"lower": lower_bound, "upper": upper_bound},
                    "method": "numerical_trapezoidal",
                }
            else:
                return {
                    "error": "Insufficient data points in integration region",
                    "bounds": {"lower": lower_bound, "upper": upper_bound},
                }
                
        except Exception as e:
            logger.warning(f"Area calculation failed: {e}")
            return {
                "error": f"Area calculation failed: {e}",
                "bounds": {"lower": str(bounds[0]), "upper": str(bounds[1])},
            }
    
    def _create_matplotlib_plot(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        expr: sp.Expr,
        input_params: PlotInput,
        area_calculation: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Create matplotlib plot.
        
        Args:
            x_data: X data array
            y_data: Y data array
            expr: SymPy expression
            input_params: Input parameters
            area_calculation: Area calculation results
        
        Returns:
            Dict[str, Any]: Plot result with base64 image
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot the function
        ax.plot(x_data, y_data, 'b-', linewidth=2, label=f'f({input_params.variable}) = {expr}')
        
        # Fill area if requested
        if input_params.fill_area and area_calculation:
            bounds = area_calculation.get("bounds", {})
            if "lower" in bounds and "upper" in bounds:
                mask = (x_data >= bounds["lower"]) & (x_data <= bounds["upper"])
                ax.fill_between(
                    x_data[mask], 0, y_data[mask], 
                    alpha=0.3, color='lightblue',
                    label=f'Area = {area_calculation.get("total_area", "N/A"):.4f}'
                )
        
        # Customize plot
        ax.set_xlabel(f'{input_params.variable}', fontsize=14)
        ax.set_ylabel(f'f({input_params.variable})', fontsize=14)
        ax.set_title(f'Plot of {expr}', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Set axis ranges
        if input_params.y_range:
            ax.set_ylim(input_params.y_range)
        
        # Add zero lines
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        # Save to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        
        return {
            "image_base64": image_base64,
            "plot_type": "matplotlib",
            "format": "png",
        }
    
    def _create_plotly_plot(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        expr: sp.Expr,
        input_params: PlotInput,
        area_calculation: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Create plotly interactive plot.
        
        Args:
            x_data: X data array
            y_data: Y data array
            expr: SymPy expression
            input_params: Input parameters
            area_calculation: Area calculation results
        
        Returns:
            Dict[str, Any]: Plot result with Plotly JSON
        """
        fig = go.Figure()
        
        # Add main function trace
        fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            mode='lines',
            name=f'f({input_params.variable}) = {expr}',
            line=dict(width=3, color='blue'),
            hovertemplate=f'<b>{input_params.variable}</b>: %{{x:.4f}}<br>' +
                         f'<b>f({input_params.variable})</b>: %{{y:.4f}}<extra></extra>'
        ))
        
        # Add area fill if requested
        if input_params.fill_area and area_calculation:
            bounds = area_calculation.get("bounds", {})
            if "lower" in bounds and "upper" in bounds:
                mask = (x_data >= bounds["lower"]) & (x_data <= bounds["upper"])
                fig.add_trace(go.Scatter(
                    x=x_data[mask],
                    y=y_data[mask],
                    fill='tonexty',
                    fillcolor='rgba(0,100,255,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f'Area = {area_calculation.get("total_area", "N/A"):.4f}',
                    hoverinfo='skip'
                ))
        
        # Customize layout
        fig.update_layout(
            title=dict(
                text=f'Interactive Plot of {expr}',
                x=0.5,
                font=dict(size=18, family="Arial Black")
            ),
            xaxis=dict(
                title=f'{input_params.variable}',
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='black'
            ),
            yaxis=dict(
                title=f'f({input_params.variable})',
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='black'
            ),
            plot_bgcolor='white',
            width=800,
            height=600,
            hovermode='x unified'
        )
        
        # Set axis ranges
        if input_params.y_range:
            fig.update_yaxes(range=input_params.y_range)
        
        return {
            "plotly_json": fig.to_dict(),
            "plot_type": "plotly",
            "format": "json",
        }
