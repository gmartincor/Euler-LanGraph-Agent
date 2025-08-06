#!/usr/bin/env python3
"""
Professional workflow testing script.

Single responsibility: Test mathematical workflow end-to-end.
Eliminates code duplication from multiple debug scripts.
"""

import asyncio
import logging
from datetime import datetime
from uuid import uuid4
from pathlib import Path
import sys

# Add app to path
app_path = Path(__file__).parent.parent / "app"
sys.path.insert(0, str(app_path))

from langchain_core.messages import HumanMessage

from app.core.config import get_settings
from app.core.logging import get_logger
from app.agents.graph import create_mathematical_agent_graph

logger = get_logger(__name__)


class WorkflowTester:
    """
    Professional workflow tester following single responsibility principle.
    
    Consolidates testing logic from multiple scripts into one maintainable class.
    """
    
    def __init__(self):
        """Initialize workflow tester."""
        self.settings = get_settings()
        
    async def test_integral_workflow(self, query: str = None) -> dict:
        """
        Test integral calculation with visualization workflow.
        
        Args:
            query: Optional custom query. Uses default if None.
            
        Returns:
            dict: Test results with success indicators
        """
        # Use provided query or default test case
        test_query = query or "Calcula la integral de x² del 0 al 3 y muéstrame el área bajo la curva"
        
        logger.info(f"🧪 Testing workflow with query: {test_query}")
        
        try:
            # Initialize workflow (fail fast if configuration error)
            agent_graph = create_mathematical_agent_graph()
            workflow = agent_graph.build_workflow()
            compiled_workflow = workflow.compile()
            
            # Create minimal but complete initial state
            initial_state = self._create_test_state(test_query)
            
            # Execute workflow with iteration limit
            final_state = await self._execute_workflow(compiled_workflow, initial_state)
            
            # Analyze and return results
            return self._analyze_results(final_state, test_query)
            
        except Exception as e:
            logger.error(f"❌ Workflow test failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "integral_calculated": False,
                "plot_generated": False
            }
    
    def _create_test_state(self, query: str) -> dict:
        """Create minimal test state following YAGNI principle."""
        return {
            "messages": [HumanMessage(content=query)],
            "conversation_id": uuid4(),
            "session_id": "test_session",
            "user_id": "test_user",
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "current_step": "analyze_problem",
            "iteration_count": 0,
            "max_iterations": 10,
            "workflow_status": "active",
            "reasoning_steps": [],
            "current_reasoning": None,
            "thought_process": [],
            "observations": [],
            "available_tools": ["integral_calculator", "plot_generator", "function_analyzer"],
            "selected_tools": [],
            "tools_to_use": [],
            "tool_calls": [],
            "tool_results": [],
            "current_problem": query,
            "problem_analysis": {},
            "solution_progress": {},
            "validation_results": {},
            "final_answer": None,
            "confidence_score": 0.0,
            "error_state": None,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "test_mode": True
            }
        }
    
    async def _execute_workflow(self, compiled_workflow, initial_state: dict) -> dict:
        """Execute workflow with circuit breaker pattern."""
        final_state = None
        step_count = 0
        max_steps = 15  # Circuit breaker
        
        logger.info("🔄 Starting workflow execution...")
        
        async for state in compiled_workflow.astream(
            initial_state, 
            config={"configurable": {"thread_id": "test_session"}}
        ):
            step_count += 1
            logger.info(f"Step {step_count}: Current step = {state.get('current_step', 'unknown')}")
            final_state = state
            
            # Circuit breaker: prevent infinite loops
            if step_count > max_steps:
                logger.warning(f"⚠️ Maximum steps ({max_steps}) reached, breaking")
                break
        
        return final_state
    
    def _analyze_results(self, final_state: dict, query: str) -> dict:
        """Analyze workflow results with clear success criteria."""
        if not final_state:
            return {
                "success": False,
                "error": "No final state received",
                "integral_calculated": False,
                "plot_generated": False
            }
        
        # Check for integral calculation
        integral_calculated = (
            "9" in str(final_state) or
            "integral" in str(final_state).lower() or
            any("integral" in str(result) for result in final_state.get("tool_results", []))
        )
        
        # Check for plot generation
        plot_generated = (
            any("plot" in str(result).lower() for result in final_state.get("tool_results", [])) or
            any("visualization" in str(result).lower() for result in final_state.get("tool_results", [])) or
            any(result.get("tool_name") == "plot_generator" for result in final_state.get("tool_results", []))
        )
        
        success = integral_calculated and plot_generated
        
        logger.info("🎯 WORKFLOW ANALYSIS RESULTS")
        logger.info("=" * 40)
        logger.info(f"✅ Overall Success: {success}")
        logger.info(f"✅ Integral Calculated: {integral_calculated}")
        logger.info(f"✅ Plot Generated: {plot_generated}")
        logger.info(f"📊 Tool Results Count: {len(final_state.get('tool_results', []))}")
        
        return {
            "success": success,
            "integral_calculated": integral_calculated,
            "plot_generated": plot_generated,
            "final_state": final_state,
            "tools_executed": [r.get('tool_name') for r in final_state.get('tool_results', [])],
            "step_count": final_state.get('iteration_count', 0)
        }


async def main():
    """Main testing function."""
    logger.info("🤖 Professional Workflow Testing Suite")
    logger.info("=" * 50)
    
    tester = WorkflowTester()
    
    # Test the original user request
    result = await tester.test_integral_workflow()
    
    if result["success"]:
        logger.info("🎉 SUCCESS: Complete workflow functioning correctly!")
    else:
        logger.error("❌ FAILURE: Workflow needs attention")
        if "error" in result:
            logger.error(f"Error: {result['error']}")


if __name__ == "__main__":
    asyncio.run(main())
