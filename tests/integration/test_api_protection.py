import os
import sys
import asyncio
import pytest
from unittest.mock import patch, MagicMock


class TestAPIProtection:
    """Test suite to validate API protection and mock infrastructure."""

    def test_no_real_api_keys_loaded(self):
        """Test: Verify that NO real API keys are loaded."""
        print("🔍 Testing: No real API keys loaded...")
        
        from tests.fixtures.mock_factory import MockFactory
        
        with MockFactory.mock_all_api_calls():
            from app.core.config import get_settings
            settings = get_settings()
            
            # Settings should be mocked
            assert settings.gemini_api_key == "mock_api_key_12345"
            assert settings.environment in ["testing", "mock"]
            print("✅ PASS: No real API keys detected")

    def test_llm_infrastructure_mocked(self):
        """Test: Verify that the LLM infrastructure is fully mocked."""
        print("🔍 Testing: LLM infrastructure is mocked...")
        
        from tests.fixtures.mock_factory import MockFactory
        
        with MockFactory.mock_all_api_calls():
            from app.agents.chains import create_chain_factory
            
            # Create chain factory with mocked settings
            chain_factory = create_chain_factory()
            
            # This should NOT make real API calls
            chain = chain_factory.create_analysis_chain()
            
            print("✅ PASS: Chain creation with mocked components")

    def test_mock_infrastructure_components(self):
        """Test: Verify that all mock components work."""
        print("🔍 Testing: Mock infrastructure components...")
        
        from tests.fixtures.mock_factory import MockFactory, TestValidationHelpers
        
        # Test centralized mock settings
        mock_settings = MockFactory.create_mock_settings()
        assert mock_settings.google_api_key == "mock_api_key_12345"
        print(f"✅ Mock settings: API key = {mock_settings.google_api_key}")
        
        # Test mock tool registry
        mock_registry = MockFactory.create_mock_tool_registry()
        tools = mock_registry.list_tools()
        assert len(tools) > 0
        print(f"✅ Mock tool registry: {len(tools)} tools available")
        
        # Test mock chain factory
        mock_factory = MockFactory.create_mock_chain_factory()
        assert mock_factory is not None
        print("✅ Mock chain factory created successfully")

    @pytest.mark.asyncio
    async def test_complete_workflow_with_mocks(self):
        """Test: Verify that the complete workflow works with mocks."""
        print("🔍 Testing: Complete workflow with mocks...")
        
        from tests.fixtures.mock_factory import MockFactory, TestValidationHelpers
        
        with MockFactory.mock_all_api_calls() as mocks:
            from app.agents.nodes import analyze_problem_node
            from app.agents.state import get_empty_math_agent_state
            from app.agents.state import WorkflowSteps
            
            # Mocked initial state
            state = get_empty_math_agent_state()
            state['user_input'] = "What is the integral of x^2?"
            state['current_step'] = WorkflowSteps.ANALYSIS  # Fix: Use correct step name
            
            # Execute node - should use mocks, NOT real API
            result = await analyze_problem_node(state)
            
            # Validate that it executed correctly with mocks
            assert isinstance(result, dict)
            
            # Validate that NO real API calls were made
            TestValidationHelpers.assert_no_real_api_calls(mocks['llm_class'])
            
            print("✅ PASS: Complete workflow execution with mocked components")

    def test_complete_workflow_protection(self):
        """Test: Complete validation of workflow protection."""
        print("🔍 Testing: Complete workflow protection...")
        
        from tests.fixtures.mock_factory import MockFactory
        
        with MockFactory.mock_all_api_calls():
            # Import main components
            from app.core.config import get_settings
            from app.agents.graph import create_agent_graph
            
            # Verify mocked configuration
            settings = get_settings()
            assert settings.gemini_api_key == "mock_api_key_12345"
            
            # Create agent graph (should use mock components)
            graph = create_agent_graph()
            assert graph is not None
            
            print("✅ PASS: Complete workflow with mocked components")


def main():
    """Run complete API protection validation as a standalone function."""
    print("🚀 STARTING API PROTECTION VALIDATION")
    print("=" * 50)
    
    try:
        test_instance = TestAPIProtection()
        
        # Test 1: No real API keys
        test_instance.test_no_real_api_keys_loaded()
        
        # Test 2: LLM calls mocked
        test_instance.test_llm_calls_are_mocked()
        
        # Test 3: Nodes with mocks
        asyncio.run(test_instance.test_nodes_with_mocks())
        
        # Test 4: Complete workflow
        test_instance.test_complete_workflow_protection()
        
        print("=" * 50)
        print("🎉 SUCCESS: API PROTECTION IS WORKING!")
        print("✅ No real API keys are being used")
        print("✅ All components are properly mocked")
        print("✅ System is safe from API consumption")
        
        return 0
        
    except Exception as e:
        print("=" * 50)
        print(f"❌ FAILURE: API Protection validation failed!")
        print(f"Error: {e}")
        print("🚨 CRITICAL: Review mock infrastructure immediately!")
        
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
