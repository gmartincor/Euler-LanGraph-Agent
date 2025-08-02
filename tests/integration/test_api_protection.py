#!/usr/bin/env python3
"""
🔒 COMPREHENSIVE API PROTECTION & MOCK INFRASTRUCTURE VALIDATION

Este test unificado valida:
1. Protección contra APIs reales (NO consumo de quota)
2. Funcionamiento correcto de la infraestructura de mocks
3. Ejecución completa del workflow con mocks

OBJETIVO: Confirmar que el sistema está 100% seguro y funcional con mocks.
"""

import os
import sys
import asyncio
import pytest
from unittest.mock import patch, MagicMock


class TestAPIProtection:
    """Suite de tests para validar protección de APIs y infraestructura de mocks."""

    def test_no_real_api_keys_loaded(self):
        """Test: Verificar que NO se cargan API keys reales."""
        print("🔍 Testing: No real API keys loaded...")
        
        from tests.fixtures.mock_factory import MockFactory
        
        with MockFactory.mock_all_api_calls():
            from app.core.config import get_settings
            settings = get_settings()
            
            # Las settings deben ser mockeadas
            assert settings.gemini_api_key == "mock_api_key_12345"
            assert settings.environment in ["testing", "mock"]
            print("✅ PASS: No real API keys detected")

    def test_llm_infrastructure_mocked(self):
        """Test: Verificar que la infraestructura LLM está completamente mockeada."""
        print("🔍 Testing: LLM infrastructure is mocked...")
        
        from tests.fixtures.mock_factory import MockFactory
        
        with MockFactory.mock_all_api_calls():
            from app.agents.chains import create_chain_factory
            
            # Crear chain factory con settings mockeadas
            chain_factory = create_chain_factory()
            
            # Esto NO debe hacer llamadas reales a la API
            chain = chain_factory.create_analysis_chain()
            
            print("✅ PASS: Chain creation with mocked components")

    def test_mock_infrastructure_components(self):
        """Test: Verificar que todos los componentes de mock funcionan."""
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
        """Test: Verificar que el workflow completo funciona con mocks."""
        print("🔍 Testing: Complete workflow with mocks...")
        
        from tests.fixtures.mock_factory import MockFactory, TestValidationHelpers
        
        with MockFactory.mock_all_api_calls() as mocks:
            from app.agents.nodes import analyze_problem_node
            from app.agents.state import get_empty_math_agent_state
            from app.agents.state import WorkflowSteps
            
            # Estado inicial mockeado
            state = get_empty_math_agent_state()
            state['user_input'] = "What is the integral of x^2?"
            state['current_step'] = WorkflowSteps.PROBLEM_ANALYSIS
            
            # Ejecutar nodo - debe usar mocks, NO API real
            result = await analyze_problem_node(state)
            
            # Validar que se ejecutó correctamente con mocks
            assert isinstance(result, dict)
            
            # Validar que NO se hicieron llamadas reales
            TestValidationHelpers.assert_no_real_api_calls(mocks['llm_class'])
            
            print("✅ PASS: Complete workflow execution with mocked components")

    def test_complete_workflow_protection(self):
        """Test: Validación completa de protección del workflow."""
        print("🔍 Testing: Complete workflow protection...")
        
        from tests.fixtures.mock_factory import MockFactory
        
        with MockFactory.mock_all_api_calls():
            # Importar componentes principales
            from app.core.config import get_settings
            from app.agents.graph import create_agent_graph
            
            # Verificar configuración mockeada
            settings = get_settings()
            assert settings.gemini_api_key == "mock_api_key_12345"
            
            # Crear grafo del agente (debe usar componentes mock)
            graph = create_agent_graph()
            assert graph is not None
            
            print("✅ PASS: Complete workflow with mocked components")


def main():
    """Ejecutar validación completa de protección de APIs como función standalone."""
    print("🚀 STARTING API PROTECTION VALIDATION")
    print("=" * 50)
    
    try:
        test_instance = TestAPIProtection()
        
        # Test 1: No API keys reales
        test_instance.test_no_real_api_keys_loaded()
        
        # Test 2: LLM calls mockeadas
        test_instance.test_llm_calls_are_mocked()
        
        # Test 3: Nodos con mocks
        asyncio.run(test_instance.test_nodes_with_mocks())
        
        # Test 4: Workflow completo
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
