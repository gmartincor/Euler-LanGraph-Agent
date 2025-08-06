#!/usr/bin/env python3
"""
Test script to verify that BigTool integration works correctly.
This script tests both mathematical calculations and visualizations.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the app directory to the Python path
app_dir = Path(__file__).parent / "app"
sys.path.insert(0, str(app_dir))

from app.core.agent_controller import AgentController
from app.core.config import get_settings
from app.tools.initialization import initialize_tools
from app.database.connection import DatabaseManager


async def test_agent_integration():
    """Test the agent's ability to calculate integrals and generate plots."""
    
    print("🧪 Testing ReAct Agent Integration")
    print("=" * 50)
    
    try:
        # Initialize settings
        settings = get_settings()
        print(f"✅ Settings loaded: {settings.app_name} v{settings.app_version}")
        
        # Initialize database
        db_manager = DatabaseManager()
        db_manager.initialize()
        print("✅ Database initialized")
        
        # Initialize tools
        tool_registry = initialize_tools()
        print(f"✅ Tools initialized: {len(tool_registry.list_tools())} tools")
        
        # Initialize agent directly
        from app.agents.interface import create_mathematical_agent
        agent = create_mathematical_agent()
        print("✅ Mathematical agent created")
        
        # Test case: Calculate integral and show visualization
        test_message = "Calculate the integral of x² from 0 to 3 and show me the area under the curve"
        print(f"\n🔍 Testing: {test_message}")
        
        # Process the message
        result = await agent.solve(test_message)
        
        print(f"\n📊 Result:")
        print(f"Answer: {result.get('answer', 'No response')}")
        print(f"Status: {result.get('status', 'Unknown')}")
        print(f"Final answer: {result.get('final_answer', 'None')}")
        
        # Check if visualization was generated
        if 'visualizations' in result and result['visualizations']:
            print("✅ Visualization generated successfully!")
            print(f"Number of visualizations: {len(result['visualizations'])}")
        else:
            print("❌ No visualization generated")
            
        # Check if calculation was performed
        if 'final_answer' in result and result['final_answer']:
            print("✅ Mathematical calculation completed!")
        else:
            print("❌ No mathematical calculation found")
            
        return result
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


async def main():
    """Main test function."""
    # Set environment variables if not set
    if not os.getenv('DATABASE_URL'):
        os.environ['DATABASE_URL'] = 'postgresql://agent_user:agent_pass@localhost:5432/react_agent_db'
    
    if not os.getenv('GOOGLE_API_KEY'):
        print("❌ GOOGLE_API_KEY environment variable not set")
        return
    
    result = await test_agent_integration()
    
    if result:
        print("\n🎉 Integration test completed successfully!")
    else:
        print("\n💥 Integration test failed!")


if __name__ == "__main__":
    asyncio.run(main())
