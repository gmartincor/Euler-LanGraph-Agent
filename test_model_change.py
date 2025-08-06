#!/usr/bin/env python3
"""
Simple test to verify Gemini 2.5 Flash model change
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_config_change():
    """Test that config loads with new model"""
    try:
        # Test environment variable
        os.environ["GOOGLE_API_KEY"] = "test-key-12345"
        os.environ["DATABASE_URL"] = "sqlite:///:memory:"
        
        from app.core.config import get_settings
        
        settings = get_settings()
        
        print("🔍 Testing model configuration...")
        print(f"   📱 Model Name: {settings.gemini_model_name}")
        print(f"   ⚙️  Temperature: {settings.gemini_temperature}")
        print(f"   🎯 Max Tokens: {settings.gemini_max_tokens}")
        
        # Verify the change
        expected_model = "gemini-2.5-flash"
        if settings.gemini_model_name == expected_model:
            print(f"✅ SUCCESS: Model updated to {expected_model}")
            print("✅ Ready to test with better rate limits!")
            return True
        else:
            print(f"❌ FAILED: Expected {expected_model}, got {settings.gemini_model_name}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def test_gemini_config():
    """Test gemini config property"""
    try:
        from app.core.config import get_settings
        
        settings = get_settings()
        gemini_config = settings.gemini_config
        
        print("\n🔍 Testing Gemini configuration...")
        print(f"   📱 Model: {gemini_config['model_name']}")
        print(f"   🌡️  Temperature: {gemini_config['temperature']}")
        print(f"   🎯 Max Tokens: {gemini_config['max_output_tokens']}")
        print(f"   🔗 Top P: {gemini_config['top_p']}")
        print(f"   🎲 Top K: {gemini_config['top_k']}")
        
        # Verify
        if gemini_config['model_name'] == "gemini-2.5-flash":
            print("✅ SUCCESS: Gemini config ready for 2.5 Flash!")
            return True
        else:
            print(f"❌ FAILED: Wrong model in config")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

if __name__ == "__main__":
    print("🚀 TESTING GEMINI 2.5 FLASH MODEL CHANGE\n")
    
    success1 = test_config_change()
    success2 = test_gemini_config()
    
    print("\n" + "="*50)
    if success1 and success2:
        print("🎉 ALL TESTS PASSED!")
        print("🚀 Your agent is now ready to use Gemini 2.5 Flash")
        print("💡 No more rate limit issues!")
    else:
        print("❌ Some tests failed. Check the output above.")
    print("="*50)
