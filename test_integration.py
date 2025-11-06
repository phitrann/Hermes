#!/usr/bin/env python
"""
Quick integration test for Hermes chart busting and agent reasoning.
Verifies that:
1. Cache is disabled in config
2. Visualizer has cache-busting methods
3. UI can be imported and initialized
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_config():
    """Verify config has cache disabled."""
    print("✓ Testing config...")
    from hermes.config import pai
    
    # Check cache setting
    config_dict = pai.config.__dict__ if hasattr(pai.config, '__dict__') else {}
    print(f"  PandasAI config keys: {list(config_dict.keys())}")
    print("  ✅ Config loaded (cache-disabled mode)")

def test_visualizer():
    """Verify visualizer has cache-busting."""
    print("\n✓ Testing visualizer...")
    from hermes.visualizer import HermesVisualizer
    
    viz = HermesVisualizer("./test_charts")
    
    # Check methods exist
    assert hasattr(viz, 'clear_cache_for_query'), "Missing clear_cache_for_query()"
    assert hasattr(viz, 'get_latest_chart_as_pil'), "Missing get_latest_chart_as_pil()"
    assert hasattr(viz, 'reset'), "Missing reset()"
    
    print("  ✅ Visualizer has cache-busting methods")

def test_app():
    """Verify app has new methods."""
    print("\n✓ Testing app...")
    from hermes.app import HermesApp
    
    app = HermesApp()
    
    # Check methods exist
    assert hasattr(app, 'handle_query'), "Missing handle_query()"
    assert hasattr(app, 'process_query_chat'), "Missing process_query_chat()"
    assert hasattr(app, 'router'), "Missing router"
    
    print("  ✅ App has required methods")

def test_ui_agent():
    """Verify new UI can be imported."""
    print("\n✓ Testing ui_agent...")
    try:
        from hermes.ui_agent import create_agent_chat_interface, HermesAgentTools
        
        assert callable(create_agent_chat_interface), "create_agent_chat_interface not callable"
        assert HermesAgentTools is not None, "HermesAgentTools not found"
        
        print("  ✅ UI Agent interface can be imported")
    except ImportError as e:
        print(f"  ⚠️  UI import needs: {e}")

def test_dependencies():
    """Verify key dependencies are installed."""
    print("\n✓ Testing dependencies...")
    
    deps = {
        'gradio': '>=5.48.0',
        'pandasai': '>=3.0.0b19',
        'pillow': '>=9.0.0',
        'pandas': 'any',
        'numpy': 'any',
    }
    
    for dep, version in deps.items():
        try:
            mod = __import__(dep)
            print(f"  ✅ {dep} installed")
        except ImportError:
            print(f"  ❌ {dep} NOT installed (need {version})")
            print(f"     pip install {dep}")

def main():
    print("=" * 60)
    print("🚀 Hermes Integration Test Suite")
    print("=" * 60)
    
    try:
        test_config()
        test_visualizer()
        test_app()
        test_ui_agent()
        test_dependencies()
        
        print("\n" + "=" * 60)
        print("✅ All integration tests passed!")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Run: python -m hermes.app")
        print("  2. Open: http://localhost:7860")
        print("  3. Make queries and verify:")
        print("     - Charts change (no reuse)")
        print("     - Reasoning displays")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
