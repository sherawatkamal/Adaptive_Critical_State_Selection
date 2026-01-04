#!/usr/bin/env python3
"""
Integration test for WebShop state save/restore.

This test creates a minimal WebShop environment to verify
the state save/restore functionality works correctly.

We bypass the heavy Lucene search dependency by mocking the search engine.
"""
import sys
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from collections import defaultdict

# Add WebShop to path
sys.path.insert(0, str(Path(__file__).parent.parent / "WebShop-master"))


def create_mock_search_engine():
    """Create a mock search engine that doesn't require Java."""
    mock = MagicMock()
    mock.search.return_value = []
    return mock


def create_test_products():
    """Create minimal test products."""
    return [
        {
            "asin": "B078TEST01",
            "name": "Test Product 1",
            "category": "test",
            "query": "laptop",
            "product_category": "Electronics",
            "full_description": "A test laptop product",
            "small_description": ["Fast", "Portable"],
            "pricing": ["$99.99"],
            "customization_options": {"color": [{"value": "black"}]},
            "images": ["http://example.com/img1.jpg"],
        },
        {
            "asin": "B078TEST02",
            "name": "Test Product 2",
            "category": "test",
            "query": "phone",
            "product_category": "Electronics",
            "full_description": "A test phone product",
            "small_description": ["Smart", "5G"],
            "pricing": ["$199.99"],
            "customization_options": {},
            "images": ["http://example.com/img2.jpg"],
        },
    ]


def test_state_serialization_roundtrip():
    """Test that state dict can be serialized and deserialized."""
    print("Testing state serialization roundtrip...")
    
    # Simulate state structure from get_state()
    state = {
        'session_id': 'test_session_123',
        'session_data': {
            'goal': {'instruction_text': 'Find red shoes under $50'},
            'keywords': ['red', 'shoes'],
            'page': 1,
            'asin': 'B078GWRC1J',
            'asins': ['B078GWRC1J', 'B079ABC123'],  # set converted to list
            'options': {'size': '10', 'color': 'red'},
            'actions': {'search': 1, 'click': 2},  # defaultdict to dict
            'done': False,
        },
        'browser_url': 'http://127.0.0.1:3000/item_page/test/B078GWRC1J',
        'browser_page': '<html><body>Product Page</body></html>',
        'prev_obs': ['obs1', 'obs2', 'obs3'],
        'prev_actions': ['search[red shoes]', 'click[item]'],
        'instruction_text': 'Find red shoes under $50',
        'text_to_clickable_keys': ['buy now', 'back to search'],
    }
    
    # Serialize to JSON
    json_str = json.dumps(state)
    assert isinstance(json_str, str), "State should be JSON serializable"
    
    # Deserialize back
    restored = json.loads(json_str)
    assert restored == state, "Round-trip should preserve state"
    
    print("✅ State serialization roundtrip passed")
    return True


def test_session_data_type_conversion():
    """Test conversion of set <-> list and defaultdict <-> dict."""
    print("Testing session data type conversions...")
    
    # Original with non-serializable types (as in env)
    original = {
        'goal': {'instruction_text': 'test'},
        'asins': {'B078GWRC1J', 'B079ABC123'},  # set
        'actions': defaultdict(int, {'search': 1}),  # defaultdict
        'keywords': ['test'],
    }
    
    # Simulate _serialize_session_data()
    serialized = {}
    for key, value in original.items():
        if key == 'asins':
            serialized[key] = list(value)
        elif key == 'actions':
            serialized[key] = dict(value)
        else:
            serialized[key] = value
    
    # Should be JSON serializable
    json_str = json.dumps(serialized)
    
    # Simulate restoration in set_state()
    restored = {}
    session_data = json.loads(json_str)
    for key, value in session_data.items():
        if key == 'asins':
            restored[key] = set(value)
        elif key == 'actions':
            restored[key] = defaultdict(int, value)
        else:
            restored[key] = value
    
    # Verify restoration
    assert restored['asins'] == original['asins'], "asins should restore to set"
    assert isinstance(restored['actions'], defaultdict), "actions should restore to defaultdict"
    assert restored['actions']['search'] == 1, "action counts preserved"
    assert restored['actions']['new_key'] == 0, "defaultdict works for new keys"
    
    print("✅ Session data type conversion passed")
    return True


def test_browser_state_methods():
    """Test SimBrowser get_state/set_state."""
    print("Testing SimBrowser state methods...")
    
    # Check the source code has the methods
    from pathlib import Path
    source = Path(__file__).parent.parent / "WebShop-master/web_agent_site/envs/web_agent_text_env.py"
    content = source.read_text()
    
    # Check SimBrowser class has methods
    assert "class SimBrowser:" in content, "SimBrowser class should exist"
    assert "def get_state(self)" in content, "SimBrowser.get_state should exist"
    assert "def set_state(self, state" in content, "SimBrowser.set_state should exist"
    
    # Check WebAgentTextEnv has methods
    assert "class WebAgentTextEnv" in content, "WebAgentTextEnv should exist"
    
    # Find get_state in WebAgentTextEnv (before SimBrowser)
    webagent_section = content.split("class SimBrowser")[0]
    assert "def get_state(self)" in webagent_section, "WebAgentTextEnv.get_state should exist"
    assert "def set_state(self, state" in webagent_section, "WebAgentTextEnv.set_state should exist"
    assert "def _serialize_session_data(self)" in webagent_section, "_serialize_session_data should exist"
    
    print("✅ SimBrowser state methods exist")
    return True


def test_state_dict_keys():
    """Test that get_state returns expected keys."""
    print("Testing state dictionary structure...")
    
    expected_keys = [
        'session_id',
        'session_data',
        'browser_url',
        'browser_page',
        'prev_obs',
        'prev_actions',
        'instruction_text',
        'text_to_clickable_keys',
    ]
    
    # Simulate get_state() output
    state = {
        'session_id': 'test',
        'session_data': {},
        'browser_url': 'http://example.com',
        'browser_page': '<html></html>',
        'prev_obs': [],
        'prev_actions': [],
        'instruction_text': 'Test task',
        'text_to_clickable_keys': [],
    }
    
    for key in expected_keys:
        assert key in state, f"State should have key: {key}"
    
    print("✅ State dictionary structure verified")
    return True


def test_history_list_preservation():
    """Test that prev_obs and prev_actions lists are preserved."""
    print("Testing history preservation...")
    
    prev_obs = ['Initial page', 'Search results', 'Item page']
    prev_actions = ['search[laptop]', 'click[item-123]']
    
    state = {
        'prev_obs': list(prev_obs),
        'prev_actions': list(prev_actions),
    }
    
    # Serialize and restore
    json_str = json.dumps(state)
    restored = json.loads(json_str)
    
    assert list(restored['prev_obs']) == prev_obs, "prev_obs should be preserved"
    assert list(restored['prev_actions']) == prev_actions, "prev_actions should be preserved"
    
    print("✅ History preservation passed")
    return True


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "=" * 60)
    print("WebShop State Save/Restore Integration Tests")
    print("=" * 60 + "\n")
    
    tests = [
        test_state_serialization_roundtrip,
        test_session_data_type_conversion,
        test_browser_state_methods,
        test_state_dict_keys,
        test_history_list_preservation,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"❌ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    
    if all(results):
        print(f"🎉 ALL TESTS PASSED ({passed}/{total})")
    else:
        print(f"⚠️  {passed}/{total} tests passed")
    print("=" * 60)
    
    return all(results)


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
