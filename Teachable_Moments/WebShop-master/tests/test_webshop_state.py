"""
Tests for WebShop environment state save/restore functionality.

These tests verify that:
1. get_state() returns a serializable dictionary
2. set_state() restores environment to exact previous state
3. Observation after set_state() matches observation when state was saved
4. Session data (goal, options, etc.) is correctly restored
5. Browser URL and page source are restored
6. History (prev_obs, prev_actions) is restored
7. Counterfactual rollouts produce different results for different actions
"""

import pytest
import json
from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv


class TestStateSaveRestore:
    """Test suite for state save/restore functionality."""
    
    @pytest.fixture
    def env(self):
        """Create a WebAgentTextEnv instance for testing."""
        env = WebAgentTextEnv(
            observation_mode='text',
            num_products=100,  # Use smaller dataset for faster tests
        )
        yield env
        env.close()
    
    def test_get_state_returns_dict(self, env):
        """Test that get_state() returns a dictionary."""
        env.reset(session=0)
        state = env.get_state()
        assert isinstance(state, dict)
    
    def test_get_state_serializable(self, env):
        """Test that get_state() returns a JSON-serializable dictionary."""
        env.reset(session=0)
        state = env.get_state()
        # Should not raise an exception
        json_str = json.dumps(state)
        assert isinstance(json_str, str)
    
    def test_get_state_contains_required_keys(self, env):
        """Test that get_state() contains all required keys."""
        env.reset(session=0)
        state = env.get_state()
        
        required_keys = [
            'session_id',
            'session_data',
            'browser_url',
            'browser_page',
            'prev_obs',
            'prev_actions',
            'instruction_text',
            'text_to_clickable_keys',
        ]
        
        for key in required_keys:
            assert key in state, f"Missing key: {key}"
    
    def test_state_save_restore(self, env):
        """Test that state can be saved and restored."""
        # Reset and take some actions
        obs1, _ = env.reset(session=0)
        env.step("search[red shoes]")
        obs2, _, _, _ = env.step("click[item - b078gwrc1j]")
        
        # Save state
        saved_state = env.get_state()
        
        # Take more actions
        env.step("click[buy now]")
        
        # Restore state
        restored_obs = env.set_state(saved_state)
        
        # Verify restoration
        assert restored_obs == obs2, "Restored observation should match saved observation"
        assert env.browser.current_url == saved_state['browser_url'], "Browser URL should match"
    
    def test_session_data_restoration(self, env):
        """Test that session data is correctly restored."""
        env.reset(session=0)
        env.step("search[laptop]")
        
        # Save state
        saved_state = env.get_state()
        original_session_data = saved_state['session_data']
        
        # Mutate state
        env.step("click[item - b078gwrc1j]")
        
        # Restore
        env.set_state(saved_state)
        
        # Check session data
        current_session = env.server.user_sessions.get(env.session, {})
        assert 'goal' in current_session, "Goal should be restored"
        assert current_session['keywords'] == original_session_data['keywords'], "Keywords should match"
    
    def test_history_restoration(self, env):
        """Test that prev_obs and prev_actions are restored."""
        env.reset(session=0)
        env.step("search[phone]")
        
        # Save state
        saved_state = env.get_state()
        saved_prev_obs_len = len(saved_state['prev_obs'])
        saved_prev_actions_len = len(saved_state['prev_actions'])
        
        # Take more actions
        env.step("click[item - b078gwrc1j]")
        env.step("click[buy now]")
        
        # Restore
        env.set_state(saved_state)
        
        # Verify history lengths match
        assert len(env.prev_obs) == saved_prev_obs_len, "prev_obs length should match"
        assert len(env.prev_actions) == saved_prev_actions_len, "prev_actions length should match"
    
    def test_counterfactual_rollout(self, env):
        """Test counterfactual simulation from saved state."""
        # Get to a decision point (search results page)
        env.reset(session=0)
        env.step("search[laptop]")
        
        # Get available actions to find valid clickable items
        available = env.get_available_actions()
        clickables = available.get('clickables', [])
        
        # Find two different item buttons if available
        items = [c for c in clickables if c.startswith('item')]
        
        if len(items) >= 2:
            # Save state at decision point
            decision_state = env.get_state()
            
            # Path A - click first item
            env.step(f"click[{items[0]}]")
            result_a = env.observation
            
            # Restore and take Path B - click second item
            env.set_state(decision_state)
            env.step(f"click[{items[1]}]")
            result_b = env.observation
            
            # Results should differ since we clicked different items
            assert result_a != result_b, "Different actions should produce different observations"
    
    def test_instruction_text_restoration(self, env):
        """Test that instruction_text is correctly restored."""
        env.reset(session=0)
        original_instruction = env.instruction_text
        
        # Save state
        saved_state = env.get_state()
        
        # Reset to a different session (changes instruction)
        env.reset(session=1)
        
        # Restore original state
        env.set_state(saved_state)
        
        # Verify instruction text matches
        assert env.instruction_text == original_instruction, "Instruction text should be restored"


class TestSimBrowserState:
    """Test suite for SimBrowser state methods."""
    
    @pytest.fixture
    def env(self):
        """Create a WebAgentTextEnv instance for testing."""
        env = WebAgentTextEnv(
            observation_mode='text',
            num_products=100,
        )
        yield env
        env.close()
    
    def test_browser_get_state(self, env):
        """Test that browser.get_state() returns expected keys."""
        env.reset(session=0)
        browser_state = env.browser.get_state()
        
        assert 'current_url' in browser_state
        assert 'page_source' in browser_state
        assert 'session_id' in browser_state
    
    def test_browser_set_state(self, env):
        """Test that browser.set_state() restores browser state."""
        env.reset(session=0)
        
        # Save browser state
        original_state = env.browser.get_state()
        
        # Take an action
        env.step("search[laptop]")
        
        # Restore browser state
        env.browser.set_state(original_state)
        
        assert env.browser.current_url == original_state['current_url']
        assert env.browser.page_source == original_state['page_source']
        assert env.browser.session_id == original_state['session_id']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
