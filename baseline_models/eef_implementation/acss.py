"""
ACSS - Adaptive Critical State Selection using Shannon Entropy
FIXED VERSION: Properly implements entropy-based selection

Baseline: Uniform skip-length from EEF paper
Novel: Shannon Entropy-based state selection
"""
import numpy as np
from typing import List, Dict, Tuple
import torch
import torch.nn.functional as F


class ACSS:
    """
    Critical State Selection for EEF
    - Baseline: Uniform skip-length (from paper, Algorithm 1)
    - Entropy: Shannon Entropy-based selection (novel contribution)
    """
    
    def __init__(self, strategy='baseline', M=5, agent=None, verbose=False):
        """
        Args:
            strategy: 'baseline' (skip-length) or 'entropy' (Shannon Entropy)
            M: Number of states to select per trajectory
            agent: Trained agent for entropy computation (optional)
            verbose: Print debug information
        """
        self.strategy = strategy
        self.M = M
        self.agent = agent
        self.verbose = verbose 
    
    def select_critical_states(self, trajectory: Dict) -> List[int]:
        """
        Select critical states from a failed trajectory
        
        Args:
            trajectory: Failed trajectory dict with 'steps' list
            
        Returns:
            List of step indices to simulate from
        """
        steps = trajectory.get('steps', [])
        traj_len = len(steps)
        
        if traj_len <= 1:
            return []
        
        if self.strategy == 'baseline':
            return self._baseline_skip_length(traj_len)
        elif self.strategy == 'entropy':
            return self._entropy_based_selection(trajectory)
        else:
            return self._baseline_skip_length(traj_len)
    
    def _baseline_skip_length(self, traj_len: int) -> List[int]:
        """
        Baseline: Uniform skip-length selection from EEF paper (Algorithm 1)
        
        Formula: l = ⌊|τe|/(M + 1)⌋
        Selects states: [s_l, s_2l, s_3l, ..., s_M×l]
        
        Args:
            traj_len: Length of trajectory
            
        Returns:
            List of uniformly spaced state indices
        """
        # Skip length formula from paper
        l = int(np.floor(traj_len / (self.M + 1)))
        
        if l == 0:
            # Trajectory too short, select middle state
            return [traj_len // 2] if traj_len > 1 else []
        
        # Select states at intervals of l
        selected_states = []
        for m in range(1, self.M + 1):
            state_idx = m * l
            if state_idx < traj_len:
                selected_states.append(state_idx)
        
        return selected_states
    
    def _entropy_based_selection(self, trajectory: Dict) -> List[int]:
        """
        ACSS: Multi-Signal State Selection with proper entropy weighting
        
        Key insight: States with HIGH entropy (uncertainty) are decision points
        where the agent was unsure - these are critical for recovery.
        """
        steps = trajectory.get('steps', [])
        traj_len = len(steps)
        
        if traj_len <= 1:
            return []
        
        # Match baseline's state count
        baseline_states = self._baseline_skip_length(traj_len)
        max_states = len(baseline_states)
        
        if max_states == 0:
            return []
        
        # Compute importance for each state
        state_scores = []
        
        for i, step in enumerate(steps[:-1]):  # Exclude last step (terminal)
            obs = step.get('observation', '')
            valid_actions = step.get('valid_actions', [])
            
            if not valid_actions:
                continue
            
            try:
                # Compute multi-signal importance
                signals = self._compute_importance_signals(
                    obs=obs,
                    valid_actions=valid_actions,
                    goal=trajectory.get('goal', ''),
                    state_idx=i,
                    traj_len=traj_len,
                    step=step
                )
                
                # FIX: Weighted combination with entropy as PRIMARY signal
                combined_score = self._combine_signals(signals)
                
                state_scores.append((i, combined_score, signals))
                
            except Exception as e:
                if self.verbose:
                    print(f"  Warning: Error computing signals for step {i}: {e}")
                continue
        
        if not state_scores:
            return self._baseline_skip_length(traj_len)
        
        # Select top-M states by combined score
        state_scores.sort(key=lambda x: x[1], reverse=True)
        selected_states = [idx for idx, score, signals in state_scores[:max_states]]
        
        if self.verbose:
            print(f"\n  ACSS Entropy Selection:")
            print(f"    Trajectory length: {traj_len}")
            print(f"    Selecting top {max_states} states:")
            for idx, score, signals in state_scores[:max_states]:
                sig_str = ", ".join([f"{k}={v:.2f}" for k, v in signals.items()])
                print(f"      Step {idx}: score={score:.3f} ({sig_str})")
        
        return sorted(selected_states)
    
    def _compute_importance_signals(self, obs: str, valid_actions: List[str],
                                    goal: str, state_idx: int, traj_len: int,
                                    step: Dict) -> Dict[str, float]:
        """
        Compute multiple importance signals for a state
        
        Returns dict of signal_name -> score (0 to 1)
        """
        signals = {}
        
        # 1. ENTROPY SIGNAL (PRIMARY) - Decision uncertainty
        #    High entropy = agent was uncertain = critical decision point
        signals['entropy'] = self._compute_entropy_signal(valid_actions, obs, goal)
        
        # 2. ACTION SPACE SIGNAL - Goldilocks principle
        #    Medium complexity is best (not too simple, not too complex)
        signals['action_space'] = self._compute_action_space_signal(valid_actions)
        
        # 3. POSITION SIGNAL - Slight preference for middle
        #    Avoid very early (not enough context) and very late (too committed)
        position = state_idx / max(traj_len - 1, 1)
        signals['position'] = self._compute_position_signal(position)
        
        # 4. OBSERVATION COMPLEXITY - Information richness
        signals['obs_complexity'] = self._compute_observation_signal(obs)
        
        # 5. NAVIGATION SIGNAL - Exploration indicators
        signals['navigation'] = self._compute_navigation_signal(valid_actions)
        
        return signals
    
    def _combine_signals(self, signals: Dict[str, float]) -> float:
        """
        Combine signals with proper weighting
        
        FIX: Entropy should be primary signal, position secondary
        """
        # Weight configuration - entropy is primary
        weights = {
            'entropy': 0.40,        # PRIMARY: Decision uncertainty
            'action_space': 0.25,   # Action complexity (goldilocks)
            'position': 0.15,       # Slight middle preference
            'obs_complexity': 0.10, # Information richness
            'navigation': 0.10,     # Exploration indicators
        }
        
        combined = 0.0
        for signal_name, weight in weights.items():
            if signal_name in signals:
                combined += weight * signals[signal_name]
        
        return combined
    
    def _compute_entropy_signal(self, valid_actions: List[str], 
                                obs: str = '', goal: str = '') -> float:
        """
        Shannon Entropy signal - measures decision uncertainty
        
        If agent is available: use actual policy distribution
        Otherwise: proxy using action space size (uniform assumption)
        """
        if not valid_actions:
            return 0.0
        
        n_actions = len(valid_actions)
        
        # If we have an agent, compute actual policy entropy
        if self.agent is not None:
            try:
                entropy = self._compute_agent_policy_entropy(obs, valid_actions, goal)
                return entropy
            except Exception:
                pass  # Fall back to proxy
        
        # Proxy: Assume uniform distribution, entropy = log(n)
        # Normalize by log(100) assuming max 100 actions
        max_entropy = np.log(100)
        entropy = np.log(n_actions)
        normalized = min(entropy / max_entropy, 1.0)
        
        return normalized
    
    def _compute_agent_policy_entropy(self, obs: str, valid_actions: List[str], 
                                      goal: str) -> float:
        """
        Compute actual policy entropy from agent's action distribution
        
        Higher entropy = more uncertainty = more critical decision point
        """
        try:
            with torch.no_grad():
                # This depends on agent interface - adapt as needed
                state = self.agent.build_state(obs, {'goal': goal})
                logits = self.agent.get_action_logits(state, valid_actions)
                
                # Convert to probabilities
                probs = F.softmax(logits, dim=-1)
                
                # Compute Shannon entropy: -sum(p * log(p))
                log_probs = torch.log(probs + 1e-10)
                entropy = -(probs * log_probs).sum().item()
                
                # Normalize by maximum entropy (uniform distribution)
                max_entropy = np.log(len(valid_actions))
                if max_entropy > 0:
                    normalized = entropy / max_entropy
                else:
                    normalized = 0.0
                
                return min(normalized, 1.0)
                
        except Exception as e:
            if self.verbose:
                print(f"    Warning: Could not compute agent entropy: {e}")
            # Fall back to action space proxy
            return min(np.log(len(valid_actions)) / np.log(100), 1.0)
    
    def _compute_action_space_signal(self, valid_actions: List[str]) -> float:
        """
        Action space complexity - GOLDILOCKS APPROACH
        
        Prefer MEDIUM complexity:
        - Too simple (few actions) = not interesting
        - Too complex (many actions) = too hard to recover
        - Medium = just right for learning
        """
        if not valid_actions:
            return 0.0
        
        n_actions = len(valid_actions)
        normalized = min(n_actions / 100.0, 1.0)
        
        # Goldilocks scoring: Peak at medium (0.3-0.5)
        if normalized < 0.1:
            # Too simple - low score
            return normalized * 3.0  # Max 0.3
        elif normalized > 0.7:
            # Too complex - penalize
            return max(0.0, (1.0 - normalized) * 2.0)  # Decreases from 0.6
        else:
            # Sweet spot (0.1 to 0.7) - reward
            # Peak at 0.3, score = 1.0
            distance_from_optimal = abs(normalized - 0.3)
            return max(0.0, 1.0 - distance_from_optimal * 1.5)
    
    def _compute_position_signal(self, position: float) -> float:
        """
        Position signal - slight preference for middle states
        
        Very early states: not enough context for decision
        Very late states: already too committed to wrong path
        Middle states: best for recovery
        """
        # Bell curve centered at 0.4 (slightly favor earlier middle)
        # Not too aggressive - we don't want to just replicate baseline
        center = 0.4
        width = 0.3
        
        return np.exp(-0.5 * ((position - center) / width) ** 2)
    
    def _compute_observation_signal(self, obs: str) -> float:
        """
        Observation complexity - information richness
        
        States with more information might be more interesting
        """
        if not obs:
            return 0.0
        
        word_count = len(obs.split())
        # Normalize assuming max 500 words
        return min(word_count / 500.0, 1.0)
    
    def _compute_navigation_signal(self, valid_actions: List[str]) -> float:
        """
        Navigation action ratio - exploration indicators
        
        States with navigation options (search, next, prev) indicate
        the agent can explore alternatives
        """
        if not valid_actions:
            return 0.0
        
        nav_keywords = ['click[', 'search[', 'next', 'prev', 'back']
        nav_count = sum(1 for a in valid_actions 
                       if any(kw in a.lower() for kw in nav_keywords))
        
        return nav_count / len(valid_actions)
    
    def batch_select(self, trajectories: List[Dict]) -> Dict[int, List[int]]:
        """
        Select critical states from multiple trajectories
        
        Args:
            trajectories: List of failed trajectory dicts
            
        Returns:
            Dict mapping task_id -> list of critical state indices
        """
        selections = {}
        
        for traj in trajectories:
            task_id = traj.get('task_id')
            critical_states = self.select_critical_states(traj)
            
            if critical_states:
                selections[task_id] = critical_states
        
        return selections


def visualize_critical_states(trajectory: Dict, selected_indices: List[int], 
                             strategy: str = 'baseline'):
    """
    Visualize which states were selected as critical
    """
    steps = trajectory.get('steps', [])
    traj_len = len(steps)
    
    print(f"\n  {'─'*50}")
    print(f"  Critical States Visualization")
    print(f"  Strategy: {strategy}")
    print(f"  Trajectory length: {traj_len}")
    print(f"  Selected {len(selected_indices)} states: {selected_indices}")
    
    # ASCII visualization
    vis = ['·'] * traj_len
    for idx in selected_indices:
        if idx < traj_len:
            vis[idx] = '█'
    
    # Print in rows of 50
    print(f"\n  Timeline (█=selected, ·=not selected):")
    for start in range(0, traj_len, 50):
        end = min(start + 50, traj_len)
        print(f"    {start:3d}: {''.join(vis[start:end])}")
    
    # Show selected actions
    if steps:
        print(f"\n  Selected actions:")
        for idx in selected_indices[:5]:  # Show first 5
            if idx < len(steps):
                action = steps[idx].get('action_taken', '')
                print(f"    Step {idx}: {action[:60]}...")
        if len(selected_indices) > 5:
            print(f"    ... and {len(selected_indices) - 5} more")
    
    print(f"  {'─'*50}\n")


def compare_strategies(trajectory: Dict, agent=None, M: int = 5):
    """
    Compare baseline and entropy strategies on a single trajectory
    """
    print(f"\n{'='*70}")
    print(f"COMPARING ACSS STRATEGIES")
    print(f"{'='*70}")
    print(f"Task ID: {trajectory.get('task_id')}")
    print(f"Trajectory length: {len(trajectory.get('steps', []))} steps")
    print(f"M (states to select): {M}")
    
    # Baseline
    acss_baseline = ACSS(strategy='baseline', M=M, verbose=True)
    baseline_states = acss_baseline.select_critical_states(trajectory)
    
    # Entropy
    acss_entropy = ACSS(strategy='entropy', M=M, agent=agent, verbose=True)
    entropy_states = acss_entropy.select_critical_states(trajectory)
    
    # Visualize both
    visualize_critical_states(trajectory, baseline_states, 'Baseline (Skip-Length)')
    visualize_critical_states(trajectory, entropy_states, 'ACSS (Entropy)')
    
    # Comparison metrics
    overlap = len(set(baseline_states) & set(entropy_states))
    unique_baseline = len(set(baseline_states) - set(entropy_states))
    unique_entropy = len(set(entropy_states) - set(baseline_states))
    
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY:")
    print(f"  Baseline selected: {baseline_states}")
    print(f"  ACSS selected:     {entropy_states}")
    print(f"  Overlap:           {overlap} states ({100*overlap/max(len(baseline_states),1):.0f}%)")
    print(f"  Unique to baseline: {unique_baseline}")
    print(f"  Unique to ACSS:     {unique_entropy}")
    print(f"{'='*70}\n")
    
    return baseline_states, entropy_states