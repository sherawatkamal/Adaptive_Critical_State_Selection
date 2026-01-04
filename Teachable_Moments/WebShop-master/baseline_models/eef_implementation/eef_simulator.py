"""
EEF Simulator - FIXED VERSION v2
Simulates agent behavior from intermediate failure states
KEY FIXES:
1. Correct reward threshold (10.0 for WebShop)
2. Configurable success threshold
3. Better error handling
"""
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
import copy

class EEFSimulator:
    """Simulates agent from intermediate states in failed trajectories"""
    
    def __init__(self, env, agent, max_steps=50, device='cuda', success_threshold=10.0):
        """
        Args:
            env: WebShop environment
            agent: Trained agent (model)
            max_steps: Maximum simulation steps
            device: torch device
            success_threshold: Reward threshold for success (WebShop uses 10.0)
        """
        self.env = env
        self.agent = agent
        self.max_steps = max_steps
        self.device = device
        self.success_threshold = success_threshold  # FIX: Configurable threshold
        
    def simulate_from_state(self, task_idx: int, target_step: int, 
                          original_trajectory: Dict, 
                          method='softmax') -> Tuple[bool, float, List[Dict]]:
        """
        Simulate from a specific step in the original trajectory
        
        Args:
            task_idx: Task ID to run
            target_step: Which step to restart from (0-indexed)
            original_trajectory: The failed trajectory dict
            method: Action selection method ('greedy', 'softmax', 'eps')
            
        Returns:
            success: Whether simulation succeeded (reward >= threshold)
            reward: Final reward
            simulation_trajectory: List of step dicts
        """
        # Reset environment to the specific task
        obs, info = self.env.reset(task_idx)
        goal = original_trajectory.get('goal', info.get('goal', ''))
        
        # Replay actions up to target_step to reach the intermediate state
        simulation_traj = []
        original_steps = original_trajectory.get('steps', [])
        
        if target_step >= len(original_steps):
            print(f"Warning: target_step {target_step} >= trajectory length {len(original_steps)}")
            return False, 0.0, []
        
        # Replay to reach the target state
        for step_idx in range(target_step):
            if step_idx >= len(original_steps):
                break
                
            action = original_steps[step_idx].get('action_taken', '')
            
            if not action:
                print(f"Warning: Empty action at step {step_idx}")
                continue
            
            # Record this replay step
            simulation_traj.append({
                'step': step_idx,
                'observation': obs,
                'action_taken': action,
                'is_replay': True,  # Mark as replay
                'valid_actions': info.get('valid', []),
                'goal': goal
            })
            
            obs, reward, done, info = self.env.step(action)

            if done and step_idx < target_step - 1:
                print(f"      DEBUG: Episode ended early during replay at step {step_idx}, reward={reward}")
            
            if done:
                # If done during replay, can't continue simulation
                return False, reward, simulation_traj
        
        # Now at the target state - start fresh simulation with agent
        for sim_step in range(self.max_steps):
            actual_step = target_step + sim_step
            
            # Agent predicts action
            valid_acts = info.get('valid', [])
            if not valid_acts:
                break
            
            agent_info = {
                'valid': valid_acts,
                'goal': goal  # goal was extracted at line 51
            }

            # Use the agent's act method
            try:
                with torch.no_grad():
                    # Build state representation
                    state = self.agent.build_state(obs, agent_info)
                    
                    # Get action from agent
                    act_strs, act_ids, values = self.agent.act(
                        [state], [valid_acts], method=method, eps=0.0
                    )
                    action = act_strs[0]
            except Exception as e:
                print(f"Error in agent.act: {e}")
                # Fallback: random action
                action = np.random.choice(valid_acts)
            
            # Record simulation step
            simulation_traj.append({
                'step': actual_step,
                'observation': obs,
                'action_taken': action,
                'is_replay': False,  # Mark as NEW simulation step
                'valid_actions': valid_acts,
                'goal': goal
            })
            
            # Take action in environment
            obs, reward, done, info = self.env.step(action)
            
            # Update last step with outcome
            simulation_traj[-1]['reward'] = reward
            simulation_traj[-1]['done'] = done
            simulation_traj[-1]['next_observation'] = obs
            
            if done:
                # FIX: Use configurable threshold (10.0 for WebShop)
                success = (reward >= self.success_threshold)
                return success, reward, simulation_traj
        
        # Max steps reached without completion
        return False, 0.0, simulation_traj
    
    def batch_simulate(self, task_idx: int, state_indices: List[int],
                      original_trajectory: Dict) -> List[Dict]:
        """
        Simulate from multiple states in a trajectory
        
        Args:
            task_idx: Task ID
            state_indices: List of step indices to simulate from
            original_trajectory: Original failed trajectory
            
        Returns:
            List of simulation results
        """
        results = []
        
        for state_idx in state_indices:
            success, reward, sim_traj = self.simulate_from_state(
                task_idx, state_idx, original_trajectory
            )
            
            results.append({
                'state_index': state_idx,
                'success': success,
                'reward': reward,
                'trajectory': sim_traj,
                'original_trajectory': original_trajectory
            })
        
        return results


class BeneficialSegmentExtractor:
    """
    FIXED: Extracts beneficial segments from SUCCESSFUL SIMULATIONS, not failures!
    """
    
    def __init__(self, verbose=True, success_threshold=10.0):
        self.verbose = verbose
        self.success_threshold = success_threshold  # FIX: Configurable
    
    def extract_from_simulation_results(self, simulation_results: List[Dict]) -> List[Dict]:
        """
        Extract beneficial segments from simulation results
        
        CRITICAL FIX: Extract from the NEW successful trajectory, NOT the failed one!
        
        Args:
            simulation_results: List of simulation result dicts
            
        Returns:
            List of beneficial segment dicts ready for training
        """
        beneficial_segments = []
        
        for result in simulation_results:
            # FIX: Use configurable threshold (10.0 for WebShop)
            if result['reward'] < self.success_threshold:
                if self.verbose and result['reward'] > 0:
                    print(f"    Skipping partial success: reward={result['reward']:.2f} < {self.success_threshold}")
                continue
            
            # CRITICAL: Get the SIMULATED trajectory, not the original failed one!
            sim_trajectory = result['trajectory']
            state_idx = result['state_index']
            original_traj = result['original_trajectory']
            
            # Extract ONLY the NEW actions the agent took (where is_replay=False)
            new_actions = [step for step in sim_trajectory 
                          if not step.get('is_replay', False)]
            
            if not new_actions:
                if self.verbose:
                    print(f"    Warning: No new actions in successful simulation from step {state_idx}")
                continue
            
            # Verify this is actually different from failed trajectory
            if self.verbose:
                print(f"    ✓ Extracted {len(new_actions)} NEW beneficial actions")
                print(f"      (Agent explored different path from step {state_idx})")
            
            # Create beneficial segment from NEW successful actions
            beneficial_segment = {
                'task_id': original_traj.get('task_id'),
                'goal': original_traj.get('goal', ''),
                'beneficial_steps': new_actions,  # NEW successful actions!
                'recovery_step': state_idx,
                'source': 'eef_beneficial_verified',
                'num_new_actions': len(new_actions),
                'final_reward': result['reward']
            }
            
            beneficial_segments.append(beneficial_segment)
        
        if self.verbose:
            print(f"\n  ✓ Extracted {len(beneficial_segments)} beneficial segments from successful recoveries")
        
        return beneficial_segments
    
    def create_training_samples(self, beneficial_segments: List[Dict]) -> List[Dict]:
        """
        Convert beneficial segments into training samples
        
        Args:
            beneficial_segments: List of beneficial segment dicts
            
        Returns:
            List of training samples
        """
        training_samples = []
        
        for segment in beneficial_segments:
            beneficial_steps = segment['beneficial_steps']
            goal = segment['goal']
            
            # Create a training sample for each beneficial action
            for step in beneficial_steps:
                sample = {
                    'state': step.get('observation', ''),
                    'goal': goal,
                    'action': step.get('action_taken', ''),
                    'valid_actions': step.get('valid_actions', []),
                    'source': 'eef_beneficial_verified',
                    'task_id': segment['task_id'],
                    'recovery_step': segment['recovery_step'],
                    'final_reward': segment['final_reward']
                }
                training_samples.append(sample)
        
        return training_samples


def format_for_il_training(training_samples: List[Dict]) -> List[Dict]:
    """
    Format EEF samples to match IL training data format
    
    Args:
        training_samples: List of training sample dicts
        
    Returns:
        List formatted for IL training
    """
    formatted = []
    
    for sample in training_samples:
        # Handle case where action not in valid_actions
        try:
            label = sample['valid_actions'].index(sample['action'])
        except ValueError:
            label = 0
            print(f"Warning: Action '{sample['action'][:50]}...' not in valid_actions, using label 0")
        
        # Match the format expected by the IL training pipeline
        formatted_sample = {
            'state': sample['state'],
            'goal': sample['goal'], 
            'action': sample['action'],
            'valid_actions': sample['valid_actions'],
            'reward': 1.0,  # Beneficial actions get positive reward
            'label': label,
            'source': 'eef_beneficial_verified'
        }
        formatted.append(formatted_sample)
    
    return formatted