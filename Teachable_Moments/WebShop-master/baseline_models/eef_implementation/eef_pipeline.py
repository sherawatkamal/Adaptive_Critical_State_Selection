"""
EEF Pipeline - FIXED VERSION
Complete EEF implementation with ACSS
Orchestrates: Load failures -> ACSS -> Simulate -> Extract -> Train

KEY FIXES:
1. Actually filters for failed trajectories
2. Configurable success threshold
3. Better statistics and logging
"""
import json
import os
from typing import List, Dict, Tuple
from collections import defaultdict
import numpy as np

# Import fixed versions
from eef_simulator import EEFSimulator, BeneficialSegmentExtractor, format_for_il_training
from acss import ACSS, visualize_critical_states


class EEFPipeline:
    """
    Complete EEF Pipeline with ACSS
    """
    
    def __init__(self, env, agent, acss_strategy='baseline', M=5, 
                 simulation_budget=100, max_failures=None, verbose=True,
                 success_threshold=10.0):
        """
        Args:
            env: WebShop environment
            agent: Trained agent model
            acss_strategy: 'baseline' (skip-length) or 'entropy' (Shannon Entropy)
            M: Number of states to select per trajectory (parameter M from paper)
            simulation_budget: Maximum number of simulations to run
            max_failures: Maximum number of failures to load (None = load all)
            verbose: Print progress information
            success_threshold: Reward threshold for success (10.0 for WebShop)
        """
        self.env = env
        self.agent = agent
        self.success_threshold = success_threshold
        
        # Initialize components with correct threshold
        self.simulator = EEFSimulator(env, agent, success_threshold=success_threshold)
        self.segment_extractor = BeneficialSegmentExtractor(
            verbose=verbose, success_threshold=success_threshold
        )
        self.acss = ACSS(strategy=acss_strategy, M=M, agent=agent, verbose=verbose)
        
        self.simulation_budget = simulation_budget
        self.max_failures = max_failures
        self.verbose = verbose
        self.M = M
        self.acss_strategy = acss_strategy
        
        # Statistics
        self.stats = {
            'total_trajectories_loaded': 0,
            'actual_failures': 0,
            'partial_successes': 0,
            'total_states_selected': 0,
            'total_simulations_run': 0,
            'successful_recoveries': 0,
            'beneficial_segments_extracted': 0,
            'training_samples_created': 0,
            'avg_failure_reward': 0.0,
            'recovery_rewards': []
        }
    
    def load_failure_trajectories(self, failure_json_path: str) -> List[Dict]:
        """
        Load failure trajectories from JSON file
        
        FIX: Actually filters for failed trajectories!
        
        Args:
            failure_json_path: Path to trajectory JSON
            
        Returns:
            List of failure trajectory dicts
        """
        if not os.path.exists(failure_json_path):
            raise FileNotFoundError(f"Failure data not found: {failure_json_path}")
        
        with open(failure_json_path, 'r') as f:
            data = json.load(f)
        
        # Handle different JSON formats
        if isinstance(data, dict):
            if 'trajectories' in data:
                trajectories = data['trajectories']
            else:
                trajectories = [data]
        else:
            trajectories = data
        
        self.stats['total_trajectories_loaded'] = len(trajectories)
        
        # FIX: Actually filter for failures!
        failures = []
        partial_successes = []
        total_reward = 0.0
        
        for traj in trajectories:
            reward = traj.get('reward', 0)
            total_reward += reward
            
            if reward >= self.success_threshold:
                # Full success - skip
                continue
            elif reward > 0:
                # Partial success - these are interesting for EEF
                partial_successes.append(traj)
                failures.append(traj)
            else:
                # Complete failure
                failures.append(traj)
        
        self.stats['actual_failures'] = len(failures) - len(partial_successes)
        self.stats['partial_successes'] = len(partial_successes)
        
        if trajectories:
            self.stats['avg_failure_reward'] = total_reward / len(trajectories)
        
        # Limit to max_failures if specified
        if self.max_failures is not None and self.max_failures < len(failures):
            failures = failures[:self.max_failures]
            if self.verbose:
                print(f"Limited to first {self.max_failures} failures")
        
        if self.verbose:
            print(f"\nTrajectory Loading Summary:")
            print(f"  Total loaded:       {self.stats['total_trajectories_loaded']}")
            print(f"  Complete failures:  {self.stats['actual_failures']}")
            print(f"  Partial successes:  {self.stats['partial_successes']}")
            print(f"  Using for EEF:      {len(failures)}")
            print(f"  Avg failure reward: {self.stats['avg_failure_reward']:.2f}")
            
            if failures:
                traj_lengths = [len(t.get('steps', [])) for t in failures]
                print(f"  Avg traj length:    {np.mean(traj_lengths):.1f}")
                rewards = [t.get('reward', 0) for t in failures]
                print(f"  Reward range:       {min(rewards):.2f} - {max(rewards):.2f}")
        
        if not failures:
            print("\nWARNING: No failure trajectories found!")
            print(f"  Success threshold: {self.success_threshold}")
            print("  Check if trajectories are already successful or threshold is wrong.")
        
        return failures
    
    def run_eef_phase(self, failure_trajectories: List[Dict]) -> List[Dict]:
        """
        Main EEF phase: ACSS -> Simulate -> Extract
        
        Args:
            failure_trajectories: List of failed trajectory dicts
            
        Returns:
            List of beneficial training samples
        """
        all_simulation_results = []
        simulations_run = 0
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"PHASE 1: ACSS - Selecting Critical States")
            print(f"Strategy: {self.acss_strategy}")
            print(f"{'='*70}")
        
        # Apply ACSS to select critical states
        for traj_idx, trajectory in enumerate(failure_trajectories):
            task_id = trajectory.get('task_id')
            
            # Select critical states using ACSS
            critical_states = self.acss.select_critical_states(trajectory)
            
            if not critical_states:
                if self.verbose:
                    print(f"  Task {task_id}: No critical states (trajectory too short)")
                continue
            
            self.stats['total_states_selected'] += len(critical_states)
            
            if self.verbose:
                print(f"\n  Task {task_id} ({traj_idx+1}/{len(failure_trajectories)}):")
                print(f"    Trajectory length: {len(trajectory.get('steps', []))} steps")
                print(f"    Original reward:   {trajectory.get('reward', 0):.2f}")
                print(f"    Selected {len(critical_states)} critical states: {critical_states}")
                visualize_critical_states(trajectory, critical_states, self.acss_strategy)
            
            # Check simulation budget
            if simulations_run >= self.simulation_budget:
                if self.verbose:
                    print(f"\n  ⚠ Simulation budget reached ({self.simulation_budget})")
                break
            
            # Simulate from each critical state
            if self.verbose:
                print(f"\n{'─'*50}")
                print(f"  Simulating from Task {task_id}")
                print(f"{'─'*50}")
            
            for state_idx in critical_states:
                if simulations_run >= self.simulation_budget:
                    break
                
                if self.verbose:
                    print(f"\n    Simulating from step {state_idx}...")
                
                success, reward, sim_traj = self.simulator.simulate_from_state(
                    task_id, state_idx, trajectory, method='greedy'
                )
                
                simulations_run += 1
                self.stats['total_simulations_run'] += 1
                self.stats['recovery_rewards'].append(reward)
                
                if success:
                    self.stats['successful_recoveries'] += 1
                    if self.verbose:
                        new_steps = len([s for s in sim_traj if not s.get('is_replay', False)])
                        print(f"      ✓ SUCCESS! Reward: {reward:.2f}")
                        print(f"        Recovery took {new_steps} new steps")
                else:
                    if self.verbose:
                        print(f"      ✗ Failed. Reward: {reward:.2f}")
                
                # Store result
                all_simulation_results.append({
                    'state_index': state_idx,
                    'success': success,
                    'reward': reward,
                    'trajectory': sim_traj,
                    'original_trajectory': trajectory
                })
        
        # Report overall statistics
        if self.stats['total_simulations_run'] > 0:
            success_rate = self.stats['successful_recoveries'] / self.stats['total_simulations_run']
            avg_recovery_reward = np.mean(self.stats['recovery_rewards'])
            
            if self.verbose:
                print(f"\n{'='*70}")
                print(f"  Simulation Summary:")
                print(f"    Total simulations:   {self.stats['total_simulations_run']}")
                print(f"    Successful:          {self.stats['successful_recoveries']}")
                print(f"    Success rate:        {success_rate:.1%}")
                print(f"    Avg recovery reward: {avg_recovery_reward:.2f}")
                print(f"{'='*70}")
        
        # Extract beneficial segments
        if self.verbose:
            print(f"\n{'='*70}")
            print("PHASE 2: Extracting Beneficial Segments")
            print(f"{'='*70}")
        
        beneficial_segments = self.segment_extractor.extract_from_simulation_results(
            all_simulation_results
        )
        
        self.stats['beneficial_segments_extracted'] = len(beneficial_segments)
        
        if self.verbose:
            print(f"  Extracted {len(beneficial_segments)} beneficial segments")
        
        # Create training samples
        training_samples = self.segment_extractor.create_training_samples(beneficial_segments)
        self.stats['training_samples_created'] = len(training_samples)
        
        if self.verbose:
            print(f"  Created {len(training_samples)} training samples")
        
        return training_samples
    
    def save_beneficial_data(self, training_samples: List[Dict], output_path: str):
        """
        Save beneficial segments and training samples
        """
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        output_data = {
            'metadata': {
                'num_samples': len(training_samples),
                'acss_strategy': self.acss_strategy,
                'M': self.M,
                'success_threshold': self.success_threshold,
                'statistics': self.stats
            },
            'samples': training_samples
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        if self.verbose:
            print(f"\n  ✓ Saved {len(training_samples)} beneficial training samples to: {output_path}")
    
    def print_statistics(self):
        """Print comprehensive EEF pipeline statistics"""
        print(f"\n{'='*70}")
        print("EEF PIPELINE STATISTICS")
        print(f"{'='*70}")
        print(f"\n  Configuration:")
        print(f"    ACSS Strategy:       {self.acss_strategy}")
        print(f"    M (states/traj):     {self.M}")
        print(f"    Success threshold:   {self.success_threshold}")
        print(f"    Simulation budget:   {self.simulation_budget}")
        
        print(f"\n  Data Loading:")
        print(f"    Trajectories loaded: {self.stats['total_trajectories_loaded']}")
        print(f"    Complete failures:   {self.stats['actual_failures']}")
        print(f"    Partial successes:   {self.stats['partial_successes']}")
        print(f"    Avg failure reward:  {self.stats['avg_failure_reward']:.2f}")
        
        print(f"\n  ACSS Selection:")
        print(f"    States selected:     {self.stats['total_states_selected']}")
        if self.stats['total_trajectories_loaded'] > 0:
            avg_states = self.stats['total_states_selected'] / max(1, self.stats['actual_failures'] + self.stats['partial_successes'])
            print(f"    Avg states/traj:     {avg_states:.1f}")
        
        print(f"\n  Simulation Results:")
        print(f"    Simulations run:     {self.stats['total_simulations_run']}")
        print(f"    Successful:          {self.stats['successful_recoveries']}")
        
        if self.stats['total_simulations_run'] > 0:
            success_rate = self.stats['successful_recoveries'] / self.stats['total_simulations_run']
            print(f"    Recovery rate:       {success_rate:.1%}")
            
            if self.stats['recovery_rewards']:
                avg_reward = np.mean(self.stats['recovery_rewards'])
                print(f"    Avg recovery reward: {avg_reward:.2f}")
        
        print(f"\n  Extraction Results:")
        print(f"    Beneficial segments: {self.stats['beneficial_segments_extracted']}")
        print(f"    Training samples:    {self.stats['training_samples_created']}")
        
        # Efficiency metrics
        if self.stats['total_simulations_run'] > 0:
            efficiency = self.stats['training_samples_created'] / self.stats['total_simulations_run']
            print(f"\n  Efficiency:")
            print(f"    Samples per sim:     {efficiency:.2f}")
        
        print(f"{'='*70}\n")


def run_eef_pipeline(env, agent, failure_json_path, output_dir='./eef_output',
                    acss_strategy='baseline', M=5, simulation_budget=100, 
                    max_failures=None, success_threshold=10.0):
    """
    Convenience function to run complete EEF pipeline
    
    Args:
        env: WebShop environment
        agent: Trained agent
        failure_json_path: Path to failure trajectories JSON
        output_dir: Directory for outputs
        acss_strategy: 'baseline' (skip-length) or 'entropy' (Shannon Entropy)
        M: States per trajectory (parameter M from paper)
        simulation_budget: Max simulations
        max_failures: Maximum number of failures to load (None = load all)
        success_threshold: Reward threshold for success (10.0 for WebShop)
        
    Returns:
        training_samples: List of beneficial training samples
        pipeline: EEFPipeline object with statistics
    """
    # Initialize pipeline
    pipeline = EEFPipeline(
        env=env,
        agent=agent,
        acss_strategy=acss_strategy,
        M=M,
        simulation_budget=simulation_budget,
        max_failures=max_failures,
        verbose=True,
        success_threshold=success_threshold
    )
    
    # Print configuration
    print(f"\n{'='*70}")
    print("EEF PIPELINE - EXPLORING EXPERT FAILURES")
    print(f"{'='*70}")
    print(f"Configuration:")
    print(f"  ACSS Strategy:      {acss_strategy}")
    print(f"  M (states/traj):    {M}")
    print(f"  Simulation Budget:  {simulation_budget}")
    print(f"  Success Threshold:  {success_threshold}")
    print(f"  Failure Data:       {failure_json_path}")
    print(f"  Output Directory:   {output_dir}")
    
    # Load failures
    failures = pipeline.load_failure_trajectories(failure_json_path)
    
    if not failures:
        print("\n⚠ No failure trajectories to process!")
        return [], pipeline
    
    # Run EEF
    training_samples = pipeline.run_eef_phase(failures)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save training samples
    output_path = os.path.join(output_dir, 'beneficial_training_samples.json')
    pipeline.save_beneficial_data(training_samples, output_path)
    
    # Also save in IL-compatible format
    formatted_samples = format_for_il_training(training_samples)
    il_path = os.path.join(output_dir, 'il_training_samples.json')
    with open(il_path, 'w') as f:
        json.dump(formatted_samples, f, indent=2)
    print(f"  ✓ Saved IL-formatted samples to: {il_path}")
    
    # Print statistics
    pipeline.print_statistics()
    
    # Save statistics
    stats_path = os.path.join(output_dir, 'eef_statistics.json')
    with open(stats_path, 'w') as f:
        # Convert numpy types for JSON serialization
        stats_dict = {}
        for k, v in pipeline.stats.items():
            if isinstance(v, np.ndarray):
                stats_dict[k] = v.tolist()
            elif isinstance(v, (np.floating, np.integer)):
                stats_dict[k] = float(v)
            else:
                stats_dict[k] = v
        json.dump(stats_dict, f, indent=2)
    
    print(f"  ✓ Saved statistics to: {stats_path}")
    
    return training_samples, pipeline