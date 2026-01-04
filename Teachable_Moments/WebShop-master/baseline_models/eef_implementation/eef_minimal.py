"""
Minimal EEF (Exploring Expert Failures) - Extract Beneficial Segments
Just the core extraction logic - no visualization, no extras
"""

import json
import logging
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class BeneficialSegment:
    """A beneficial action segment extracted from a failed trajectory"""
    trajectory_id: str
    segment_type: str  # 'initial_success' or 'recovery'
    start_state_idx: int
    end_state_idx: int
    actions: List[str]
    num_actions: int


class MinimalEEFExtractor:
    """
    Minimal EEF implementation - extracts beneficial segments from failed trajectories
    
    Algorithm:
    1. For each failed trajectory, simulate from M evenly-spaced states
    2. Identify which states lead to success
    3. Extract the action sequences that got there
    """
    
    def __init__(self, num_simulations: int = 5):
        """
        Args:
            num_simulations: Number M of states to simulate per trajectory
        """
        self.M = num_simulations
        self.beneficial_segments = []
        self.stats = {'processed': 0, 'with_segments': 0, 'total_segments': 0}
    
    def extract_from_file(self, json_path: str) -> List[BeneficialSegment]:
        """
        Extract beneficial segments from a JSON file of failed trajectories
        
        Args:
            json_path: Path to JSON file with failed trajectories
            
        Returns:
            List of BeneficialSegment objects
        """
        logger.info(f"Loading failed trajectories from: {json_path}")
        
        # Load trajectories
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Handle different JSON formats
        if isinstance(data, list):
            trajectories = data
        elif isinstance(data, dict) and 'trajectories' in data:
            trajectories = data['trajectories']
        else:
            trajectories = [data]
        
        # Filter for failed trajectories
        # Check both top-level reward and last step reward
        failed = []
        for t in trajectories:
            # Check top-level reward/success
            if t.get('reward', 0) == 0 or t.get('success', True) == False:
                failed.append(t)
                continue
            # Check last step reward (for WebShop format)
            steps = t.get('steps', [])
            if steps and steps[-1].get('reward', 0) == 0:
                failed.append(t)
                continue
        
        logger.info(f"Found {len(failed)} failed trajectories")
        
        # Process each trajectory
        for i, traj in enumerate(failed):
            logger.info(f"Processing trajectory {i+1}/{len(failed)}")
            
            # Debug: show first action of this trajectory
            if i < 3:  # Only for first 3 trajectories
                if 'steps' in traj and traj['steps']:
                    first_action = traj['steps'][0].get('action_taken', 'N/A')
                    logger.info(f"  → First action: {first_action[:80]}...")
                elif 'actions' in traj and traj['actions']:
                    first_action = traj['actions'][0]
                    logger.info(f"  → First action: {first_action[:80]}...")
            
            segments = self._extract_from_trajectory(traj)
            self.beneficial_segments.extend(segments)
            self.stats['processed'] += 1
            if segments:
                self.stats['with_segments'] += 1
        
        self.stats['total_segments'] = len(self.beneficial_segments)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"EXTRACTION COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Processed: {self.stats['processed']} trajectories")
        logger.info(f"Found segments in: {self.stats['with_segments']} trajectories")
        logger.info(f"Total beneficial segments: {self.stats['total_segments']}")
        logger.info(f"{'='*60}\n")
        
        return self.beneficial_segments
    
    def _extract_from_trajectory(self, trajectory: Dict) -> List[BeneficialSegment]:
        """
        Extract beneficial segments from a single trajectory
        
        In a real implementation, this would:
        1. Compute simulation states (every l steps)
        2. Simulate from each state using a policy
        3. Identify which segments led to success
        
        For now, we use a heuristic approach based on trajectory structure:
        - Extract segments that follow good patterns
        - Identify potential recovery points
        """
        segments = []
        
        # Handle different trajectory formats
        if 'task_id' in trajectory:
            traj_id = f"task_{trajectory['task_id']}"
            # Extract actions from steps
            steps = trajectory.get('steps', [])
            actions = [step.get('action_taken', '') for step in steps if step.get('action_taken')]
        else:
            traj_id = trajectory.get('id', 'unknown')
            actions = trajectory.get('actions', [])
        
        # Skip if too few actions or empty
        if not actions or len(actions) < 2:
            return segments
        
        # Compute simulation indices
        T = len(actions)
        l = max(1, T // (self.M + 1))
        sim_indices = [l * m for m in range(1, self.M + 1) if l * m < T]
        
        # HEURISTIC: Identify beneficial patterns in WebShop trajectories
        
        # Pattern 1: Good search queries (first 1-2 actions)
        # WebShop starts with search, so first actions often show good query formulation
        if len(actions) >= 2 and 'search[' in actions[0].lower():
            early_segment = BeneficialSegment(
                trajectory_id=traj_id,
                segment_type='initial_success',
                start_state_idx=0,
                end_state_idx=min(2, len(actions)),
                actions=actions[0:min(2, len(actions))],
                num_actions=min(2, len(actions))
            )
            segments.append(early_segment)
        
        # Pattern 2: Item selection + navigation (actions 1-4)
        # After search, good agents click on relevant items or navigate
        if len(actions) >= 4:
            mid_segment = BeneficialSegment(
                trajectory_id=traj_id,
                segment_type='initial_success',
                start_state_idx=1,
                end_state_idx=min(4, len(actions)),
                actions=actions[1:min(4, len(actions))],
                num_actions=min(3, len(actions) - 1)
            )
            segments.append(mid_segment)
        
        # Pattern 3: Recovery segments - look for navigation actions
        # "click[back to search]", "click[next >]", or new search after progress
        for i, action in enumerate(actions[2:], 2):  # Start from action 2
            action_lower = action.lower()
            
            # Identify recovery actions
            is_recovery = (
                'back to search' in action_lower or
                'click[next' in action_lower or
                'click[< prev]' in action_lower or
                (i > 3 and 'search[' in action_lower)  # New search after exploration
            )
            
            if is_recovery and i + 2 <= len(actions):
                # Found potential recovery action
                recovery_segment = BeneficialSegment(
                    trajectory_id=traj_id,
                    segment_type='recovery',
                    start_state_idx=i,
                    end_state_idx=min(i + 3, len(actions)),
                    actions=actions[i:min(i + 3, len(actions))],
                    num_actions=min(3, len(actions) - i)
                )
                segments.append(recovery_segment)
                break  # Only first recovery per trajectory
        
        return segments
    
    def save_results(self, output_path: str = 'beneficial_segments.json'):
        """Save extracted segments to JSON file"""
        output = {
            'metadata': {
                'num_segments': len(self.beneficial_segments),
                'num_simulations': self.M,
                'statistics': self.stats
            },
            'segments': [asdict(seg) for seg in self.beneficial_segments]
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"✓ Saved {len(self.beneficial_segments)} segments to: {output_path}")
        return output_path
    
    def print_segments(self, max_display: int = 5):
        """Print segments in human-readable format"""
        print("\n" + "="*70)
        print("BENEFICIAL SEGMENTS FOUND")
        print("="*70)
        
        if not self.beneficial_segments:
            print("No segments found.")
            return
        
        # Group by type
        initial = [s for s in self.beneficial_segments if s.segment_type == 'initial_success']
        recovery = [s for s in self.beneficial_segments if s.segment_type == 'recovery']
        
        if initial:
            print(f"\n📊 INITIAL SUCCESS SEGMENTS ({len(initial)} found):")
            print("-"*70)
            for i, seg in enumerate(initial[:max_display], 1):
                print(f"\n{i}. From trajectory: {seg.trajectory_id}")
                print(f"   States: {seg.start_state_idx} → {seg.end_state_idx}")
                print(f"   Actions ({seg.num_actions}):")
                for j, action in enumerate(seg.actions, 1):
                    print(f"      {j}. {action}")
            
            if len(initial) > max_display:
                print(f"\n   ... and {len(initial) - max_display} more")
        
        if recovery:
            print(f"\n\n🔄 RECOVERY SEGMENTS ({len(recovery)} found):")
            print("-"*70)
            for i, seg in enumerate(recovery[:max_display], 1):
                print(f"\n{i}. From trajectory: {seg.trajectory_id}")
                print(f"   States: {seg.start_state_idx} → {seg.end_state_idx}")
                print(f"   Actions ({seg.num_actions}):")
                for j, action in enumerate(seg.actions, 1):
                    print(f"      {j}. {action}")
            
            if len(recovery) > max_display:
                print(f"\n   ... and {len(recovery) - max_display} more")
        
        print("\n" + "="*70)


def extract_beneficial_segments(
    input_json: str,
    output_json: str = 'beneficial_segments.json',
    num_simulations: int = 5,
    print_results: bool = True
) -> str:
    """
    Main function to extract beneficial segments
    
    Args:
        input_json: Path to failed trajectories JSON file
        output_json: Path to save extracted segments
        num_simulations: Number M of simulations per trajectory
        print_results: Whether to print segments to console
        
    Returns:
        Path to output file
    """
    # Create extractor
    extractor = MinimalEEFExtractor(num_simulations=num_simulations)
    
    # Extract segments
    segments = extractor.extract_from_file(input_json)
    
    # Print results
    if print_results and segments:
        extractor.print_segments(max_display=5)
    
    # Save to file
    output_path = extractor.save_results(output_json)
    
    return output_path


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python eef_minimal.py <input_json> [output_json] [num_simulations]")
        print("\nExample:")
        print("  python eef_minimal.py failed_trajectories.json")
        print("  python eef_minimal.py failed_trajectories.json segments.json 10")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'beneficial_segments.json'
    num_sims = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    
    extract_beneficial_segments(input_file, output_file, num_sims)