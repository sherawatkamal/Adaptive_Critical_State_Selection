"""AgentDebug Baseline Implementation.

Implements the AgentDebug method (Global and Fine-grained analysis)
as a baseline for Teachable Moments.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import logging

from ..data.snapshot import TeacherHint, ErrorType
from ..teacher.client import TeacherClient

logger = logging.getLogger(__name__)


@dataclass
class AgentDebugConfig:
    """Configuration for AgentDebug."""
    mode: str = "global"  # global, fine, critical
    max_steps_context: int = 5


class AgentDebugBaseline:
    """AgentDebug baseline implementation."""
    
    def __init__(self, client: TeacherClient, config: AgentDebugConfig):
        self.client = client
        self.config = config
        
    def analyze_trajectory(self, trajectory: Dict[str, Any]) -> List[TeacherHint]:
        """
        Analyze a full trajectory to find bugs/failures.
        
        Args:
            trajectory: Dict with steps, states, actions
            
        Returns:
            List of generated hints/patches
        """
        if self.config.mode == "global":
            return self._analyze_global(trajectory)
        elif self.config.mode == "fine":
            return self._analyze_fine(trajectory)
        else:
            raise ValueError(f"Unknown mode: {self.config.mode}")
            
    def _analyze_global(self, trajectory: Dict[str, Any]) -> List[TeacherHint]:
        """Global analysis: Read entire trace and summarize one key error."""
        steps = trajectory.get("steps", [])
        if not steps:
            return []
            
        # Format trace for LLM
        trace_str = self._format_trace(steps)
        
        prompt = f"""Analyze this agent trajectory and identify the primary cause of failure.

TRACE:
{trace_str}

Provide the single most critical correction as a JSON hint:
{{
    "suggested_action": "action string",
    "rationale": "explanation",
    "error_type": "planning_error"
}}"""

        try:
            if hasattr(self.client, "generate_text"):
                resp = self.client.generate_text(prompt)
            else:
                resp = self.client.generate(prompt)
                
            # Parse JSON (simplified)
            import json
            import re
            
            match = re.search(r'\{.*\}', resp, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return [TeacherHint(
                    suggested_action=data.get("suggested_action", ""),
                    rationale=data.get("rationale", ""),
                    error_type=ErrorType.PLANNING_ERROR,
                    confidence="high"
                )]
        except Exception as e:
            logger.error(f"AgentDebug Global failed: {e}")
            
        return []

    def _analyze_fine(self, trajectory: Dict[str, Any]) -> List[TeacherHint]:
        """Fine analysis: Step-by-step verification (expensive)."""
        # Simplified placeholder for fine-grained analysis
        # In a real impl, this would loop over steps and ask teacher for each
        return []

    def _format_trace(self, steps: List[Dict]) -> str:
        lines = []
        for i, step in enumerate(steps):
            lines.append(f"Step {i}:")
            lines.append(f"  Action: {step.get('action_taken')}")
            lines.append(f"  Obs: {step.get('observation', '')[:200]}...")
        return "\n".join(lines)
