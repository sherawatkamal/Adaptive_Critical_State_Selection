"""
Teacher model API client.

Handles communication with GPT-4o for hint generation with caching
to minimize API costs.
"""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import json
import hashlib
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class TeacherConfig:
    """Configuration for teacher API client."""
    
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 512
    cache_dir: str = ".teacher_cache"
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    @classmethod
    def from_yaml(cls, path: str) -> "TeacherConfig":
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data.get("teacher", {}))


class TeacherCache:
    """Disk-based cache for teacher responses."""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0
    
    def _get_key(self, prompt: str) -> str:
        """Generate cache key from prompt."""
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]
    
    def _get_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"
    
    def get(self, prompt: str) -> Optional[str]:
        """Get cached response for prompt."""
        key = self._get_key(prompt)
        path = self._get_path(key)
        
        if path.exists():
            self.hits += 1
            with open(path) as f:
                data = json.load(f)
            return data.get("response")
        
        self.misses += 1
        return None
    
    def set(self, prompt: str, response: str) -> None:
        """Cache response for prompt."""
        key = self._get_key(prompt)
        path = self._get_path(key)
        
        with open(path, "w") as f:
            json.dump({
                "prompt": prompt,
                "response": response,
                "timestamp": time.time(),
            }, f)
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "cache_size": len(list(self.cache_dir.glob("*.json"))),
        }


class TeacherClient:
    """Client for teacher model API with caching."""
    
    def __init__(self, config: Optional[TeacherConfig] = None):
        self.config = config or TeacherConfig()
        self.cache = TeacherCache(self.config.cache_dir)
        self._client = None
    
    def _get_client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI()
            except ImportError:
                raise ImportError("openai package required for teacher client")
        return self._client
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        use_cache: bool = True,
    ) -> str:
        """
        Generate response from teacher model.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            use_cache: Whether to use caching
            
        Returns:
            Generated response text
        """
        # Build full prompt for caching
        full_prompt = f"{system_prompt or ''}\n\n{prompt}"
        
        # Check cache
        if use_cache:
            cached = self.cache.get(full_prompt)
            if cached is not None:
                logger.debug("Cache hit for teacher request")
                return cached
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Make API call with retries
        client = self._get_client()
        
        for attempt in range(self.config.retry_attempts):
            try:
                response = client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                
                result = response.choices[0].message.content
                
                # Cache result
                if use_cache:
                    self.cache.set(full_prompt, result)
                
                return result
            
            except Exception as e:
                logger.warning(f"Teacher API attempt {attempt + 1} failed: {e}")
                if attempt < self.config.retry_attempts - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    raise

    # ------------------------------------------------------------------
    # Backwards-compatible alias
    # ------------------------------------------------------------------
    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        use_cache: bool = True,
    ) -> str:
        """Alias for older code paths that expect `generate_text()`.

        The canonical API is :meth:`generate`. Some legacy scripts/modules in
        this repo call `generate_text(prompt)`.
        """
        return self.generate(prompt=prompt, system_prompt=system_prompt, use_cache=use_cache)
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        return self.cache.get_stats()


class MockTeacherClient:
    """Mock teacher client for testing."""
    
    def __init__(self, responses: Optional[dict] = None):
        self.responses = responses or {}
        self.calls = []
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        use_cache: bool = True,
    ) -> str:
        self.calls.append({"prompt": prompt, "system_prompt": system_prompt})
        
        # Check for predefined response
        for key, response in self.responses.items():
            if key in prompt:
                return response
        
        # Default mock response
        # Default mock response: emit a minimally-valid JSON hint so that
        # `structured_hint.generate_teacher_hint()` works end-to-end in smoke tests.
        import json
        import re

        actions: list[str] = []
        for line in prompt.splitlines():
            m = re.match(r"^\s*-\s*(.+?)\s*$", line)
            if m:
                actions.append(m.group(1))

        suggested_action = actions[0] if actions else "search[query]"

        return json.dumps(
            {
                "suggested_action": suggested_action,
                "rationale": "Mock rationale (testing only).",
                "error_type": "planning_error",
                "confidence": "low",
            }
        )

    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        use_cache: bool = True,
    ) -> str:
        """Alias for older code paths that expect `generate_text()`."""
        return self.generate(prompt=prompt, system_prompt=system_prompt, use_cache=use_cache)
    
    def get_cache_stats(self) -> dict:
        return {"hits": 0, "misses": len(self.calls), "hit_rate": 0.0, "cache_size": 0}


def create_teacher_client(
    config: Optional[TeacherConfig] = None,
    mock: bool = False,
) -> TeacherClient:
    """
    Factory function to create teacher client.
    
    Args:
        config: Teacher configuration
        mock: Whether to create mock client for testing
        
    Returns:
        TeacherClient instance
    """
    if mock:
        return MockTeacherClient()
    return TeacherClient(config)
