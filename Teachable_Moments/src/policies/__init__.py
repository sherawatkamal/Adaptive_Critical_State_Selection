"""Policy interfaces used across labeling / evaluation.

This package is intentionally lightweight so that the rest of the
codebase can depend on a stable API:

- ModelFactoryPolicy: wraps src.utils.model_factory.ModelFactory
- RandomPolicy: tiny, dependency-free policy for smoke tests

"""

from .model_factory_policy import ModelFactoryPolicy
from .random_policy import RandomPolicy

__all__ = ["ModelFactoryPolicy", "RandomPolicy"]
