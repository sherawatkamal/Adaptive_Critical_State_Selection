"""
Supervision generation for SFT training.
"""

from .format_router import (
    generate_supervision_single,
    generate_supervision,
    SupervisionFormat,
)
from .sft_data import (
    SFTDataset,
    create_sft_dataset,
    save_sft_dataset,
)
from .patch_templates import (
    DEMO_FORMAT,
    CONTRAST_FORMAT,
    HINT_FORMAT,
)

__all__ = [
    "generate_supervision_single",
    "generate_supervision",
    "SupervisionFormat",
    "SFTDataset",
    "create_sft_dataset",
    "save_sft_dataset",
    "DEMO_FORMAT",
    "CONTRAST_FORMAT",
    "HINT_FORMAT",
]
