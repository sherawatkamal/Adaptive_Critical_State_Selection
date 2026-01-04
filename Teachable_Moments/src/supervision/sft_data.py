"""
SFT dataset creation and management.
"""

from dataclasses import dataclass, field
from typing import Optional, Iterator, Union
from pathlib import Path
import json
import random
from .format_router import SupervisionExample, SupervisionFormat

@dataclass
class SFTDataset:
    """Dataset for supervised fine-tuning."""
    
    examples: list[SupervisionExample] = field(default_factory=list)
    name: str = ""
    format: str = ""
    quadrant: str = ""
    metadata: dict = field(default_factory=dict)
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __iter__(self) -> Iterator[SupervisionExample]:
        return iter(self.examples)
    
    def __getitem__(self, idx: int) -> SupervisionExample:
        return self.examples[idx]
    
    def add(self, example: SupervisionExample) -> None:
        """Add an example to the dataset."""
        self.examples.append(example)
    
    def extend(self, examples: list[SupervisionExample]) -> None:
        """Add multiple examples."""
        self.examples.extend(examples)
    
    def shuffle(self, seed: Optional[int] = None) -> "SFTDataset":
        """Shuffle examples in place."""
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.examples)
        return self
    
    def split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: Optional[int] = None,
    ) -> tuple["SFTDataset", "SFTDataset", "SFTDataset"]:
        """
        Split into train, validation, and test sets.
        
        Args:
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train, val, test) datasets
        """
        if seed is not None:
            random.seed(seed)
        
        examples = self.examples.copy()
        random.shuffle(examples)
        
        n = len(examples)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_examples = examples[:n_train]
        val_examples = examples[n_train:n_train + n_val]
        test_examples = examples[n_train + n_val:]
        
        return (
            SFTDataset(examples=train_examples, name=f"{self.name}_train"),
            SFTDataset(examples=val_examples, name=f"{self.name}_val"),
            SFTDataset(examples=test_examples, name=f"{self.name}_test"),
        )
    
    def to_dict_list(self) -> list[dict]:
        """Convert to list of dictionaries."""
        return [ex.to_dict() for ex in self.examples]
    
    def to_hf_format(self) -> list[dict]:
        """
        Convert to HuggingFace datasets format.
        
        Returns:
            List of dicts with 'input' and 'output' keys
        """
        return [
            {"input": ex.input_text, "output": ex.output_text}
            for ex in self.examples
        ]
    
    def save(self, path: Union[Path, str], format: str = "jsonl") -> None:
        """
        Save dataset to file.
        
        Args:
            path: Output file path
            format: "jsonl" or "json"
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "jsonl":
            with open(path, "w") as f:
                for ex in self.examples:
                    f.write(json.dumps(ex.to_dict()) + "\n")
        elif format == "json":
            with open(path, "w") as f:
                json.dump({
                    "name": self.name,
                    "format": self.format,
                    "quadrant": self.quadrant,
                    "metadata": self.metadata,
                    "examples": self.to_dict_list(),
                }, f, indent=2)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    @classmethod
    def load(cls, path: Union[Path, str]) -> "SFTDataset":
        """Load dataset from file."""
        path = Path(path)
        
        if path.suffix == ".jsonl":
            examples = []
            with open(path) as f:
                for line in f:
                    data = json.loads(line)
                    examples.append(SupervisionExample(
                        input_text=data["input"],
                        output_text=data["output"],
                        snapshot_id=data.get("snapshot_id", ""),
                        format=SupervisionFormat(data.get("format", "demo")),
                        quadrant=data.get("quadrant", ""),
                        metadata=data.get("metadata", {}),
                    ))
            return cls(examples=examples, name=path.stem)
        
        elif path.suffix == ".json":
            with open(path) as f:
                data = json.load(f)
            
            examples = [
                SupervisionExample(
                    input_text=ex["input"],
                    output_text=ex["output"],
                    snapshot_id=ex.get("snapshot_id", ""),
                    format=SupervisionFormat(ex.get("format", "demo")),
                    quadrant=ex.get("quadrant", ""),
                    metadata=ex.get("metadata", {}),
                )
                for ex in data["examples"]
            ]
            return cls(
                examples=examples,
                name=data.get("name", ""),
                format=data.get("format", ""),
                quadrant=data.get("quadrant", ""),
                metadata=data.get("metadata", {}),
            )
        
        else:
            raise ValueError(f"Unknown file format: {path.suffix}")


def create_sft_dataset(
    snapshots: list[dict],
    format: Union[SupervisionFormat, str],
    name: str = "",
    quadrant: str = "",
) -> SFTDataset:
    """
    Create SFT dataset from snapshots.
    
    Args:
        snapshots: List of labeled snapshots
        format: Supervision format
        name: Dataset name
        quadrant: Quadrant label if applicable
        
    Returns:
        SFTDataset object
    """
    from .format_router import generate_supervision
    
    examples = generate_supervision(snapshots, format)
    
    return SFTDataset(
        examples=examples,
        name=name,
        format=format if isinstance(format, str) else format.value,
        quadrant=quadrant,
    )


def save_sft_dataset(
    dataset: SFTDataset,
    output_dir: Union[Path, str],
    splits: bool = True,
    seed: int = 42,
) -> dict[str, Path]:
    """
    Save dataset with optional train/val/test splits.
    
    Args:
        dataset: SFT dataset to save
        output_dir: Output directory
        splits: Whether to create train/val/test splits
        seed: Random seed for splitting
        
    Returns:
        Dict mapping split names to file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paths = {}
    
    if splits and len(dataset) > 10:
        train, val, test = dataset.split(seed=seed)
        
        train_path = output_dir / f"{dataset.name}_train.jsonl"
        train.save(train_path)
        paths["train"] = train_path
        
        if len(val) > 0:
            val_path = output_dir / f"{dataset.name}_val.jsonl"
            val.save(val_path)
            paths["val"] = val_path
        
        if len(test) > 0:
            test_path = output_dir / f"{dataset.name}_test.jsonl"
            test.save(test_path)
            paths["test"] = test_path
    
    else:
        full_path = output_dir / f"{dataset.name}.jsonl"
        dataset.save(full_path)
        paths["full"] = full_path
    
    return paths


def merge_datasets(*datasets: SFTDataset, name: str = "merged") -> SFTDataset:
    """
    Merge multiple datasets into one.
    
    Args:
        *datasets: Datasets to merge
        name: Name for merged dataset
        
    Returns:
        Merged SFTDataset
    """
    all_examples = []
    for ds in datasets:
        all_examples.extend(ds.examples)
    
    return SFTDataset(examples=all_examples, name=name)
