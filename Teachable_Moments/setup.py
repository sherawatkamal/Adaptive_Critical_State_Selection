#!/usr/bin/env python3
"""Setup script for teachable-moments package."""

from pathlib import Path

from setuptools import find_packages, setup

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    requirements = [
        line.strip()
        for line in requirements_path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
else:
    requirements = []

# Core requirements (subset for basic installation)
core_requirements = [
    "torch>=2.0.0",
    "transformers>=4.30.0",
    "numpy>=1.24.0",
    "scipy>=1.11.0",
    "scikit-learn>=1.3.0",
    "pandas>=2.0.0",
    "gymnasium>=0.29.0",
    "minigrid>=2.3.0",
    "tqdm>=4.65.0",
    "pyyaml>=6.0",
]

setup(
    name="teachable-moments",
    version="0.1.0",
    author="SAGE Lab",
    author_email="sage-lab@vt.edu",
    description="Identifying and leveraging teachable moments in RL agent training",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sage-lab/teachable-moments",
    project_urls={
        "Bug Reports": "https://github.com/sage-lab/teachable-moments/issues",
        "Source": "https://github.com/sage-lab/teachable-moments",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=core_requirements,
    extras_require={
        "full": requirements,
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "ruff>=0.0.280",
            "mypy>=1.4.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ],
        "tracking": [
            "wandb>=0.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "tm-label=scripts.phase1.run_labeling:main",
            "tm-train=scripts.phase2.train_per_quadrant:main",
            "tm-eval=scripts.phase3.evaluate_end2end:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
