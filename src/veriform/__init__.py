"""
Veriform: Verification-Oriented Formalization Benchmarking

A framework for measuring the faithfulness of autoformalization systems.
"""

__version__ = "0.1.0"

from veriform.config import BenchmarkConfig
from veriform.pipeline import StandardPipeline

__all__ = ["BenchmarkConfig", "StandardPipeline", "__version__"]
