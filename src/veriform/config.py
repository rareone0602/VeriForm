"""Pydantic configuration schema for the experiment runner.

A YAML matching this schema is the single source of truth for one
benchmark run. Override individual fields from the CLI via dotted paths
(e.g. ``--set perturbation.p=1.0 formalization.type=goedel``).
"""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class DatasetSection(BaseModel):
    """How to load CoT chains. Currently the active path is ProcessBench."""
    type: Literal["processbench"] = "processbench"
    file_path: str = "./data/processed/dags.json"
    num_samples: Optional[int] = 1179
    seed: int = 42


class PerturbationSection(BaseModel):
    """Which perturber to instantiate and its parameters."""
    type: Literal["regex", "deepseek_pre", "gemini", "openai"] = "regex"
    p: float = Field(0.5, ge=0.0, le=1.0)
    params: Dict[str, Any] = Field(default_factory=dict)


class FormalizationSection(BaseModel):
    type: Literal["stepfun", "kimina", "goedel", "herald"] = "stepfun"
    sampling: Literal["recommended", "deterministic"] = "deterministic"
    params: Dict[str, Any] = Field(default_factory=dict)


class ProvingSection(BaseModel):
    """DeepSeekProver settings. Set ``enabled: false`` to skip the prover stage
    (useful for smoke-testing the perturbation+formalization pipeline)."""
    enabled: bool = True
    type: Literal["deepseek"] = "deepseek"
    batch_size: int = 2
    negation: Literal["strong", "full"] = "full"
    base_url: str = "http://localhost:8001/v1"
    params: Dict[str, Any] = Field(default_factory=dict)


class OutputSection(BaseModel):
    dir: str = "experiments/outputs"
    save_lean_files: bool = True
    save_intermediate: bool = True
    save_heatmap: bool = True


class SemanticsSection(BaseModel):
    """ProofBridge-based semantic-faithfulness re-weighting. Disabled until
    the wrapper in src/veriform/semantics/ is implemented."""
    enabled: bool = False
    encoder_path: Optional[str] = None
    drift_threshold: Optional[float] = None


class RunConfig(BaseModel):
    """Top-level config consumed by scripts/run_benchmark.py."""
    model_config = ConfigDict(extra="forbid")

    dataset: DatasetSection = DatasetSection()
    perturbation: PerturbationSection = PerturbationSection()
    formalization: FormalizationSection = FormalizationSection()
    proving: ProvingSection = ProvingSection()
    output: OutputSection = OutputSection()
    semantics: SemanticsSection = SemanticsSection()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        import yaml
        with open(path) as f:
            return cls(**(yaml.safe_load(f) or {}))


# --- Backwards-compat alias for any external code importing BenchmarkConfig ---
BenchmarkConfig = RunConfig
