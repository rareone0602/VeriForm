"""Dataset loaders.

Only ProcessBenchLoader is in active use — it reads the pre-processed
DAG JSON file produced by the upstream LLM-based DAG-construction step
and yields ReasoningChain objects to feed the rest of the pipeline.

Other dataset-specific loaders (GSM8K, MATH, Custom) were removed in
favour of the single ProcessBench-derived path; restore from git history
if a different dataset becomes relevant.
"""

import json
import random
from abc import ABC, abstractmethod
from pprint import pprint
from typing import Any, Dict, List, Optional

from .reasoning_chain import ReasoningChain, ReasoningStep, StepType


class DatasetLoader(ABC):
    """Abstract base class for dataset loaders."""

    def __init__(self, split: str = "train", num_samples: Optional[int] = None, seed: int = 42):
        self.split = split
        self.num_samples = num_samples
        self.seed = seed
        random.seed(seed)

    @abstractmethod
    def load(self) -> List[ReasoningChain]:
        """Load and parse the dataset into reasoning chains."""

    @abstractmethod
    def parse_reasoning_steps(self, example: Dict[str, Any]) -> List[ReasoningStep]:
        """Parse an example into reasoning steps."""


class ProcessBenchLoader(DatasetLoader):
    """Reads a pre-processed dags.json file (column-major-then-row-major,
    one entry per ProcessBench reasoning chain) and yields ReasoningChain
    objects with the per-step DAG already populated."""

    def __init__(self, file_path: str, **kwargs):
        super().__init__(**kwargs)
        self.file_path = file_path

    def load(self) -> List[ReasoningChain]:
        with open(self.file_path, "r") as f:
            data = json.load(f)
        if self.num_samples:
            data = random.sample(data, min(self.num_samples, len(data)))

        chains: List[ReasoningChain] = []
        for example in data:
            try:
                chain = ReasoningChain(
                    chain_id=example["id"],
                    problem_statement=example["problem"],
                    steps=self.parse_reasoning_steps(example["dags"]),
                    final_answer=example["dags"]["final_answer"],
                    source_dataset=example["split"],
                    metadata={
                        "generator": example.get("generator", ""),
                        "notes": example.get("dags", {}).get("metadata", {}).get("notes", ""),
                        "difficulty": example.get("dags", {}).get("metadata", {}).get("difficulty", "unknown"),
                        "final_answer_correct": example.get("final_answer_correct", None),
                    },
                )
                chains.append(chain)
            except Exception:
                pprint(example)
                raise

        return chains

    def parse_reasoning_steps(self, example: Dict[str, Any]) -> List[ReasoningStep]:
        steps_data = example["nodes"]
        steps: List[ReasoningStep] = []
        for step_idx, step_content in enumerate(steps_data):
            assert step_content["node_id"] == f"step_{step_idx + 1}", "Step IDs are not in expected order."
            steps.append(
                ReasoningStep(
                    step_id=step_idx,
                    content=step_content["content"],
                    step_type=(
                        StepType.OTHER
                        if step_content["statement_type"] == "declarative"
                        else StepType.CALCULATION
                    ),
                    previous_steps=list(range(step_idx)),
                )
            )
        return steps
