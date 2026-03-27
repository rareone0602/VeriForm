# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

This is the implementaiton of the paper.pdf. Use can also refer to README.md but if there is conflicting information, please adhere to paper.pdf.

## Environment

- **Python**: `~/projects/ruqola/VeriForm/veriform/bin/python` (torch 2.9.0+cu128, transformers 5.3, datasets 4.8)
- **Storage**: Home has 100 GB quota, so some large files like `experiments` and `data` are symlink to `/scratch/users/$USER/`. HF_HOME is set in the user environment;
- **GPUs**: 3× H200 NVL (143 GB each). They are occupied so far, but you can refactor the part that uses Lean 4. 



## Improvement

1. Directory redesign: The current directory structure is a bit messy. Consider organizing it into clearer modules, e.g., `data_processing/`, `model/`, `training/`, `evaluation/`. `src/veriform/autoformalization/` is no longer in used, we basically use `src/veriform/autoformalization_v2` only.
2. The way we negate a Lean statement is based on a python regex to parse Lean code, but I believe there are some alternative ways to do it, e.g., using Lean's own parser. This part is a bit hacky and may not be robust to all cases. Discuss it with me before you start working on it.
3. Other improvements you can think of. The code is not very clean and there are many places that can be refactored. Discuss with me.
4. Git commit after you read the codebase and understand the overall structure. You can also create a new branch for your work.