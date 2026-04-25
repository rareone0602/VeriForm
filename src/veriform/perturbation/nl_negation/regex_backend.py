"""Regex-based negation of natural-language statements.

Identical behaviour to the original StandardPerturber.logical_negation_str.
Operates on English; the Lean-side counterpart lives in
src/veriform/proving/negation.py.
"""

import re
from typing import Tuple


class RegexNegationBackend:
    """Negate a natural-language CoT statement using regex heuristics.

    Strategy (in order):
    1. If a ``not`` already appears, drop the first one (double-negation).
    2. Otherwise insert ``not`` after the first modal/copula verb.
    3. Otherwise prepend ``It is false that ``.
    """

    _MODALS = re.compile(
        r"\b(is|are|was|were|has|have|had|can|could|will|would|shall|should|must|might)\b",
        re.IGNORECASE,
    )

    def negate(self, content: str) -> Tuple[str, bool]:
        # 1. Removal: drop the first existing 'not'
        if re.search(r"\bnot\b", content):
            return re.sub(r"\bnot\s+", "", content, count=1), True

        # 2. Insertion: after the first modal/copula
        match = self._MODALS.search(content)
        if match:
            head = content[: match.end()]
            tail = content[match.end() :]
            return f"{head} not{tail}", True

        # 3. Fallback: prepend 'It is false that'
        return f"It is false that {content}", True
