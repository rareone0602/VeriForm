"""Heatmap rendering for per-dataset flag distributions."""

from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import matplotlib.pyplot as plt


def draw_heatmap(
    p: float,
    formalizer_name: str,
    heatmap_data: Dict[str, Any],
    output_path: Union[str, Path],
) -> None:
    """Render per-dataset flag distribution as a percentage heatmap."""
    raw_data = np.array(heatmap_data["data"])
    rows = heatmap_data["rows"]
    cols = heatmap_data["columns"][:-1]

    category_data = raw_data[:, :-1]
    row_sums = category_data.sum(axis=1, keepdims=True)
    safe_sums = np.where(row_sums == 0, 1, row_sums)
    pct_data = (category_data / safe_sums) * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pct_data, cmap="Blues", vmin=0, vmax=100)

    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(rows)))
    ax.set_xticklabels(cols)
    ax.set_yticklabels(rows)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(rows)):
        for j in range(len(cols)):
            pct_val = pct_data[i, j]
            raw_val = category_data[i, j]
            text_str = f"{pct_val:.1f}%\n({int(raw_val)})"
            text_color = "white" if pct_val > 50 else "black"
            ax.text(
                j, i, text_str,
                ha="center", va="center",
                color=text_color,
                fontsize=9, fontweight="medium",
            )

    ax.set_title(f"Theorem Status Distribution (p={p}, formalizer={formalizer_name})")
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Percentage (%)", rotation=-90, va="bottom")

    fig.tight_layout()
    plt.savefig(output_path)
    plt.close()
