from __future__ import annotations

from typing import Dict, Optional, Any
from pathlib import Path


class DistributionPlotter:
    """
    Pie chart + bar chart pour la distribution des classes.
    """

    def plot_pie(self, counts: Dict[str, int], title: str, save_to: Optional[Path] = None) -> None:
        raise NotImplementedError

    def plot_bar(self, counts: Dict[str, int], title: str, save_to: Optional[Path] = None) -> None:
        raise NotImplementedError


class GridPlotter:
    """
    Affiche une grille (original + variantes).
    Pratique pour Augmentation et Transformation.
    """

    def show_grid(
        self,
        title: str,
        images: Dict[str, Any],
        original: Optional[Any] = None,
        save_to: Optional[Path] = None,
        max_cols: int = 3
    ) -> None:
        raise NotImplementedError
