from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
import keras
import tensorflow as tf


@dataclass
class PredictConfig:
    show_transforms: bool = True
    top_k: int = 1
    extra: Dict[str, Any] = field(default_factory=dict)


class Predictor:
    """
    Charge un ModelBundle puis prédit sur une image.
    """

    def __init__(self, bundle_loader: Any, transformations_engine: Any) -> None:
        self.bundle_loader = bundle_loader
        self.transformations_engine = transformations_engine

    def predict(self, bundle_zip: Path, image_path: Path, cfg: PredictConfig) -> Tuple[str, Dict[str, float]]:
        """
        Retour:
          (predicted_label, {label: prob, ...})
        """
        raise NotImplementedError


class PredictionVisualiser:
    """
    Affiche: original + transformed + résultat.
    """

    def show(self, original: Any, transformed: Dict[str, Any], predicted_label: str) -> None:
        raise NotImplementedError
