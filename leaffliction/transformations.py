from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Protocol
import tensorflow as tf


class Transformation(Protocol):
    @property
    def name(self) -> str:
        ...

    def apply(self, img: Any) -> Any:
        ...


@dataclass
class GrayscaleTf:
    name: str = "Grayscale"

    def apply(self, img: Any) -> Any:
        raise NotImplementedError


@dataclass
class CannyEdgesTf:
    name: str = "Canny"

    def apply(self, img: Any) -> Any:
        raise NotImplementedError


@dataclass
class HistogramEqualisationTf:
    name: str = "HistEq"

    def apply(self, img: Any) -> Any:
        raise NotImplementedError


@dataclass
class SharpenTf:
    name: str = "Sharpen"

    def apply(self, img: Any) -> Any:
        raise NotImplementedError


@dataclass
class ThresholdTf:
    name: str = "Threshold"

    def apply(self, img: Any) -> Any:
        raise NotImplementedError


@dataclass
class MorphologyTf:
    mode: str = "erode"  # ou "dilate", "open", "close"
    name: str = "Morphology"

    def apply(self, img: Any) -> Any:
        raise NotImplementedError


class TransformationEngine:
    def __init__(self, tfs: List[Transformation]) -> None:
        self.tfs = tfs

    @classmethod
    def default_six(cls) -> "TransformationEngine":
        raise NotImplementedError

    def apply_all(self, img: tf.Tensor) -> Dict[str, tf.Tensor]:
        raise NotImplementedError


class BatchTransformer:
    """
    Mode dossier:
      Transformation.py -src ... -dst ...
    Sauvegarde toutes les transformations dans dst.
    """

    def __init__(self, engine: TransformationEngine, path_manager: Any) -> None:
        self.engine = engine
        self.path_manager = path_manager

    def run(self, src: Path, dst: Path, recursive: bool = True) -> None:
        raise NotImplementedError
