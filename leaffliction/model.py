from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import tensorflow as tf
import keras


@dataclass
class ModelConfig:
    img_size: Tuple[int, int] = (224, 224)
    num_classes: int = 0
    seed: int = 42
    framework: str = "tf"  # tu restes sur "tf"
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelPaths:
    model_file: str = "model.keras"
    labels_file: str = "labels.json"
    config_file: str = "config.json"
    preprocess_file: str = "preprocess.json"


class LabelEncoder:
    """
    Mapping stable:
      class_name -> id
      id -> class_name
    """

    def __init__(self) -> None:
        self.class_to_id: Dict[str, int] = {}
        self.id_to_class: Dict[int, str] = {}

    def fit(self, class_names: List[str]) -> None:
        raise NotImplementedError

    def encode(self, class_name: str) -> int:
        raise NotImplementedError

    def decode(self, class_id: int) -> str:
        raise NotImplementedError

    def to_json_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def from_json_dict(cls, data: Dict[str, Any]) -> "LabelEncoder":
        raise NotImplementedError


class ModelFactory:
    """
    Construit un modèle (TensorFlow/Keras).
    """

    def build(self, cfg: ModelConfig) -> keras.Model:
        raise NotImplementedError


class ModelBundle:
    """
    Ce qui est sauvegardé pour predict:
    - modèle (model.keras)
    - label encoder (labels.json)
    - config (config.json)
    - preprocess (preprocess.json)
    """

    def __init__(
        self,
        model: keras.Model,
        labels: LabelEncoder,
        cfg: ModelConfig,
        preprocess: Optional[Dict[str, Any]] = None,
        paths: Optional[ModelPaths] = None
    ) -> None:
        self.model = model
        self.labels = labels
        self.cfg = cfg
        self.preprocess = preprocess or {}
        self.paths = paths or ModelPaths()

    def save(self, out_dir: Path) -> None:
        """
        out_dir/
          model.keras
          labels.json
          config.json
          preprocess.json
        """
        raise NotImplementedError

    @classmethod
    def load(cls, in_dir: Path) -> "ModelBundle":
        """
        model = keras.models.load_model(...)
        """
        raise NotImplementedError

    @classmethod
    def load_from_zip(cls, zip_path: Path, extract_dir: Path) -> "ModelBundle":
        raise NotImplementedError
