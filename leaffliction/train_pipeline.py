from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple, List
import tensorflow as tf
import keras


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-3
    valid_ratio: float = 0.2
    seed: int = 42
    img_size: Tuple[int, int] = (224, 224)
    augment_in_train: bool = True
    export_increased_images: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Metrics:
    train_accuracy: float
    valid_accuracy: float
    valid_count: int
    notes: Dict[str, Any] = field(default_factory=dict)


class Trainer:
    """
    Orchestrateur d'entraînement.
    """

    def __init__(self, dataset_scanner: Any, dataset_splitter: Any, model_factory: Any, labels: Any) -> None:
        self.dataset_scanner = dataset_scanner
        self.dataset_splitter = dataset_splitter
        self.model_factory = model_factory
        self.labels = labels

    def train(self, dataset_dir: Path, out_dir: Path, cfg: TrainConfig) -> Metrics:
        """
        - scan dataset
        - split train/valid
        - build tf.data datasets
        - build keras model
        - compile + fit
        - evaluate accuracy
        - save bundle
        """
        raise NotImplementedError


class TrainingPackager:
    """
    Prépare les artefacts (model, labels, cfg, images augmentées si demandé)
    puis zip le tout.
    """

    def __init__(self, zip_packager: Any) -> None:
        self.zip_packager = zip_packager

    def prepare_artifacts_dir(self, tmp_dir: Path) -> Path:
        raise NotImplementedError

    def build_zip(self, artifacts_dir: Path, out_zip: Path) -> None:
        raise NotImplementedError


class RequirementsGate:
    """
    Valide les contraintes:
    - valid_accuracy > 0.90
    - valid_count >= 100
    """

    def assert_ok(self, metrics: Metrics) -> None:
        raise NotImplementedError


class KerasCallbacksFactory:
    def build(self, out_dir: Path) -> List[keras.callbacks.Callback]:
        """
        EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard (optionnel)
        """
        raise NotImplementedError
