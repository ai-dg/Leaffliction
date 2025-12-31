from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import tensorflow as tf


@dataclass
class DatasetIndex:
    """
    Représentation standard du dataset:
    - root: dossier racine
    - class_names: nom des sous-dossiers (classes)
    - items: [(path_image, class_id), ...]
    - counts: {class_name: count}
    """
    root: Path
    class_names: List[str]
    items: List[Tuple[Path, int]]
    counts: Dict[str, int]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def size(self) -> int:
        return len(self.items)


class DatasetScanner:
    """
    Scan un dataset de type:
    root/
      class_a/
        img1.jpg
      class_b/
        img2.jpg
    """

    def scan(self, root: Path) -> DatasetIndex:
        """Construit un DatasetIndex depuis un dossier racine."""
        raise NotImplementedError


class DatasetSplitter:
    """
    Split train/valid à partir de la liste items.
    """

    def split(
        self,
        items: List[Tuple[Path, int]],
        valid_ratio: float,
        seed: int,
        stratified: bool = True
    ) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]:
        """
        Retourne (train_items, valid_items)
        - stratified: conserve approx les proportions de classes
        """
        raise NotImplementedError


@dataclass
class TFDataConfig:
    img_size: Tuple[int, int] = (224, 224)
    batch_size: int = 32
    shuffle: bool = True
    seed: int = 42
    cache: bool = False
    prefetch: bool = True


class TFDatasetBuilder:
    """
    Transforme une liste [(Path, label_id), ...] en tf.data.Dataset.
    augmentor: typiquement un keras.Model / keras.Sequential d'augmentations.
    """

    def __init__(self, cfg: TFDataConfig, augmentor: Optional[Any] = None) -> None:
        self.cfg = cfg
        self.augmentor = augmentor

    def build(
        self,
        items: List[Tuple[Path, int]],
        training: bool
    ) -> tf.data.Dataset:
        """
        Retourne un tf.data.Dataset qui yield (image_tensor, label_id)
        image_tensor: tf.float32, shape (H,W,3)
        label: tf.int32
        """
        raise NotImplementedError
