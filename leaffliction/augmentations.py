from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Protocol
import tensorflow as tf
import keras


class KerasAugmentationsFactory:
    def build(self) -> keras.Sequential:
        """
        Retourne un keras.Sequential de layers d'augmentation
        (RandomFlip, RandomRotation, RandomZoom, etc.)
        Utilisé uniquement en training.
        """
        raise NotImplementedError


class Augmentation(Protocol):
    """
    Interface d'une augmentation.
    'name' sert au suffix (_Flip, _Rotate, etc.)
    """

    @property
    def name(self) -> str:
        ...

    def apply(self, img: Any) -> Any:
        ...


@dataclass
class FlipHorizontalAug:
    name: str = "FlipH"

    def apply(self, img: Any) -> Any:
        raise NotImplementedError


@dataclass
class FlipVerticalAug:
    name: str = "FlipV"

    def apply(self, img: Any) -> Any:
        raise NotImplementedError


@dataclass
class RotateAug:
    angle: float
    name: str = "Rotate"

    def apply(self, img: Any) -> Any:
        raise NotImplementedError


@dataclass
class BrightnessContrastAug:
    brightness: float = 0.0
    contrast: float = 0.0
    name: str = "BrightContrast"

    def apply(self, img: Any) -> Any:
        raise NotImplementedError


@dataclass
class GaussianBlurAug:
    sigma: float = 1.0
    name: str = "Blur"

    def apply(self, img: Any) -> Any:
        raise NotImplementedError


@dataclass
class RandomCropResizeAug:
    crop_ratio: float = 0.9
    name: str = "CropResize"

    def apply(self, img: Any) -> Any:
        raise NotImplementedError


class AugmentationEngine:
    """
    Applique une liste d'augmentations et retourne un dict:
    { "FlipH": img1, "Rotate": img2, ... }
    """

    def __init__(self, augs: List[Augmentation]) -> None:
        self.augs = augs

    @classmethod
    def default_six(cls) -> "AugmentationEngine":
        """
        Factory: les 6 augmentations mandatory.
        Ajuste les paramètres si tu veux.
        """
        raise NotImplementedError

    def apply_all(self, img: Any) -> Dict[str, Any]:
        raise NotImplementedError


class AugmentationSaver:
    """
    Sauvegarde les images augmentées dans le même dossier
    avec suffixes conformes au sujet.
    """

    def __init__(self, path_manager: Any) -> None:
        self.path_manager = path_manager

    def save_all(self, image_path: Path, results: Dict[str, Any]) -> List[Path]:
        """
        Renvoie la liste des paths écrits.
        Exemple attendu: image (1)_Flip.JPG, image (1)_Rotate.JPG, etc.
        """
        raise NotImplementedError
