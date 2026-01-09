from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Protocol, Tuple
import numpy as np
import albumentations as A

# TODO - remove commented out code & every reference/dependencies in the
#        codebase
# class Augmentation(Protocol):
#     """
#     Interface d'une augmentation.
#     'name' sert au suffix (_Flip, _Rotate, etc.)
#     """

#     @property
#     def name(self) -> str:
#         ...

#     def apply(self, img: np.ndarray) -> np.ndarray:
#         ...


# @dataclass
# class FlipHorizontalAug:
#     name: str = "FlipH"

#     def apply(self, img: np.ndarray) -> np.ndarray:
#         raise NotImplementedError


# @dataclass
# class FlipVerticalAug:
#     name: str = "FlipV"

#     def apply(self, img: np.ndarray) -> np.ndarray:
#         raise NotImplementedError


# @dataclass
# class RotateAug:
#     angle: float
#     name: str = "Rotate"

#     def apply(self, img: np.ndarray) -> np.ndarray:
#         raise NotImplementedError


# @dataclass
# class BrightnessContrastAug:
#     brightness: float = 0.0
#     contrast: float = 0.0
#     name: str = "BrightContrast"

#     def apply(self, img: np.ndarray) -> np.ndarray:
#         raise NotImplementedError


# @dataclass
# class GaussianBlurAug:
#     sigma: float = 1.0
#     name: str = "Blur"

#     def apply(self, img: np.ndarray) -> np.ndarray:
#         raise NotImplementedError


# @dataclass
# class RandomCropResizeAug:
#     crop_ratio: float = 0.9
#     name: str = "CropResize"

#     def apply(self, img: np.ndarray) -> np.ndarray:
#         raise NotImplementedError


class AugmentationEngine:
    """
    Applique une liste d'augmentations et retourne un dict:
    { "FlipH": img1, "Rotate": img2, ... }
    
    Utilisé pour:
    1. Visualisation (Augmentation.py) - apply_all()
    2. Augmentation du dataset de training - augment_dataset()
    """

    augs = {
        "FlipH": A.HorizontalFlip(p=1.0),
        "Rotate": A.Rotate(
            limit=(-15,15),
            p=1.0
        ),
        "Shear": None,
        "Blur": None,
        "Crop": None,
        "Distortion": None,
    }

    # TODO - Remove default_six everywhere in the codebase and refac
    # @classmethod
    # def default_six(cls) -> "AugmentationEngine":
    #     """
    #     Factory: les 6 augmentations mandatory.
    #     """
    #     raise NotImplementedError

    def apply_all(self, img: np.ndarray) -> Dict[str, np.ndarray]:
        """Applique toutes les augmentations pour visualisation"""
        return { name: result["image"] for (name, result) in self.augs.items()}
    
    # TODO - Remove apply_random everywhere in the codebase and refac
    # def apply_random(self, img: np.ndarray, n: int = 2) -> np.ndarray:
    #     """Applique n augmentations aléatoires"""
    #     raise NotImplementedError
    
    def augment_dataset(
        self,
        train_items: List[Tuple[Path, int]],
        train_items_grouped: Dict[int, List[Tuple[Path, int]]],
        output_dir: Path,
    ) -> List[Tuple[Path, int]]:
        """
        Pour chaque image de train:
        1. Applique N augmentations différentes
        2. Sauvegarde les nouvelles images sur disque
        3. Retourne la liste étendue: originales + augmentées
        
        Exemple:
        Input: 400 images Apple_healthy
        Output: 400 originales + 1200 augmentées = 1600 images
        """
        raise NotImplementedError


class AugmentationSaver:
    """
    Sauvegarde les images augmentées dans le même dossier
    avec suffixes conformes au sujet.
    Utilisé par Augmentation.py pour la visualisation.
    """

    def __init__(self, path_manager: Any) -> None:
        self.path_manager = path_manager

    def save_all(self, image_path: Path, results: Dict[str, np.ndarray]) -> List[Path]:
        """
        Renvoie la liste des paths écrits.
        Exemple attendu: image (1)_Flip.JPG, image (1)_Rotate.JPG, etc.
        """
        raise NotImplementedError
