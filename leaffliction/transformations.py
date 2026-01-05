from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Protocol, Tuple
import numpy as np
import cv2
import torch


class Transformation(Protocol):
    """
    Interface d'une transformation.
    Utilisé pour créer des canaux de features pour PyTorch.
    """
    
    @property
    def name(self) -> str:
        ...

    def apply(self, img: np.ndarray) -> np.ndarray:
        ...


@dataclass
class GrayscaleTf:
    name: str = "Grayscale"

    def apply(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@dataclass
class CannyEdgesTf:
    threshold1: float = 100
    threshold2: float = 200
    name: str = "Canny"

    def apply(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@dataclass
class HistogramEqualisationTf:
    name: str = "HistEq"

    def apply(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@dataclass
class SharpenTf:
    name: str = "Sharpen"

    def apply(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@dataclass
class ThresholdTf:
    threshold: int = 127
    name: str = "Threshold"

    def apply(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@dataclass
class MorphologyTf:
    mode: str = "erode"  # ou "dilate", "open", "close"
    kernel_size: int = 5
    name: str = "Morphology"

    def apply(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class TransformationEngine:
    """
    Moteur de transformations.
    Utilisé pour:
    1. Visualisation (Transformation.py) - apply_all()
    2. Création de tensors PyTorch (train/predict) - apply_all_as_tensor()
    """
    
    def __init__(self, tfs: List[Transformation]) -> None:
        self.tfs = tfs

    @classmethod
    def default_six(cls) -> "TransformationEngine":
        """Factory: les 6 transformations par défaut"""
        raise NotImplementedError

    def apply_all(self, img: np.ndarray) -> Dict[str, np.ndarray]:
        """Applique toutes les transformations pour visualisation"""
        raise NotImplementedError
    
    def apply_all_as_tensor(self, img: np.ndarray) -> torch.Tensor:
        """
        Applique toutes les transformations et retourne un tensor PyTorch.
        
        Args:
            img: Image RGB (H, W, 3)
        
        Returns:
            tensor: (n_transforms, H, W) avec les transformations comme canaux
        """
        channels = []
        
        for tf in self.tfs:
            # Appliquer transformation
            transformed = tf.apply(img)
            
            # Convertir en grayscale si nécessaire
            if len(transformed.shape) == 3:
                transformed = cv2.cvtColor(transformed, cv2.COLOR_RGB2GRAY)
            
            # Normaliser [0, 255] → [0, 1]
            transformed = transformed.astype(np.float32) / 255.0
            
            channels.append(transformed)
        
        # Stack en tensor (n_transforms, H, W)
        tensor = torch.from_numpy(np.stack(channels, axis=0))
        return tensor
    
    def batch_transform(
        self, 
        items: List[Tuple[Path, int]], 
        img_size: Tuple[int, int] = (224, 224)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transforme un batch d'images en tensors PyTorch.
        
        Args:
            items: [(path, label), ...]
            img_size: (H, W) taille de redimensionnement
        
        Returns:
            X: (n, n_transforms, H, W) tensor des transformations
            y: (n,) tensor des labels
        """
        X_list = []
        y_list = []
        
        for img_path, label in items:
            # Charger image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"⚠️  Warning: Could not load {img_path}")
                continue
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size)
            
            # Transformer en tensor
            tensor = self.apply_all_as_tensor(img)
            
            X_list.append(tensor)
            y_list.append(label)
        
        # Stack en batch
        X = torch.stack(X_list)  # (n, n_transforms, H, W)
        y = torch.tensor(y_list, dtype=torch.long)  # (n,)
        
        return X, y


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
        """
        Transforme toutes les images de src et sauvegarde dans dst.
        """
        raise NotImplementedError
