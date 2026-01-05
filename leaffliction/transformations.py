from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Protocol, Tuple
import numpy as np


class Transformation(Protocol):
    """
    Interface d'une transformation.
    Utilisé pour extraire des features des images.
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
    2. Extraction de features (train/predict) - via FeatureExtractor
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


class FeatureExtractor:
    """
    Extrait des features numériques depuis une image.
    Ces features seront utilisées par le modèle ML (SVM, Random Forest, etc.)
    
    Features extraites:
    - Histogrammes couleur
    - Statistiques des transformations
    - Textures (optionnel)
    - Contours (optionnel)
    """
    
    def __init__(self, transformations: List[Transformation]) -> None:
        self.transformations = transformations
    
    def extract_features(self, img_path: Path) -> np.ndarray:
        """
        Extrait un vecteur de features depuis une image.
        
        Retourne: np.ndarray de shape (n_features,)
        
        Exemple de features:
        - Histogramme couleur (256 bins × 3 channels = 768 features)
        - Histogramme grayscale (256 features)
        - Statistiques (mean, std, min, max par transformation)
        - Total: ~1000-2000 features
        """
        raise NotImplementedError
    
    def extract_batch(
        self,
        items: List[Tuple[Path, int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extrait features pour un batch d'images.
        
        Retourne:
        - X: np.ndarray de shape (n_samples, n_features)
        - y: np.ndarray de shape (n_samples,)
        """
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
        """
        Transforme toutes les images de src et sauvegarde dans dst.
        """
        raise NotImplementedError
