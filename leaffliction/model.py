from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class ModelConfig:
    """Configuration du modèle ML"""
    num_classes: int = 0
    seed: int = 42
    model_type: str = "svm"  # "svm", "random_forest", "knn"
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelPaths:
    """Chemins des fichiers du bundle"""
    model_file: str = "model.pkl"
    scaler_file: str = "scaler.pkl"
    labels_file: str = "labels.json"
    config_file: str = "config.json"
    feature_config_file: str = "feature_config.json"


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
        """Crée le mapping depuis une liste de noms de classes"""
        raise NotImplementedError

    def encode(self, class_name: str) -> int:
        """Convertit un nom de classe en ID"""
        raise NotImplementedError

    def decode(self, class_id: int) -> str:
        """Convertit un ID en nom de classe"""
        raise NotImplementedError

    def to_json_dict(self) -> Dict[str, Any]:
        """Sérialise en dict pour JSON"""
        raise NotImplementedError

    @classmethod
    def from_json_dict(cls, data: Dict[str, Any]) -> "LabelEncoder":
        """Désérialise depuis un dict JSON"""
        raise NotImplementedError


class MLModelFactory:
    """
    Construit un modèle ML traditionnel (sklearn).
    Supporte: SVM, Random Forest, KNN
    """

    def build(self, cfg: ModelConfig) -> Any:
        """
        Construit un modèle sklearn selon cfg.model_type
        
        model_type:
        - "svm": SVC avec kernel RBF
        - "random_forest": RandomForestClassifier
        - "knn": KNeighborsClassifier
        
        Retourne: modèle sklearn non entraîné
        """
        raise NotImplementedError


class MLModelBundle:
    """
    Bundle complet pour sauvegarder/charger un modèle ML.
    
    Contient:
    - model: modèle sklearn entraîné
    - scaler: StandardScaler pour normaliser les features
    - labels: LabelEncoder pour les classes
    - feature_extractor: FeatureExtractor pour extraire features
    - cfg: ModelConfig
    """

    def __init__(
        self,
        model: Any,  # sklearn model
        scaler: Any,  # StandardScaler
        labels: LabelEncoder,
        feature_extractor: Any,  # FeatureExtractor
        cfg: ModelConfig,
        paths: Optional[ModelPaths] = None
    ) -> None:
        self.model = model
        self.scaler = scaler
        self.labels = labels
        self.feature_extractor = feature_extractor
        self.cfg = cfg
        self.paths = paths or ModelPaths()

    def save(self, out_dir: Path) -> None:
        """
        Sauvegarde le bundle dans out_dir/:
        - model.pkl (sklearn model avec joblib)
        - scaler.pkl (StandardScaler avec joblib)
        - labels.json
        - config.json
        - feature_config.json
        """
        raise NotImplementedError

    @classmethod
    def load(cls, in_dir: Path) -> "MLModelBundle":
        """
        Charge le bundle depuis in_dir/
        """
        raise NotImplementedError

    @classmethod
    def load_from_zip(cls, zip_path: Path, extract_dir: Optional[Path] = None) -> "MLModelBundle":
        """
        Extrait le zip puis charge le bundle.
        """
        raise NotImplementedError
    
    def predict(self, features: np.ndarray) -> Tuple[int, Dict[str, float]]:
        """
        Prédit la classe depuis un vecteur de features.
        
        Args:
            features: np.ndarray de shape (n_features,) ou (1, n_features)
        
        Retourne:
            - pred_id: int (ID de la classe prédite)
            - probs: Dict[str, float] (probabilités par classe)
        """
        raise NotImplementedError
