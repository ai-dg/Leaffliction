from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
import cv2
import tempfile


@dataclass
class PredictConfig:
    show_transforms: bool = True
    top_k: int = 3
    extra: Dict[str, Any] = field(default_factory=dict)


class MLPredictor:
    """
    Charge un MLModelBundle puis prédit sur une image (ML traditionnel).
    """

    def __init__(self, bundle_loader: Any, transformations_engine: Any) -> None:
        self.bundle_loader = bundle_loader
        self.transformations_engine = transformations_engine

    def predict(self, bundle_zip: Path, image_path: Path, cfg: PredictConfig) -> Tuple[str, Dict[str, float], Dict[str, np.ndarray]]:
        """
        Pipeline de prédiction ML traditionnel:
        1. Charger le bundle depuis le zip (model.pkl, scaler.pkl, labels.json)
        2. Extraire features de l'image
        3. Normaliser features avec le scaler
        4. Prédire avec le modèle ML
        5. Décoder le label
        6. (Optionnel) Appliquer transformations pour visualisation
        
        Retourne:
          - predicted_label: str (nom de la classe)
          - probs: Dict[str, float] (probabilités par classe)
          - transformed: Dict[str, np.ndarray] (transformations pour visualisation)
        """
        # 1. Charger le bundle
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = self.bundle_loader.load_from_zip(bundle_zip, Path(temp_dir))
        
        # 2. Extraire features de l'image
        features = bundle.feature_extractor.extract_features(image_path)
        
        # 3. Prédire
        pred_id, probs = bundle.predict(features)
        
        # 4. Décoder le label
        predicted_label = bundle.labels.decode(pred_id)
        
        # 5. (Optionnel) Appliquer transformations pour visualisation
        transformed = {}
        if cfg.show_transforms:
            img = cv2.imread(str(image_path))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                transformed = self.transformations_engine.apply_all(img)
        
        return predicted_label, probs, transformed


class PredictionVisualiser:
    """
    Affiche: original + transformed + résultat.
    Utilisé pour montrer les différentes vues de l'image.
    """

    def show(self, original: np.ndarray, transformed: Dict[str, np.ndarray], predicted_label: str) -> None:
        """
        Affiche une grille avec:
        - Image originale
        - Transformations (Grayscale, Canny, etc.)
        - Résultat de prédiction
        
        Utilise GridPlotter pour l'affichage.
        """
        from leaffliction.plotting import GridPlotter
        
        grid = GridPlotter()
        title = f"Prediction: {predicted_label}"
        grid.show_grid(title, transformed, original=original)
