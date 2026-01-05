from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
import cv2
import torch
import tempfile


@dataclass
class PredictConfig:
    show_transforms: bool = True
    top_k: int = 3
    extra: Dict[str, Any] = field(default_factory=dict)


class PyTorchPredictor:
    """
    Charge un PyTorchModelBundle puis prédit sur une image.
    """

    def __init__(self, bundle_loader: Any, transformation_engine: Any) -> None:
        self.bundle_loader = bundle_loader
        self.transformation_engine = transformation_engine

    def predict(
        self, 
        bundle_zip: Path, 
        image_path: Path, 
        cfg: PredictConfig
    ) -> Tuple[str, Dict[str, float], Dict[str, np.ndarray]]:
        """
        Pipeline de prédiction PyTorch:
        1. Charger le bundle depuis le zip (model.pth, labels.json)
        2. Charger et transformer l'image en tensor
        3. Prédire avec le modèle PyTorch
        4. Décoder le label
        5. (Optionnel) Appliquer transformations pour visualisation
        
        Retourne:
          - predicted_label: str (nom de la classe)
          - probs: Dict[str, float] (probabilités par classe)
          - transformed: Dict[str, np.ndarray] (transformations pour visualisation)
        """
        # 1. Charger le bundle
        print("📦 Loading model bundle...")
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = self.bundle_loader.load_from_zip(bundle_zip, Path(temp_dir))
        print("   Model loaded successfully")
        print()
        
        # 2. Charger et transformer l'image
        print("🔍 Processing image...")
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, bundle.cfg.img_size)
        
        # Créer tensor avec transformations
        tensor = self.transformation_engine.apply_all_as_tensor(img_resized)
        print(f"   Tensor shape: {tensor.shape}")
        print()
        
        # 3. Prédire
        print("🎯 Predicting...")
        pred_id, probs = bundle.predict(tensor)
        predicted_label = bundle.labels.decode(pred_id)
        print(f"   Predicted: {predicted_label}")
        print()
        
        # 4. (Optionnel) Appliquer transformations pour visualisation
        transformed = {}
        if cfg.show_transforms:
            transformed = self.transformation_engine.apply_all(img_resized)
        
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
