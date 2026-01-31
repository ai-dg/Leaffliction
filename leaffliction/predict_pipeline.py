from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
import cv2
import torch
import sys
import tempfile
from Distribution import DatasetScanner
from leaffliction.model import InferenceManager


@dataclass
class PredictConfig:
    show_transforms: bool = True
    top_k: int = 3
    extra: Dict[str, Any] = field(default=dict)


class Predictor:
    def __init__(self, model_loader: InferenceManager, transformation_engine: Any) -> None:
        self.model_loader = model_loader
        self.transformation_engine = transformation_engine

    def predict(
        self, 
        bundle_zip: Path, 
        image_path: Path,
        model_path: Path, 
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

        
        print(f"bundle_zip={bundle_zip}")
        print(f"model_path={model_path}")

        # 1) validationvprint = Logger(self.verbose)
        if bundle_zip is None and model_path is None:
            raise ValueError("You must provide either bundle_zip or model_path")

        if bundle_zip is not None and model_path is not None:
            raise ValueError("Choose ONE: bundle_zip OR model_path (not both)")

        # 2) load
        if bundle_zip is not None:
            if not bundle_zip.exists():
                raise FileNotFoundError(f"Bundle zip not found: {bundle_zip}")

            print("Loading model bundle from zip...")
            with tempfile.TemporaryDirectory() as temp_dir:
                bundle = self.model_loader.load_from_zip(bundle_zip, Path(temp_dir))
            print("   Model loaded successfully")
            print()

        else:  # model_path is not None
            if not model_path.exists():
                raise FileNotFoundError(f"Model path not found: {model_path}")

            print("Loading model bundle from directory...")
            bundle = self.model_loader.load(model_path)
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
        print(f"Tensor shape: {tensor.shape}")
        print()

        print("Labels mapping (id -> label):")
        print(bundle.labels.class_to_id)
        print(bundle.labels.id_to_class)

        
        # 3. Prédire
        print("Predicting...")
        pred_id, probs = bundle.predict(tensor)
        predicted_label = bundle.labels.decode(pred_id)
        print(f"   Predicted: {predicted_label}")
        print()

        
        
        # 4. (Optionnel) Appliquer transformations pour visualisation
        transformed = {}
        if cfg.show_transforms:
            transformed = self.transformation_engine.apply_all(img_resized)
        
        return predicted_label, probs, transformed
    
    def predict_with_dir(
            self,
            dir_path : Path,
        ):
        scanner = DatasetScanner()
        index = scanner.scan(dir_path)
        items = index.items







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
        
        Utilise Plotter pour l'affichage.
        """
        from leaffliction.plotting import Plotter
        
        grid = Plotter()
        title = f"Prediction: {predicted_label}"
        grid.plot_grid(title, transformed, original=original)
