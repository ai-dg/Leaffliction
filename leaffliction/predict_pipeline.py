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
from leaffliction.utils import Logger


@dataclass
class PredictConfig:
    show_transforms: bool = True
    top_k: int = 3
    extra: Dict[str, Any] = field(default_factory=dict)


class Predictor:
    def __init__(self, model_loader: InferenceManager, transformation_engine: Any, verbose : bool = True) -> None:
        self.model_loader = model_loader
        self.transformation_engine = transformation_engine
        self.verbose = verbose
        self.logger = Logger(self.verbose)

    def predict(
        self, 
        model_zip: Path, 
        image_path: Path,
        model_path: Path, 
        cfg: PredictConfig
    ) -> Tuple[str, Dict[str, float], Dict[str, np.ndarray]]:

        if model_zip is None and model_path is None:
            self.logger.error("You must provide either model_zip or model_path")
            sys.exit(1)

        if model_zip is not None and model_path is not None:
            self.logger.error("Choose ONE: model_zip OR model_path (not both)")
            sys.exit(1)


        if model_zip is not None:
            if not model_zip.exists():
                self.logger.error(f"loader zip not found: {model_zip}")
                sys.exit(1)

            self.logger.info("Loading model loader from zip...")
            with tempfile.TemporaryDirectory() as temp_dir:
                loader = self.model_loader.load_from_zip(model_zip, Path(temp_dir))
            self.logger.info("   Model loaded successfully")
            self.logger.info()
        else:
            if not model_path.exists():
                raise FileNotFoundError(f"Model path not found: {model_path}")

            self.logger.info("Loading model loader from directory...")
            loader = self.model_loader.load(model_path)
            self.logger.info("   Model loaded successfully")
            self.logger.info()
        
        self.logger.info("Processing image...")
        img = cv2.imread(str(image_path))
        if img is None:
            self.logger.error(f"Could not load image: {image_path}")
            sys.exit(1)
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, loader.cfg.img_size)
        
        tensor = self.transformation_engine.apply_all_as_tensor(img_resized)
        self.logger.info(f"Tensor shape: {tensor.shape}")
        self.logger.info()

        self.logger.info("Labels mapping (id -> label):")
        for k, v in loader.labels.class_to_id.items():
            self.logger.info(f"{k}: {v}")
        self.logger.info("")
        
        self.logger.info("Predicting...")
        pred_id, probs = loader.predict(tensor)
        predicted_label = loader.labels.decode(pred_id)
        self.logger.info(f"   Predicted: {predicted_label}")
        self.logger.info()

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
    def show(self, original: np.ndarray, transformed: Dict[str, np.ndarray], predicted_label: str) -> None:
        from leaffliction.plotting import Plotter
        
        grid = Plotter()
        title = f"Prediction: {predicted_label}"
        grid.plot_grid(title, transformed, original=original)
