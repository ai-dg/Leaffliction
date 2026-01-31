from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List
import numpy as np
import cv2
import sys
import tempfile
import re
from leaffliction.dataset import DatasetScanner
from leaffliction.model import InferenceManager
from leaffliction.utils import Logger


@dataclass
class PredictConfig:
    """
    Configuration for prediction display and output options.
    """
    show_transforms: bool = True
    top_k: int = 3
    extra: Dict[str, Any] = field(default_factory=dict)


class Predictor:
    """
    Handles image prediction using a trained model and transformations.
    """

    def __init__(
            self,
            model_loader: InferenceManager,
            transformation_engine: Any,
            verbose: bool = True) -> None:
        """
        Initialize the predictor.

        :param model_loader: Model loader class for inference.
        :type model_loader: InferenceManager
        :param transformation_engine: Engine for image transformations.
        :type transformation_engine: Any
        :param verbose: Enable detailed logging.
        :type verbose: bool
        :return: None
        :rtype: None
        """
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
        """
        Predict the class of a plant leaf image.

        :param model_zip: Path to model ZIP archive (optional).
        :type model_zip: Path
        :param image_path: Path to the input image.
        :type image_path: Path
        :param model_path: Path to model directory (optional).
        :type model_path: Path
        :param cfg: Prediction configuration.
        :type cfg: PredictConfig
        :return: Tuple of (predicted_label, probabilities_dict,
            transformed_images_dict).
        :rtype: Tuple[str, Dict[str, float], Dict[str, np.ndarray]]
        """

        if model_zip is None and model_path is None:
            self.logger.error(
                "You must provide either model_zip or model_path")
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
                loader = self.model_loader.load_from_zip(
                    model_zip, Path(temp_dir))
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

    def infer_true_label(self, image_path: Path) -> Optional[str]:
        """
        Infer the ground-truth label from either the parent folder name
        or the filename.

        This function supports both dataset structures:
        - Structured folders:  .../Apple_Black_rot1/img.jpg  -> Apple_Black_rot
        - Flat folders:       .../Apple_Black_rot1.JPG       -> Apple_Black_rot

        It removes common numeric suffixes and separators such as:
        - Apple_Black_rot1, Apple_Black_rot_1, Apple_Black_rot (1),
          Apple_Black_rot-1

        :param image_path: Path to an image file.
        :type image_path: Path
        :return: Inferred class label, or None if it cannot be determined.
        :rtype: Optional[str]
        """

        def _clean(name: str) -> Optional[str]:
            if not name:
                return None

            s = name.strip()
            s = s.rsplit(".", 1)[0]
            s = re.sub(r"\s*\(\d+\)\s*$", "", s)
            s = re.sub(r"[\s_\-]*\d+\s*$", "", s)
            s = s.strip(" _-")
            return s if s else None

        parent_label = _clean(image_path.parent.name)
        if parent_label:
            return parent_label

        file_label = _clean(image_path.stem)
        if file_label:
            return file_label

        return None

    def predict_with_dir(
        self,
        dir_path: Path,
        model_zip: Path,
        model_path: Path,
        cfg: PredictConfig,
    ) -> Dict[Path, Tuple[str, str, Dict[str, float], Dict[str, np.ndarray]]]:
        """
        Predict classes for all images in a directory (recursive if
        DatasetScanner does).

        Returns the same thing as `predict()` but for each image:
        { image_path: (predicted_label, probabilities_dict,
          transformed_images_dict) }

        :param dir_path: Path to directory containing images.
        :param model_zip: Path to model ZIP archive (optional).
        :param model_path: Path to model directory (optional).
        :param cfg: Prediction configuration.
        :return: Dict mapping image path to (predicted_label, probs,
            transformed)
        """
        if not dir_path.exists() or not dir_path.is_dir():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        if model_zip is None and model_path is None:
            self.logger.error(
                "You must provide either model_zip or model_path")
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
                loader = self.model_loader.load_from_zip(
                    model_zip, Path(temp_dir))
            self.logger.info("   Model loaded successfully")
            self.logger.info()
        else:
            if not model_path.exists():
                raise FileNotFoundError(f"Model path not found: {model_path}")

            self.logger.info("Loading model loader from directory...")
            loader = self.model_loader.load(model_path)
            self.logger.info("   Model loaded successfully")
            self.logger.info()

        scanner = DatasetScanner()
        index = scanner.scan(dir_path)
        items: List[Tuple[Path, int]] = index.items

        if len(items) == 0:
            self.logger.warn(f"No images found in: {dir_path}")
            return {}

        self.logger.info(f"Found {len(items)} images in: {dir_path}")
        self.logger.info()

        self.logger.info("Labels mapping (id -> label):")
        for k, v in loader.labels.class_to_id.items():
            self.logger.info(f"{k}: {v}")
        self.logger.info("")

        results: Dict[Path, Tuple[str, str,
                                  Dict[str, float],
                                  Dict[str, np.ndarray]]] = {}

        for image_path, _true_id in items:
            self.logger.info(f"Processing image: {image_path}")

            img = cv2.imread(str(image_path))
            if img is None:
                self.logger.error(f"Could not load image: {image_path}")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, loader.cfg.img_size)

            tensor = self.transformation_engine.apply_all_as_tensor(
                img_resized)

            self.logger.info("Predicting...")
            pred_id, probs = loader.predict(tensor)
            predicted_label = loader.labels.decode(pred_id)

            true_label = self.infer_true_label(image_path) or "unknown"

            self.logger.info(
                f"   True: {true_label} | Predicted: {predicted_label}")

            transformed = {}
            if cfg.show_transforms:
                transformed = self.transformation_engine.apply_all(img_resized)

            results[image_path] = (
                true_label, predicted_label, probs, transformed)
            self.logger.info("")

        return results


class PredictionVisualiser:
    """
    Visualizes prediction results with original and transformed images.
    """

    def show(self,
             original: np.ndarray,
             transformed: Dict[str,
                               np.ndarray],
             predicted_label: str) -> None:
        """
        Display prediction results in a grid layout.

        :param original: Original input image.
        :type original: np.ndarray
        :param transformed: Dictionary of transformed images.
        :type transformed: Dict[str, np.ndarray]
        :param predicted_label: Predicted class label.
        :type predicted_label: str
        :return: None
        :rtype: None
        """
        from leaffliction.plotting import Plotter

        grid = Plotter()
        title = f"Prediction: {predicted_label}"
        grid.plot_grid(title, transformed, original=original)
