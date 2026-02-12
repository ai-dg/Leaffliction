from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import zipfile
import tempfile
import shutil
import random

import numpy as np
import torch
import torch.nn as nn
from leaffliction.utils import Logger
import sys
from leaffliction.transformations import TransformationEngine


@dataclass
class ModelConfig:
    """
    Configuration parameters for the neural network model.
    """
    num_classes: int = 0
    input_channels: int = 4
    img_size: Tuple[int, int] = (224, 224)
    seed: int = 42
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelPaths:
    """
    File paths for model artifacts (weights, labels, config).
    """
    model_file: str = "model.pth"
    labels_file: str = "labels.json"
    config_file: str = "config.json"


class LabelMapper:
    """
    Maps between class names and numeric IDs for classification.
    """

    def __init__(self, verbose: bool = True) -> None:
        """
        Initialize the label mapper.

        :param verbose: Enable detailed logging output.
        :type verbose: bool
        :return: None
        :rtype: None
        """
        self.class_to_id: Dict[str, int] = {}
        self.id_to_class: Dict[int, str] = {}
        self.verbose = verbose
        self.logger = Logger(verbose)

    def fit(self, class_names: List[str]) -> None:
        """
        Build the mapping from a list of class names.

        :param class_names: List of class name strings.
        :type class_names: List[str]
        :return: None
        :rtype: None
        """
        uniq = sorted(set(class_names))
        self.class_to_id = {name: i for i, name in enumerate(uniq)}
        self.id_to_class = {i: name for name, i in self.class_to_id.items()}

    def encode(self, class_name: str) -> int:
        """
        Convert a class name to its numeric ID.

        :param class_name: Name of the class to encode.
        :type class_name: str
        :return: Numeric class ID.
        :rtype: int
        """
        if class_name not in self.class_to_id:
            self.logger.error(
                f"Unknown class_name: {class_name}. "
                f"Known: {list(self.class_to_id.keys())}")
            sys.exit(1)
        return self.class_to_id[class_name]

    def decode(self, class_id: int) -> str:
        """
        Convert a numeric ID back to its class name.

        :param class_id: Numeric class ID to decode.
        :type class_id: int
        :return: Class name string.
        :rtype: str
        """
        if class_id not in self.id_to_class:
            self.logger.error(
                f"Unknown class_id: {class_id}. "
                f"Known: {list(self.id_to_class.keys())}")
            sys.exit(1)
        return self.id_to_class[class_id]

    def to_json_dict(self) -> Dict[str, Any]:
        """
        Export the mapping to a JSON-serializable dictionary.

        :return: Dictionary containing class-to-ID mapping.
        :rtype: Dict[str, Any]
        """
        return {"class_to_id": self.class_to_id}

    @classmethod
    def from_json_dict(cls, data: Dict[str, Any]) -> "LabelMapper":
        """
        Create a LabelMapper from a JSON dictionary.

        :param data: Dictionary containing class-to-ID mapping.
        :type data: Dict[str, Any]
        :return: Initialized LabelMapper instance.
        :rtype: LabelMapper
        """
        enc = cls()
        c2i = data.get("class_to_id", {})
        if not isinstance(c2i, dict):
            raise ValueError("labels.json: 'class_to_id' must be a dict.")
        enc.class_to_id = {str(k): int(v) for k, v in c2i.items()}
        enc.id_to_class = {int(v): str(k) for k, v in enc.class_to_id.items()}
        return enc


class ConvolutionalNeuralNetwork(nn.Module):
    """
    Custom CNN architecture for plant disease classification.
    """

    def __init__(self, num_classes: int, input_channels: int = 6):
        """
        Initialize the CNN with specified architecture.

        :param num_classes: Number of output classes.
        :type num_classes: int
        :param input_channels: Number of input channels (default: 6).
        :type input_channels: int
        :return: None
        :rtype: None
        """
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        Forward pass through the network.

        :param x: Input tensor of shape (batch, channels, height, width).
        :type x: torch.Tensor
        :return: Output logits of shape (batch, num_classes).
        :rtype: torch.Tensor
        """
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x


def _set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    :param seed: Random seed value.
    :type seed: int
    :return: None
    :rtype: None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class InferenceManager:
    """
    Manages model loading, saving, and inference operations.
    """

    def __init__(
        self,
        model: ConvolutionalNeuralNetwork,
        labels: LabelMapper,
        transformation_engine: Optional[TransformationEngine],
        cfg: ModelConfig,
        paths: Optional[ModelPaths] = None,
        verbose: bool = True
    ) -> None:
        """
        Initialize the inference manager.

        :param model: The neural network model.
        :type model: ConvolutionalNeuralNetwork
        :param labels: Label mapper for class encoding/decoding.
        :type labels: LabelMapper
        :param transformation_engine: Engine for image transformations.
        :type transformation_engine: Optional[TransformationEngine]
        :param cfg: Model configuration.
        :type cfg: ModelConfig
        :param paths: File paths for model artifacts.
        :type paths: Optional[ModelPaths]
        :param verbose: Enable detailed logging.
        :type verbose: bool
        :return: None
        :rtype: None
        """
        self.model = model
        self.labels = labels
        self.transformation_engine = transformation_engine
        self.cfg = cfg
        self.paths = paths or ModelPaths()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.verbose = verbose
        self.logger = Logger(self.verbose)

    def save(self, out_dir: Path) -> None:
        """
        Save model, config, and labels to disk.

        :param out_dir: Output directory for model artifacts.
        :type out_dir: Path
        :return: None
        :rtype: None
        """
        out_dir.mkdir(parents=True, exist_ok=True)

        cfg_dict = {
            "num_classes": self.cfg.num_classes,
            "input_channels": self.cfg.input_channels,
            "img_size": list(self.cfg.img_size),
            "seed": self.cfg.seed,
            "extra": self.cfg.extra,
        }
        (out_dir / self.paths.config_file).write_text(
            json.dumps(cfg_dict, indent=2),
            encoding="utf-8")

        (
            out_dir /
            self.paths.labels_file).write_text(
            json.dumps(
                self.labels.to_json_dict(),
                indent=2,
                ensure_ascii=False),
            encoding="utf-8")

        model_path = out_dir / self.paths.model_file
        torch.save(self.model.state_dict(), model_path)

    @classmethod
    def load(cls, in_dir: Path, verbose: bool = True) -> "InferenceManager":
        """
        Load a model from a directory.

        :param in_dir: Directory containing model artifacts.
        :type in_dir: Path
        :param verbose: Enable detailed logging.
        :type verbose: bool
        :return: Loaded InferenceManager instance.
        :rtype: InferenceManager
        """
        in_dir = Path(in_dir)
        paths = ModelPaths()
        logger = Logger(verbose=verbose)

        cfg_path = in_dir / paths.config_file
        labels_path = in_dir / paths.labels_file
        model_path = in_dir / paths.model_file

        if not cfg_path.exists():
            logger.error(f"Missing config file: {cfg_path}")
            sys.exit(1)

        if not labels_path.exists():
            logger.error(f"Missing labels file: {labels_path}")
            sys.exit(1)

        if not model_path.exists():
            logger.error(f"Missing model file: {model_path}")
            sys.exit(1)

        cfg_data = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg = ModelConfig(
            num_classes=int(cfg_data["num_classes"]),
            input_channels=int(cfg_data.get("input_channels", 6)),
            img_size=tuple(cfg_data.get("img_size", [224, 224])),
            seed=int(cfg_data.get("seed", 42)),
            extra=dict(cfg_data.get("extra", {})),
        )
        _set_seed(cfg.seed)

        labels_data = json.loads(labels_path.read_text(encoding="utf-8"))
        labels = LabelMapper.from_json_dict(labels_data)

        model = model = ConvolutionalNeuralNetwork(
            num_classes=cfg.num_classes,
            input_channels=cfg.input_channels
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()

        bundle = cls(
            model=model,
            labels=labels,
            transformation_engine=None,
            cfg=cfg,
            paths=paths,
            verbose=verbose)
        return bundle

    @classmethod
    def load_from_zip(
            cls,
            zip_path: Path,
            extract_dir: Optional[Path] = None,
            verbose: bool = True) -> InferenceManager:
        """
        Load a model from a ZIP archive.

        :param zip_path: Path to the ZIP file containing model artifacts.
        :type zip_path: Path
        :param extract_dir: Optional directory for extraction (uses temp
            if None).
        :type extract_dir: Optional[Path]
        :param verbose: Enable detailed logging.
        :type verbose: bool
        :return: Loaded InferenceManager instance.
        :rtype: InferenceManager
        """
        zip_path = Path(zip_path)
        logger = Logger(verbose)
        if not zip_path.exists():
            logger.error(f"Zip not found: {zip_path}")
            sys.exit(1)

        paths = ModelPaths()

        def find_root(root: Path) -> Path:
            hits = list(root.rglob(paths.config_file))
            if not hits:
                logger.error(
                    f"Missing config file inside zip extraction: {root}")
                sys.exit(1)
            return hits[0].parent

        if extract_dir is None:
            tmp = Path(tempfile.mkdtemp(prefix="leaffliction_bundle_"))
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(tmp)
                return cls.load(find_root(tmp), verbose=verbose)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        else:
            extract_dir = Path(extract_dir)
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)
            return cls.load(find_root(extract_dir), verbose=verbose)

    def predict(self, tensor: torch.Tensor):
        """
        Perform inference on an input tensor.

        :param tensor: Input tensor of shape (channels, height, width) or
            (batch, channels, height, width).
        :type tensor: torch.Tensor
        :return: Tuple of (predicted_class_id, probability_dict).
        :rtype: Tuple[int, Dict[str, float]]
        """
        self.model.eval()

        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)

        expected_c = int(getattr(self.cfg, "input_channels", tensor.size(1)))

        got_c = int(tensor.size(1))
        if got_c != expected_c:
            self.logger.error(
                f"Channel mismatch: model expects C={expected_c} "
                f"but got C={got_c}. ")
            sys.exit(1)

        tensor = tensor.to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs_tensor = torch.softmax(outputs, dim=1)
            pred_id = torch.argmax(probs_tensor, dim=1).item()

            probs_np = probs_tensor.detach().cpu().numpy()[0]
            probs = {
                self.labels.decode(i): float(
                    probs_np[i]) for i in range(
                    len(probs_np))}

        return pred_id, probs
