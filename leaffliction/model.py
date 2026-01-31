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


@dataclass
class ModelConfig:
    """Configuration du modèle PyTorch"""
    num_classes: int = 0
    input_channels: int = 7  # Nombre de transformations
    img_size: Tuple[int, int] = (224, 224)
    seed: int = 42
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelPaths:
    """Chemins des fichiers du bundle"""
    model_file: str = "model.pth"
    labels_file: str = "labels.json"
    config_file: str = "config.json"


class LabelMapper:
    """
    Mapping stable:
      class_name -> id
      id -> class_name
    """

    def __init__(self) -> None:
        self.class_to_id: Dict[str, int] = {}
        self.id_to_class: Dict[int, str] = {}

    def fit(self, class_names: List[str]) -> None:
        """Crée le mapping depuis une liste de noms de classes (tri stable)."""
        # tri pour stabilité (même ordre entre machines/runs)
        uniq = sorted(set(class_names))
        self.class_to_id = {name: i for i, name in enumerate(uniq)}
        self.id_to_class = {i: name for name, i in self.class_to_id.items()}

    def encode(self, class_name: str) -> int:
        """Convertit un nom de classe en ID"""
        if class_name not in self.class_to_id:
            raise KeyError(f"Unknown class_name: {class_name}. Known: {list(self.class_to_id.keys())}")
        return self.class_to_id[class_name]

    def decode(self, class_id: int) -> str:
        """Convertit un ID en nom de classe"""
        if class_id not in self.id_to_class:
            raise KeyError(f"Unknown class_id: {class_id}. Known: {list(self.id_to_class.keys())}")
        return self.id_to_class[class_id]

    def to_json_dict(self) -> Dict[str, Any]:
        """Sérialise en dict pour JSON"""
        # on stocke class_to_id ; id_to_class se reconstruit
        return {"class_to_id": self.class_to_id}

    @classmethod
    def from_json_dict(cls, data: Dict[str, Any]) -> "LabelMapper":
        """Désérialise depuis un dict JSON"""
        enc = cls()
        c2i = data.get("class_to_id", {})
        if not isinstance(c2i, dict):
            raise ValueError("labels.json: 'class_to_id' must be a dict.")
        # sécurise types
        enc.class_to_id = {str(k): int(v) for k, v in c2i.items()}
        enc.id_to_class = {int(v): str(k) for k, v in enc.class_to_id.items()}
        return enc


class ConvolutionalNeuralNetwork(nn.Module):
    """
    Modèle PyTorch qui prend les transformations en entrée.
    Input: (batch, n_transforms, H, W) où n_transforms = 6
    """
    def __init__(self, num_classes: int, input_channels: int = 6):
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
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class InferenceManager:
    """
    Bundle complet pour sauvegarder/charger un modèle PyTorch.
    """

    def __init__(
        self,
        model: ConvolutionalNeuralNetwork,
        labels: LabelMapper,
        transformation_engine: Any,  # TransformationEngine
        cfg: ModelConfig,
        paths: Optional[ModelPaths] = None
    ) -> None:
        self.model = model
        self.labels = labels
        self.transformation_engine = transformation_engine
        self.cfg = cfg
        self.paths = paths or ModelPaths()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def save(self, out_dir: Path) -> None:
        """
        Sauvegarde le bundle dans out_dir/:
        - model.pth (PyTorch state_dict)
        - labels.json
        - config.json
        """
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1) config.json
        cfg_dict = {
            "num_classes": self.cfg.num_classes,
            "input_channels": self.cfg.input_channels,
            "img_size": list(self.cfg.img_size),
            "seed": self.cfg.seed,
            "extra": self.cfg.extra,
        }
        (out_dir / self.paths.config_file).write_text(json.dumps(cfg_dict, indent=2), encoding="utf-8")

        # 2) labels.json
        (out_dir / self.paths.labels_file).write_text(
            json.dumps(self.labels.to_json_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        # 3) model.pth
        # on sauvegarde uniquement state_dict pour portabilité
        model_path = out_dir / self.paths.model_file
        torch.save(self.model.state_dict(), model_path)

    @classmethod
    def load(cls, in_dir: Path) -> "InferenceManager":
        """
        Charge le bundle depuis in_dir/
        Note: ne peut pas reconstruire transformation_engine sans ta factory -> on met None par défaut.
        """
        in_dir = Path(in_dir)
        paths = ModelPaths()

        cfg_path = in_dir / paths.config_file
        labels_path = in_dir / paths.labels_file
        model_path = in_dir / paths.model_file

        if not cfg_path.exists():
            raise FileNotFoundError(f"Missing config file: {cfg_path}")
        if not labels_path.exists():
            raise FileNotFoundError(f"Missing labels file: {labels_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model file: {model_path}")

        # 1) config
        cfg_data = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg = ModelConfig(
            num_classes=int(cfg_data["num_classes"]),
            input_channels=int(cfg_data.get("input_channels", 6)),
            img_size=tuple(cfg_data.get("img_size", [224, 224])),
            seed=int(cfg_data.get("seed", 42)),
            extra=dict(cfg_data.get("extra", {})),
        )
        _set_seed(cfg.seed)

        # 2) labels
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

        # transformation_engine: à injecter plus tard si tu veux prédire depuis paths
        bundle = cls(model=model, labels=labels, transformation_engine=None, cfg=cfg, paths=paths)
        return bundle

    @classmethod
    def load_from_zip(cls, zip_path: Path, extract_dir: Optional[Path] = None) -> "InferenceManager":
        zip_path = Path(zip_path)
        if not zip_path.exists():
            raise FileNotFoundError(f"Zip not found: {zip_path}")

        paths = ModelPaths()

        def find_root(root: Path) -> Path:
            hits = list(root.rglob(paths.config_file))
            if not hits:
                raise FileNotFoundError(f"Missing config file inside zip extraction: {root}")
            return hits[0].parent  # dossier contenant config.json

        if extract_dir is None:
            tmp = Path(tempfile.mkdtemp(prefix="leaffliction_bundle_"))
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(tmp)
                return cls.load(find_root(tmp))
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        else:
            extract_dir = Path(extract_dir)
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)
            return cls.load(find_root(extract_dir))


    def predict(self, tensor: torch.Tensor) -> Tuple[int, Dict[str, float]]:
        self.model.eval()

        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)  # [1,C,H,W]

        # Nombre de channels attendus par le modèle
        expected_c = int(getattr(self.cfg, "input_channels", tensor.size(1)))

        got_c = int(tensor.size(1))
        if got_c != expected_c:
            raise ValueError(
                f"Channel mismatch: model expects C={expected_c} but got C={got_c}. "
                f"If the model expects 5, you must provide a 5-channel tensor "
                f"(e.g., from TransformationEngine.apply_all_as_tensor)."
            )

        tensor = tensor.to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs_tensor = torch.softmax(outputs, dim=1)
            pred_id = torch.argmax(probs_tensor, dim=1).item()

            probs_np = probs_tensor.detach().cpu().numpy()[0]
            probs = {self.labels.decode(i): float(probs_np[i]) for i in range(len(probs_np))}

        return pred_id, probs

