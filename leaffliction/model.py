from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    """Configuration du modèle PyTorch"""
    num_classes: int = 0
    input_channels: int = 6  # Nombre de transformations
    img_size: Tuple[int, int] = (224, 224)
    seed: int = 42
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelPaths:
    """Chemins des fichiers du bundle"""
    model_file: str = "model.pth"
    labels_file: str = "labels.json"
    config_file: str = "config.json"


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


class TransformationClassifier(nn.Module):
    """
    Modèle PyTorch qui prend les transformations en entrée.
    Input: (batch, n_transforms, H, W) où n_transforms = 6
    """
    def __init__(self, num_classes: int, input_channels: int = 6):
        super().__init__()
        
        # Convolutions pour extraire features des transformations
        self.features = nn.Sequential(
            # Conv1: (6, 224, 224) → (32, 112, 112)
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv2: (32, 112, 112) → (64, 56, 56)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv3: (64, 56, 56) → (128, 28, 28)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv4: (128, 28, 28) → (256, 14, 14)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # x: (batch, 6, 224, 224)
        x = self.features(x)      # (batch, 256, 14, 14)
        x = self.gap(x)           # (batch, 256, 1, 1)
        x = self.classifier(x)    # (batch, num_classes)
        return x


class PyTorchModelFactory:
    """
    Construit un modèle PyTorch.
    """

    def build(self, cfg: ModelConfig) -> TransformationClassifier:
        """
        Construit un TransformationClassifier PyTorch.
        
        Retourne: modèle PyTorch non entraîné
        """
        model = TransformationClassifier(
            num_classes=cfg.num_classes,
            input_channels=cfg.input_channels
        )
        return model


class PyTorchModelBundle:
    """
    Bundle complet pour sauvegarder/charger un modèle PyTorch.
    
    Contient:
    - model: modèle PyTorch entraîné
    - labels: LabelEncoder pour les classes
    - transformation_engine: TransformationEngine pour créer tensors
    - cfg: ModelConfig
    """

    def __init__(
        self,
        model: TransformationClassifier,
        labels: LabelEncoder,
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
        raise NotImplementedError

    @classmethod
    def load(cls, in_dir: Path) -> "PyTorchModelBundle":
        """
        Charge le bundle depuis in_dir/
        """
        raise NotImplementedError

    @classmethod
    def load_from_zip(cls, zip_path: Path, extract_dir: Optional[Path] = None) -> "PyTorchModelBundle":
        """
        Extrait le zip puis charge le bundle.
        """
        raise NotImplementedError
    
    def predict(self, tensor: torch.Tensor) -> Tuple[int, Dict[str, float]]:
        """
        Prédit la classe depuis un tensor de transformations.
        
        Args:
            tensor: torch.Tensor de shape (n_transforms, H, W) ou (1, n_transforms, H, W)
        
        Retourne:
            - pred_id: int (ID de la classe prédite)
            - probs: Dict[str, float] (probabilités par classe)
        """
        self.model.eval()
        
        # Ajouter batch dimension si nécessaire
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)  # (1, n_transforms, H, W)
        
        tensor = tensor.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(tensor)
            probs_tensor = torch.softmax(outputs, dim=1)
            pred_id = torch.argmax(probs_tensor, dim=1).item()
            
            # Convertir probs en dict
            probs_np = probs_tensor.cpu().numpy()[0]
            probs = {
                self.labels.decode(i): float(probs_np[i])
                for i in range(len(probs_np))
            }
        
        return pred_id, probs
