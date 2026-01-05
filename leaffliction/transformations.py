from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Protocol, Tuple
import numpy as np
import cv2
import torch
from plantcv import plantcv as pcv
# from leaffliction.dataset import DatasetScanner

class Transformation(Protocol):
    """
    Interface d'une transformation.
    Utilisé pour créer des canaux de features pour PyTorch.
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
    2. Création de tensors PyTorch (train/predict) - apply_all_as_tensor()
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
    
    def apply_all_as_tensor(self, img: np.ndarray) -> torch.Tensor:
        """
        Applique toutes les transformations et retourne un tensor PyTorch.
        
        Args:
            img: Image RGB (H, W, 3)
        
        Returns:
            tensor: (n_transforms, H, W) avec les transformations comme canaux
        """
        channels = []
        
        for tf in self.tfs:
            # Appliquer transformation
            transformed = tf.apply(img)
            
            # Convertir en grayscale si nécessaire
            if len(transformed.shape) == 3:
                transformed = cv2.cvtColor(transformed, cv2.COLOR_RGB2GRAY)
            
            # Normaliser [0, 255] → [0, 1]
            transformed = transformed.astype(np.float32) / 255.0
            
            channels.append(transformed)
        
        # Stack en tensor (n_transforms, H, W)
        tensor = torch.from_numpy(np.stack(channels, axis=0))
        return tensor
    
    def batch_transform(
        self, 
        items: List[Tuple[Path, int]], 
        img_size: Tuple[int, int] = (224, 224)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transforme un batch d'images en tensors PyTorch.
        
        Args:
            items: [(path, label), ...]
            img_size: (H, W) taille de redimensionnement
        
        Returns:
            X: (n, n_transforms, H, W) tensor des transformations
            y: (n,) tensor des labels
        """
        X_list = []
        y_list = []
        
        for img_path, label in items:
            # Charger image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"⚠️  Warning: Could not load {img_path}")
                continue
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size)
            
            # Transformer en tensor
            tensor = self.apply_all_as_tensor(img)
            
            X_list.append(tensor)
            y_list.append(label)
        
        # Stack en batch
        X = torch.stack(X_list)  # (n, n_transforms, H, W)
        y = torch.tensor(y_list, dtype=torch.long)  # (n,)
        
        return X, y


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

def main():
    path = "../leaves/images/Apple_Black_rot/image (100).JPG"
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"cv2.imread a échoué: {path}")

    original = img.copy()

    # --- Basique (déjà OK) ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    edges = cv2.Canny(blur, 50, 150)

    # --- MASK (PlantCV) ---
    # Si le résultat est inversé, teste "light"
    mask = pcv.threshold.binary(gray_img=blur, threshold=0, object_type="dark")
    mask = (mask > 0).astype(np.uint8) * 255  # compat OpenCV

    # --- CONTOUR principal ---
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv2.contourArea) if cnts else None

    # --- ROI / ANALYZE / LANDMARKS ---
    roi_vis = original.copy()
    analyze_vis = original.copy()
    landmarks_vis = original.copy()
    roi_crop = None

    if cnt is not None and cv2.contourArea(cnt) > 500:
        # ROI objects = bbox + crop
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(roi_vis, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_crop = original[y:y + h, x:x + w].copy()

        # Analyze object = contour + centroid + mesures
        cv2.drawContours(analyze_vis, [cnt], -1, (255, 0, 255), 2)

        area = cv2.contourArea(cnt)
        perim = cv2.arcLength(cnt, True)

        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(analyze_vis, (cx, cy), 6, (0, 0, 255), -1)

        cv2.putText(
            analyze_vis,
            f"area={area:.0f}px  perimeter={perim:.0f}px",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (40, 40, 40),
            2
        )

        # --- PSEUDOLANDMARKS (PlantCV) ---
        # Selon version PlantCV: pseudolandmarks(contour=..., num_landmarks=...)
        # ou pseudolandmarks(contour, num_landmarks)
        try:
            landmarks = pcv.pseudolandmarks(contour=cnt, num_landmarks=40)
        except TypeError:
            landmarks = pcv.pseudolandmarks(cnt, 40)

        # landmarks est généralement un array/list de points (x,y)
        for pt in landmarks:
            px, py = int(pt[0]), int(pt[1])
            cv2.circle(landmarks_vis, (px, py), 3, (0, 165, 255), -1)

    # --- DISPLAY ---
    cv2.imshow("original", original)
    cv2.imshow("gray", gray)
    cv2.imshow("blur", blur)
    cv2.imshow("bw (Otsu)", bw)
    cv2.imshow("edges (Canny)", edges)

    cv2.imshow("mask (PlantCV)", mask)
    cv2.imshow("roi objects (bbox)", roi_vis)
    if roi_crop is not None:
        cv2.imshow("roi crop", roi_crop)

    cv2.imshow("analyze object", analyze_vis)
    cv2.imshow("pseudolandmarks (PlantCV)", landmarks_vis)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    






if __name__ == "__main__":
    main()