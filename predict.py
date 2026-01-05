from __future__ import annotations

from pathlib import Path
import cv2
import torch

from leaffliction.cli import CLIBuilder
from leaffliction.predict_pipeline import PyTorchPredictor, PredictConfig
from leaffliction.transformations import TransformationEngine
from leaffliction.model import PyTorchModelBundle


def main() -> None:
    parser = CLIBuilder().build_predict_parser()
    args = parser.parse_args()

    bundle_zip = Path(args.bundle_zip)
    image_path = Path(args.image_path)

    cfg = PredictConfig(
        show_transforms=getattr(args, "show_transforms", True),
        top_k=getattr(args, "top_k", 3),
    )

    # Transformation engine
    tf_engine = TransformationEngine.default_six()

    # Predictor PyTorch
    predictor = PyTorchPredictor(
        bundle_loader=PyTorchModelBundle,
        transformation_engine=tf_engine
    )

    print("=" * 60)
    print("🍃 LEAFFLICTION - PyTorch Prediction")
    print("=" * 60)
    print(f"   Model: {bundle_zip}")
    print(f"   Image: {image_path}")
    print("=" * 60)
    print()

    # Prédiction
    predicted_label, probs, transformed = predictor.predict(
        bundle_zip=bundle_zip,
        image_path=image_path,
        cfg=cfg
    )

    # Affichage résultat
    print("🎯 Prediction Result")
    print("=" * 60)
    print(f"   Predicted class: {predicted_label}")
    print()
    
    # Top K prédictions
    print(f"   Top {cfg.top_k} predictions:")
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    for i, (label, prob) in enumerate(sorted_probs[:cfg.top_k], 1):
        bar = "█" * int(prob * 30)
        print(f"   {i}. {label:30s} {prob:6.2%} {bar}")
    print("=" * 60)
    print()

    # Affichage transformations (optionnel)
    if cfg.show_transforms and transformed:
        print("📊 Showing transformations...")
        from leaffliction.plotting import GridPlotter
        
        # Charger image originale
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        grid = GridPlotter()
        grid.show_grid(f"Transformations - Predicted: {predicted_label}", transformed, original=img)


if __name__ == "__main__":
    main()
