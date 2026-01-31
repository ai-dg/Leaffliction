from __future__ import annotations

from os import path
from pathlib import Path
import cv2
import torch
import sys

from leaffliction.cli import ArgsManager
from leaffliction.predict_pipeline import Predictor, PredictConfig
from leaffliction.transformations import TransformationEngine
from leaffliction.model import InferenceManager


def main() -> None:
    parser = ArgsManager().build_predict_parser()
    args = parser.parse_args()

    bundle_zip: Path | None = args.bundle_zip
    model_path: Path | None = args.model_path
    image_path: Path = args.image_path

    if bundle_zip is None and model_path is None:
        print("Error: you must provide either --bundle-zip or --model-path")
        sys.exit(1)

    if bundle_zip is not None and model_path is not None:
        print("Error: choose ONE method between --bundle-zip OR --model-path")
        sys.exit(1)

    if bundle_zip is not None and not bundle_zip.exists():
        print(f"Error: bundle zip not found: {bundle_zip}")
        sys.exit(1)

    if model_path is not None and not model_path.exists():
        print(f"Error: model path not found: {model_path}")
        sys.exit(1)

    if not image_path.exists():
        print(f"Error: image not found: {image_path}")
        sys.exit(1)

    print("✅ CLI arguments OK")
    print(f"  image_path = {image_path}")
    if bundle_zip:
        print(f"  bundle_zip = {bundle_zip}")
    else:
        print(f"  model_path = {model_path}")




    

    cfg = PredictConfig(
        show_transforms=getattr(args, "show_transforms", True),
        top_k=getattr(args, "top_k", 3),
    )

    # Transformation engine
    tf_engine = TransformationEngine.trainning()

    # Predictor PyTorch
    predictor = Predictor(
        model_loader=InferenceManager,
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
        model_path=model_path,
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
        from leaffliction.plotting import Plotter
        
        # Charger image originale
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        grid = Plotter()
        grid.plot_grid(f"Transformations - Predicted: {predicted_label}", transformed, original=img)


if __name__ == "__main__":
    main()
