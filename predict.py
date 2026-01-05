from __future__ import annotations

from pathlib import Path
import cv2

from leaffliction.cli import CLIBuilder
from leaffliction.predict_pipeline import MLPredictor, PredictConfig, PredictionVisualiser
from leaffliction.transformations import TransformationEngine
from leaffliction.model import MLModelBundle


def main() -> None:
    parser = CLIBuilder().build_predict_parser()
    args = parser.parse_args()

    bundle_zip = Path(args.bundle_zip)
    image_path = Path(args.image_path)

    cfg = PredictConfig(
        show_transforms=getattr(args, "show_transforms", True),
        top_k=getattr(args, "top_k", 3),  # Top 3 prédictions par défaut
    )

    # Moteur de transformations
    engine = TransformationEngine.default_six()

    # Predictor ML
    predictor = MLPredictor(bundle_loader=MLModelBundle, transformations_engine=engine)

    print(f"🔍 Predicting disease for: {image_path.name}")
    print(f"📦 Using model from: {bundle_zip}")
    print()

    # Prédiction
    label, probs, transformed = predictor.predict(
        bundle_zip=bundle_zip, 
        image_path=image_path, 
        cfg=cfg
    )

    # Affichage résultat principal
    print("=" * 60)
    print("✅ PREDICTION RESULT")
    print("=" * 60)
    print(f"🍃 Predicted class: {label}")
    print(f"📊 Confidence: {probs[label]:.1%}")
    print()
    
    # Top K prédictions
    if cfg.top_k > 1:
        print(f"Top {cfg.top_k} predictions:")
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        for i, (class_name, prob) in enumerate(sorted_probs[:cfg.top_k], 1):
            print(f"   {i}. {class_name}: {prob:.1%}")
        print()

    # Affichage visuel des transformations (optionnel)
    if cfg.show_transforms and transformed:
        print("📊 Showing transformations...")
        
        # Charger l'image originale pour affichage
        img = cv2.imread(str(image_path))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            vis = PredictionVisualiser()
            vis.show(original=img, transformed=transformed, predicted_label=label)
    
    print("=" * 60)


if __name__ == "__main__":
    main()
