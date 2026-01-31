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
from leaffliction.utils import Logger


def main() -> None:
    parser = ArgsManager().build_predict_parser()
    args = parser.parse_args()

    model_zip: Path | None = args.model_zip
    model_path: Path | None = args.model_path
    image_path: Path = args.image_path
    logger = Logger(args.verbose)

    if model_zip is None and model_path is None:
        logger.error("Error: you must provide either --bundle-zip or --model-path")
        sys.exit(1)

    if model_zip is not None and model_path is not None:
        logger.error("Error: choose ONE method between --bundle-zip OR --model-path")
        sys.exit(1)

    if model_zip is not None and not model_zip.exists():
        logger.error(f"Error: bundle zip not found: {model_zip}")
        sys.exit(1)

    if model_path is not None and not model_path.exists():
        logger.error(f"Error: model path not found: {model_path}")
        sys.exit(1)

    if not image_path.exists():
        logger.error(f"Error: image not found: {image_path}")
        sys.exit(1)

    logger.info("CLI arguments OK")
    logger.info(f"  image_path = {image_path}")
    if model_zip:
        logger.info(f"  model_zip = {model_zip}")
    else:
        logger.info(f"  model_path = {model_path}")


    cfg = PredictConfig(
        show_transforms=getattr(args, "show_transforms", True),
        top_k=getattr(args, "top_k", 3),
    )
    tf_engine = TransformationEngine.trainning()


    predictor = Predictor(
        model_loader=InferenceManager,
        transformation_engine=tf_engine,
        verbose=args.verbose
    )

    logger.info("=" * 60)
    logger.info("LEAFFLICTION - PyTorch Prediction")
    logger.info("=" * 60)
    logger.info(f"   Model: {model_zip}")
    logger.info(f"   Image: {image_path}")
    logger.info("=" * 60)
    logger.info()

    predicted_label, probs, transformed = predictor.predict(
        model_zip=model_zip,
        model_path=model_path,
        image_path=image_path,
        cfg=cfg
    )

    logger.info("Prediction Result")
    logger.info("=" * 60)
    logger.info(f"   Predicted class: {predicted_label}")
    logger.info()
    

    logger.info(f"   Top {cfg.top_k} predictions:")
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    
    bar_width = 10
    full_char = "▰"
    empty_char = "▱"

    for i, (label, prob) in enumerate(sorted_probs[:cfg.top_k], 1):
        filled = int(round(prob * bar_width))
        filled = max(0, min(bar_width, filled))
        bar = full_char * filled + empty_char * (bar_width - filled)

        logger.info(f"   {i}. {label:30s} {prob:6.2%} {bar} ")

    logger.info("=" * 60)
    logger.info()

    # Affichage transformations (optionnel)
    if cfg.show_transforms and transformed:
        logger.info("Showing transformations...")
        from leaffliction.plotting import Plotter
        
        # Charger image originale
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        grid = Plotter()
        grid.plot_grid(f"Transformations - Predicted: {predicted_label}", transformed, original=img)


if __name__ == "__main__":
    main()
