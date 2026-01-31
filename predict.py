from __future__ import annotations

from pathlib import Path
import cv2
import sys
from typing import Dict, Tuple
import numpy as np

from leaffliction.cli import ArgsManager
from leaffliction.predict_pipeline import Predictor, PredictConfig
from leaffliction.transformations import TransformationEngine
from leaffliction.model import InferenceManager
from leaffliction.utils import Logger
from leaffliction.plotting import Plotter


def print_dir_predictions(
    logger,
    results: Dict[Path, Tuple[str, str, Dict[str, float],
                              Dict[str, np.ndarray]]],
    cfg
):
    """
    Display prediction results for a directory of images and compute
    overall accuracy.

    For each image, this function prints:
    - the ground-truth label (True label),
    - the predicted label,
    - a correctness flag (TP / FP),
    - the top-k class probabilities as a textual bar chart,
    and optionally displays the corresponding image transformations
    in a grid layout.

    At the end, it prints a summary including total accuracy and class counts.

    :param results: Mapping between image paths and prediction outputs.
    :type results: Dict[Path, Tuple[str, str, Dict[str, float],
        Dict[str, np.ndarray]]]
    :param logger: Logger instance used to display formatted output.
    :type logger: Logger
    :param cfg: Prediction configuration (top_k, show_transforms, etc.).
    :type cfg: PredictConfig
    :return: None
    :rtype: None
    """
    if not results:
        logger.warn("No prediction results to display.")
        return

    bar_width = 10
    full_char = "▰"
    empty_char = "▱"

    pred_counts = {}
    true_counts = {}

    total = 0
    correct = 0

    for image_path, (true_label, predicted_label, probs,
                     transformed) in results.items():
        total += 1

        is_correct = (predicted_label == true_label)
        correct += int(is_correct)

        # For single-label classification, "TP/FP" naming is a bit imprecise,
        # but if you want exactly that wording:
        verdict = "TP" if is_correct else "FP"

        pred_counts[predicted_label] = pred_counts.get(predicted_label, 0) + 1
        true_counts[true_label] = true_counts.get(true_label, 0) + 1

        logger.info("Prediction Result")
        logger.info("=" * 60)
        logger.info(f"Image: {image_path}")
        logger.info(f"   True class     : {true_label}")
        logger.info(f"   Predicted class: {predicted_label}   [{verdict}]")
        logger.info()

        logger.info(f"   Top {cfg.top_k} predictions:")
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)

        for i, (label, prob) in enumerate(sorted_probs[:cfg.top_k], 1):
            filled = int(round(prob * bar_width))
            filled = max(0, min(bar_width, filled))
            bar = full_char * filled + empty_char * (bar_width - filled)
            logger.info(f"   {i}. {label:30s} {prob:6.2%} {bar} ")

        logger.info("=" * 60)
        logger.info("")

        if cfg.show_transforms and transformed:
            logger.info("Showing transformations...")

            img = cv2.imread(str(image_path))
            if img is None:
                logger.warn(f"Could not reload image for grid: {image_path}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            grid = Plotter()
            grid.plot_grid(
                f"Transformations - True: {true_label} | "
                f"Pred: {predicted_label} [{verdict}]",
                transformed,
                original=img)

    # --- Global summary ---
    accuracy = correct / total if total else 0.0

    logger.info("Directory Summary")
    logger.info("=" * 60)
    logger.info(f"Total images : {total}")
    logger.info(f"Correct      : {correct}")
    logger.info(f"Accuracy     : {accuracy:.2%}")
    logger.info("-" * 60)

    logger.info("True label distribution:")
    for label, n in sorted(
            true_counts.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"   {label:30s} {n:4d} ({(n/total):.1%})")

    logger.info("-" * 60)

    logger.info("Predicted label distribution:")
    for label, n in sorted(
            pred_counts.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"   {label:30s} {n:4d} ({(n/total):.1%})")

    logger.info("=" * 60)
    logger.info("")


def main() -> None:
    """
    Predict plant disease class from a leaf image.

    Loads a trained model, applies transformations to the input image,
    performs inference, and displays prediction results with probabilities.

    :return: None
    :rtype: None
    """
    parser = ArgsManager().build_predict_parser()
    args = parser.parse_args()

    image_path: Path = args.image_path
    dir_path: str = args.dir_path
    model_zip: Path | None = args.model_zip
    model_path: Path | None = args.model_path
    logger = Logger(args.verbose)

    if dir_path is None:
        if model_zip is None and model_path is None:
            logger.error(
                "Error: you must provide either "
                "--bundle-zip or --model-path")
            sys.exit(1)

        if model_zip is not None and model_path is not None:
            logger.error(
                "Error: choose ONE method between "
                "--bundle-zip OR --model-path")
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
        show_transforms=args.show_transforms,
        top_k=args.top_k,
    )
    tf_engine = TransformationEngine.trainning(args.verbose)

    predictor = Predictor(
        model_loader=InferenceManager,
        transformation_engine=tf_engine,
        verbose=args.verbose
    )

    logger.info("=" * 60)
    logger.info("LEAFFLICTION - PyTorch Prediction")
    logger.info("=" * 60)
    logger.info(f"   Model: {model_zip}")
    if dir_path is None:
        logger.info(f"   Image: {image_path}")
    else:
        logger.info(f"   Dir path: {dir_path}")
    logger.info("=" * 60)
    logger.info()

    if dir_path is not None:
        results = predictor.predict_with_dir(
            dir_path=dir_path,
            model_zip=model_zip,
            model_path=model_path,
            cfg=cfg
        )
        print_dir_predictions(
            logger=logger,
            results=results,
            cfg=cfg
        )

    else:
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

        if cfg.show_transforms and transformed:
            logger.info("Showing transformations...")

            img = cv2.imread(str(image_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            grid = Plotter()
            grid.plot_grid(
                f"Transformations - Predicted: {predicted_label}",
                transformed,
                original=img)


if __name__ == "__main__":
    main()
