"""
Train a classification model on leaf disease images.

This module provides the main entry point for training a plant disease
classification model from the command line.
"""

from __future__ import annotations
from pathlib import Path
from leaffliction.cli import ArgsManager
from leaffliction.dataset import DatasetScanner, DatasetSplitter
from leaffliction.augmentations import AugmentationEngine
from leaffliction.transformations import TransformationEngine
from leaffliction.model import LabelMapper
from leaffliction.train_pipeline import (
    Trainer,
    TrainConfig,
    ModelChecker,
    TrainingPackager
)
from leaffliction.utils import ZipPackager, Hasher, Logger
from leaffliction.plotting import Plotter


def main() -> None:
    """
    Train a model from the command-line.

    Parses CLI arguments, prepares dataset, trains the model, plots metrics,
    checks model validity, packages artifacts into a ZIP and writes a SHA1
    signature file.
    """
    parser = ArgsManager().build_train_parser()
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(getattr(args, "out_dir", "training_artifacts"))
    out_zip = Path(getattr(args, "out_zip", "train_output.zip"))

    cfg = TrainConfig(
        epochs=getattr(args, "epochs", 70),
        batch_size=getattr(args, "batch_size", 8),
        lr=getattr(args, "learning_rate", getattr(args, "lr", 1e-3)),
        valid_ratio=getattr(args, "valid_ratio", 0.2),
        seed=getattr(args, "seed", 42),
        img_size=(getattr(args, "img_h", 224), getattr(args, "img_w", 224)),
        augment_train=getattr(args, "augment", True),
        transform_train=getattr(args, "transform", True),
    )

    scanner = DatasetScanner()
    splitter = DatasetSplitter()
    labels = LabelMapper()
    aug_engine = AugmentationEngine()
    tf_engine = TransformationEngine.trainning()

    trainer = Trainer(
        dataset_scanner=scanner,
        dataset_splitter=splitter,
        augmentation_engine=aug_engine,
        transformation_engine=tf_engine,
        labels=labels,
        verbose=args.verbose
    )

    logger = Logger(args.verbose)
    logger.info("Training information:")
    logger.info(f"   Dataset: {dataset_dir}")
    aug_status = 'Enabled' if cfg.augment_train else 'Disabled'
    logger.info(f"   Augmentation: {aug_status}")
    logger.info(f"   Validation ratio: {cfg.valid_ratio:.0%}")
    logger.info(f"   Epochs: {cfg.epochs}")
    logger.info(f"   Batch size: {cfg.batch_size}")

    metrics = trainer.train(dataset_dir=dataset_dir, out_dir=out_dir, cfg=cfg)
    plotter = Plotter()
    plotter.plot_learning_curve(
        metrics.history_train_acc,
        metrics.history_valid_acc)
    plotter.plot_learning_curve_loss(metrics.history_train_loss)

    checker = ModelChecker(verbose=args.verbose)
    checker.assert_ok(metrics)

    packager = TrainingPackager(
        zip_packager=ZipPackager(),
        verbose=args.verbose)
    artifacts_dir = packager.prepare_artifacts_dir(tmp_dir=out_dir)
    packager.build_zip(artifacts_dir=artifacts_dir, out_zip=out_zip)

    logger.info("Generating signature...")
    hasher = Hasher()
    signature = hasher.sha1_file(out_zip)
    signature_file = Path("signature.txt")
    signature_file.write_text(signature + "\n")

    logger.info("Training completed successfully!")
    logger.info(f"- Train accuracy: {metrics.train_accuracy:.2%}")
    logger.info(f"- Valid accuracy: {metrics.valid_accuracy:.2%}")
    logger.info(f"- Valid count: {metrics.valid_count}")
    logger.info(f"- Model artifact: {out_zip}")
    logger.info(f"- Signature: {signature_file}")
    logger.info(f"   SHA1: {signature}")


if __name__ == "__main__":
    main()
