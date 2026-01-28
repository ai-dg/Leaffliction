from __future__ import annotations

from pathlib import Path

from leaffliction.cli import CLIBuilder
from leaffliction.dataset import DatasetScanner, DatasetSplitter
from leaffliction.augmentations import AugmentationEngine
from leaffliction.transformations import TransformationEngine
from leaffliction.model import PyTorchModelFactory, LabelEncoder
from leaffliction.train_pipeline import PyTorchTrainer, TrainConfig, RequirementsGate, TrainingPackager
from leaffliction.utils import ZipPackager, Hasher


def main() -> None:
    parser = CLIBuilder().build_train_parser()
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(getattr(args, "out_dir", "artifacts"))
    out_zip = Path(getattr(args, "out_zip", "learnings.zip"))

    # Configuration pour PyTorch
    cfg = TrainConfig(
        epochs=getattr(args, "epochs", 50),
        batch_size=getattr(args, "batch_size", 32),
        lr=getattr(args, "lr", 1e-3),
        valid_ratio=getattr(args, "valid_ratio", 0.2),
        seed=getattr(args, "seed", 42),
        img_size=(getattr(args, "img_h", 224), getattr(args, "img_w", 224)),
        augment_train=getattr(args, "augment", True),
        transform_train=getattr(args, "transform", True),
        augmentations_per_image=getattr(args, "aug_per_image", 3),
    )

    # Composants
    scanner = DatasetScanner()
    splitter = DatasetSplitter()
    labels = LabelEncoder()
    aug_engine = AugmentationEngine()
    tf_engine = TransformationEngine.default_six()
    model_factory = PyTorchModelFactory()

    # Trainer PyTorch
    trainer = PyTorchTrainer(
        dataset_scanner=scanner,
        dataset_splitter=splitter,
        augmentation_engine=aug_engine,
        transformation_engine=tf_engine,
        model_factory=model_factory,
        labels=labels,
    )

    print("=" * 60)
    print("🍃 LEAFFLICTION - PyTorch Training Pipeline")
    print("=" * 60)
    print(f"   Dataset: {dataset_dir}")
    print(f"   Augmentation: {'Enabled' if cfg.augment_train else 'Disabled'}")
    print(f"   Validation ratio: {cfg.valid_ratio:.0%}")
    print(f"   Epochs: {cfg.epochs}")
    print(f"   Batch size: {cfg.batch_size}")
    print("=" * 60)
    print()

    # Entraînement
    metrics = trainer.train(dataset_dir=dataset_dir, out_dir=out_dir, cfg=cfg)

    # Vérification des contraintes
    gate = RequirementsGate()
    gate.assert_ok(metrics)

    # Zip final
    packager = TrainingPackager(zip_packager=ZipPackager())
    artifacts_dir = packager.prepare_artifacts_dir(tmp_dir=out_dir)
    packager.build_zip(artifacts_dir=artifacts_dir, out_zip=out_zip)

    # Signature SHA1
    print("🔐 Generating signature...")
    hasher = Hasher()
    signature = hasher.sha1_file(out_zip)
    signature_file = Path("signature.txt")
    signature_file.write_text(signature + "\n")
    print(f"   Signature saved to {signature_file}")
    print()

    print("=" * 60)
    print("✅ Training completed successfully!")
    print("=" * 60)
    print(f"📊 Train accuracy: {metrics.train_accuracy:.2%}")
    print(f"📊 Valid accuracy: {metrics.valid_accuracy:.2%}")
    print(f"📊 Valid count: {metrics.valid_count}")
    print(f"📦 Model bundle: {out_zip}")
    print(f"🔐 Signature: {signature_file}")
    print(f"   SHA1: {signature}")
    print("=" * 60)


if __name__ == "__main__":
    main()
