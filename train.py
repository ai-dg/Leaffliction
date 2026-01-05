from __future__ import annotations

from pathlib import Path

from leaffliction.cli import CLIBuilder
from leaffliction.dataset import DatasetScanner, DatasetSplitter
from leaffliction.augmentations import AugmentationEngine
from leaffliction.transformations import TransformationEngine, FeatureExtractor
from leaffliction.model import MLModelFactory, LabelEncoder
from leaffliction.train_pipeline import MLTrainer, TrainConfig, RequirementsGate, TrainingPackager
from leaffliction.utils import ZipPackager, Hasher


def main() -> None:
    parser = CLIBuilder().build_train_parser()
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(getattr(args, "out_dir", "artifacts"))
    out_zip = Path(getattr(args, "out_zip", "learnings.zip"))

    # Configuration pour ML traditionnel
    cfg = TrainConfig(
        # Paramètres ML (pas d'epochs/batch_size comme CNN)
        model_type=getattr(args, "model_type", "svm"),  # svm, random_forest, knn
        valid_ratio=getattr(args, "valid_ratio", 0.2),
        seed=getattr(args, "seed", 42),
        augment_train=getattr(args, "augment", True),
        augmentations_per_image=getattr(args, "aug_per_image", 3),
        export_augmented_images=getattr(args, "export_images", False),
    )

    # Composants ML traditionnels
    scanner = DatasetScanner()
    splitter = DatasetSplitter()
    labels = LabelEncoder()
    
    # Moteurs d'augmentation et transformation
    aug_engine = AugmentationEngine.default_six()
    tf_engine = TransformationEngine.default_six()
    
    # Feature extractor (cœur du système ML)
    feature_extractor = FeatureExtractor(tf_engine.tfs)
    
    # Model factory pour ML
    model_factory = MLModelFactory()

    # Trainer ML
    trainer = MLTrainer(
        dataset_scanner=scanner,
        dataset_splitter=splitter,
        augmentation_engine=aug_engine,
        feature_extractor=feature_extractor,
        model_factory=model_factory,
        labels=labels,
    )

    print(f"🚀 Starting ML training with {cfg.model_type.upper()}...")
    print(f"   Dataset: {dataset_dir}")
    print(f"   Augmentation: {'Enabled' if cfg.augment_train else 'Disabled'}")
    print(f"   Validation ratio: {cfg.valid_ratio:.0%}")
    print()

    # Entraînement
    metrics = trainer.train(dataset_dir=dataset_dir, out_dir=out_dir, cfg=cfg)

    # Validation des contraintes (accuracy > 90%, valid >= 100)
    gate = RequirementsGate()
    gate.assert_ok(metrics)

    # Packaging final
    packager = TrainingPackager(zip_packager=ZipPackager())
    artifacts_dir = packager.prepare_artifacts_dir(tmp_dir=out_dir)
    packager.build_zip(artifacts_dir=artifacts_dir, out_zip=out_zip)

    # Génération signature.txt
    hasher = Hasher()
    sha1_hash = hasher.ft_sha1_file(out_zip)
    signature_file = Path("signature.txt")
    signature_file.write_text(sha1_hash + "\n")

    print()
    print("=" * 60)
    print("✅ Training completed successfully!")
    print("=" * 60)
    print(f"📊 Train accuracy: {metrics.train_accuracy:.2%}")
    print(f"📊 Valid accuracy: {metrics.valid_accuracy:.2%}")
    print(f"📊 Valid count: {metrics.valid_count}")
    print(f"📦 Model saved to: {out_zip}")
    print(f"📝 Signature saved to: {signature_file}")
    print(f"   SHA1: {sha1_hash}")
    print("=" * 60)


if __name__ == "__main__":
    main()
