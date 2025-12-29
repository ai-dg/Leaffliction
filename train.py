from __future__ import annotations

from pathlib import Path

from leaffliction.cli import CLIBuilder
from leaffliction.dataset import DatasetScanner, DatasetSplitter, TFDataConfig, TFDatasetBuilder
from leaffliction.augmentations import KerasAugmentationsFactory
from leaffliction.model import ModelFactory, LabelEncoder
from leaffliction.train_pipeline import Trainer, TrainConfig, RequirementsGate, TrainingPackager
from leaffliction.utils import ZipPackager


def main() -> None:
    parser = CLIBuilder().build_train_parser()
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(getattr(args, "out_dir", "artifacts"))
    out_zip = Path(getattr(args, "out_zip", "learnings.zip"))

    cfg = TrainConfig(
        epochs=getattr(args, "epochs", 10),
        batch_size=getattr(args, "batch_size", 32),
        lr=getattr(args, "lr", 1e-3),
        valid_ratio=getattr(args, "valid_ratio", 0.2),
        seed=getattr(args, "seed", 42),
        img_size=(getattr(args, "img_h", 224), getattr(args, "img_w", 224)),
        augment_in_train=getattr(args, "augment", True),
        export_increased_images=getattr(args, "export_images", True),
    )

    scanner = DatasetScanner()
    splitter = DatasetSplitter()
    labels = LabelEncoder()
    model_factory = ModelFactory()

    trainer = Trainer(
        dataset_scanner=scanner,
        dataset_splitter=splitter,
        model_factory=model_factory,
        labels=labels,
    )

    metrics = trainer.train(dataset_dir=dataset_dir, out_dir=out_dir, cfg=cfg)

    gate = RequirementsGate()
    gate.assert_ok(metrics)

    # Zip final
    packager = TrainingPackager(zip_packager=ZipPackager())
    artifacts_dir = packager.prepare_artifacts_dir(tmp_dir=out_dir)
    packager.build_zip(artifacts_dir=artifacts_dir, out_zip=out_zip)


if __name__ == "__main__":
    main()
