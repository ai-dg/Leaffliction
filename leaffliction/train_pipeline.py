"""
Training pipeline orchestration.

Provides configuration, metrics tracking, model training, validation,
artifact packaging, and model quality checks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from Transformation import TransformationDirectory
import sys
import shutil
from typing import Optional
from leaffliction.model import ModelConfig
from leaffliction.model import ConvolutionalNeuralNetwork
from leaffliction.utils import Logger

@dataclass
class TrainConfig:
    """
    Configuration for training.

    Attributes:
        epochs: Number of passes over the training set.
        batch_size: Number of samples processed per batch.
        lr: Learning rate.
        valid_ratio: Fraction of the dataset used for validation.
        seed: Random seed.
        img_size: Image size as (height, width).
        augment_train: If True, augmented images are included in the training set.
        transform_train: If True, transformed images are included in the training set.
        extra: Extra keyword arguments.
    """
    epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-3
    valid_ratio: float = 0.2
    seed: int = 42
    img_size: Tuple[int, int] = (224, 224)
    augment_train: bool = True
    transform_train: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Metrics:
    """
    Training metrics.

    Attributes:
        train_accuracy: Training accuracy for the final evaluation.
        valid_accuracy: Validation accuracy for the final evaluation.
        valid_count: Number of samples in the validation set.
        history_valid_acc: Mapping epoch -> validation accuracy.
        history_train_acc: Mapping epoch -> training accuracy.
        history_train_loss: Mapping epoch -> training loss.
        notes: Extra info as keyword args.
    """
    train_accuracy: float
    valid_accuracy: float
    valid_count: int
    history_valid_acc: Dict[int, float] = field(default_factory=dict)
    history_train_acc: Dict[int, float] = field(default_factory=dict)
    history_train_loss: Dict[int, float] = field(default_factory=dict)    
    notes: Dict[str, Any] = field(default_factory=dict)


class Trainer:
    """
    Training orchestrator for the full ML pipeline.
    
    Handles dataset scanning, splitting, augmentation, transformation, 
    model training, and evaluation.
    """

    def __init__(
        self, 
        dataset_scanner: Any, 
        dataset_splitter: Any, 
        augmentation_engine: Any,
        transformation_engine: Any,
        labels: Any,
        verbose: bool = True,
    ) -> None:
        self.dataset_scanner = dataset_scanner
        self.dataset_splitter = dataset_splitter
        self.augmentation_engine = augmentation_engine
        self.transformation_engine = transformation_engine
        self.labels = labels
        self.logger = Logger(verbose)

    def train(self, dataset_dir: Path, out_dir: Path, cfg: TrainConfig) -> Metrics:
        logger = self.logger
        index = self.dataset_scanner.scan(dataset_dir)
        logger.info(f"Dataset scan: Found {index.num_classes} classes, {index.size} images")
        
        self.labels.fit(index.class_names)
        
        logger.info("Splitting dataset...")
        train_items, valid_items = self.dataset_splitter.split(
            index.items,
            cfg.valid_ratio,
            cfg.seed,
            stratified=True
        )
        logger.info(
            f"Dataset split: Train: {len(train_items)} images, Valid: {len(valid_items)} images"
        )

        X_train = None
        y_train = None
        X_valid = None
        y_valid = None

        aug_dir = None
        if cfg.augment_train:
            logger.info("Augmenting train set...")
            aug_dir = out_dir / "augmented"
            train_items = self.augmentation_engine.augment_dataset(
                train_items,
                cfg.seed,
                dataset_dir=dataset_dir,
                output_dir=aug_dir,
            )
            logger.info(f"    Created {len(train_items)} total images (original + augmented)")

        if cfg.transform_train:
            transform_dir = out_dir / "transform"
            batch_engine = TransformationDirectory(self.transformation_engine)
            batch_engine.run(dataset_dir, transform_dir)

            if aug_dir:
                batch_engine.run(aug_dir, transform_dir)

            train_items = self.transformation_engine.extract_transformed_items(train_items, transform_dir)
            valid_items = self.transformation_engine.extract_transformed_items(valid_items, transform_dir)

            logger.info(
                f"Transforms extracted -> "
                f"train: {len(train_items)}, valid: {len(valid_items)}"
            )
            
            X_train, y_train = self.transformation_engine.load_transformer_items(train_items, capacity=0.05)
            X_valid, y_valid = self.transformation_engine.load_transformer_items(valid_items, capacity=1)
        
        if X_train is None or \
            y_train is None or \
            X_valid is None or \
            y_valid is None:
            raise ValueError("X_train must be initialized before use")


        logger.info("Creating DataLoaders...")
        train_dataset = TensorDataset(X_train, y_train)
        valid_dataset = TensorDataset(X_valid, y_valid)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=cfg.batch_size, 
            shuffle=True
        )
        valid_loader = DataLoader(
            valid_dataset, 
            batch_size=cfg.batch_size, 
            shuffle=False
        )
        logger.info(f"   Train batches: {len(train_loader)}")
        logger.info(f"   Valid batches: {len(valid_loader)}")

        logger.info(f"Number of channels: {X_train.shape[1]}")
        
        model_cfg = ModelConfig(
            num_classes=index.num_classes,
            input_channels=X_train.shape[1],
            img_size=cfg.img_size,
            seed=cfg.seed
        )
        model = ConvolutionalNeuralNetwork(
            num_classes=model_cfg.num_classes,
            input_channels=model_cfg.input_channels
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        logger.info(f"Model: {type(model).__name__}")
        logger.info(f"Device: {device}")
        logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        logger.info("Training model...")
        import time
        start_time = time.time()
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        
        best_valid_acc = 0.0

        history_train_acc = {}
        history_valid_acc = {}
        best_valid_acc = float("-inf")
        best_epoch = -1

        history_train_loss = {}
        
        
        for epoch in range(cfg.epochs):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == y_batch).sum().item()
                train_total += y_batch.size(0)
            
            train_acc = train_correct / train_total
            history_train_acc[epoch + 1] = train_acc
            history_train_loss[epoch + 1] = train_loss
            
            model.eval()
            valid_correct = 0
            valid_total = 0
            
            with torch.no_grad():
                for X_batch, y_batch in valid_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    _, predicted = torch.max(outputs, 1)
                    valid_correct += (predicted == y_batch).sum().item()
                    valid_total += y_batch.size(0)
            
            valid_acc = valid_correct / valid_total
            history_valid_acc[epoch + 1] = valid_acc
            
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                torch.save(model.state_dict(), out_dir / "best_model.pth")
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(f"   Epoch {epoch+1}/{cfg.epochs} - "
                                f"Train Acc: {train_acc:.2%} - "
                                f"Valid Acc: {valid_acc:.2%}")
        
        training_time = time.time() - start_time
        logger.info(f"   Training completed in {training_time:.1f}s")
        
        model.load_state_dict(torch.load(out_dir / "best_model.pth"))
        
        logger.info("Final evaluation...")
        model.eval()
        train_correct = 0
        valid_correct = 0
        
        with torch.no_grad():
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == y_batch).sum().item()
            
            for X_batch, y_batch in valid_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                _, predicted = torch.max(outputs, 1)
                valid_correct += (predicted == y_batch).sum().item()
        
        train_acc = train_correct / len(train_dataset)
        valid_acc = valid_correct / len(valid_dataset)
        
        logger.info(f"   Train accuracy: {train_acc:.2%}")
        logger.info(f"   Valid accuracy: {valid_acc:.2%}")
        
        metrics = Metrics(
            train_accuracy=train_acc,
            valid_accuracy=valid_acc,
            valid_count=len(valid_items),
            history_train_acc=history_train_acc,
            history_valid_acc=history_valid_acc,
            history_train_loss=history_train_loss,
            notes={
                "training_time": training_time,
                "epochs": cfg.epochs,
                "best_epoch": epoch + 1,
                "n_transforms": X_train.shape[1]
            }
        )
        
        logger.info("Saving model...")
        from leaffliction.model import InferenceManager
        inference = InferenceManager(
            model=model,
            labels=self.labels,
            transformation_engine=self.transformation_engine,
            cfg=model_cfg
        )
        inference.save(out_dir / "model")
        logger.info(f"   Model saved to {out_dir / 'model'}")
        
        return metrics


class TrainingPackager:
    """
    Prepares and packages training artifacts into a zip file.
    """

    def __init__(
            self,
            zip_packager: Any,
            verbose: bool = True
            ) -> None:
        self.zip_packager = zip_packager
        self.logger = Logger(verbose)

    def prepare_artifacts_dir(self, tmp_dir: Path) -> Path:
        """
        Prepare the artificats directory to be zipped.

        It creates the 'artificats' directory if it doesn't exist.
        A 'model' is created under that directory, whom purpose is
        to contain the model artificats.

        Args:
            tmp_dir: the parent directory of the 'artifacts' directory.
        """
        

        artifacts_dir = tmp_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        model_dir = tmp_dir / "model"
        dst_model_dir = artifacts_dir / "model"

        if model_dir.exists():
            if dst_model_dir.exists():
                shutil.rmtree(dst_model_dir)

            shutil.copytree(model_dir, dst_model_dir)

        return artifacts_dir

    def build_zip(self, artifacts_dir: Path, out_zip: Path) -> None:
        """
        Zips the artificat directory
        
        Args:
            artifacts_dir: Path of the 'artifacts' directory
            out_zip: Output Path of the zip archive.
        """
        self.logger.info("Creating learnings.zip...")
        self.zip_packager.zip_dir(artifacts_dir, out_zip)
        self.logger.info(f"   ZIP created: {out_zip}")



class ModelChecker:
    """
    Validates trained model against project constraints:
    - Validation accuracy >= 90%
    - Validation set size >= 100
    """

    def __init__(self, verbose: bool = True):
        self.logger = Logger(verbose)

    def assert_ok(self, metrics: Metrics) -> None:
        """
        Ensure metrics respect constraints.
        Exit if non compliant.
        """
        
        if metrics.valid_accuracy < 0.90:
            self.logger.error(
                f"Validation accuracy {metrics.valid_accuracy:.2%} < 90%. "
                f"Training failed to meet requirements."
            )
            exit()
        self.logger.info(f"   [OK] Validation accuracy: {metrics.valid_accuracy:.2%} >= 90%")
        
        if metrics.valid_count < 100:
            self.logger.error(
                f"Validation set has {metrics.valid_count} images < 100. "
                f"Increase dataset size or reduce valid_ratio."
            )
            exit()
        self.logger.info(f"   [OK] Validation set size: {metrics.valid_count} >= 100")
