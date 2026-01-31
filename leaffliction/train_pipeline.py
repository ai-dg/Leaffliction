from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from Transformation import TransformationDirectory
import sys
from typing import Optional
from leaffliction.model import ModelConfig
from leaffliction.model import ConvolutionalNeuralNetwork

@dataclass
class TrainConfig:
    # Paramètres PyTorch
    epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-3
    valid_ratio: float = 0.2
    seed: int = 42
    img_size: Tuple[int, int] = (224, 224)
    augment_train: bool = True
    transform_train: bool = True
    augmentations_per_image: int = 3
    extra: Dict[str, Any] = field(default=dict)


@dataclass
class Metrics:
    train_accuracy: float
    valid_accuracy: float
    valid_count: int
    history_valid_acc : Dict[int, int]
    history_train_acc : Dict[int, int]
    history_train_loss : Dict[int, int]    
    notes: Dict[str, Any] = field(default=dict)



class Trainer:
    """
    Orchestrateur d'entraînement PyTorch.
    """

    def __init__(
        self, 
        dataset_scanner: Any, 
        dataset_splitter: Any, 
        augmentation_engine: Any,
        transformation_engine: Any,
        labels: Any
    ) -> None:
        self.dataset_scanner = dataset_scanner
        self.dataset_splitter = dataset_splitter
        self.augmentation_engine = augmentation_engine
        self.transformation_engine = transformation_engine
        self.labels = labels

    def train(self, dataset_dir: Path, out_dir: Path, cfg: TrainConfig) -> Metrics:
        print("📂 Scanning dataset...")
        index = self.dataset_scanner.scan(dataset_dir)
        print(f"   Found {index.num_classes} classes, {index.size} images")
        print()
        
        self.labels.fit(index.class_names)
        
        print("✂️  Splitting dataset...")
        train_items, valid_items = self.dataset_splitter.split(
            index.items,
            cfg.valid_ratio,
            cfg.seed,
            stratified=True
        )
        print(f"   Train: {len(train_items)} images")
        print(f"   Valid: {len(valid_items)} images")
        print()

        X_train = None
        y_train = None
        X_valid = None
        y_valid = None

        aug_dir = None
        if cfg.augment_train:
            print("🔄 Augmenting train set...")
            aug_dir = out_dir / "augmented"
            train_items = self.augmentation_engine.augment_dataset(
                train_items,
                cfg.seed,
                dataset_dir=dataset_dir,
                output_dir=aug_dir,
            )
            print(f"   Created {len(train_items)} total images (original + augmented)")
            print()

        if cfg.transform_train:
            transform_dir = out_dir / "transform"
            batch_engine = TransformationDirectory(self.transformation_engine)
            batch_engine.run(dataset_dir, transform_dir)

            if aug_dir:
                batch_engine.run(aug_dir, transform_dir)

            train_items = self.transformation_engine.extract_transformed_items(train_items, transform_dir)
            valid_items = self.transformation_engine.extract_transformed_items(valid_items, transform_dir)

            print(
                f"Transforms extracted → "
                f"train: {len(train_items)}, valid: {len(valid_items)}"
            )
            
            X_train, y_train = self.transformation_engine.load_transformer_items(train_items, capacity=0.5)
            X_valid, y_valid = self.transformation_engine.load_transformer_items(valid_items, capacity=1)
        
        if X_train is None or \
            y_train is None or \
            X_valid is None or \
            y_valid is None:
            raise ValueError("X_train must be initialized before use")


        print("📦 Creating DataLoaders...")
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
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Valid batches: {len(valid_loader)}")
        print()

        print(f"Number of channels: {X_train.shape[1]}")
        
        # 7. Construire modèle
        print("🤖 Building PyTorch model...")
        
        model_cfg = ModelConfig(
            num_classes=index.num_classes,
            input_channels=X_train.shape[1],  # Nombre de transformations
            img_size=cfg.img_size,
            seed=cfg.seed
        )
        model = ConvolutionalNeuralNetwork(
            num_classes=model_cfg.num_classes,
            input_channels=model_cfg.input_channels
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        print(f"   Model: {type(model).__name__}")
        print(f"   Device: {device}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print()
        
        # 8. Entraîner
        print("🚀 Training model...")
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
            # Training
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
                print(f"   Epoch {epoch+1}/{cfg.epochs} - "
                      f"Train Acc: {train_acc:.2%} - "
                      f"Valid Acc: {valid_acc:.2%}")
        
        training_time = time.time() - start_time
        print(f"   Training completed in {training_time:.1f}s")
        print()
        
        model.load_state_dict(torch.load(out_dir / "best_model.pth"))
        
        print("📈 Final evaluation...")
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
        
        print(f"   Train accuracy: {train_acc:.2%}")
        print(f"   Valid accuracy: {valid_acc:.2%}")
        print()
        
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
        
        # 11. Sauvegarder bundle
        print("💾 Saving model...")
        from leaffliction.model import InferenceManager
        bundle = InferenceManager(
            model=model,
            labels=self.labels,
            transformation_engine=self.transformation_engine,
            cfg=model_cfg
        )
        bundle.save(out_dir / "model")
        print(f"   Model saved to {out_dir / 'model'}")
        print()
        
        return metrics


class TrainingPackager:
    """
    Prépare les artefacts puis zip le tout.
    """

    def __init__(self, zip_packager: Any) -> None:
        self.zip_packager = zip_packager

    def prepare_artifacts_dir(self, tmp_dir: Path) -> Path:
        """
        Prépare le dossier d'artefacts à zipper.
        Pour PyTorch: model/
        """
        import shutil

        artifacts_dir = tmp_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Copier le dossier model/
        model_dir = tmp_dir / "model"
        dst_model_dir = artifacts_dir / "model"

        if model_dir.exists():
            # 🔥 Si destination existe, on la supprime pour éviter FileExistsError
            if dst_model_dir.exists():
                shutil.rmtree(dst_model_dir)

            shutil.copytree(model_dir, dst_model_dir)

        return artifacts_dir

    def build_zip(self, artifacts_dir: Path, out_zip: Path) -> None:
        print("📦 Creating learnings.zip...")
        self.zip_packager.zip_dir(artifacts_dir, out_zip)
        print(f"   ZIP created: {out_zip}")



class ModelChecker:
    """
    Valide les contraintes du sujet:
    - valid_accuracy > 0.90
    - valid_count >= 100
    """

    def assert_ok(self, metrics: Metrics) -> None:
        """
        Vérifie que les métriques respectent les contraintes.
        Lève ValueError si non conforme.
        """
        print("✅ Checking requirements...")
        
        # Contrainte 1: accuracy > 90%
        if metrics.valid_accuracy < 0.90:
            raise ValueError(
                f"❌ Validation accuracy {metrics.valid_accuracy:.2%} < 90%. "
                f"TraiZipPackagerning failed to meet requirements."
            )
        print(f"   ✓ Validation accuracy: {metrics.valid_accuracy:.2%} >= 90%")
        
        # Contrainte 2: validation set >= 100 images
        if metrics.valid_count < 100:
            raise ValueError(
                f"❌ Validation set has {metrics.valid_count} images < 100. "
                f"Increase dataset size or reduce valid_ratio."
            )
        print(f"   ✓ Validation set size: {metrics.valid_count} >= 100")
        print()
