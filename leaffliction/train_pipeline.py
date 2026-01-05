from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List
import numpy as np


@dataclass
class TrainConfig:
    # Paramètres ML traditionnels (pas d'epochs/batch_size comme CNN)
    model_type: str = "svm"  # svm, random_forest, knn
    valid_ratio: float = 0.2
    seed: int = 42
    augment_train: bool = True
    augmentations_per_image: int = 3
    export_augmented_images: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Metrics:
    train_accuracy: float
    valid_accuracy: float
    valid_count: int
    notes: Dict[str, Any] = field(default_factory=dict)


class MLTrainer:
    """
    Orchestrateur d'entraînement ML traditionnel.
    """

    def __init__(
        self, 
        dataset_scanner: Any, 
        dataset_splitter: Any, 
        augmentation_engine: Any,
        feature_extractor: Any,
        model_factory: Any, 
        labels: Any
    ) -> None:
        self.dataset_scanner = dataset_scanner
        self.dataset_splitter = dataset_splitter
        self.augmentation_engine = augmentation_engine
        self.feature_extractor = feature_extractor
        self.model_factory = model_factory
        self.labels = labels

    def train(self, dataset_dir: Path, out_dir: Path, cfg: TrainConfig) -> Metrics:
        """
        Pipeline complet ML traditionnel:
        1. Scanner le dataset
        2. Split train/valid (stratifié)
        3. (Optionnel) Augmenter le train set (images physiques)
        4. Extraire features (train + valid)
        5. Normaliser features (StandardScaler)
        6. Construire le modèle ML (SVM/RF/KNN)
        7. Entraîner
        8. Évaluer accuracy
        9. Sauvegarder bundle (model.pkl, scaler.pkl, labels.json)
        
        Retourne: Metrics avec train_accuracy, valid_accuracy, valid_count
        """
        from sklearn.preprocessing import StandardScaler
        
        # 1. Scanner dataset
        print("📂 Scanning dataset...")
        index = self.dataset_scanner.scan(dataset_dir)
        print(f"   Found {index.num_classes} classes, {index.size} images")
        print()
        
        # 2. Fitter le LabelEncoder
        self.labels.fit(index.class_names)
        
        # 3. Split train/valid
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
        
        # 4. (Optionnel) Augmenter train set
        if cfg.augment_train:
            print("🔄 Augmenting train set...")
            aug_dir = out_dir / "augmented"
            train_items = self.augmentation_engine.augment_dataset(
                train_items,
                aug_dir,
                cfg.augmentations_per_image
            )
            print(f"   Created {len(train_items)} total images (original + augmented)")
            print()
        
        # 5. Extraire features
        print("🔍 Extracting features...")
        print("   Train features...")
        X_train, y_train = self.feature_extractor.extract_batch(train_items)
        print(f"   Train features: {X_train.shape}")
        
        print("   Valid features...")
        X_valid, y_valid = self.feature_extractor.extract_batch(valid_items)
        print(f"   Valid features: {X_valid.shape}")
        print()
        
        # 6. Normaliser features
        print("📊 Normalizing features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_valid_scaled = scaler.transform(X_valid)
        print("   StandardScaler fitted")
        print()
        
        # 7. Construire modèle
        print(f"🤖 Building {cfg.model_type.upper()} model...")
        from leaffliction.model import ModelConfig
        model_cfg = ModelConfig(
            num_classes=index.num_classes,
            seed=cfg.seed
        )
        model = self.model_factory.build(model_cfg, cfg.model_type)
        print(f"   Model: {type(model).__name__}")
        print()
        
        # 8. Entraîner
        print("🚀 Training model...")
        import time
        start_time = time.time()
        model.fit(X_train_scaled, y_train)
        training_time = time.time() - start_time
        print(f"   Training completed in {training_time:.1f}s")
        print()
        
        # 9. Évaluer
        print("📈 Evaluating...")
        train_acc = model.score(X_train_scaled, y_train)
        valid_acc = model.score(X_valid_scaled, y_valid)
        print(f"   Train accuracy: {train_acc:.2%}")
        print(f"   Valid accuracy: {valid_acc:.2%}")
        print()
        
        # 10. Créer métriques
        metrics = Metrics(
            train_accuracy=train_acc,
            valid_accuracy=valid_acc,
            valid_count=len(valid_items),
            notes={
                "model_type": cfg.model_type,
                "training_time": training_time,
                "n_features": X_train.shape[1]
            }
        )
        
        # 11. Sauvegarder bundle
        print("💾 Saving model...")
        from leaffliction.model import MLModelBundle
        bundle = MLModelBundle(
            model=model,
            scaler=scaler,
            labels=self.labels,
            feature_extractor=self.feature_extractor,
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
        Pour ML traditionnel: model/, (optionnel) augmented/
        """
        artifacts_dir = tmp_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Copier le dossier model/
        model_dir = tmp_dir / "model"
        if model_dir.exists():
            import shutil
            shutil.copytree(model_dir, artifacts_dir / "model")
        
        return artifacts_dir

    def build_zip(self, artifacts_dir: Path, out_zip: Path) -> None:
        """
        Crée le fichier learnings.zip depuis artifacts_dir.
        """
        print("📦 Creating learnings.zip...")
        self.zip_packager.ft_zip_dir(artifacts_dir, out_zip)
        print(f"   ZIP created: {out_zip}")


class RequirementsGate:
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
                f"Training failed to meet requirements."
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
