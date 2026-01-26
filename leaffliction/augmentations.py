from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple
import cv2
import numpy as np
import albumentations as A
from leaffliction.utils import PathManager
from random import Random
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class AugmentationEngine:
    """
    Engine responsible for applying image augmentations and balancing
    a training dataset through image transforms.

    This class defines a fixed set of augmentation transforms and applies
    them iteratively to under-represented classes until all classes reach
    a comparable number of samples.

    Augmented images are generated with controlled randomness and saved
    to disk as new dataset items.
    """

    augs = {
        "Rotate": A.Rotate(
            limit=(-15, 15),
            p=1.0
        ),
        "Blur": A.Blur(
            blur_limit=(5, 9),
            p=1.0
        ),
        "Contrast": A.ColorJitter(
            contrast=(1.4, 2),
            p=1.0
        ),
        "Scaling": A.Affine(
            scale=(1.2, 1.5),
            p=1.0
        ),
        "Illumination": A.ColorJitter(
            brightness=(1.4, 2),
            p=1.0
        ),
        "Projective": A.Perspective(
            fit_output=True,
            scale=(0.05, 0.2),
            p=1.0
        ),
    }

    def apply_all(self, img: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Apply all defined augmentations to a single image.

        This method is primarily intended for visualization and inspection
        of the available transformations.

        Args:
            img: the image to process in a Matlike format.

        Returns:
            A dictionary mapping augmentation names to their resulting images.
        """
        return {
            name: augmentation(image=img)['image']
            for name, augmentation in self.augs.items()
            }

    def augment_dataset(
            self,
            train_items: List[Tuple[Path, int]],
            seed: int,
            dataset_dir: Path,
            output_dir: Path,
            ) -> List[Tuple[Path, int]]:
        """
        Augment the training dataset to balance class distributions.

        For each under-represented class, augmentations are applied
        iteratively to existing images until the class reaches the
        target number of samples. If all images are exhausted and the
        deficit remains, the process restarts with randomized transforms.

        Args:
            train_items: Training set as (image_path, class_id) tuples.
            seed: Random seed for reproducible shuffling.
            dataset_dir: Root directory of the original dataset
            output_dir: Root directory where augmented images are saved.

        Returns:
            The augmented training set, including newly generated samples.
        """
        pm = PathManager()

        train_items_grouped = defaultdict(list)
        for item in train_items:
            train_items_grouped[item[1]].append(item)

        target_count = len(max(train_items_grouped.values(), key=len))

        augmentations = list(self.augs.keys())
        nb_augs = len(augmentations)

        augmented_items = []
        for class_id, items in train_items_grouped.items():
            current_count = len(items)
            deficit = target_count - current_count

            total_nb_pass = target_count // (current_count * nb_augs) + 1
            # Use a round-robin strategy to cycle through augmentations
            # and source images evenly when generating new samples.
            for gen_img_count in range(deficit):
                augm_name = augmentations[gen_img_count % nb_augs]
                item = items[(gen_img_count // nb_augs) % current_count]

                image = cv2.imread(str(item[0]), cv2.IMREAD_COLOR_RGB)

                result = self.augs[augm_name](image=image)
                transformed_image = result['image']

                pass_id = (gen_img_count // nb_augs) // current_count
                suffix = (
                    f"_{augm_name}{pass_id}"
                    if total_nb_pass > 1
                    else f"_{augm_name}"
                )
                augm_path = pm.make_suffixed_path(
                    pm.mirror_path(item[0], dataset_dir, output_dir),
                    suffix
                )
                pm.ensure_dir(augm_path.parent)
                cv2.imwrite(str(augm_path), transformed_image)
                augmented_items.append((augm_path, item[1]))

        train_items.extend(augmented_items)
        rdm = Random(seed)
        rdm.shuffle(train_items)
        return train_items
    
    def load_augmented_items(
            self,
            items: List[Tuple[Path, int]],
            img_size: Tuple[int, int] = (224, 224)
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Charge simplement les images (augmentées ou non) depuis le disque
        et retourne X, y pour PyTorch, sans transformation.

        Args:
            items: [(image_path, class_id), ...]
            img_size: taille de redimensionnement

        Returns:
            X: torch.Tensor (N, 3, H, W)
            y: torch.Tensor (N,)
        """
        X_list = []
        y_list = []

        for img_path, label in items:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"⚠️  Could not load image: {img_path}")
                continue

            # BGR → RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Resize
            img = cv2.resize(img, img_size)

            # HWC → CHW
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))

            X_list.append(img)
            y_list.append(label)

        if not X_list:
            raise ValueError("No images could be loaded from augmented items.")

        X = torch.from_numpy(np.stack(X_list, axis=0))   # (N, 3, H, W)
        y = torch.tensor(y_list, dtype=torch.long)

        return X, y



class AugmentationSaver:
    """
    Utility class for saving augmented images to disk.

    Augmented images are written to a separate directory from the
    original dataset, with filename suffixes indicating the applied
    transformations. This class is primarily intended for evaluation
    and visualization purposes.
    """

    def __init__(self, path_manager: Any) -> None:
        self.path_manager = path_manager

    def save_all(
            self,
            image_path: Path,
            output_dir: Path,
            results: Dict[str, np.ndarray]
            ) -> List[Path]:
        """
        Saves all augmented versions of an image to disk.

        Args:
            image_path: Path to the original image
            output_dir: Root directory where augmented images are saved.
            results: Mapping of augmentation names to transformed images.

        Returns:
            List of paths to the saved augmented images.
        """
        saved_paths = []
        for transformation in results:

            transformed_image = results[transformation]

            augm_path = self.path_manager.make_suffixed_path(
                output_dir / image_path.name,
                f"_{transformation}",
            )
            self.path_manager.ensure_dir(augm_path.parent)
            cv2.imwrite(str(augm_path), transformed_image)
            saved_paths.append(augm_path)

        return saved_paths
