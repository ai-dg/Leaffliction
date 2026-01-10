from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple
import cv2
import numpy as np
import albumentations as A
from leaffliction.utils import PathManager
from random import Random
from collections import defaultdict

# TODO - remove commented out code & every reference/dependencies in the
#        codebase
# class Augmentation(Protocol):
#     """
#     Interface d'une augmentation.
#     'name' sert au suffix (_Flip, _Rotate, etc.)
#     """

#     @property
#     def name(self) -> str:
#         ...

#     def apply(self, img: np.ndarray) -> np.ndarray:
#         ...


# @dataclass
# class FlipHorizontalAug:
#     name: str = "FlipH"

#     def apply(self, img: np.ndarray) -> np.ndarray:
#         raise NotImplementedError


# @dataclass
# class FlipVerticalAug:
#     name: str = "FlipV"

#     def apply(self, img: np.ndarray) -> np.ndarray:
#         raise NotImplementedError


# @dataclass
# class RotateAug:
#     angle: float
#     name: str = "Rotate"

#     def apply(self, img: np.ndarray) -> np.ndarray:
#         raise NotImplementedError


# @dataclass
# class BrightnessContrastAug:
#     brightness: float = 0.0
#     contrast: float = 0.0
#     name: str = "BrightContrast"

#     def apply(self, img: np.ndarray) -> np.ndarray:
#         raise NotImplementedError


# @dataclass
# class GaussianBlurAug:
#     sigma: float = 1.0
#     name: str = "Blur"

#     def apply(self, img: np.ndarray) -> np.ndarray:
#         raise NotImplementedError


# @dataclass
# class RandomCropResizeAug:
#     crop_ratio: float = 0.9
#     name: str = "CropResize"

#     def apply(self, img: np.ndarray) -> np.ndarray:
#         raise NotImplementedError


class AugmentationEngine:
    """
    Applique une liste d'augmentations et retourne un dict:
    { "FlipH": img1, "Rotate": img2, ... }
    
    Utilisé pour:
    1. Visualisation (Augmentation.py) - apply_all()
    2. Augmentation du dataset de training - augment_dataset()
    """

    augs = {
        "Rotate": A.Rotate(
            limit=(-15,15),
            p=1.0
        ),
        "Blur": A.Blur(
            blur_limit=(5,9),
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

    # TODO - Remove default_six everywhere in the codebase and refac
    # @classmethod
    # def default_six(cls) -> "AugmentationEngine":
    #     """
    #     Factory: les 6 augmentations mandatory.
    #     """
    #     raise NotImplementedError

    def apply_all(self, img: np.ndarray) -> Dict[str, np.ndarray]:
        """Applique toutes les augmentations pour visualisation"""
        return { name: augmentation(image=img)['image'] for (name, augmentation) in self.augs.items()}
    
    # TODO - Remove apply_random everywhere in the codebase and refac
    # def apply_random(self, img: np.ndarray, n: int = 2) -> np.ndarray:
    #     """Applique n augmentations aléatoires"""
    #     raise NotImplementedError
    
    def augment_dataset(
        self,
        train_items: List[Tuple[Path, int]],
        seed: int,
        dataset_dir: Path,
        output_dir: Path,
    ) -> List[Tuple[Path, int]]:
        """
        Pour chaque image de train:
        1. Applique N augmentations différentes
        2. Sauvegarde les nouvelles images sur disque
        3. Retourne la liste étendue: originales + augmentées
        
        Exemple:
        Input: 400 images Apple_healthy
        Output: 400 originales + 1200 augmentées = 1600 images
        """
        pm = PathManager()

        train_items_grouped = defaultdict(list)
        for item in train_items:
            train_items_grouped[item[1]].append(item)

        target_count = len(max(train_items_grouped.values(), key=len))

        augmentations = list(self.augs.keys())
        nb_augs = len(augmentations)

        # print(target_count)
        # print(nb_augs)

        for class_item in train_items_grouped:
            items = train_items_grouped[class_item]
            current_count = len(items)
            deficit = target_count - current_count
            # print("class: ", class_item)
            # print("current count: ", current_count)
            # print("deficit: ", deficit)

            for gen_img_count in range(deficit):
                augm_name = augmentations[gen_img_count % nb_augs]
                item = items[(gen_img_count // nb_augs) % current_count]

                image = cv2.imread(str(item[0]), cv2.IMREAD_COLOR_RGB)
                

                result = self.augs[augm_name](image=image)
                transformed_image = result['image']

                img_transform_id = (gen_img_count // nb_augs) % current_count
                augm_path = pm.make_suffixed_path(
                    pm.mirror_path(item[0], dataset_dir, output_dir),
                    f"_{augm_name}{img_transform_id}"
                )
                pm.ensure_dir(augm_path.parent)
                cv2.imwrite(str(augm_path), transformed_image)
                train_items.append((augm_path, item[1]))
        
        rdm = Random(seed) 
        rdm.shuffle(train_items)
        return train_items


class AugmentationSaver:
    """
    Sauvegarde les images augmentées dans le même dossier
    avec suffixes conformes au sujet.
    Utilisé par Augmentation.py pour la visualisation.
    """

    def __init__(self, path_manager: Any) -> None:
        self.path_manager = path_manager

    def save_all(
            self,
            image_path: Path,
            dataset_dir: Path,
            output_dir: Path,
            results: Dict[str, np.ndarray]
        ) -> List[Path]:
        """
        Renvoie la liste des paths écrits.
        Exemple attendu: image (1)_Flip.JPG, image (1)_Rotate.JPG, etc.
        """
        pm = PathManager()
        saved_paths = []
        for transformation in results:
            
            transformed_image = results[transformation]

            augm_path = pm.make_suffixed_path(
                pm.mirror_path(image_path, dataset_dir, output_dir),
                f"_{transformation}",
            )
            pm.ensure_dir(augm_path.parent)
            cv2.imwrite(str(augm_path), transformed_image)
            saved_paths.append(augm_path)
        
        return saved_paths


def main():
    from leaffliction.dataset import DatasetScanner, DatasetIndex, DatasetSplitter

    dataset_dir = Path("leaves")
    out_dir = Path("artifacts")
    aug_dir = out_dir / "augmented"
    scanner = DatasetScanner()
    index = scanner.scan(dataset_dir)
    
    splitter = DatasetSplitter()
    train_items, valid_items = splitter.split(index.items, 0.2, 42, stratified=True)

    augmentation_engine = AugmentationEngine()
    train_items = augmentation_engine.augment_dataset(
        train_items,
        42,
        dataset_dir,
        aug_dir
    )

    index_aug = scanner.scan(out_dir)
    print(index_aug.counts)
    print(index.counts)

if __name__ == "__main__":
    main()