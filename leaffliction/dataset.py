from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import tensorflow as tf
from utils import PathManager
from collections import Counter


@dataclass
class DatasetIndex:
    """
    Représentation standard du dataset:
    - root: dossier racine
    - class_names: nom des sous-dossiers (classes)
    - items: [(path_image, class_id), ...]
    - counts: {class_name: count}
    """
    root: Path
    class_names: List[str]
    items: List[Tuple[Path, int]]
    counts: Dict[str, int]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def size(self) -> int:
        return len(self.items)


class DatasetScanner:
    """
    Scan un dataset de type:
    root/
      class_a/
        img1.jpg
      class_b/
        img2.jpg
    """

    def scan(self, root: Path) -> DatasetIndex:
        """Construit un DatasetIndex depuis un dossier racine."""
        root = root.resolve()
        pm = PathManager()
        
        lst_img = pm.iter_images(root, recursive=True)

        class_names = []
        items = []
        counts = {}
        for img in lst_img:
            name = img.parent.name
            if name not in class_names:
                class_names.append(name)
            items.append((img, class_names.index(name)))
            if name not in counts:
                counts[name] = 1
            else:
                counts[name] = counts[name] + 1

        return DatasetIndex(root, class_names, items, counts)


class DatasetSplitter:
    """
    Split train/valid à partir de la liste items.
    """

    def split(
        self,
        items: List[Tuple[Path, int]],
        valid_ratio: float,
        seed: int,
        stratified: bool = True
    ) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]:
        """
        Retourne (train_items, valid_items)
        - stratified: conserve approx les proportions de classes
        """
        import random
        random.seed(seed)
        
        if not stratified:
            # Simple shuffle et split
            items_copy = items.copy()
            random.shuffle(items_copy)
            split_idx = int(len(items_copy) * (1 - valid_ratio))
            return items_copy[:split_idx], items_copy[split_idx:]
        
        # Stratifié: grouper par classe
        class_items = {}
        for path, label in items:
            if label not in class_items:
                class_items[label] = []
            class_items[label].append((path, label))
        
        train_items = []
        valid_items = []
        
        for label, items_list in class_items.items():
            items_copy = items_list.copy()
            random.shuffle(items_copy)
            n_valid = int(len(items_copy) * valid_ratio)
            valid_items.extend(items_copy[:n_valid])
            train_items.extend(items_copy[n_valid:])
        
        random.shuffle(train_items)
        random.shuffle(valid_items)
        
        return train_items, valid_items
