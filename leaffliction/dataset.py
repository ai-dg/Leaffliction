from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from leaffliction.utils import PathManager
from random import Random
from collections import defaultdict

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

        rdm = Random(seed)
        train_items, valid_items = [], []
        if stratified:
            items_grouped = defaultdict(list)
            for item in items:
                items_grouped[item[1]].append(item)
        
            for class_id in items_grouped:
                rdm.shuffle(items_grouped[class_id])

            for class_id in items_grouped:
                nb_valid_items = int(len(items_grouped[class_id]) * valid_ratio)
                valid_items += items_grouped[class_id][:nb_valid_items + 1]
                train_items += items_grouped[class_id][nb_valid_items:]

            rdm.shuffle(valid_items)
            rdm.shuffle(train_items)
        else:
            nb_valid_items = int(len(items) * valid_ratio)
            valid_items = items[:nb_valid_items + 1]
            train_items = items[nb_valid_items:]

        return (train_items, valid_items)
