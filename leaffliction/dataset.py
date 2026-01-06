from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from utils import PathManager
# from leaffliction.utils import PathManager
from random import Random
from collections import defaultdict, Counter

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
                valid_items.extend(items_grouped[class_id][:nb_valid_items])
                train_items.extend(items_grouped[class_id][nb_valid_items:])

            rdm.shuffle(valid_items)
            rdm.shuffle(train_items)
        else:
            shuffled_items = items.copy()
            rdm.shuffle(shuffled_items)
            nb_valid_items = int(len(shuffled_items) * valid_ratio)
            valid_items = shuffled_items[:nb_valid_items]
            train_items = shuffled_items[nb_valid_items:]

        return (train_items, valid_items)

    def display_split(
        self,
        train_items: List[Tuple[Path, int]],
        valid_items: List[Tuple[Path, int]],
        n_samples: int = 3
    ) -> None:
        """Display split statistics and sample items."""
        train_classes = Counter(item[1] for item in train_items)
        valid_classes = Counter(item[1] for item in valid_items)
        
        print("=" * 60)
        print("DATASET SPLIT STATISTICS")
        print("=" * 60)
        print(f"Train items: {len(train_items)}")
        print(f"Valid items: {len(valid_items)}")
        print(f"Total: {len(train_items) + len(valid_items)}")
        print()
        print(f"Train class distribution: {dict(train_classes)}")
        print(f"Valid class distribution: {dict(valid_classes)}")
        print()
        print(f"First {n_samples} TRAIN items:")
        for item in train_items[:n_samples]:
            print(f"  {item[0].name} (class_id: {item[1]})")
        print()
        print(f"First {n_samples} VALID items:")
        for item in valid_items[:n_samples]:
            print(f"  {item[0].name} (class_id: {item[1]})")
        print("=" * 60)

def main():
    scanner = DatasetScanner()
    dataset = scanner.scan(Path("leaves"))
    
    print(dataset.counts)

    splitter = DatasetSplitter()
    train_items, valid_items = splitter.split(dataset.items, 0.2, 4)
    
    splitter.display_split(train_items, valid_items, n_samples=5)

if __name__ == "__main__":
    main()
