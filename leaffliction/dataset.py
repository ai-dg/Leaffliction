"""Dataset scanning, indexing, and splitting.

Provides utilities to scan directory-based image datasets, build indices,
and split data into training and validation sets.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from leaffliction.utils import PathManager
from random import Random
from collections import defaultdict, Counter


@dataclass
class DatasetIndex:
    """
    Immutable index describing an image classification dataset.

    Attributes:
        root: Root directory of the dataset.
        class_names: Names of the class subdirectories.
        items: List of (image_path, class_id) tuples.
        counts: {class_name: count}
    """
    root: Path
    class_names: List[str]
    items: List[Tuple[Path, int]]
    counts: Dict[str, int]

    @property
    def num_classes(self) -> int:
        """Number of distinct classes in the dataset."""
        return len(self.class_names)

    @property
    def size(self) -> int:
        """Total number of items in the dataset."""
        return len(self.items)


class DatasetScanner:
    """
    Scans a directory-based image dataset and builds a DatasetIndex.

    Expected directory structure::

        root/
            class_a/
                img1.jpg
                img2.jpg
            class_b/
                img2.jpg
    """

    def scan(self, root: Path) -> DatasetIndex:
        """
        Scan a dataset directory and build a DatasetIndex.

        Args:
            root: Path to the dataset root directory.

        Returns:
            A DatasetIndex containing paths, class IDs,
            and dataset statistics.
        """
        root = root.resolve()
        pm = PathManager()

        lst_img = pm.iter_images(root, recursive=True)

        class_names = sorted({img.parent.name for img in lst_img})

        class_to_id = {name: i for i, name in enumerate(class_names)}

        items = [(img, class_to_id[img.parent.name]) for img in lst_img]
        counts = dict(Counter(img.parent.name for img in lst_img))

        return DatasetIndex(root, class_names, items, counts)


class DatasetSplitter:
    """
    Utility class to split a dataset into training and validation sets.
    """

    def split(
        self,
        items: List[Tuple[Path, int]],
        valid_ratio: float,
        seed: int,
        stratified: bool = True
    ) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]:
        """
        Split items into training and validation subsets.

        If stratified is True, a least one sample per class is assigned
        to the validation set when possible.

        Args:
            items: List of (image_path, class_id) tuples.
            valid_ratio: Fraction of items to use for validation.
            seed: Random seed for reproducible shuffling.
            stratified: If True, preserve class proportions approximately.

        Returns:
            A tuple (train_items, valid_items).
        """

        rdm = Random(seed)
        if not stratified:
            items_copy = items.copy()
            rdm.shuffle(items_copy)
            split_idx = max(1, int(len(items_copy) * (1 - valid_ratio)))
            return items_copy[:split_idx], items_copy[split_idx:]

        train_items, valid_items = [], []
        items_grouped = defaultdict(list)
        for item in items:
            items_grouped[item[1]].append(item)

        for class_id in sorted(items_grouped):
            rdm.shuffle(items_grouped[class_id])

        for class_id in sorted(items_grouped):
            split_idx = max(1, int(len(items_grouped[class_id]) * valid_ratio))
            valid_items.extend(items_grouped[class_id][:split_idx])
            train_items.extend(items_grouped[class_id][split_idx:])

        rdm.shuffle(valid_items)
        rdm.shuffle(train_items)
        return (train_items, valid_items)
