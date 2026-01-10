from __future__ import annotations

from pathlib import Path
import cv2

from leaffliction.cli import CLIBuilder
from leaffliction.utils import PathManager
from leaffliction.augmentations import AugmentationEngine, AugmentationSaver
from leaffliction.plotting import GridPlotter


def main() -> None:
    parser = CLIBuilder().build_augmentation_parser()
    args = parser.parse_args()
    # TODO - add the two varaibales below as parameter in cli
    output_dir = Path("evaluation/augmented")
    dataset_dir = Path("leaves")

    image_path = Path(args.image_path)

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    engine = AugmentationEngine()
    
    results = engine.apply_all(img)

    grid = GridPlotter()
    grid.show_grid("Augmentations", results, original=img)

    pm = PathManager()
    saver = AugmentationSaver(pm)
    saved_paths = saver.save_all(image_path, dataset_dir, output_dir, results)
    
    print(f"\n✅ Augmentations saved:")
    for path in saved_paths:
        print(f"   - {path.name}")


if __name__ == "__main__":
    main()
