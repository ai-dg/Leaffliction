"""Apply image augmentations to a single image and save results.

Provides command-line interface for testing augmentation transforms,
visualizing results in a grid, and saving augmented images to disk.
"""
from __future__ import annotations

from pathlib import Path
import cv2

from leaffliction.cli import ArgsManager
from leaffliction.utils import PathManager
from leaffliction.augmentations import AugmentationEngine, AugmentationSaver
from leaffliction.plotting import Plotter
from leaffliction.utils import Logger

def main() -> None:
    """
    Apply augmentations to a single image from the command-line.
    
    Parses CLI arguments, loads the image, applies augmentation transforms,
    displays results in a grid, and saves augmented images to disk.
    """
    parser = ArgsManager().build_augmentation_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    image_path = Path(args.image_path)
    logger = Logger(args.verbose)
    

    img = cv2.imread(str(image_path))
    if img is None:
        logger.error(f"Error: Could not load image from {image_path}")
        return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    engine = AugmentationEngine()
    
    results = engine.apply_all_script(img)

    grid = Plotter()
    grid.plot_grid("Augmentations", results, original=img)

    pm = PathManager()
    saver = AugmentationSaver(pm)
    saved_paths = saver.save_all(image_path, output_dir, results)
    
    logger.info("Augmentations saved:")
    for path in saved_paths:
        logger.info(f"   - {path.parent}/{path.name}")


if __name__ == "__main__":
    main()
