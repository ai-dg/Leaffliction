from __future__ import annotations

from pathlib import Path
from tabnanny import verbose
import cv2
from plantcv import plantcv as pcv
import numpy as np

from leaffliction.cli import ArgsManager
from leaffliction.utils import PathManager
from leaffliction.transformations import TransformationEngine, TransformationDirectory
from leaffliction.plotting import Plotter
from leaffliction.utils import Logger


def main() -> None:
    """
    Apply image transformations to single images or directories.

    Processes plant leaf images through various transformations (grayscale,
    hue extraction, masking, etc.) and displays or saves results.

    :return: None
    :rtype: None
    """
    parser = ArgsManager().build_transformation_parser()
    args = parser.parse_args()

    if args.only:
        engine = TransformationEngine.only_selected(args.only, args.verbose)
    else:
        engine = TransformationEngine.default_six(args.verbose)
    grid = Plotter()
    pm = PathManager()
    logger = Logger(args.verbose)

    if getattr(args, "src", None) and getattr(args, "dst", None):
        src = Path(args.src)
        dst = Path(args.dst)
        group_images = TransformationDirectory(engine=engine, verbose=args.verbose)
        group_images.run(src=src, dst=dst, recursive=getattr(args, "recursive", True))
        logger.info(f"Directory of images transformation completed: {src} → {dst}")
        return

    image_path = Path(args.image_path)

    img = cv2.imread(str(image_path))
    if img is None:
        logger.error(f"Error: Could not load image from {image_path}")
        return
    

    results = engine.apply_all(img)

    grid.plot_grid("Transformations", results, original=img)

    
    logger.info(f"Transformations applied:")
    for name in results.keys():
        logger.info(f"   - {name}")


if __name__ == "__main__":
    main()
