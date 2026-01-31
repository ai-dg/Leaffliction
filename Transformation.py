from __future__ import annotations

from pathlib import Path
import cv2
from plantcv import plantcv as pcv
import numpy as np

from leaffliction.cli import ArgsManager
from leaffliction.utils import PathManager
from leaffliction.transformations import TransformationEngine, TransformationDirectory
from leaffliction.plotting import Plotter


def main() -> None:
    parser = ArgsManager().build_transformation_parser()
    args = parser.parse_args()

    engine = TransformationEngine.default_six()
    grid = Plotter()
    pm = PathManager()

    # Mode batch: -src / -dst
    if getattr(args, "src", None) and getattr(args, "dst", None):
        src = Path(args.src)
        dst = Path(args.dst)
        batch = TransformationDirectory(engine=engine, path_manager=pm)
        batch.run(src=src, dst=dst, recursive=getattr(args, "recursive", True))
        print(f"✅ Batch transformation completed: {src} → {dst}")
        return

    # Mode single image
    image_path = Path(args.image_path)

    # Charger l'image avec OpenCV (ML traditionnel)
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Convertir BGR → RGB pour affichage correct
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Appliquer toutes les transformations
    results = engine.apply_all(img)
    # pcv.plot_image(results['GaussianMask'])

    # Afficher la grille (original + 6 transformations)
    grid.plot_grid("Transformations", results, original=img)
    
    print(f"\n✅ Transformations applied:")
    for name in results.keys():
        print(f"   - {name}")


if __name__ == "__main__":
    main()
