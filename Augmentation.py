from __future__ import annotations

from pathlib import Path

from leaffliction.cli import CLIBuilder
from leaffliction.utils import PathManager
from leaffliction.augmentations import AugmentationEngine, AugmentationSaver
from leaffliction.plotting import GridPlotter


def main() -> None:
    parser = CLIBuilder().build_augmentation_parser()
    args = parser.parse_args()

    image_path = Path(args.image_path)

    # TODO: charge l'image (tf / PIL / cv2) dans ta logique (utils/ImageIO par ex.)
    # img = ...

    engine = AugmentationEngine.default_six()
    # results = engine.apply_all(img)
    results = {}  # placeholder

    # Affichage (original + 6)
    grid = GridPlotter()
    # grid.show_grid("Augmentations", results, original=img)
    grid.show_grid("Augmentations", results, original=None)

    # Sauvegarde des 6 images dans le même dossier
    pm = PathManager()
    saver = AugmentationSaver(pm)
    saver.save_all(image_path, results)


if __name__ == "__main__":
    main()
