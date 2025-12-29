from __future__ import annotations

from pathlib import Path

from leaffliction.cli import CLIBuilder
from leaffliction.utils import PathManager
from leaffliction.transformations import TransformationEngine, BatchTransformer
from leaffliction.plotting import GridPlotter


def main() -> None:
    parser = CLIBuilder().build_transformation_parser()
    args = parser.parse_args()

    engine = TransformationEngine.default_six()
    grid = GridPlotter()
    pm = PathManager()

    # Mode batch: -src / -dst
    if getattr(args, "src", None) and getattr(args, "dst", None):
        src = Path(args.src)
        dst = Path(args.dst)
        batch = BatchTransformer(engine=engine, path_manager=pm)
        batch.run(src=src, dst=dst, recursive=getattr(args, "recursive", True))
        return

    # Mode single image
    image_path = Path(args.image_path)

    # TODO: charge l'image
    # img = ...

    # results = engine.apply_all(img)
    results = {}  # placeholder

    # grid.show_grid("Transformations", results, original=img)
    grid.show_grid("Transformations", results, original=None)


if __name__ == "__main__":
    main()
