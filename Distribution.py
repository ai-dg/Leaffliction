from __future__ import annotations

from pathlib import Path

from leaffliction.cli import ArgsManager
from leaffliction.dataset import DatasetScanner
from leaffliction.plotting import Plotter


def main() -> None:
    """
    Analyze and visualize dataset class distribution.

    Scans the dataset directory, counts images per class, and displays
    distribution charts (pie, bar, or both) based on user selection.

    :return: None
    :rtype: None
    """
    parser = ArgsManager().build_distribution_parser()
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)

    scanner = DatasetScanner()
    index = scanner.scan(dataset_dir)

    title = f"Dataset distribution: {index.root.name}"

    plotter = Plotter(args.verbose)
    if args.mode == "both":
        plotter.plot_both(index.counts, title, save_to=args.save)
    if args.mode == "pie":
        plotter.plot_pie(index.counts, title=title, save_to=args.save)
    if args.mode == "bar":
        plotter.plot_bar(index.counts, title=title, save_to=args.save)


if __name__ == "__main__":
    main()
