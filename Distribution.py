from __future__ import annotations

from pathlib import Path

from leaffliction.cli import CLIBuilder
from leaffliction.dataset import DatasetScanner
from leaffliction.plotting import DistributionPlotter


def main() -> None:
    parser = CLIBuilder().build_distribution_parser()
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)

    scanner = DatasetScanner()
    index = scanner.scan(dataset_dir)

    title = f"Dataset distribution: {index.root.name}"

    plotter = DistributionPlotter()
    if args.mode == "both":
        plotter.plot_both(index.counts, title)
    if args.mode == "pie":
        plotter.plot_pie(index.counts, title=title)
    if args.mode == "bar":
        plotter.plot_bar(index.counts, title=title)


if __name__ == "__main__":
    main()
