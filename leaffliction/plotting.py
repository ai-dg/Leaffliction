from __future__ import annotations
from leaffliction.utils import Logger
import cv2
from math import ceil
import numpy as np

from typing import Dict, Optional
from pathlib import Path
import matplotlib.pyplot as plt
plt.style.use("./style/leaffliction.mplstyle")


class Plotter:
    """
    Visualization utility for dataset distribution and training metrics.
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize the plotter.

        :param verbose: Enable detailed logging.
        :type verbose: bool
        :return: None
        :rtype: None
        """
        self.verbose = verbose
        self.logger = Logger(self.verbose)

    def plot_pie(self, counts: Dict[str, int], title: str,
                 save_to: Optional[str] = None) -> None:
        """
        Plot class distribution as a pie chart.

        :param counts: Dictionary mapping class names to counts.
        :type counts: Dict[str, int]
        :param title: Chart title.
        :type title: str
        :param save_to: Optional path to save the figure.
        :type save_to: Optional[str]
        :return: None
        :rtype: None
        """
        counts_array = [value for key, value in counts.items()]

        self.logger.info(
            f"Number of elements to classify: "
            f"{sum([v for k, v in counts.items()])}")
        self.logger.info(f"Number of classes to plot : {len(counts_array)}")
        self.class_names = list(counts.keys())
        for k, v in counts.items():
            self.logger.info(f"{k}:{v}")

        plt.figure()
        class_names = list(counts.keys())
        plt.pie(
            counts_array,
            labels=class_names,
            autopct="%1.1f%%",
            startangle=90
        )
        plt.title(title)
        plt.axis("equal")

        if save_to is not None:
            plt.savefig(save_to + "/pie.png", bbox_inches="tight")

        plt.show()
        plt.close()

    def plot_bar(self, counts: Dict[str, int], title: str,
                 save_to: Optional[str] = None) -> None:
        """
        Plot class distribution as a bar chart.

        :param counts: Dictionary mapping class names to counts.
        :type counts: Dict[str, int]
        :param title: Chart title.
        :type title: str
        :param save_to: Optional path to save the figure.
        :type save_to: Optional[str]
        :return: None
        :rtype: None
        """
        counts_array = [value for key, value in counts.items()]
        self.logger.info(
            f"Number of elements to classify: "
            f"{sum([v for k, v in counts.items()])}")
        self.logger.info(f"Number of classes to plot : {len(counts_array)}")
        class_names = list(counts.keys())
        for k, v in counts.items():
            self.logger.info(f"{k}:{v}")

        plt.figure(figsize=(12, 17))
        plt.bar(class_names, counts_array)
        plt.title(title)
        plt.xlabel("Classes")
        plt.ylabel("Number of images")
        plt.xticks(rotation=45, ha="right")

        if save_to is not None:
            plt.savefig(save_to, bbox_inches="tight")

        plt.show()
        plt.close()

    def plot_both(self,
                  counts: Dict[str,
                               int],
                  title: str,
                  save_to: Optional[str] = None) -> None:
        """
        Plot class distribution as both pie and bar charts side by side.

        :param counts: Dictionary mapping class names to counts.
        :type counts: Dict[str, int]
        :param title: Chart title.
        :type title: str
        :param save_to: Optional path to save the figure.
        :type save_to: Optional[str]
        :return: None
        :rtype: None
        """
        counts_array = list(counts.values())
        class_names = list(counts.keys())
        self.logger.info(
            f"Number of elements to classify: "
            f"{sum([v for k, v in counts.items()])}")
        self.logger.info(f"Number of classes to plot : {len(counts_array)}")
        for k, v in counts.items():
            self.logger.info(f"{k}:{v}")

        plt.suptitle(title)

        plt.subplot(1, 2, 1)
        plt.pie(
            counts_array,
            labels=class_names,
            autopct="%1.1f%%",
            startangle=90
        )
        plt.title("Class distribution (pie)")
        plt.axis("equal")

        plt.subplot(1, 2, 2)
        plt.bar(class_names, counts_array)
        plt.title("Class distribution (bar)")
        plt.xlabel("Classes")
        plt.ylabel("Number of images")
        plt.xticks(rotation=45, ha="right")

        plt.tight_layout(
            rect=[0, 0, 1, 0.95]
        )

        if save_to is not None:
            plt.savefig(save_to, bbox_inches="tight")

        plt.show()
        plt.close()

    def plot_learning_curve(
            self, train: Dict[int, int], valid: Dict[int, int]):
        """
        Plot training and validation accuracy curves.

        :param train: Dictionary mapping epochs to training accuracy.
        :type train: Dict[int, int]
        :param valid: Dictionary mapping epochs to validation accuracy.
        :type valid: Dict[int, int]
        :return: None
        :rtype: None
        """

        x_train = [epoch for epoch, acc in train.items()]
        y_train = [acc for epoch, acc in train.items()]

        x_valid = [epoch for epoch, acc in valid.items()]
        y_valid = [acc for epoch, acc in valid.items()]

        plt.plot(x_train, y_train, label="train curve")
        plt.plot(x_valid, y_valid, label="valid curve")

        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.savefig("./learning curve.jpg")
        plt.close()

    def plot_learning_curve_loss(self, loss: Dict[int, int]):
        """
        Plot training loss curve.

        :param loss: Dictionary mapping epochs to training loss.
        :type loss: Dict[int, int]
        :return: None
        :rtype: None
        """
        x_loss = [epoch for epoch, acc in loss.items()]
        y_loss = [acc for epoch, acc in loss.items()]

        plt.plot(x_loss, y_loss, label="train curve loss")

        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("loss")
        plt.savefig("./learning curve_loss.jpg")
        plt.close()

    def _imshow_safe(self, img: np.ndarray) -> None:
        """
        Safely display an image handling different formats
        (grayscale, RGB, RGBA).

        :param img: Image array to display.
        :type img: np.ndarray
        :return: None
        :rtype: None
        """
        if img.ndim == 2:
            plt.imshow(img, cmap="gray", vmin=0, vmax=255)
            return

        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            plt.imshow(img)
            return

        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img)
            return

        plt.imshow(img)

    def plot_grid(
        self,
        title: str,
        images: Dict[str, np.ndarray],
        original: Optional[np.ndarray] = None,
        save_to: Optional[Path] = None,
        max_cols: int = 3
    ) -> None:
        """
        Display multiple images in a grid layout.

        :param title: Grid title.
        :type title: str
        :param images: Dictionary mapping image names to arrays.
        :type images: Dict[str, np.ndarray]
        :param original: Optional original image to display first.
        :type original: Optional[np.ndarray]
        :param save_to: Optional path to save the figure.
        :type save_to: Optional[Path]
        :param max_cols: Maximum number of columns in the grid.
        :type max_cols: int
        :return: None
        :rtype: None
        """

        plt.suptitle(title)

        nb_images = len(images)
        if original is not None:
            nb_images += 1

        rows = ceil(nb_images / max_cols)

        plot_idx = 1

        if original is not None:
            plt.subplot(rows, max_cols, plot_idx)
            plt.axis('off')
            self._imshow_safe(original)
            plt.title("Original")
            plot_idx += 1

        for transformation, img in images.items():
            plt.subplot(rows, max_cols, plot_idx)
            plt.axis('off')
            self._imshow_safe(img)
            plt.title(transformation)
            plot_idx += 1

        if save_to is not None:
            plt.savefig(save_to)
        plt.show()
        plt.close()


def main():
    """
    Example usage of the Plotter class.

    :return: None
    :rtype: None
    """
    counts = {
        "Apple Healthy": 1600,
        "Apple Black Rot": 800,
        "Apple Rust": 400,
        "Apple Scab": 600,
        "Grape Black Rot": 500,
        "Grape Esca": 1500,
        "Grape Healthy": 400,
        "Grape Spot": 700
    }

    title = "Fruits graph"

    save_to = Path("plots")

    distribution = Plotter()

    distribution.plot_pie(counts, title, save_to)
    distribution.plot_bar(counts, title, save_to)
    distribution.plot_both(counts, title, save_to)


if __name__ == "__main__":
    main()
