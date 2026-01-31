from __future__ import annotations

from typing import Dict, Optional, Any
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
plt.style.use("./style/leaffliction.mplstyle")
import numpy as np
from math import ceil
import cv2



class DistributionPlotter:
    """
    Pie chart + bar chart pour la distribution des classes.
    """

    def plot_pie(self, counts: Dict[str, int], title: str, save_to: Optional[Path] = None, verbose : bool = True) -> None:
        counts_array = [value for key, value in counts.items()]
        
        print(f"Number of classes to plot : {len(counts_array)}")
        class_names = list(counts.keys())
        print(f"{counts.keys()}")

        plt.figure()
        plt.pie(
            counts_array,
            labels=class_names,
            autopct="%1.1f%%",
            startangle=90
        )
        plt.title(title)
        plt.axis("equal")

        if save_to is not None:
            plt.savefig(save_to, bbox_inches="tight")

        plt.show()
        plt.close()



    def plot_bar(self, counts: Dict[str, int], title: str, save_to: Optional[Path] = None) -> None:
        counts_array = [value for key, value in counts.items()]
        class_names = list(counts.keys())

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

    def plot_both(self, counts: Dict[str, int], title: str, save_to: Optional[Path] = None) -> None:
        counts_array = list(counts.values())
        class_names = list(counts.keys())

    
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

    def plot_learning_curve(self, train : Dict[int, int], valid : Dict[int, int]):
        
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

    def plot_learning_curve_loss(self, loss : Dict[int,int]):
        x_loss = [epoch for epoch, acc in loss.items()]
        y_loss = [acc for epoch, acc in loss.items()]

        
        plt.plot(x_loss, y_loss, label="train curve loss")

        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("loss")
        plt.savefig("./learning curve_loss.jpg")
        plt.close()        


class GridPlotter:
    """
    Affiche une grille (original + variantes).
    Pratique pour Augmentation et Transformation.
    """
    def _imshow_safe(self, img: np.ndarray) -> None:
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

    def show_grid(
        self,
        title: str,
        images: Dict[str, np.ndarray],
        original: Optional[np.ndarray] = None,
        save_to: Optional[Path] = None,
        max_cols: int = 3
    ) -> None:

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
    counts = {
        "Apple Healthy" : 1600,
        "Apple Black Rot" : 800,
        "Apple Rust" : 400,
        "Apple Scab" : 600,
        "Grape Black Rot" : 500,
        "Grape Esca" : 1500,
        "Grape Healthy" : 400,
        "Grape Spot" : 700
    }

    title = "Fruits graph"

    save_to = Path("plots")

    distribution = DistributionPlotter()

    distribution.plot_pie(counts, title, save_to)
    distribution.plot_bar(counts, title, save_to)
    distribution.plot_both(counts, title, save_to)




if __name__ == "__main__":
    main()