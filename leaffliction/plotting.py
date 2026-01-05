from __future__ import annotations

from typing import Dict, Optional, Any
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
plt.style.use("./style/leaffliction.mplstyle")



class DistributionPlotter:
    """
    Pie chart + bar chart pour la distribution des classes.
    """

    def plot_pie(self, counts: Dict[str, int], title: str, save_to: Optional[Path] = None) -> None:
        counts_array = [value for key, value in counts.items()]
        class_names = list(counts.keys())
        
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




class GridPlotter:
    """
    Affiche une grille (original + variantes).
    Pratique pour Augmentation et Transformation.
    """

    def show_grid(
        self,
        title: str,
        images: Dict[str, Any],
        original: Optional[Any] = None,
        save_to: Optional[Path] = None,
        max_cols: int = 3
    ) -> None:
        raise NotImplementedError



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