# leaffliction/transformations.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Protocol, Tuple, Any, Optional

import cv2
import numpy as np
import torch
from plantcv import plantcv as pcv
import rembg
import matplotlib.pyplot as plt


def _ensure_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _to_bgr_if_bgra(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def _to_gray_if_color(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] in (3, 4):
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


@dataclass
class TFContext:
    img: np.ndarray  # BGR ou RGB, on ne touche pas ici
    cache: Dict[str, np.ndarray]

    def get(self, key: str) -> np.ndarray:
        return self.cache[key]

    def has(self, key: str) -> bool:
        return key in self.cache

    def set(self, key: str, value: np.ndarray) -> np.ndarray:
        self.cache[key] = value
        return value

    def ensure_base(self, threshold: int = 35, fill_size: int = 200) -> None:
        if self.has("img_no_bg"):
            return

        img = _ensure_uint8(self.img)

        img_no_bg = rembg.remove(img)
        img_no_bg = _to_bgr_if_bgra(_ensure_uint8(img_no_bg))
        self.set("img_no_bg", img_no_bg)


        grayscale = pcv.rgb2gray_lab(rgb_img=img_no_bg, channel="l")
        grayscale = _ensure_uint8(grayscale)
        
        # pcv.plot_image(grayscale)

        mean_L = float(grayscale.mean())
        p10 = float(np.percentile(grayscale, 10))
        self.set("mean_L", np.array(mean_L, dtype=np.float32))

        if mean_L < 70 or p10 < 30:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            L = clahe.apply(grayscale)
            gamma = 0.7
            L = np.clip(((L / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)

            grayscale = L

        self.set("grayscale_l", grayscale)



        thresh = pcv.threshold.binary(gray_img=grayscale, threshold=threshold, object_type="light")
        thresh = _ensure_uint8(thresh)
        white_ratio = (thresh > 0).mean()
        if white_ratio < 0.02 or white_ratio > 0.98:
            thresh = pcv.threshold.otsu(gray_img=grayscale, object_type="light")
            thresh = _ensure_uint8(thresh)
        
        self.set("thresh", thresh)


        filled = pcv.fill(bin_img=thresh, size=fill_size)
        filled = _ensure_uint8(filled)
        self.set("filled", filled)


        gaussian = pcv.gaussian_blur(img=filled, ksize=(3, 3))
        gaussian = _ensure_uint8(gaussian)
        self.set("gaussian", gaussian)

        masked = pcv.apply_mask(img=img, mask=gaussian, mask_color="white")
        masked = _ensure_uint8(masked)
        self.set("masked", masked)

    def ensure_roi(self) -> None:
        if self.has("roi_image") and self.has("kept_mask"):
            return

        if not self.has("filled") or not self.has("masked"):
            raise RuntimeError("Base pipeline not computed. Call ensure_base() first.")

        image = _ensure_uint8(self.img)
        masked = self.get("masked")
        filled = self.get("filled")

        roi_start_x = 0
        roi_start_y = 0
        roi_h = image.shape[0]  # rows
        roi_w = image.shape[1]  # cols
        roi_line_w = 5

        # PlantCV ROI rectangle
        roi = pcv.roi.rectangle(
            img=masked,
            x=roi_start_x,
            y=roi_start_y,
            w=roi_w,
            h=roi_h
        )

        kept_mask = pcv.roi.filter(mask=filled, roi=roi, roi_type="partial")
        kept_mask = _ensure_uint8(kept_mask)
        self.set("kept_mask", kept_mask)

        roi_image = image.copy()

        if roi_image.ndim == 3 and roi_image.shape[2] == 4:
            roi_image[kept_mask != 0] = (0, 255, 0, 255)
        else:
            roi_image[kept_mask != 0] = (0, 255, 0)

        cv2.rectangle(
            roi_image,
            (roi_start_x, roi_start_y),
            (roi_start_x + roi_w - 1, roi_start_y + roi_h - 1),
            (255, 0, 0),
            thickness=roi_line_w
        )

        self.set("roi_image", _ensure_uint8(roi_image))


class Transformation(Protocol):
    @property
    def name(self) -> str:
        ...

    def apply(self, ctx: TFContext) -> np.ndarray:
        ...



@dataclass
class NoBg:
    name: str = "NoBg"

    def apply(self, ctx: TFContext) -> np.ndarray:
        ctx.ensure_base()
        return ctx.get("img_no_bg")


@dataclass
class GrayscaleL:
    name: str = "LAB_L"

    def apply(self, ctx: TFContext) -> np.ndarray:
        ctx.ensure_base()
        return ctx.get("grayscale_l")


@dataclass
class Thresh:
    threshold: int = 35
    fill_size: int = 200
    name: str = "Thresh"

    def apply(self, ctx: TFContext) -> np.ndarray:
        ctx.ensure_base(threshold=self.threshold, fill_size=self.fill_size)
        return ctx.get("thresh")


@dataclass
class Filled:
    threshold: int = 35
    fill_size: int = 200
    name: str = "Filled"

    def apply(self, ctx: TFContext) -> np.ndarray:
        ctx.ensure_base(threshold=self.threshold, fill_size=self.fill_size)
        return ctx.get("filled")


@dataclass
class GaussianMask:
    threshold: int = 35
    fill_size: int = 200
    name: str = "GaussianMask"

    def apply(self, ctx: TFContext) -> np.ndarray:
        ctx.ensure_base(threshold=self.threshold, fill_size=self.fill_size)
        return ctx.get("gaussian")


@dataclass
class Masked:
    threshold: int = 35
    fill_size: int = 200
    name: str = "Masked"

    def apply(self, ctx: TFContext) -> np.ndarray:
        ctx.ensure_base(threshold=self.threshold, fill_size=self.fill_size)
        return ctx.get("masked")


@dataclass
class RoiImage:
    threshold: int = 35
    fill_size: int = 200
    name: str = "RoiImage"

    def apply(self, ctx: TFContext) -> np.ndarray:
        ctx.ensure_base(threshold=self.threshold, fill_size=self.fill_size)
        ctx.ensure_roi()
        return ctx.get("roi_image")


def _draw_pseudolandmarks(image: np.ndarray, pseudolandmarks, color_bgr, radius: int) -> np.ndarray:
    out = image.copy()
    # gérer 4 canaux
    if out.ndim == 3 and out.shape[2] == 4 and len(color_bgr) == 3:
        color = (*color_bgr, 255)
    else:
        color = color_bgr

    for p in pseudolandmarks:
        if p is None:
            continue
        p = np.asarray(p)
        if p.size < 2:
            continue

        row = int(p[0][0])  # y
        col = int(p[0][1])  # x

        cv2.circle(out, (row, col), radius, color, thickness=-1)

    return out

@dataclass
class AnalyzeImage:
    name: str = "AnalyzeImage"

    def apply(self, ctx: TFContext) -> np.ndarray:
        ctx.ensure_base()
        ctx.ensure_roi()

        img = _ensure_uint8(ctx.img)
        labeled_mask = ctx.get("kept_mask")


        out = pcv.analyze.size(img=img, labeled_mask=labeled_mask)

        
        if out is None:
        
            out = img.copy()

        return _ensure_uint8(out)


@dataclass
class PseudoLandmarks:
    threshold: int = 35
    fill_size: int = 200
    name: str = "PseudoLandmarks"

    def apply(self, ctx: TFContext) -> np.ndarray:
        ctx.ensure_base(threshold=self.threshold, fill_size=self.fill_size)
        ctx.ensure_roi()

        img = _ensure_uint8(ctx.img)
        kept_mask = ctx.get("kept_mask")

        pseudo_img = img.copy()
        top_x, bottom_x, center_v_x = pcv.homology.x_axis_pseudolandmarks(
            img=pseudo_img,
            mask=kept_mask,
            label="default"
        )
        # Red
        pseudo_img = _draw_pseudolandmarks(pseudo_img, top_x, (255, 0, 0), 5)       # red
        # Magenta
        pseudo_img = _draw_pseudolandmarks(pseudo_img, bottom_x, (255, 0, 255), 5)  # magenta
        # Blue
        pseudo_img = _draw_pseudolandmarks(pseudo_img, center_v_x, (0, 0, 255), 5)  # blue

        return _ensure_uint8(pseudo_img)


class TransformationEngine:
    def __init__(self, tfs: List[Transformation]) -> None:
        self.tfs = tfs

    @classmethod
    def default_six(cls) -> "TransformationEngine":
        tfs: List[Transformation] = [
            # GrayscaleL(),
            # Thresh(threshold=35, fill_size=200),
            # Filled(threshold=35, fill_size=200),
            GaussianMask(threshold=120, fill_size=200),
            Masked(threshold=35, fill_size=200),
            RoiImage(),
            AnalyzeImage(),
            PseudoLandmarks(threshold=35, fill_size=200),
        ]
        return cls(tfs=tfs)

    def apply_all(self, img: np.ndarray) -> Dict[str, np.ndarray]:
        ctx = TFContext(img=_ensure_uint8(img), cache={})

        results: Dict[str, np.ndarray] = {}
        for tf in self.tfs:
            results[tf.name] = tf.apply(ctx)

        # if ctx.has("kept_mask"):
        #     results["kept_mask"] = ctx.get("kept_mask")

        return results

    def apply_all_as_tensor(self, img: np.ndarray) -> torch.Tensor:
        ctx = TFContext(img=_ensure_uint8(img), cache={})

        channels: List[np.ndarray] = []
        for tf in self.tfs:
            transformed = tf.apply(ctx)

  
            gray = _to_gray_if_color(transformed)
            gray = _ensure_uint8(gray)

            channels.append(gray.astype(np.float32) / 255.0)

        return torch.from_numpy(np.stack(channels, axis=0))

    def batch_transform(
        self,
        items: List[Tuple[Path, int]],
        img_size: Tuple[int, int] = (224, 224)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        X_list: List[torch.Tensor] = []
        y_list: List[int] = []

        for img_path, label in items:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"⚠️  Warning: Could not load {img_path}")
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size)

            tensor = self.apply_all_as_tensor(img)
            X_list.append(tensor)
            y_list.append(label)

        X = torch.stack(X_list)
        y = torch.tensor(y_list, dtype=torch.long)
        return X, y


class BatchTransformer:
    def __init__(self, engine: TransformationEngine, path_manager: PathManager) -> None:
        self.engine = engine
        self.pm = path_manager

    def run(self, src: Path, dst: Path, recursive: bool = True) -> None:
        self.pm.ensure_dir(dst)

        # 1) liste images
        if src.is_file():
            paths = [src]
        else:
            paths = self.pm.iter_images(src, recursive=recursive)

        for p in paths:
            img = cv2.imread(str(p))
            if img is None:
                print(f"⚠️  Could not load {p}")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.engine.apply_all(img_rgb)

            mirrored_file = (
                self.pm.mirror_path(p, src_root=src, target_root=dst)
                if src.is_dir()
                else (dst / p.name)
            )
            out_dir = self.pm.ensure_dir(mirrored_file.parent)

            stem = p.stem

            orig_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            orig_path = out_dir / f"{stem}_original.png"
            cv2.imwrite(str(orig_path), _ensure_uint8(orig_bgr))

            for name, out_img in results.items():
                out = out_img

                if out.ndim == 3 and out.shape[2] == 3:
                    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
                elif out.ndim == 3 and out.shape[2] == 4:
                    out_bgr = out
                else:
                    out_bgr = out

                out_path = out_dir / f"{stem}_{name}.png"
                cv2.imwrite(str(out_path), _ensure_uint8(out_bgr))




def main():
    [...]


if __name__ == "__main__":
    main()




# def plot_stat_hist(label, sc=1):

#     """
#     Retrieve the histogram x and y values and plot them
#     """

#     y = pcv.outputs.observations['default_1'][label]['value']
#     x = [
#         i * sc
#         for i in pcv.outputs.observations['default_1'][label]['label']
#     ]
#     if label == "hue_frequencies":
#         x = x[:int(255 / 2)]
#         y = y[:int(255 / 2)]
#     if (
#         label == "blue-yellow_frequencies" or
#         label == "green-magenta_frequencies"
#     ):
#         x = [x + 128 for x in x]
#     plt.plot(x, y, label=label)


# def plot_histogram(image, kept_mask):

#     """
#     Plot the histogram of the image
#     """

#     dict_label = {
#         "blue_frequencies": 1,
#         "green_frequencies": 1,
#         "green-magenta_frequencies": 1,
#         "lightness_frequencies": 2.55,
#         "red_frequencies": 1,
#         "blue-yellow_frequencies": 1,
#         "hue_frequencies": 1,
#         "saturation_frequencies": 2.55,
#         "value_frequencies": 2.55
#     }

#     labels, _ = pcv.create_labels(mask=kept_mask)
#     pcv.analyze.color(
#         rgb_img=image,
#         colorspaces="all",
#         labeled_mask=labels,
#         label="default"
#     )

#     plt.subplots(figsize=(16, 9))
#     for key, val in dict_label.items():
#         plot_stat_hist(key, val)

#     plt.legend()

#     plt.title("Color Histogram")
#     plt.xlabel("Pixel intensity")
#     plt.ylabel("Proportion of pixels (%)")
#     plt.grid(
#         visible=True,
#         which='major',
#         axis='both',
#         linestyle='--',
#     )
#     plt.show()
#     plt.close()



# def create_roi_image(image, masked, filled):
#     """
#     Create an image with the ROI rectangle and the mask
#     """
#     roi_start_x = 0
#     roi_start_y = 0
#     roi_h = image.shape[0]  
#     roi_w = image.shape[1]  
#     roi_line_w = 5

  
#     roi = pcv.roi.rectangle(
#         img=masked,
#         x=roi_start_x,
#         y=roi_start_y,
#         w=roi_w,
#         h=roi_h
#     )

#     kept_mask = pcv.roi.filter(mask=filled, roi=roi, roi_type="partial")

#     roi_image = image.copy()


#     roi_image[kept_mask != 0] = (0, 255, 0)


#     cv2.rectangle(
#         roi_image,
#         (roi_start_x, roi_start_y),
#         (roi_start_x + roi_w - 1, roi_start_y + roi_h - 1),
#         (255, 0, 0),
#         thickness=roi_line_w
#     )

#     return roi_image, kept_mask


# def draw_pseudolandmarks(image, pseudolandmarks, color, radius):
#     out = image.copy()

#     if out.ndim == 3 and out.shape[2] == 4 and len(color) == 3:
#         color = (*color, 255)

#     for p in pseudolandmarks:
#         if p is None:
#             continue
#         p = np.asarray(p)
#         if p.size < 2:
#             continue

#         row = int(p[0][0])  # y
#         col = int(p[0][1])  # x

#         cv2.circle(out, (row, col), radius, color, thickness=-1)

#     return out



# def create_pseudolandmarks_image(image, kept_mask):
#     """
#     Create a displayable image with the pseudolandmarks
#     """
#     pseudo_img = image.copy()

#     top_x, bottom_x, center_v_x = pcv.homology.x_axis_pseudolandmarks(
#         img=pseudo_img, mask=kept_mask, label="default"
#     )

#     pseudo_img = draw_pseudolandmarks(pseudo_img, top_x, (255, 0, 0), 5)       # red
#     pseudo_img = draw_pseudolandmarks(pseudo_img, bottom_x, (255, 0, 255), 5)  # magenta
#     pseudo_img = draw_pseudolandmarks(pseudo_img, center_v_x, (0, 0, 255), 5)  # blue

#     return pseudo_img

# def main():
    
#     # path = "../leaves/images/Apple_Black_rot/image (100).JPG"
#     # path = "../leaves/images/Apple_rust/image (100).JPG"
#     # path = "../leaves/images/Apple_scab/image (100).JPG"
#     path = "../leaves/images/Apple_healthy/image (100).JPG"
#     # path = "../leaves/test.JPG"

#     # img, _, _ = pcv.readimage(filename=path, mode='rgb')

#     img = cv2.imread(path)

#     img_no_bg = rembg.remove(img)
    

#     grayscale = pcv.rgb2gray_lab(rgb_img=img_no_bg, channel="l")

#     thresh = pcv.threshold.binary(
#         gray_img=grayscale, threshold=120, object_type='light'
#     )

#     filled = pcv.fill(
#         bin_img=thresh, size=200
#     )

#     gaussian_blur = pcv.gaussian_blur(img=filled, ksize=(3,3))

#     masked = pcv.apply_mask(img=img, mask=gaussian_blur, mask_color="white")


#     roi_image, kept_mask = create_roi_image(img, masked, filled)

#     analyze_image = pcv.analyze.size(img=img, labeled_mask=kept_mask)

#     pseudolandmarks = create_pseudolandmarks_image(img, kept_mask)


#     pcv.plot_image(img, title="original")
#     pcv.plot_image(gaussian_blur, title="Gaussian")
#     pcv.plot_image(masked, title="masked")
#     pcv.plot_image(roi_image, title="roi image")
#     pcv.plot_image(analyze_image, title="analyze image")
#     pcv.plot_image(pseudolandmarks, title="pseudo")

  

#     # Create the figure to plot
#     fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(16, 9))


#     images = {
#         "Original": cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
#         "Gaussian blur": cv2.cvtColor(gaussian_blur, cv2.COLOR_BGR2RGB),
#         "Mask": cv2.cvtColor(masked, cv2.COLOR_BGR2RGB),
#         "ROI Objects": cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB),
#         "Analyze object": cv2.cvtColor(analyze_image, cv2.COLOR_BGR2RGB),
#         "Pseudolandmarks": cv2.cvtColor(pseudolandmarks, cv2.COLOR_BGR2RGB),
#     }

#     # Title of the plot
#     fig.suptitle(f"Transformation of {path}")

#     # Put the images on the plot
#     for (label, img), axe in zip(images.items(), ax.flat):

#         if label in [
#             "Pseudowithoutbg",
#             "Doublewithoutbg",
#             "Doublewithoutbg"
#         ]:
#             continue

#         axe.imshow(img)
#         axe.set_title(label)
#         axe.set(xticks=[], yticks=[])
#         axe.label_outer()

#     plt.show()
#     plt.close()

#     plot_histogram(img, kept_mask)

