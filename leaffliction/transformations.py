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
from leaffliction.utils import PathManager
from collections import defaultdict
from random import Random


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
            else:
                print(f"Processing: {img_path}")

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size)

            tensor = self.apply_all_as_tensor(img)
            X_list.append(tensor)
            y_list.append(label)

        X = torch.stack(X_list)
        y = torch.tensor(y_list, dtype=torch.long)
        return X, y
    
    def load_transformer_items(
        self,
        items: List[Tuple[Path, int]],
        img_size: Tuple[int, int] = (224, 224),
        capacity: float = 1.0,
        seed: int = 42
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if not (0 < capacity <= 1.0):
            raise ValueError("capacity must be in (0, 1]")

        # ---------- 1) Stratified subsampling ----------
        if capacity < 1.0:
            rdm = Random(seed)

            items_by_class = defaultdict(list)
            for path, label in items:
                items_by_class[label].append((path, label))

            limited_items: List[Tuple[Path, int]] = []

            print(f"📉 Applying capacity limit: {int(capacity * 100)}%")

            for label, class_items in items_by_class.items():
                n_total = len(class_items)
                n_keep = max(1, int(n_total * capacity))

                rdm.shuffle(class_items)
                kept = class_items[:n_keep]

                limited_items.extend(kept)

                print(
                    f"  class {label}: "
                    f"{n_keep}/{n_total} images kept"
                )

            items = limited_items
            rdm.shuffle(items)

            print(f"📦 Total images after limit: {len(items)}")

        # ---------- 2) Load images ----------
        X_list = []
        y_list = []

        for idx, (img_path, label) in enumerate(items, start=1):
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"⚠️  Could not load image: {img_path}")
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size)

            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))

            X_list.append(img)
            y_list.append(label)

            if idx % 500 == 0:
                print(f"  Loaded {idx}/{len(items)} images")

        if not X_list:
            raise ValueError("No images could be loaded from augmented items.")

        X = torch.from_numpy(np.stack(X_list, axis=0))
        y = torch.tensor(y_list, dtype=torch.long)

        return X, y
    

class BatchTransformer:
    def __init__(self, engine: TransformationEngine) -> None:
        self.engine = engine
        self.pm = PathManager()

    def _build_label_map(self, src: Path, recursive: bool) -> Dict[str, int]:
        print(f"\n🔎 Building label → id map (recursive={recursive})")

        if src.is_file():
            label = src.parent.name
            print(f"  • Single file mode → label '{label}' → id 0")
            return {label: 0}

        labels = set()
        for p in self.pm.iter_images(src, recursive=recursive):
            labels.add(p.parent.name)

        sorted_labels = sorted(labels)
        label_to_id = {label: i for i, label in enumerate(sorted_labels)}

        print("  ✔ Labels found:")
        for label, idx in label_to_id.items():
            print(f"    - {label} → {idx}")

        return label_to_id

    def run(
        self,
        src: Path,
        dst: Path,
        recursive: bool = True
    ) -> List[Tuple[Path, int]]:

        print(f"\n🚀 BatchTransformer.run")
        print(f"  src = {src}")
        print(f"  dst = {dst}")
        print(f"  recursive = {recursive}")

        self.pm.ensure_dir(dst)
        items: List[Tuple[Path, int]] = []

        # 1) list images
        if src.is_file():
            paths = [src]
            print("📄 Source is a single file")
        else:
            paths = list(self.pm.iter_images(src, recursive=recursive))
            print(f"📁 Found {len(paths)} images under {src}")

        # 2) label -> id mapping
        label_to_id = self._build_label_map(src, recursive=recursive)

        # 3) transformation names
        tf_names = [tf.name for tf in self.engine.tfs]
        print(f"\n🧪 Transformations ({len(tf_names)}): {tf_names}")

        # 4) iterate images
        for idx, p in enumerate(paths, start=1):
            print(f"\n➡️  [{idx}/{len(paths)}] Processing image:")
            print(f"    path = {p}")

            label = p.parent.name
            if label not in label_to_id:
                print(f"    ⚠️  Unknown label '{label}', skipping")
                continue

            class_id = label_to_id[label]
            print(f"    label = '{label}' → class_id = {class_id}")

            # output directory mirroring
            mirrored_file = (
                self.pm.mirror_path(p, src_root=src, target_root=dst)
                if src.is_dir()
                else (dst / p.name)
            )
            out_dir = self.pm.ensure_dir(mirrored_file.parent)
            stem = p.stem

            print(f"    output dir = {out_dir}")
            print(f"    stem = {stem}")

            # 5) check missing outputs (NO ORIGINAL)
            missing_tf_names = [
                name for name in tf_names
                if not (out_dir / f"{stem}_{name}.png").exists()
            ]

            if missing_tf_names:
                print(f"    missing transforms = {missing_tf_names}")
            else:
                print(f"    ✅ all transforms already exist (nothing to do)")

                # On retourne quand même les paths existants des transforms
                for name in tf_names:
                    items.append((out_dir / f"{stem}_{name}.png", class_id))
                continue

            # 6) load image only if needed
            print("    📥 Loading image")
            img = cv2.imread(str(p))
            if img is None:
                print(f"    ❌ Could not load image, skipping")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 7) compute ONLY missing transforms, keep ctx/cache consistent
            print("    🔄 Applying transformations")
            ctx = TFContext(img=_ensure_uint8(img_rgb), cache={})

            for tf in self.engine.tfs:
                out_img = tf.apply(ctx)

                if tf.name not in missing_tf_names:
                    print(f"      ⏭️  {tf.name} (already exists)")
                    continue

                out_path = out_dir / f"{stem}_{tf.name}.png"
                if out_path.exists():
                    print(f"      ⏭️  {tf.name} appeared meanwhile, skip")
                    continue

                print(f"      ✨ Writing {tf.name} → {out_path.name}")

                if out_img.ndim == 3 and out_img.shape[2] == 3:
                    out_bgr = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
                else:
                    out_bgr = out_img

                cv2.imwrite(str(out_path), _ensure_uint8(out_bgr))

            # 8) register all transform outputs (existing + newly created)
            for name in tf_names:
                items.append((out_dir / f"{stem}_{name}.png", class_id))

        print(f"\n✅ BatchTransformer finished")
        print(f"📦 Total transformed items returned: {len(items)}")

        return items


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

