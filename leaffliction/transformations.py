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
from leaffliction.utils import Logger
from leaffliction.plotting import Plotter
import sys

def _ensure_uint8(img: np.ndarray) -> np.ndarray:
    """
    Convert image to uint8 data type if needed.

    :param img: Input image array.
    :type img: np.ndarray
    :return: Image as uint8 array.
    :rtype: np.ndarray
    """
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _to_bgr_if_bgra(img: np.ndarray) -> np.ndarray:
    """
    Convert BGRA image to BGR by removing alpha channel.

    :param img: Input image array.
    :type img: np.ndarray
    :return: BGR image without alpha channel.
    :rtype: np.ndarray
    """
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def _to_gray_if_color(img: np.ndarray) -> np.ndarray:
    """
    Convert color image to grayscale if needed.

    :param img: Input image array.
    :type img: np.ndarray
    :return: Grayscale image.
    :rtype: np.ndarray
    """
    if img.ndim == 3 and img.shape[2] in (3, 4):
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


@dataclass
class TransformationPipeline:
    """
    Pipeline context for applying sequential transformations with caching.
    """
    img: np.ndarray
    cache: Dict[str, np.ndarray]

    def get(self, key: str) -> np.ndarray:
        """
        Retrieve a cached transformation result.

        :param key: Cache key name.
        :type key: str
        :return: Cached image array.
        :rtype: np.ndarray
        """
        return self.cache[key]

    def has(self, key: str) -> bool:
        """
        Check if a transformation result is cached.

        :param key: Cache key name.
        :type key: str
        :return: True if key exists in cache.
        :rtype: bool
        """
        return key in self.cache

    def set(self, key: str, value: np.ndarray) -> np.ndarray:
        """
        Store a transformation result in cache.

        :param key: Cache key name.
        :type key: str
        :param value: Image array to cache.
        :type value: np.ndarray
        :return: The cached value.
        :rtype: np.ndarray
        """
        self.cache[key] = value
        return value

    def ensure_base(self, threshold: int = 35, fill_size: int = 200) -> None:
        """
        Compute and cache base transformations (background removal, grayscale, etc.).

        :param threshold: Threshold value for binary operations.
        :type threshold: int
        :param fill_size: Size parameter for fill operations.
        :type fill_size: int
        :return: None
        :rtype: None
        """
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

        masked_bgr = _ensure_uint8(masked)
        hsv = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        hue_255 = (hue.astype(np.float32) * (255.0 / 179.0)).astype(np.uint8)
        self.set("hue", hue_255)


    def ensure_roi(self) -> None:
        """
        Compute and cache region of interest (ROI) transformations.

        :return: None
        :rtype: None
        """
        if self.has("roi_image") and self.has("kept_mask"):
            return

        if not self.has("filled") or not self.has("masked"):
            raise RuntimeError("Base pipeline not computed. Call ensure_base() first.")

        image = _ensure_uint8(self.img)
        masked = self.get("masked")
        filled = self.get("filled")

        roi_start_x = 0
        roi_start_y = 0
        roi_h = image.shape[0]
        roi_w = image.shape[1]
        roi_line_w = 5

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
    """
    Protocol defining the interface for transformation operations.
    """
    @property
    def name(self) -> str:
        ...

    def apply(self, ctx: TransformationPipeline) -> np.ndarray:
        ...



@dataclass
class NoBg:
    """
    Transformation that removes image background.
    """
    name: str = "NoBg"

    def apply(self, ctx: TransformationPipeline) -> np.ndarray:
        """
        Apply background removal transformation.

        :param ctx: Transformation pipeline context.
        :type ctx: TransformationPipeline
        :return: Image with background removed.
        :rtype: np.ndarray
        """
        ctx.ensure_base()
        return ctx.get("img_no_bg")


@dataclass
class GrayscaleL:
    """
    Transformation that converts image to grayscale using L channel.
    """
    name: str = "GrayScale"

    def apply(self, ctx: TransformationPipeline) -> np.ndarray:
        """
        Apply grayscale conversion.

        :param ctx: Transformation pipeline context.
        :type ctx: TransformationPipeline
        :return: Grayscale image.
        :rtype: np.ndarray
        """
        ctx.ensure_base()
        return ctx.get("grayscale_l")


@dataclass
class Thresh:
    """
    Transformation that applies binary thresholding.
    """
    threshold: int = 35
    fill_size: int = 200
    name: str = "Thresh"

    def apply(self, ctx: TransformationPipeline) -> np.ndarray:
        """
        Apply binary threshold transformation.

        :param ctx: Transformation pipeline context.
        :type ctx: TransformationPipeline
        :return: Thresholded binary image.
        :rtype: np.ndarray
        """
        ctx.ensure_base(threshold=self.threshold, fill_size=self.fill_size)
        return ctx.get("thresh")


@dataclass
class Filled:
    """
    Transformation that fills small holes in binary mask.
    """
    threshold: int = 35
    fill_size: int = 200
    name: str = "Filled"

    def apply(self, ctx: TransformationPipeline) -> np.ndarray:
        """
        Apply hole filling transformation.

        :param ctx: Transformation pipeline context.
        :type ctx: TransformationPipeline
        :return: Filled binary mask.
        :rtype: np.ndarray
        """
        ctx.ensure_base(threshold=self.threshold, fill_size=self.fill_size)
        return ctx.get("filled")


@dataclass
class GaussianMask:
    """
    Transformation that applies Gaussian blur to mask.
    """
    threshold: int = 35
    fill_size: int = 200
    name: str = "GaussianMask"

    def apply(self, ctx: TransformationPipeline) -> np.ndarray:
        """
        Apply Gaussian blur to mask.

        :param ctx: Transformation pipeline context.
        :type ctx: TransformationPipeline
        :return: Blurred mask.
        :rtype: np.ndarray
        """
        ctx.ensure_base(threshold=self.threshold, fill_size=self.fill_size)
        return ctx.get("gaussian")


@dataclass
class Masked:
    """
    Transformation that applies mask to isolate leaf region.
    """
    threshold: int = 35
    fill_size: int = 200
    name: str = "Masked"

    def apply(self, ctx: TransformationPipeline) -> np.ndarray:
        """
        Apply masking transformation.

        :param ctx: Transformation pipeline context.
        :type ctx: TransformationPipeline
        :return: Masked image showing only leaf region.
        :rtype: np.ndarray
        """
        ctx.ensure_base(threshold=self.threshold, fill_size=self.fill_size)
        return ctx.get("masked")

@dataclass
class Hue:
    """
    Transformation that extracts hue channel from HSV color space.
    """
    name: str = "Hue"

    def apply(self, ctx: TransformationPipeline) -> np.ndarray:
        """
        Extract hue channel.

        :param ctx: Transformation pipeline context.
        :type ctx: TransformationPipeline
        :return: Hue channel image.
        :rtype: np.ndarray
        """
        ctx.ensure_base()
        return ctx.get("hue")

@dataclass
class RoiImage:
    """
    Transformation that highlights region of interest with bounding box.
    """
    threshold: int = 35
    fill_size: int = 200
    name: str = "RoiImage"

    def apply(self, ctx: TransformationPipeline) -> np.ndarray:
        """
        Apply ROI visualization.

        :param ctx: Transformation pipeline context.
        :type ctx: TransformationPipeline
        :return: Image with ROI highlighted.
        :rtype: np.ndarray
        """
        ctx.ensure_base(threshold=self.threshold, fill_size=self.fill_size)
        ctx.ensure_roi()
        return ctx.get("roi_image")


def _draw_pseudolandmarks(image: np.ndarray, pseudolandmarks, color_bgr, radius: int) -> np.ndarray:
    """
    Draw pseudolandmark points on an image.

    :param image: Input image array.
    :type image: np.ndarray
    :param pseudolandmarks: List of landmark coordinates.
    :type pseudolandmarks: Any
    :param color_bgr: Color for drawing landmarks in BGR format.
    :type color_bgr: tuple
    :param radius: Radius of landmark circles.
    :type radius: int
    :return: Image with landmarks drawn.
    :rtype: np.ndarray
    """
    out = image.copy()
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

        row = int(p[0][0])
        col = int(p[0][1])

        cv2.circle(out, (row, col), radius, color, thickness=-1)

    return out

@dataclass
class AnalyzeImage:
    """
    Transformation that performs morphological analysis on leaf.
    """
    name: str = "AnalyzeImage"

    def apply(self, ctx: TransformationPipeline) -> np.ndarray:
        """
        Apply morphological analysis.

        :param ctx: Transformation pipeline context.
        :type ctx: TransformationPipeline
        :return: Analyzed image with measurements.
        :rtype: np.ndarray
        """
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
    """
    Transformation that detects and draws pseudolandmarks on leaf.
    """
    threshold: int = 35
    fill_size: int = 200
    name: str = "PseudoLandmarks"

    def apply(self, ctx: TransformationPipeline) -> np.ndarray:
        """
        Detect and draw pseudolandmarks.

        :param ctx: Transformation pipeline context.
        :type ctx: TransformationPipeline
        :return: Image with pseudolandmarks drawn.
        :rtype: np.ndarray
        """
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
        pseudo_img = _draw_pseudolandmarks(pseudo_img, top_x, (255, 0, 0), 5)
        # Magenta
        pseudo_img = _draw_pseudolandmarks(pseudo_img, bottom_x, (255, 0, 255), 5)
        # Blue
        pseudo_img = _draw_pseudolandmarks(pseudo_img, center_v_x, (0, 0, 255), 5)

        return _ensure_uint8(pseudo_img)


def _base_stem_and_tf_name(stem: str, tf_names: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract base filename and transformation name from a stem.

    :param stem: Filename stem to parse.
    :type stem: str
    :param tf_names: List of known transformation names.
    :type tf_names: List[str]
    :return: Tuple of (base_stem, transformation_name) or (None, None).
    :rtype: Tuple[Optional[str], Optional[str]]
    """
    for tf in tf_names:
        suf = "_" + tf
        if stem.endswith(suf):
            base = stem[: -len(suf)]
            return base, tf
    return None, None


class TransformationEngine:
    """
    Engine for applying multiple transformations to images.
    """

    def __init__(self, tfs: List[Transformation], verbose : bool = True) -> None:
        """
        Initialize the transformation engine.

        :param tfs: List of transformations to apply.
        :type tfs: List[Transformation]
        :param verbose: Enable detailed logging.
        :type verbose: bool
        :return: None
        :rtype: None
        """
        self.tfs = tfs
        self.verbose = verbose
        self.logger = Logger(verbose)

    @classmethod
    def default_six(cls, verbose : bool = True) -> "TransformationEngine":
        """
        Create engine with default six transformations for visualization.

        :param verbose: Enable detailed logging.
        :type verbose: bool
        :return: Configured TransformationEngine instance.
        :rtype: TransformationEngine
        """
        tfs: List[Transformation] = [
            GrayscaleL(),
            Hue(),
            GaussianMask(threshold=120, fill_size=200),
            Masked(threshold=35, fill_size=200),
            RoiImage(),
            AnalyzeImage(),
            PseudoLandmarks(threshold=35, fill_size=200),
        ]
        return cls(tfs=tfs, verbose=verbose)
    
    @classmethod
    def trainning(cls, verbose : bool = True) -> "TransformationEngine":
        """
        Create engine with transformations optimized for training.

        :param verbose: Enable detailed logging.
        :type verbose: bool
        :return: Configured TransformationEngine instance.
        :rtype: TransformationEngine
        """
        tfs: List[Transformation] = [
            Hue(),
            Masked(threshold=35, fill_size=200),
            AnalyzeImage(),
            PseudoLandmarks(threshold=35, fill_size=200),
        ]
        return cls(tfs=tfs, verbose=verbose)
    
    @classmethod
    def only_selected(cls, only: list[str], verbose : bool = True) -> "TransformationEngine":
        """
        Create engine with only specified transformations.

        :param only: List of transformation names to include.
        :type only: list[str]
        :param verbose: Enable detailed logging.
        :type verbose: bool
        :return: Configured TransformationEngine instance.
        :rtype: TransformationEngine
        """
        if not isinstance(only, list) or len(only) == 0:
            raise ValueError("only_selected expects a non-empty list of transformations")

        mapping = {
            "grayscale": GrayscaleL(),
            "gaussian": GaussianMask(threshold=120, fill_size=200),
            "mask": Masked(threshold=35, fill_size=200),
            "hue": Hue(),
            "roi": RoiImage(),
            "analyze": AnalyzeImage(),
            "pseudo": PseudoLandmarks(threshold=35, fill_size=200),
        }

        tfs: list[Transformation] = []

        for name in only:
            key = name.lower()
            if key not in mapping:
                raise ValueError(
                    f"Unknown transformation '{name}'. "
                    f"Available: {list(mapping.keys())}"
                )
            tfs.append(mapping[key])

        return cls(tfs=tfs, verbose=verbose)


    def apply_all(self, img: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Apply all transformations and return results as dictionary.

        :param img: Input image array.
        :type img: np.ndarray
        :return: Dictionary mapping transformation names to result images.
        :rtype: Dict[str, np.ndarray]
        """
        ctx = TransformationPipeline(img=_ensure_uint8(img), cache={})

        results: Dict[str, np.ndarray] = {}
        for tf in self.tfs:
            results[tf.name] = tf.apply(ctx)

        # if ctx.has("kept_mask"):
        #     results["kept_mask"] = ctx.get("kept_mask")


        return results
    


    def apply_all_as_tensor(self, img: np.ndarray) -> torch.Tensor:
        """
        Apply all transformations and return as stacked tensor.

        :param img: Input image array.
        :type img: np.ndarray
        :return: Tensor of shape (num_transforms, height, width).
        :rtype: torch.Tensor
        """
        ctx = TransformationPipeline(img=_ensure_uint8(img), cache={})

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
        """
        Transform a batch of images from file paths.

        :param items: List of (image_path, label) tuples.
        :type items: List[Tuple[Path, int]]
        :param img_size: Target image size as (height, width).
        :type img_size: Tuple[int, int]
        :return: Tuple of (features_tensor, labels_tensor).
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        X_list: List[torch.Tensor] = []
        y_list: List[int] = []

        for img_path, label in items:
            img = cv2.imread(str(img_path))
            if img is None:
                self.logger.warn(f" Warning: Could not load {img_path}")
                continue
            else:
                self.logger.info(f"Processing: {img_path}")

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
        """
        Load pre-transformed images from disk and group by base filename.

        :param items: List of (image_path, label) tuples.
        :type items: List[Tuple[Path, int]]
        :param img_size: Target image size as (height, width).
        :type img_size: Tuple[int, int]
        :param capacity: Fraction of data to load (0 < capacity <= 1.0).
        :type capacity: float
        :param seed: Random seed for sampling.
        :type seed: int
        :return: Tuple of (features_tensor, labels_tensor).
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """

        if not (0 < capacity <= 1.0):
            self.logger.error("capacity must be in (0, 1]")
            sys.exit(1)


        tf_names = [tf.name for tf in self.tfs]
        tf_set = set(tf_names)

        groups: Dict[Tuple[int, str, str], Dict[str, Path]] = defaultdict(dict)

        skipped_non_tf = 0
        for p, label in items:
            stem = p.stem

            base_stem, tf_name = _base_stem_and_tf_name(stem, tf_names)
            if base_stem is None or tf_name is None:
                skipped_non_tf += 1
                continue

            key = (int(label), str(p.parent), base_stem)
            groups[key][tf_name] = p

        if not groups:
            self.logger.error("No transformed items found in 'items'.")
            sys.exit(1)

        samples: List[Tuple[List[Path], int]] = []
        missing_groups = 0

        for (label, parent_str, base_stem), tf_map in groups.items():
            if tf_set.issubset(tf_map.keys()):
                ordered_paths = [tf_map[name] for name in tf_names]
                samples.append((ordered_paths, label))
            else:
                missing_groups += 1

        if not samples:
            self.logger.error(
                "No complete samples with all transforms were found. "
                f"Groups total={len(groups)}, missing_groups={missing_groups}.")
            sys.exit(1)

        self.logger.info(f"Grouping done: {len(groups)} groups")
        self.logger.info(f"Complete samples (all {len(tf_names)} transforms): {len(samples)}")
        if missing_groups:
            self.logger.info(f"Incomplete groups skipped: {missing_groups}")
        if skipped_non_tf:
            self.logger.info(f"Non-transform files ignored: {skipped_non_tf}")


        if capacity < 1.0:
            rdm = Random(seed)
            samples_by_class: Dict[int, List[Tuple[List[Path], int]]] = defaultdict(list)
            for s in samples:
                samples_by_class[s[1]].append(s)

            limited_samples: List[Tuple[List[Path], int]] = []
            self.logger.info(f"Applying capacity limit on samples: {int(capacity * 100)}%")

            for label, class_samples in samples_by_class.items():
                n_total = len(class_samples)
                n_keep = max(1, int(n_total * capacity))
                rdm.shuffle(class_samples)
                limited_samples.extend(class_samples[:n_keep])
                self.logger.info(f"  class {label}: {n_keep}/{n_total} samples kept")

            samples = limited_samples
            rdm.shuffle(samples)
            self.logger.info(f"Total samples after limit: {len(samples)}")


        X_list: List[torch.Tensor] = []
        y_list: List[int] = []

        for idx, (paths_5, label) in enumerate(samples, start=1):
            channels: List[np.ndarray] = []
            ok = True

            for p in paths_5:
                ch = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)  # (H,W)
                if ch is None:
                    self.logger.warn(f"Could not read: {p}")
                    ok = False
                    break

                ch = cv2.resize(ch, img_size)
                ch = ch.astype(np.float32) / 255.0
                channels.append(ch)

            if not ok:
                continue

            x = torch.from_numpy(np.stack(channels, axis=0))  # (5,H,W)
            X_list.append(x)
            y_list.append(int(label))

            if idx % 500 == 0:
                self.logger.info(f"  Loaded {idx}/{len(samples)} samples")

        if not X_list:
            self.logger.error("No samples could be loaded (all failed during cv2.imread).")
            sys.exit(1)

        X = torch.stack(X_list, dim=0)  # (N, 5, H, W)
        y = torch.tensor(y_list, dtype=torch.long)

        expected_c = len(self.tfs)
        if X.shape[1] != expected_c:
            self.logger.error(f"Expected {expected_c} channels, got {tuple(X.shape)}")
            sys.exit(1)

        return X, y
    
    def extract_transformed_items(
        self,
        items: List[Tuple[Path, int]],
        transform_dir: Path
    ) -> List[Tuple[Path, int]]:
        """
        Map original image paths to their transformed versions.

        :param items: List of (original_image_path, label) tuples.
        :type items: List[Tuple[Path, int]]
        :param transform_dir: Directory containing transformed images.
        :type transform_dir: Path
        :return: List of (transformed_image_path, label) tuples.
        :rtype: List[Tuple[Path, int]]
        """
        tf_names = [tf.name for tf in self.tfs]
        transformed_items: List[Tuple[Path, int]] = []

        for img_path, class_id in items:

            class_dir = img_path.parent.name
            stem = img_path.stem

            out_dir = transform_dir / class_dir

            for tf_name in tf_names:
                tf_path = out_dir / f"{stem}_{tf_name}.png"
                if tf_path.exists():
                    transformed_items.append((tf_path, int(class_id)))
                else:

                    pass

        return transformed_items

    

class TransformationDirectory:
    """
    Batch processor for applying transformations to entire directories.
    """

    def __init__(self, engine: TransformationEngine, verbose : bool = True) -> None:
        """
        Initialize the directory transformer.

        :param engine: Transformation engine to use.
        :type engine: TransformationEngine
        :param verbose: Enable detailed logging.
        :type verbose: bool
        :return: None
        :rtype: None
        """
        self.engine = engine
        self.pm = PathManager()
        self.logger = Logger(verbose)

    def _build_label_map(self, src: Path, recursive: bool) -> Dict[str, int]:
        """
        Build mapping from class names to numeric IDs.

        :param src: Source directory or file path.
        :type src: Path
        :param recursive: Whether to scan recursively.
        :type recursive: bool
        :return: Dictionary mapping class names to IDs.
        :rtype: Dict[str, int]
        """
        self.logger.info(f"Building label → id map (recursive={recursive})")

        if src.is_file():
            label = src.parent.name
            self.logger.info(f"  • Single file mode → label '{label}' → id 0")
            return {label: 0}

        labels = set()
        for p in self.pm.iter_images(src, recursive=recursive):
            labels.add(p.parent.name)

        sorted_labels = sorted(labels)
        label_to_id = {label: i for i, label in enumerate(sorted_labels)}

        self.logger.info("  ✔ Labels found:")
        for label, idx in label_to_id.items():
            self.logger.info(f"    - {label} → {idx}")

        return label_to_id

    def run(
        self,
        src: Path,
        dst: Path,
        recursive: bool = True
    ) -> List[Tuple[Path, int]]:
        """
        Process all images in source directory and save transformations.

        :param src: Source directory containing images.
        :type src: Path
        :param dst: Destination directory for transformed images.
        :type dst: Path
        :param recursive: Whether to process subdirectories.
        :type recursive: bool
        :return: List of (transformed_image_path, label) tuples.
        :rtype: List[Tuple[Path, int]]
        """

        self.logger.info(f"TransformationDirectory.run")
        self.logger.info(f"  src = {src}")
        self.logger.info(f"  dst = {dst}")
        self.logger.info(f"  recursive = {recursive}")

        self.pm.ensure_dir(dst)
        items: List[Tuple[Path, int]] = []

        if src.is_file():
            paths = [src]
            self.logger.info("Source is a single file")
        else:
            paths = list(self.pm.iter_images(src, recursive=recursive))
            self.logger.info(f"Found {len(paths)} images under {src}")

        label_to_id = self._build_label_map(src, recursive=recursive)

        tf_names = [tf.name for tf in self.engine.tfs]
        self.logger.info(f"Transformations ({len(tf_names)}): {tf_names}")

        for idx, p in enumerate(paths, start=1):
            self.logger.info(f"[{idx}/{len(paths)}] Processing image:")
            self.logger.info(f"    path = {p}")

            label = p.parent.name
            if label not in label_to_id:
                self.logger.warn(f"    Unknown label '{label}', skipping")
                continue

            class_id = label_to_id[label]
            self.logger.info(f"    label = '{label}' → class_id = {class_id}")

            mirrored_file = (
                self.pm.mirror_path(p, src_root=src, target_root=dst)
                if src.is_dir()
                else (dst / p.name)
            )
            out_dir = self.pm.ensure_dir(mirrored_file.parent)
            stem = p.stem

            self.logger.info(f"    output dir = {out_dir}")
            self.logger.info(f"    stem = {stem}")

            missing_tf_names = [
                name for name in tf_names
                if not (out_dir / f"{stem}_{name}.png").exists()
            ]

            if missing_tf_names:
                self.logger.info(f"    missing transforms = {missing_tf_names}")
            else:
                self.logger.info(f"    all transforms already exist (nothing to do)")

                for name in tf_names:
                    items.append((out_dir / f"{stem}_{name}.png", class_id))
                continue

            self.logger.info("    Loading image")
            img = cv2.imread(str(p))
            if img is None:
                self.logger.error(f"   Could not load image, skipping")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            self.logger.info("    Applying transformations")
            ctx = TransformationPipeline(img=_ensure_uint8(img_rgb), cache={})

            for tf in self.engine.tfs:
                out_img = tf.apply(ctx)

                if tf.name not in missing_tf_names:
                    self.logger.info(f"      {tf.name} (already exists)")
                    continue

                out_path = out_dir / f"{stem}_{tf.name}.png"
                if out_path.exists():
                    self.logger.info(f"      {tf.name} appeared meanwhile, skip")
                    continue

                self.logger.info(f"      Writing {tf.name} → {out_path.name}")

                if out_img.ndim == 3 and out_img.shape[2] == 3:
                    out_bgr = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
                else:
                    out_bgr = out_img

                cv2.imwrite(str(out_path), _ensure_uint8(out_bgr))

            
            for name in tf_names:
                items.append((out_dir / f"{stem}_{name}.png", class_id))

        self.logger.info(f"\nTransformationDirectory finished")
        self.logger.info(f"Total transformed items returned: {len(items)}")

        return items



def main():
    """
    Main function placeholder for transformation module.

    :return: None
    :rtype: None
    """
    [...]


if __name__ == "__main__":
    main()

