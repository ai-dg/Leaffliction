"""
Production-ready Gradio app for Leaffliction — leaf disease classification.

Hugging Face Space demo: upload an image, get predicted class and confidence.
Reuses existing model loading and prediction pipeline; CPU-only.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np
import torch

# Repo root so that "leaffliction" and model path resolve correctly
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from leaffliction.model import (
    ConvolutionalNeuralNetwork,
    InferenceManager,
    LabelMapper,
    ModelConfig,
    ModelPaths,
)
from leaffliction.transformations import TransformationEngine

# -----------------------------------------------------------------------------
# Model loading (CPU-only; graceful when model is missing)
# -----------------------------------------------------------------------------

DEFAULT_MODEL_DIR = REPO_ROOT / "worked" / "model"
TOP_K = 5

# Example images from test_images (Unit_test1 & Unit_test2)
EXAMPLE_DIRS = [
    REPO_ROOT / "test_images" / "Unit_test1",
    REPO_ROOT / "test_images" / "Unit_test2",
]
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def _get_example_images() -> Tuple[List[List[str]], List[str]]:
    """
    Return (list of [path] for gr.Examples, list of display names in same order).
    Only includes existing files. Display name is just the basename (no Unit_test1/2).
    """
    out: List[List[str]] = []
    names: List[str] = []
    for dir_path in EXAMPLE_DIRS:
        if not dir_path.is_dir():
            continue
        for p in sorted(dir_path.iterdir()):
            if p.suffix in SUPPORTED_IMAGE_EXTENSIONS and p.is_file():
                out.append([str(p.resolve())])
                names.append(p.name)
    return out, names

# Loaded at startup (None if model dir missing or invalid)
_loader: Optional[InferenceManager] = None
_transformation_engine: Optional[TransformationEngine] = None


def _load_model_cpu(model_dir: Path) -> Optional[InferenceManager]:
    """
    Load model from directory with map_location="cpu". Reuses project classes
    and logic; returns None if any required file is missing (no sys.exit).
    """
    model_dir = Path(model_dir)
    paths = ModelPaths()
    cfg_path = model_dir / paths.config_file
    labels_path = model_dir / paths.labels_file
    model_path = model_dir / paths.model_file

    if not cfg_path.exists() or not labels_path.exists() or not model_path.exists():
        return None

    try:
        cfg_data = cfg_path.read_text(encoding="utf-8")
        labels_data = labels_path.read_text(encoding="utf-8")
    except Exception:
        return None

    import json
    cfg_dict = json.loads(cfg_data)
    labels_dict = json.loads(labels_data)

    cfg = ModelConfig(
        num_classes=int(cfg_dict["num_classes"]),
        input_channels=int(cfg_dict.get("input_channels", 4)),
        img_size=tuple(cfg_dict.get("img_size", [224, 224])),
        seed=int(cfg_dict.get("seed", 42)),
        extra=dict(cfg_dict.get("extra", {})),
    )
    labels = LabelMapper.from_json_dict(labels_dict)
    model = ConvolutionalNeuralNetwork(
        num_classes=cfg.num_classes,
        input_channels=cfg.input_channels,
    )
    try:
        state = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.to("cpu")
    model.eval()

    loader = InferenceManager(
        model=model,
        labels=labels,
        transformation_engine=None,
        cfg=cfg,
        paths=paths,
        verbose=False,
    )
    loader.device = torch.device("cpu")
    loader.model.to("cpu")
    return loader


def get_loader() -> Optional[InferenceManager]:
    """Return the globally loaded inference manager (or None)."""
    return _loader


def get_transformation_engine() -> Optional[TransformationEngine]:
    """Return the transformation engine for preprocessing (or None)."""
    return _transformation_engine


# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------


def predict_from_image(
    image: Optional[np.ndarray],
) -> Tuple[Optional[np.ndarray], str, float, List[Tuple[str, float]]]:
    """
    Run prediction on an image. Expects RGB numpy array (H, W, 3).

    Returns:
        (original_image, predicted_label, confidence, top_k_list)
        or (None, error_message, 0.0, []) on failure.
    """
    if image is None:
        return None, "No image provided.", 0.0, []

    loader = get_loader()
    if loader is None:
        return image, "Model not loaded. Place model in `worked/model/` (config.json, labels.json, model.pth).", 0.0, []

    engine = get_transformation_engine()
    if engine is None:
        return image, "Transformation engine not available.", 0.0, []

    try:
        if image.ndim != 3 or image.shape[2] != 3:
            return image, "Invalid image: expected RGB (H, W, 3).", 0.0, []
        img_rgb = image
        img_size = loader.cfg.img_size
        img_resized = cv2.resize(img_rgb, img_size)
        tensor = engine.apply_all_as_tensor(img_resized)
        pred_id, probs = loader.predict(tensor)
        predicted_label = loader.labels.decode(pred_id)
        confidence = probs.get(predicted_label, 0.0)
        sorted_probs = sorted(probs.items(), key=lambda x: -x[1])[:TOP_K]
        return image, predicted_label, confidence, sorted_probs
    except Exception as e:
        return image, f"Prediction error: {e}", 0.0, []


def run_interface(image: Any) -> Tuple[Optional[np.ndarray], str, str, str]:
    """
    Gradio inference: one input (Image), outputs (image, label, confidence, top-k).
    Handles file path (str) or numpy array from Gradio.
    """
    if image is None:
        return None, "—", "—", "Upload an image to get a prediction."
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            return None, "—", "—", "Could not load image from file."
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(image, np.ndarray):
        img = image
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    else:
        return None, "—", "—", "Unsupported image input type."
    orig, label, conf, top_k = predict_from_image(img)
    conf_str = f"{conf:.1%}" if isinstance(conf, (int, float)) else str(conf)
    top_k_str = "\n".join(f"{name}: {p:.1%}" for name, p in top_k) if top_k else "—"
    return orig, label, conf_str, top_k_str


# -----------------------------------------------------------------------------
# Gradio UI (Blocks)
# -----------------------------------------------------------------------------

MODEL_DETAILS_MD = """
- **Architecture:** Custom CNN (*ConvolutionalNeuralNetwork*): 4× (Conv2d → ReLU → MaxPool2d), AdaptiveAvgPool2d(1), Linear(256→128) → ReLU → Dropout(0.5) → Linear(128→num_classes). Input: 4-channel transformed images, 224×224.
- **Validation accuracy (reference run):** 98.06% (best checkpoint); train accuracy 99.98%. Training: 70 epochs, capacity=0.5 on train set, stratified 80/20 split.
- **Training data:** Leaf images grouped by class (Apple/Grape diseases and healthy). Augmentation (rotation, blur, contrast, etc.) and 4 visual transforms (Hue, Masked, AnalyzeImage, PseudoLandmarks) used to build the 4-channel input.
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="Leaffliction — Leaf Disease Classification",
        theme=gr.themes.Soft(),
        css="footer {text-align: center; margin-top: 1em;}",
    ) as demo:
        gr.Markdown(
            """
            # 🌿 Leaffliction — Leaf Disease Classification

            Upload a leaf image (JPG or PNG) to get a **predicted class** and **confidence** for plant disease recognition.  
            The model classifies into 8 classes (e.g. Apple_Black_rot, Apple_healthy, Grape_Esca, Grape_spot).
            """
        )
        gr.Markdown(
            "**Supported:** Apple & Grape leaf diseases and healthy leaves. Best results with clear, single-leaf photos."
        )

        example_images, example_names = _get_example_images()

        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(
                    type="numpy",
                    label="Upload leaf image",
                    sources=["upload", "clipboard"],
                )
                run_btn = gr.Button("Predict", variant="primary")
            with gr.Column(scale=1):
                image_out = gr.Image(label="Original image", interactive=False)
                label_out = gr.Textbox(label="Predicted class", interactive=False)
                confidence_out = gr.Textbox(label="Confidence", interactive=False)
                top_k_out = gr.Textbox(
                    label=f"Top-{TOP_K} predictions",
                    lines=TOP_K,
                    interactive=False,
                )

        run_btn.click(
            fn=run_interface,
            inputs=[image_in],
            outputs=[image_out, label_out, confidence_out, top_k_out],
        )
        image_in.upload(
            fn=run_interface,
            inputs=[image_in],
            outputs=[image_out, label_out, confidence_out, top_k_out],
        )
        if example_images and example_names:
            gr.Markdown(
                "**Sample images (click one):**  "
                + "  ·  ".join(f"*{n}*" for n in example_names)
            )
            gr.Examples(
                examples=example_images,
                inputs=image_in,
                outputs=[image_out, label_out, confidence_out, top_k_out],
                fn=run_interface,
                label="",
                run_on_click=True,
            )

        gr.Markdown("---")
        gr.Markdown("### Results")
        gr.Markdown(
            "**Predicted class** is the most likely disease or healthy label. **Confidence** is the softmax probability for that class. **Top-k** lists the same probability for the top classes."
        )

        with gr.Accordion("Model Details", open=False):
            gr.Markdown(MODEL_DETAILS_MD)

        gr.Markdown("---")
        gr.Markdown(
            """
            **Links**
            - 📂 [GitHub repository](https://github.com/username/Leaffliction)  *(replace `username` with your GitHub org/user)*
            - 📓 [Binder notebook (predict_demo)](https://mybinder.org/v2/gh/username/Leaffliction/HEAD?labpath=predict_demo.ipynb)  *(replace `username`)*
            - 🌐 [Portfolio](https://example.com)  *(placeholder — set your URL)*
            """
        )

    return demo


# -----------------------------------------------------------------------------
# Startup & launch
# -----------------------------------------------------------------------------


def main() -> None:
    global _loader, _transformation_engine
    model_dir = Path(__file__).resolve().parent / "worked" / "model"
    _loader = _load_model_cpu(model_dir)
    _transformation_engine = TransformationEngine.trainning(verbose=False)
    if _loader is None:
        print("Warning: Model not loaded. Place config.json, labels.json, model.pth in worked/model/", file=sys.stderr)
    demo = build_ui()
    demo.launch()


if __name__ == "__main__":
    main()
