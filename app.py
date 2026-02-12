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


def _get_example_data() -> Tuple[List[str], Dict[str, Path]]:
    """
    Return (list of display names, mapping name -> file path).
    Used for the sample dropdown; images are loaded on the server when selected
    so they display correctly in the output (no broken Gradio temp URL in upload area).
    """
    names: List[str] = []
    name_to_path: Dict[str, Path] = {}
    for dir_path in EXAMPLE_DIRS:
        if not dir_path.is_dir():
            continue
        for p in sorted(dir_path.iterdir()):
            if p.suffix in SUPPORTED_IMAGE_EXTENSIONS and p.is_file():
                names.append(p.name)
                name_to_path[p.name] = p
    return names, name_to_path

# Loaded at startup (None if model dir missing or invalid)
_loader: Optional[InferenceManager] = None
_transformation_engine: Optional[TransformationEngine] = None
# Sample name -> path for dropdown (set in build_ui)
_sample_name_to_path: Dict[str, Path] = {}


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


def run_sample(selected_name: Optional[str]) -> Tuple[Optional[np.ndarray], str, str, str]:
    """
    Run prediction when user selects a sample from the dropdown.
    Loads the image from disk and returns (image, label, confidence, top-k).
    Image is shown in the output panel only (avoids broken display in upload area).
    """
    if not selected_name or not selected_name.strip():
        return None, "—", "—", "—"
    path = _sample_name_to_path.get(selected_name.strip())
    if path is None or not path.exists():
        return None, "—", "—", f"Sample not found: {selected_name}"
    img = cv2.imread(str(path))
    if img is None:
        return None, "—", "—", f"Could not load: {path.name}"
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig, label, conf, top_k = predict_from_image(img_rgb)
    conf_str = f"{conf:.1%}" if isinstance(conf, (int, float)) else str(conf)
    top_k_str = "\n".join(f"{name}: {p:.1%}" for name, p in top_k) if top_k else "—"
    return orig, label, conf_str, top_k_str


# -----------------------------------------------------------------------------
# Gradio UI (Blocks)
# -----------------------------------------------------------------------------

MODEL_ARCHITECTURE_MD = """
**Architecture** (`leaffliction.model.ConvolutionalNeuralNetwork`)

```python
features = Sequential(
    Conv2d(in_ch=4, out=32, k=3, padding=1), ReLU(), MaxPool2d(2),   # block 1
    Conv2d(32, 64, 3, padding=1), ReLU(), MaxPool2d(2),              # block 2
    Conv2d(64, 128, 3, padding=1), ReLU(), MaxPool2d(2),             # block 3
    Conv2d(128, 256, 3, padding=1), ReLU(), MaxPool2d(2),            # block 4
)
gap = AdaptiveAvgPool2d(1)
classifier = Sequential(Flatten(), Linear(256, 128), ReLU(), Dropout(0.5), Linear(128, num_classes))
# Input: 4-channel transformed images, 224×224
```
"""

MODEL_VALIDATION_MD = """
**Validation (reference run)**

| Metric | Value |
|--------|-------|
| Valid accuracy (best checkpoint) | 98.06 % |
| Train accuracy | 99.98 % |
| Epochs | 70 |
| Split | Stratified 80 % train / 20 % valid |
| Transforms extracted (train) | 41,984 |
| Transforms extracted (valid) | 5,768 |
| Groups (train, complete 4 transforms) | 10,496 |
| **Capacity 50 %** | **5,248** train samples (656 per class out of 1,312 before limit) |
"""

MODEL_TRAINING_MD = """
**Training data**

- **Classes:** 8 (Apple/Grape diseases and healthy).
- **Augmentation:** rotation, blur, contrast, scaling, illumination, perspective (Albumentations).
- **Transforms (4 channels):** Hue, Masked, AnalyzeImage, PseudoLandmarks → 4-channel input, 224×224.
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="Leaffliction — Leaf Disease Classification",
        theme=gr.themes.Soft(),
        css=(
            "footer {text-align: center; margin-top: 1em;} "
            ".upload-container.reduced-height { min-height: 380px !important; height: auto !important; } "
            ".upload-container .image-frame, .image-container .image-frame { "
            "min-height: 360px !important; display: flex !important; align-items: center !important; justify-content: center !important; } "
            ".upload-container .image-frame img, .image-container .image-frame img { "
            "min-width: 320px !important; min-height: 320px !important; max-width: 100% !important; "
            "width: auto !important; height: auto !important; object-fit: contain !important; "
            "}"
        ),
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

        example_names, name_to_path = _get_example_data()
        _sample_name_to_path.clear()
        _sample_name_to_path.update(name_to_path)
        sample_dropdown: Optional[gr.Dropdown] = None

        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(
                    type="numpy",
                    label="Upload leaf image",
                    sources=["upload", "clipboard"],
                    height=400,
                )
                run_btn = gr.Button("Predict", variant="primary")
                if example_names:
                    sample_dropdown = gr.Dropdown(
                        choices=[""] + example_names,
                        value="",
                        label="Or pick a sample image",
                    )
            with gr.Column(scale=1):
                image_out = gr.Image(label="Original image", interactive=False, height=400)
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
        if example_names and sample_dropdown is not None:
            sample_dropdown.change(
                fn=run_sample,
                inputs=[sample_dropdown],
                outputs=[image_out, label_out, confidence_out, top_k_out],
            )

        gr.Markdown("---")
        gr.Markdown("### Results")
        gr.Markdown(
            "**Predicted class** is the most likely disease or healthy label. **Confidence** is the softmax probability for that class. **Top-k** lists the same probability for the top classes."
        )

        with gr.Accordion("Model Details", open=False):
            gr.Markdown(MODEL_ARCHITECTURE_MD)
            gr.Markdown(MODEL_VALIDATION_MD)
            gr.Markdown(MODEL_TRAINING_MD)

        gr.Markdown("---")
        gr.Markdown(
            """
            **Authors** · [ai-dg](https://github.com/ai-dg) · [s-t-e-v](https://github.com/s-t-e-v)

            **Links**
            - 📂 [GitHub repository](https://github.com/ai-dg/Leaffliction)
            - 📓 [Binder notebook (predict_demo)](https://mybinder.org/v2/gh/ai-dg/Leaffliction/HEAD?labpath=predict_demo.ipynb)
            - 🌐 [Portfolio](https://dagudelo.dev/)
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
