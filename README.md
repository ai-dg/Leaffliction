# Leaffliction
![Score](https://img.shields.io/badge/Score-125%25-brightgreen)  
**Image classification by disease recognition on leaves**

A PyTorch pipeline for multi-class classification of plant leaf diseases (e.g. Apple and Grape) using a custom CNN, image augmentations, and hand-crafted visual transformations (Hue, Masked, AnalyzeImage, PseudoLandmarks) as input channels.

**Réalisateurs :** [ai-dg](https://github.com/ai-dg) · [s-t-e-v](https://github.com/s-t-e-v) ·

---

## 📚 Table of Contents

- [▌ Project Overview](#project-overview)
- [▌ Features](#features)
- [▌ Getting Started](#getting-started) — [Demo Notebook](predict_demo.ipynb)
- [▌ Usage](#usage)
- [▌ Pipeline Overview](#pipeline-overview)
- [▌ Hyperparameters and Data Split](#hyperparameters-and-data-split)
- [▌ Performance Results](#performance-results)
- [▌ Learning Curves](#learning-curves)
- [▌ Project Structure](#project-structure)
- [▌ Reproducibility](#reproducibility)
- [▌ Troubleshooting](#troubleshooting)
- [▌ Limitations & Next Steps](#limitations--next-steps)
- [▌ Sources and References](#sources-and-references)

---

## ▌ Project Overview

Leaffliction implements an end-to-end ML workflow for leaf-disease recognition:

1. **Dataset** — Directory-based layout (one folder per class).
2. **Distribution** — Analyse and visualise class counts (pie/bar).
3. **Augmentation** — Balance classes via Albumentations (rotation, blur, contrast, scaling, illumination, perspective).
4. **Transformation** — Build multi-channel inputs from plant-specific transforms (Hue, Masked, AnalyzeImage, PseudoLandmarks) for the CNN.
5. **Training** — Stratified train/validation split, optional train-data cap via a **capacity** parameter, Adam optimizer, CrossEntropyLoss.
6. **Prediction** — Single image or directory, using a saved model (directory or ZIP bundle) and SHA1 signature.

The model is a **Convolutional Neural Network** (Conv2d blocks + GAP + classifier). Training is relatively heavy due to the CNN and the number of transformed samples.

---

## ▌ Features

✔️ **Custom CNN** — PyTorch `ConvolutionalNeuralNetwork` with configurable input channels and number of classes.\
✔️ **Stratified split** — Train/validation split with optional class balance.\
✔️ **Augmentation** — Albumentations-based augmentation and class balancing.\
✔️ **Transform pipeline** — Hue, Masked, AnalyzeImage, PseudoLandmarks (and optional Grayscale, Gaussian, ROI, Pseudo) for visual feature channels.\
✔️ **Training metrics** — Train/validation accuracy per epoch; final evaluation with best checkpoint.\
✔️ **Model export** — Saved as `model.pth` + `config.json` + `labels.json`, then packaged into a ZIP with SHA1 in `signature.txt`.\
✔️ **CLI** — Entry points: `Distribution.py`, `Augmentation.py`, `Transformation.py`, `train.py`, `predict.py`.

---

## ▌ Getting Started

### ■ Requirements

- Python 3.11+ (see `pyproject.toml`: `>=3.11,<3.12`)
- PyTorch, torchvision, OpenCV, Albumentations, PlantCV, matplotlib, numpy

### ■ Environment (Conda)

```bash
conda activate tf
```

Or use the project’s venv and install dependencies (e.g. from `pyproject.toml` or `requirements.txt` if aligned).

### ■ Dataset layout

Images must be grouped by class in subdirectories:

```
<dataset_dir>/
  Apple_Black_rot/
    image (1).JPG
    ...
  Apple_healthy/
  Apple_rust/
  Apple_scab/
  Grape_Black_rot/
  Grape_Esca/
  Grape_healthy/
  Grape_spot/
```

### ■ Quick commands

```bash
# Distribution (analyse classes)
python Distribution.py ./leaves/images --mode both --save . --verbose

# Augmentation (single image demo)
python Augmentation.py "<path_to_image>" --output-dir augmented_directory --verbose

# Transformation (single image or directory)
python Transformation.py "<path_to_image>" --only hue mask analyze pseudo --verbose
python Transformation.py --src ./leaves/images --dst ./training_artifacts/transform --verbose

# Training (full pipeline; heavy, uses CNN)
python train.py ./leaves/images --out-dir training_artifacts --out-zip train_output.zip --verbose

# Prediction (single image)
python predict.py --model-path ./worked/model ./test_images/Unit_test1/Apple_Black_rot1.JPG

# Prediction (directory; accuracy summary)
python predict.py --model-path ./worked/model --dir-path ./test_images/100 --verbose
```

**Unit tests (10 images):** The model in `./worked/model` is expected to predict correctly the 10 images in `test_images/Unit_test1` and `test_images/Unit_test2`, e.g.:

```bash
python predict.py --model-path ./worked/model ./test_images/Unit_test1/Apple_Black_rot1.JPG
```

### ■ Demo Notebook

The **[predict_demo.ipynb](predict_demo.ipynb)** Jupyter notebook provides a recruiter-friendly, data-scientist-style demo of the **prediction pipeline only** (no training). It runs on CPU and is suitable for Binder-like environments.

**What it demonstrates:**

- Loading the trained model from `worked/model/` (with a clear message if the model is missing).
- Loading an image from a local sample folder (`test_images/Unit_test1`, `Unit_test2`, or `test_images/100/<class>/`) or using a custom path.
- Preprocessing with the project’s **TransformationEngine** (Hue, Masked, AnalyzeImage, PseudoLandmarks) and visualising the original image and transformed channels.
- Running inference and displaying the **predicted class**, **confidence**, and **top-k** class probabilities.
- Short explanations of the problem, data assumptions, model architecture, and how metrics are produced in the project (train/valid accuracy, directory accuracy), plus limitations and next steps.

Open the notebook from the repo root so that relative paths (`worked/model`, `test_images/`) resolve correctly. Dependencies are those in `requirements.txt` (including `prompt_toolkit>=3.0.48` for the Jupyter kernel; if the kernel fails with “missing module prompt_toolkit.cursor_shapes”, run `pip install 'prompt_toolkit>=3.0.48'`).

### ■ Binder Demo (Lightweight Environment)

Binder is configured to use a **minimal, inference-only** environment so that the image stays **around 1–2 GB** (instead of ~8 GB) and builds remain within resource limits. Configuration lives in the **`binder/`** folder at the repository root:

- **`binder/runtime.txt`** — fixes the Python version (e.g. 3.11) for the Binder build.
- **`binder/requirements.txt`** — installs **CPU-only** PyTorch (via `--extra-index-url https://download.pytorch.org/whl/cpu`) and only the packages needed to run **predict_demo.ipynb**: torch, torchvision, numpy, pillow, matplotlib, opencv-python-headless. **PlantCV, rembg and onnxruntime are not installed** to keep the image small; the notebook detects their absence and uses a **simplified preprocessing fallback** (4× grayscale channel), so predictions on Binder may differ slightly from the full pipeline run locally with the full transformation engine.

The rest of the project (local development, training, full CLI) continues to use the root-level environment (e.g. `requirements.txt` or `pyproject.toml`). The root **environment.yml**, if present, is not modified; Binder uses **only** the files under `binder/`.

---

## ▌ Usage

### ■ Distribution — `Distribution.py`

- **Input:** `dataset_dir` (positional).
- **Options:** `--mode {both,bar,pie}`, `--save <path>`, `--verbose`.
- **Output:** Plots (and optional save) of class distribution.

### ■ Augmentation — `Augmentation.py`

- **Input:** `image_path` (positional).
- **Options:** `--output-dir` (default `augmented_directory`), `--verbose`.
- **Output:** Grid of augmented images and files in the output directory.

### ■ Transformation — `Transformation.py`

- **Input:** Either `image_path` or `--src` + `--dst` for batch.
- **Options:** `--only grayscale|gaussian|mask|hue|roi|analyze|pseudo`, `--verbose`.
- **Output:** Visualisation and/or transformed images (e.g. `*_<TransformName>.png`) under `--dst`.

### ■ Training — `train.py`

- **Input:** `dataset_dir` (positional).
- **Options:**
  - `--out-dir` / `-o` (default `training_artifacts`)
  - `--out-zip` (default `train_output.zip`)
  - `--valid-ratio` (default `0.2`)
  - `--seed` (default `42`)
  - `--learning-rate` (default `0.0314`)
  - `--epochs` (default `70`)
  - `--batch-size` (default `8`)
  - `--verbose`
- **Output:** `out_dir/best_model.pth`, `out_dir/model/` (config, labels, weights), ZIP artifact, `signature.txt` (SHA1), and learning curves (see below).

### ■ Prediction — `predict.py`

- **Input:** Either an image path or `--dir-path` for a directory.
- **Model:** `--model-path <dir>` or `--model-zip <file>` (one of the two).
- **Options:** `--top-k`, `--show-transforms`, `--verbose`.
- **Output:** Predicted class (and top-k probs); for directory, an accuracy summary.

---

## ▌ Pipeline Overview

```
Dataset (dir per class)
    → DatasetScanner.scan()
    → DatasetSplitter.split(valid_ratio, seed, stratified=True)
    → [Train] AugmentationEngine.augment_dataset()  (class balancing)
    → TransformationDirectory.run() on original + augmented
    → TransformationEngine.extract_transformed_items() / load_transformer_items()
         • Train: capacity=0.5  (see Hyperparameters)
         • Valid: capacity=1.0
    → DataLoader (shuffle train)
    → ConvolutionalNeuralNetwork + CrossEntropyLoss + Adam
    → Best checkpoint (validation accuracy) → best_model.pth
    → InferenceManager.save() → model/ + ZIP + signature.txt
```

---

## ▌ Hyperparameters and Data Split

### ■ Train script (`train.py`)

`TrainConfig` is built from CLI (with fallbacks):

| Parameter       | CLI / default in code | Description                    |
|----------------|------------------------|--------------------------------|
| `epochs`       | `--epochs`, default 70 | Number of training epochs.     |
| `batch_size`   | `--batch-size`, default 8 | Batch size.                 |
| `lr`           | `--learning-rate` → 0.0314 or 1e-3 (see code) | Learning rate. |
| `valid_ratio`  | `--valid-ratio`, default 0.2 | Fraction of data for validation. |
| `seed`         | `--seed`, default 42   | Random seed.                   |
| `img_size`     | (224, 224)             | Input spatial size.            |
| `augment_train`| True                   | Use augmentation for training. |
| `transform_train` | True                | Use transform pipeline.       |

### ■ Data capacity (`train_pipeline.py`)

Training uses a **capacity** limit when loading transformed items:

- **Train:** `load_transformer_items(train_items, capacity=0.5)` — keeps **50%** of complete samples per class (randomly), so only part of the transformed training set is used. This acts as a **data split / subsample** to control dataset size and training time.
- **Validation:** `capacity=1.0` — all complete validation samples are used.

So effectively: after grouping by (class, base stem) and keeping only “complete” samples (all 4 transforms), 50% of those train groups are sampled per class; validation uses 100%.

---

## ▌ Performance Results

### ■ Official training run (reference)

- **Dataset:** Directory with 8 classes (Apple/Grape diseases and healthy).
- **Split:** Stratified; validation ratio 20%. After augmentation and transforms: **10 496** train groups, **1 442** validation groups.
- **Train data cap:** `capacity=0.5` → **5 248** training samples (50% of complete train groups per class).
- **Epochs:** 70. **Device:** CUDA. **Parameters:** 422 632.

**Final metrics (best checkpoint):**

| Metric            | Value    |
|-------------------|----------|
| **Train accuracy** | 99.98% |
| **Valid accuracy** | 98.06% |
| **Training time**  | ~717 s  |

**Epoch progression (excerpt):**

- Epoch 1/70 — Train Acc: 19.38%, Valid Acc: 28.02%
- Epoch 35/70 — Train Acc: 98.00%, Valid Acc: 96.46%
- Epoch 70/70 — Train Acc: 98.91%, Valid Acc: 96.88%

Validation accuracy is required to be ≥ 90% and validation set size ≥ 100 (enforced by `ModelChecker`).

### ■ Prediction on test sets

- **Directory `./test_images/100`** (152 images): accuracy **94.08%** with `--model-path ./worked/model`.
- **Unit tests:** The model in `./worked/model` is expected to predict correctly all **10** images in `test_images/Unit_test1` and `test_images/Unit_test2` (e.g. `Apple_Black_rot1.JPG` in Unit_test1).

---

## ▌ Learning Curves

Learning curves are generated at the end of training and saved in the project root (`learning curve.jpg`, `learning curve_loss.jpg`).

| | |
|:---:|:---:|
| ![Train/valid accuracy](learning%20curve.jpg) | ![Train loss](learning%20curve_loss.jpg) |

For headless environments (e.g. no Tk), set:

```bash
MPLBACKEND=Agg python train.py ./leaves/images
```

*(Plots can later be moved to something like `assets/curves/` and linked from this README.)*

---

## ▌ Project Structure

```
Leaffliction/
├── Distribution.py      # Class distribution analysis
├── Augmentation.py      # Single-image augmentation demo
├── Transformation.py   # Single or batch image transforms
├── train.py             # Training entry point
├── predict.py           # Prediction (image or directory)
├── leaffliction/
│   ├── cli.py           # Argument parsers
│   ├── dataset.py       # DatasetScanner, DatasetSplitter
│   ├── augmentations.py # AugmentationEngine, AugmentationSaver
│   ├── transformations.py # TransformationEngine, TransformationDirectory
│   ├── model.py         # CNN, LabelMapper, InferenceManager
│   ├── train_pipeline.py # Trainer, TrainConfig, Metrics, ModelChecker, TrainingPackager
│   ├── predict_pipeline.py # Predictor, PredictConfig
│   ├── plotting.py     # Plotter (distribution, learning curves, grids)
│   └── utils.py         # PathManager, Logger, Hasher, ZipPackager
├── style/
│   └── leaffliction.mplstyle
├── pyproject.toml
├── requirements.txt
├── signature.txt        # SHA1 of the built ZIP
├── learning curve.jpg   # Train/valid accuracy curves
├── learning curve_loss.jpg
├── worked/model/        # Pre-trained model (model.pth, config.json, labels.json)
└── test_images/         # Unit_test1, Unit_test2, 100, etc.
```

---

## ▌ Reproducibility

- ■ **Seed:** `--seed` (default 42) is used for dataset split and shuffling; the pipeline uses it where applicable (e.g. in `DatasetSplitter` and capacity sampling in `load_transformer_items`).
- ■ **Versions:** Pin Python and dependencies (e.g. via `pyproject.toml` or `requirements.txt`) for reproducible runs.
- ■ **Capacity:** Using the same `capacity=0.5` in `train_pipeline.py` and the same dataset/split produces comparable train size and behaviour.

---

## ▌ Troubleshooting

- ■ **Jupyter kernel: "No module named 'prompt_toolkit.cursor_shapes'"** — The kernel's `prompt_toolkit` may be old or incomplete; use `--force-reinstall` if a simple upgrade fails. In the **same environment as the kernel** (e.g. conda `tf`), run:
  ```bash
  pip install --force-reinstall --no-cache-dir 'prompt_toolkit>=3.0.48'
  ```
  Then restart the kernel (or restart Jupyter). The repo’s `requirements.txt` already pins `prompt_toolkit>=3.0.48`.
- ■ **`albumentations` / `albucore` conflict:** If pip reports *"albumentations 2.0.8 requires albucore==0.0.24, but you have albucore 0.0.36"*, this project uses **albumentationsx** only (not the old `albumentations` package). Run: `pip uninstall albumentations` so only albumentationsx remains; the code still uses `import albumentations`.

---

## ▌ Limitations & Next Steps

- ■ **Confusion matrix / classification report** are not computed in the scripts; could be added for validation or test.
- ■ **Learning rate:** Ensure the CLI `--learning-rate` is wired to `TrainConfig.lr` in `train.py` if you want to change it from the command line.
- ■ **Headless:** Use `MPLBACKEND=Agg` when running training without a display to avoid Tk errors.
- ■ **Possible extensions:** export metrics to JSON/CSV, add confusion matrix, move learning curves to `assets/curves/`, and add a small demo dataset under version control for quick runs.

---

## ▌ Sources and References

The **ConvolutionalNeuralNetwork** class in `leaffliction/model.py` and the overall training/inference setup are inspired by standard PyTorch examples and tutorials: Conv2d stacks, ReLU, MaxPool2d, Global Average Pooling (GAP), and a small MLP classifier with dropout. The architecture is:

- ■ **Features:** four blocks of `Conv2d → ReLU → MaxPool2d` (channels 32 → 64 → 128 → 256), then `AdaptiveAvgPool2d(1)` (GAP).
- ■ **Classifier:** `Flatten → Linear(256, 128) → ReLU → Dropout(0.5) → Linear(128, num_classes)`.

**Inspiration links (CNN, training loop, ResNet-style structure):**

- ■ [PyTorch examples — MNIST main](https://github.com/pytorch/examples/blob/main/mnist/main.py)
- ■ [PyTorch-CIFAR — main](https://github.com/kuangliu/pytorch-cifar/blob/master/main.py)
- ■ [PyTorch-CIFAR — ResNet](https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py)
- ■ [PyTorch tutorial — CNN](https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/convolutional_neural_network/main.py)
- ■ [PyTorch tutorial (repo)](https://github.com/yunjey/pytorch-tutorial)
