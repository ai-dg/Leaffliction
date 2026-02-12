# Multilayer Perceptron 🧠
![Score](https://img.shields.io/badge/Score-125%25-brightgreen)  
**A neural network implementation from scratch for breast cancer diagnosis using backpropagation**

> Build and train your own multilayer perceptron to classify breast cancer tumors as malignant or benign, implementing the fundamentals of deep learning without high-level ML libraries.

---

## 📚 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [How It Works](#how-it-works)
- [Getting Started](#getting-started)
- [Usage Instructions](#usage-instructions)
- [Project Structure](#project-structure)
- [Performance Results](#performance-results)
- [Deep Dive: Mathematical Foundations](#deep-dive-mathematical-foundations)
- [Sources and References](#sources-and-references)

---

## ▌Project Overview

This project implements a complete **multilayer perceptron (MLP)** from scratch for binary classification of breast cancer diagnoses.\
The implementation mimics the architecture and API of **Keras/TensorFlow**, providing a deep understanding of how neural networks work under the hood.\
It serves as an introduction to **deep learning fundamentals**, including forward propagation, backpropagation, gradient descent, and optimization algorithms.

📘 Educational AI project: **you'll understand every line of code behind a neural network**.

<div align="center">

| Learning curve (Loss vs Epochs) | Learning curve (Metrics vs Epochs) |
|:---:|:---:|
| <img src="https://github.com/user-attachments/assets/334fc0d1-b26e-4fbf-a97e-70eea901ee41" width="500"> | <img src="https://github.com/user-attachments/assets/9fd40fad-1f6d-421f-b133-657a12994b8e" width="500"> |

</div>


---

## ▌Features

✔️ **Custom Neural Network**: Complete MLP implementation from scratch\
✔️ **Keras-like API**: Familiar interface with `CustomSequential`, `DenseLayer`, `compile()`, `fit()`, `predict()`\
✔️ **Multiple Activations**: ReLU, Sigmoid, Softmax with proper gradient computation\
✔️ **Flexible Architecture**: Configurable hidden layers and neurons\
✔️ **Data Preprocessing**: Normalization, train/validation split, feature visualization\
✔️ **Learning Curves**: Real-time visualization of training progress\
✔️ **Model Persistence**: Save and load trained models using pickle\
✔️ **Command Line Interface**: Full CLI with all required parameters\
✔️ **Modular Architecture**: Separate modules for layers, models, optimizers, losses, metrics

---

## ▌Bonus Features

- ■ **Advanced Optimizers**: SGD and Adam optimization algorithms
- ■ **Multiple Loss Functions**: Binary Cross-Entropy, Categorical Cross-Entropy, MSE
- ■ **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score
- ■ **Callbacks System**: History tracking and Early Stopping
- ■ **Data Visualization**: Scatter plots, histograms, and boxplots for exploratory analysis
- ■ **Learning Curves**: Automatic plotting of loss and metrics over epochs
- ■ **History Export**: Save training history to text files

> ⚠️ These features are only evaluated if the core program works flawlessly.

---

## ▌How it works

### ■ Neural Network Architecture

The multilayer perceptron consists of:
- **Input Layer**: 30 features (tumor characteristics)
- **Hidden Layers**: 2+ fully connected layers with ReLU activation
- **Output Layer**: 1 neuron with Sigmoid (binary) or 2 neurons with Softmax (categorical)

### ■ Training Cycle

Each epoch follows this sequence:

```
EPOCH 1
  → Forward pass
  → Loss computation
  → Backward pass
  → Weight update

EPOCH 2
  → Forward pass
  → Loss computation
  → Backward pass
  → Weight update
  ...
```

### ■ Forward Propagation

Each layer performs a linear transformation followed by an activation:

**Linear Transformation:**  

$$
Z = XW + b
$$

**Activation Function:**  

$$
A = f(Z)
$$

**Complete Forward Pass:**  

$$
X \rightarrow \text{Dense}_1 \rightarrow \text{Dense}_2 \rightarrow \cdots \rightarrow \text{Dense}_L \rightarrow \hat{y}
$$

### ■ Backpropagation

Gradients flow backward through the network:

$$
\text{Loss} \rightarrow \frac{\partial L}{\partial \hat{y}} \rightarrow \text{Dense}_L \rightarrow \cdots \rightarrow \text{Dense}_1 \rightarrow \text{gradients}
$$

For each layer:
1. **Gradient after activation**: $dZ = dA \cdot f'(Z)$
2. **Weight gradient**: $dW = X^T dZ$
3. **Bias gradient**: $db = \sum dZ$
4. **Input gradient**: $dX = dZ W^T$

### ■ Weight Update

After computing gradients, the optimizer updates weights:

$$
W_{\text{new}} = W_{\text{old}} - \alpha \cdot dW
$$

---

## ▌Getting Started

### ■ Requirements

- Python 3.8+
- `numpy` (numerical operations)
- `pandas` (data loading)
- `matplotlib` (visualization)
- `pickle` (model serialization)

### ■ Installation

1. Clone the repository

```bash
git clone https://github.com/yourusername/multilayer-perceptron.git
cd multilayer-perceptron
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Verify the dataset

```bash
ls datasets/data.csv
```

### ■ Dataset Information

**Wisconsin Breast Cancer Dataset**
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- **Instances**: 569 samples
- **Features**: 30 real-valued features computed from cell nucleus images
- **Classes**: 2 (Malignant: 212, Benign: 357)
- **Missing Values**: None

**Features include:**
- Radius, texture, perimeter, area
- Smoothness, compactness, concavity
- Concave points, symmetry, fractal dimension
- Mean, standard error, and worst values for each

---

## ▌Usage Instructions

### ■ Basic Workflow

The program operates in three modes:

1. **Split**: Divide dataset into training and validation sets
2. **Train**: Train the neural network on the training set
3. **Predict**: Evaluate the trained model on new data

### ■ Mode 1: Data Splitting

Split the dataset into training and validation sets with visualization.

```bash
python mlp.py --mode split \
  --dataset ./datasets/data.csv \
  --valid_ratio 0.2 \
  --seed 42
```

**Parameters:**
- `--dataset`: Path to the CSV file containing the full dataset
- `--valid_ratio`: Proportion of data for validation (default: 0.2)
- `--seed`: Random seed for reproducibility (default: 42)

**Output:**
- `./datasets/train.npz`: Training set (X_train, y_train)
- `./datasets/valid.npz`: Validation set (X_valid, y_valid)
- Scatter plot, histogram, and boxplot for data exploration

### ■ Mode 2: Training

Train the multilayer perceptron with specified architecture and hyperparameters.

#### Basic Training

```bash
python mlp.py --mode train \
  --model_path ./models/model.pkl \
  --layers 24 24 \
  --epochs 70 \
  --batch_size 8 \
  --optimizer Adam \
  --learning_rate 0.0314 \
  --loss bce \
  --metrics Accuracy \
  --curve_prefix mlp
```

#### Training with Multiple Metrics

```bash
python mlp.py --mode train \
  --model_path ./models/model_metrics.pkl \
  --layers 24 24 \
  --epochs 50 \
  --batch_size 8 \
  --optimizer Adam \
  --learning_rate 0.0314 \
  --loss bce \
  --metrics Accuracy Precision Recall F1Score \
  --curve_prefix mlp_full
```

#### Training with Early Stopping

```bash
python mlp.py --mode train \
  --model_path ./models/model_earlystop.pkl \
  --layers 24 24 \
  --epochs 100 \
  --batch_size 8 \
  --optimizer Adam \
  --learning_rate 0.0314 \
  --loss bce \
  --metrics Accuracy \
  --early_stopping \
  --patience 10 \
  --min_delta 0.001 \
  --curve_prefix mlp_earlystop
```

**Training Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--model_path` | string | ./models/model.pkl | Path to save the trained model |
| `--layers` | int list | [24, 24] | Hidden layer sizes (e.g., 24 24 24) |
| `--epochs` | integer | 70 | Number of training epochs |
| `--batch_size` | integer | 8 | Mini-batch size for training |
| `--optimizer` | choice | Adam | Optimizer: SGD or Adam |
| `--learning_rate` | float | 0.0314 | Learning rate for optimization |
| `--loss` | choice | bce | Loss function: bce, cce, or mse |
| `--metrics` | string list | [Accuracy] | Metrics to track during training |
| `--curve_prefix` | string | mlp | Prefix for saved plots |
| `--early_stopping` | flag | False | Enable early stopping |
| `--patience` | integer | 5 | Epochs to wait before stopping |
| `--min_delta` | float | 0.0 | Minimum improvement threshold |

### ■ Mode 3: Prediction

Evaluate a trained model on validation or test data.

```bash
python mlp.py --mode predict \
  --model_path ./models/model.pkl \
  --predict_data ./datasets/valid.npz
```

**Parameters:**
- `--model_path`: Path to the trained model (.pkl file)
- `--predict_data`: Path to data for prediction (.npz or .csv)

**Output:**
- Loss and metrics on the dataset
- Predictions saved to `mlp_predictions.txt`
- Confusion matrix and classification report

---

## ▌Project Structure

```
multilayer-perceptron/
├── mlp.py                    # Main script (entry point)
├── custom_model.py           # CustomSequential model (like keras.Sequential)
├── custom_layer.py           # DenseLayer implementation with activations
├── optimizers.py             # SGD and Adam optimizers
├── losses.py                 # Loss functions (BCE, CCE, MSE)
├── metrics.py                # Evaluation metrics (Accuracy, Precision, Recall, F1)
├── callbacks.py              # History and EarlyStopping callbacks
├── data_processor.py         # Data loading, preprocessing, and splitting
├── plotting.py               # Visualization utilities
├── requirements.txt          # Python dependencies
├── tests.md                  # Test commands and examples
├── en.subject.pdf            # Project subject (42 School)
├── datasets/
│   ├── data.csv             # Original Wisconsin Breast Cancer dataset
│   ├── train.npz            # Training set (generated by split mode)
│   └── valid.npz            # Validation set (generated by split mode)
├── models/                   # Trained models directory
└── plots/                    # Generated plots and history files
    ├── mlp_loss.png         # Loss curves
    ├── mlp_metrics.png      # Metrics curves
    ├── mlp_history.txt      # Training history
    └── mlp_predictions.txt  # Prediction results
```

---

## ▌Performance Results

### ■ Expected Performance

On the Wisconsin Breast Cancer Dataset:

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | 95-98% |
| **Validation Loss** | 0.04-0.08 |
| **Training Time** | 10-30 seconds |
| **Convergence** | 40-60 epochs |

### ■ Optimizer Comparison

| Optimizer | Convergence Speed | Final Accuracy | Notes |
|-----------|------------------|----------------|-------|
| **Adam** | Fast | 96-98% | Recommended for most cases |
| **SGD** | Slower | 94-96% | May need more epochs |

### ■ Architecture Comparison

| Architecture | Parameters | Accuracy | Training Time | Notes |
|--------------|-----------|----------|---------------|-------|
| **24-24** | ~1,500 | 96-97% | Fast | Good baseline |
| **24-24-24** | ~2,000 | 96-98% | Medium | Slightly better |
| **48-48** | ~4,000 | 96-98% | Slower | Risk of overfitting |

---

## ▌Deep Dive: Mathematical Foundations

This section provides detailed mathematical explanations of each component.

### 🧱 Dense Layer (`custom_layer.py`)

#### Linear Transformation

Every dense layer applies:

$$
Z = XW + b
$$

Where:
- $X \in \mathbb{R}^{(m, d)}$: input batch
- $W \in \mathbb{R}^{(d, u)}$: weights
- $b \in \mathbb{R}^{(u)}$: bias
- $Z \in \mathbb{R}^{(m, u)}$: pre-activation

#### Weight Initialization

To prevent vanishing/exploding gradients, weights are initialized with:

```math
\text{limit} = \sqrt{\frac{2.0}{\text{input\_dim}}}
```

This is ideal for ReLU activations (He initialization).

#### Activation Functions

**Sigmoid** (binary output):  

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

Derivative:  

$$
\sigma'(z) = \sigma(z)(1 - \sigma(z))
$$

**ReLU** (hidden layers):  

$$
\text{ReLU}(z) = \max(0, z)
$$

Derivative:  

$$
\text{ReLU}'(z) = \begin{cases} 1 & z > 0 \\ 0 & z \le 0 \end{cases}
$$

**Softmax** (multi-class output):  

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

#### Backward Pass

Given gradient $dA$ from the next layer:

1. **Gradient after activation**:  

   $$dZ = dA \cdot f'(Z)$$

3. **Weight gradient**:  

   $$dW = X^T dZ$$

5. **Bias gradient**:  

   $$db = \sum dZ$$

7. **Input gradient**:  

   $$dX = dZ W^T$$

---

### 🔧 Optimizers (`optimizers.py`)

#### SGD (Stochastic Gradient Descent)

Basic update rule:  

$$
W := W - \alpha \cdot dW
$$

Where $\alpha$ is the learning rate.

#### Adam (Adaptive Moment Estimation)

Adam combines momentum and adaptive learning rates:

**First moment (momentum)**:  

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) dW
$$

**Second moment (adaptive)**:  

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) dW^2
$$

**Bias correction**:  

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

**Weight update**:  

$$
W := W - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

Default parameters: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$

---

### 📉 Loss Functions (`losses.py`)

#### Binary Cross-Entropy (BCE)

For binary classification with sigmoid output:

$$
L = -\frac{1}{N}\sum_{i=1}^{N} [y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]
$$

**Gradient** (simplified with sigmoid):  

$$
\frac{\partial L}{\partial Z} = \hat{y} - y
$$

**Numerical stability**: Clip predictions to $[\epsilon, 1-\epsilon]$ where $\epsilon = 10^{-12}$

#### Categorical Cross-Entropy (CCE)

For multi-class classification with softmax:

$$
L = -\frac{1}{N}\sum_{i=1}^{N}\sum_{k=1}^{K} y_{ik}\log(\hat{y}_{ik})
$$

**Gradient** (simplified with softmax):  

$$
\frac{\partial L}{\partial Z} = \hat{y} - y
$$

#### Mean Squared Error (MSE)

For regression tasks:

$$
L = \frac{1}{N}\sum_{i=1}^{N} (\hat{y}_i - y_i)^2
$$

**Gradient**:  

$$
\frac{\partial L}{\partial \hat{y}} = \frac{2}{N}(\hat{y} - y)
$$

---

### 📊 Metrics (`metrics.py`)

#### Confusion Matrix Elements

- **TP** (True Positives): $\sum [\hat{y}=1 \land y=1]$
- **FP** (False Positives): $\sum [\hat{y}=1 \land y=0]$
- **TN** (True Negatives): $\sum [\hat{y}=0 \land y=0]$
- **FN** (False Negatives): $\sum [\hat{y}=0 \land y=1]$

#### Accuracy

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

#### Precision

Measures reliability of positive predictions:  

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

#### Recall (Sensitivity)

Measures ability to detect positives:  

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

#### F1-Score

Harmonic mean of precision and recall:  

$$
F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

---

### 🔄 Callbacks (`callbacks.py`)

#### History

Stores metrics at each epoch:

```python
{
  "loss": [0.55, 0.39, 0.28, 0.19],
  "val_loss": [0.60, 0.41, 0.32, 0.25],
  "accuracy": [0.75, 0.82, 0.88, 0.92],
  "val_accuracy": [0.72, 0.81, 0.86, 0.90]
}
```

#### Early Stopping

Stops training when validation loss stops improving:

**Stopping condition**:  

```math
\text{val\_loss}_{\text{epoch}} > \text{best\_val\_loss} - \text{min\_delta}
```

for `patience` consecutive epochs.

---

### 📊 Data Processing (`data_processor.py`)

#### Normalization (Standardization)

Z-score normalization:

$$
X_{\text{norm}} = \frac{X - \mu}{\sigma}
$$

Where:
- $\mu$: mean computed on training set
- $\sigma$: standard deviation computed on training set

**Critical**: Statistics are computed ONLY on training data, then applied to both train and validation sets.

#### Train/Validation Split

Random split with reproducibility:

```python
np.random.seed(seed)
indices = np.random.permutation(len(X))
split_idx = int((1 - valid_ratio) * len(X))
train_idx = indices[:split_idx]
valid_idx = indices[split_idx:]
```

---

## ▌Complete Usage Example

Here's a complete workflow from data splitting to prediction:

```bash
# Step 1: Split the dataset
python mlp.py --mode split \
  --dataset ./datasets/data.csv \
  --valid_ratio 0.2 \
  --seed 42

# Step 2: Train the model
python mlp.py --mode train \
  --model_path ./models/model.pkl \
  --layers 24 24 \
  --epochs 70 \
  --batch_size 8 \
  --optimizer Adam \
  --learning_rate 0.0314 \
  --loss bce \
  --metrics Accuracy Precision Recall F1Score \
  --early_stopping \
  --patience 10 \
  --curve_prefix mlp

# Step 3: Evaluate the model
python mlp.py --mode predict \
  --model_path ./models/model.pkl \
  --predict_data ./datasets/valid.npz
```

**Expected Output:**

```
Binary cross-entropy on dataset: 0.051234
Accuracy on dataset: 0.9737
Precision on dataset: 0.9756
Recall on dataset: 0.9524
F1Score on dataset: 0.9639
```

---

## ▌Sources and References

This implementation was inspired by and references the following sources:

### Keras/TensorFlow Architecture

- [TensorFlow Repository](https://github.com/tensorflow/tensorflow)
- [Keras Repository](https://github.com/keras-team/keras)
- [Keras Source Code](https://github.com/keras-team/keras/tree/master/keras/src)

### Specific Components

**`custom_layer.py`** - Core Dense Layer
- [Keras Dense Layer](https://github.com/keras-team/keras/blob/master/keras/src/layers/core/dense.py)
- [Keras Base Layer](https://github.com/keras-team/keras/blob/master/keras/src/layers/layer.py)
- [Keras Activations](https://github.com/keras-team/keras/blob/master/keras/src/activations/activations.py)

**`custom_model.py`** - Sequential Model
- [Keras Sequential](https://github.com/keras-team/keras/blob/master/keras/src/models/sequential.py)
- [Keras Model Base](https://github.com/keras-team/keras/blob/master/keras/src/models/model.py)

**`optimizers.py`** - Optimization Algorithms
- [Keras Base Optimizer](https://github.com/keras-team/keras/blob/master/keras/src/optimizers/optimizer.py)
- [Keras SGD](https://github.com/keras-team/keras/blob/master/keras/src/optimizers/sgd.py)
- [Keras Adam](https://github.com/keras-team/keras/blob/master/keras/src/optimizers/adam.py)

**`losses.py`** - Loss Functions
- [Keras Base Loss](https://github.com/keras-team/keras/blob/master/keras/src/losses/loss.py)
- [Keras Loss Functions](https://github.com/keras-team/keras/blob/master/keras/src/losses/losses.py)

**`metrics.py`** - Evaluation Metrics
- [Keras Base Metric](https://github.com/keras-team/keras/blob/master/keras/src/metrics/metric.py)
- [Keras Accuracy Metrics](https://github.com/keras-team/keras/blob/master/keras/src/metrics/accuracy_metrics.py)
- [Keras Probabilistic Metrics](https://github.com/keras-team/keras/blob/master/keras/src/metrics/probabilistic_metrics.py)

**`callbacks.py`** - Training Callbacks
- [Keras History](https://github.com/keras-team/keras/blob/master/keras/src/callbacks/history.py)
- [Keras Early Stopping](https://github.com/keras-team/keras/blob/master/keras/src/callbacks/early_stopping.py)

**`data_processor.py`** - Data Processing
- [TensorFlow Dataset Operations](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/data/ops/dataset_ops.py)
- [Keras Preprocessing](https://github.com/keras-team/keras/tree/master/keras/src/layers/preprocessing)

### Dataset

- [Wisconsin Breast Cancer Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.names)

---

## ▌Evaluation

The project meets all mandatory requirements:
- ✅ Multilayer perceptron implementation from scratch
- ✅ Binary classification on breast cancer dataset
- ✅ At least 2 hidden layers
- ✅ Backpropagation algorithm
- ✅ Gradient descent optimization
- ✅ Data preprocessing and normalization
- ✅ Train/validation split
- ✅ Model save/load functionality
- ✅ Learning curves visualization
- ✅ Command line interface

Bonus features implemented:
- ✅ Adam optimizer (advanced optimization)
- ✅ Multiple loss functions (BCE, CCE, MSE)
- ✅ Multiple metrics (Accuracy, Precision, Recall, F1)
- ✅ History tracking during training
- ✅ Early stopping callback
- ✅ Comprehensive data visualization
- ✅ Modular and extensible architecture

---

## 📜 License

This project was completed as part of the **42 School** curriculum.\
It is intended for **academic purposes only** and follows the evaluation requirements set by 42.

Unauthorized public sharing or direct copying for **grading purposes** is discouraged.\
If you wish to use or study this code, please ensure it complies with **your school's policies**.

---

## Acknowledgments

Special thanks to:
- The Keras and TensorFlow teams for their excellent documentation
- The UCI Machine Learning Repository for providing the dataset
- The 42 School community for support and feedback

---

**Built with ❤️ for learning and understanding neural networks from the ground up.**
