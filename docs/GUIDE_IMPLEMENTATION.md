# 📖 Guide Conceptuel d'Implémentation — Leaffliction (PyTorch)

> **Objectif** : Expliquer **littéralement** ce que chaque classe doit faire, **sans code**, pour une approche **PyTorch avec transformations comme features**.

---

## 📑 Table des matières

1. [Vue d'ensemble : PyTorch avec Transformations](#vue-densemble)
2. [Pipeline PyTorch](#pipeline-pytorch)
3. [leaffliction/cli.py — Parsers d'arguments](#cli)
4. [leaffliction/utils.py — Utilitaires](#utils)
5. [leaffliction/dataset.py — Gestion du dataset](#dataset)
6. [leaffliction/plotting.py — Visualisations](#plotting)
7. [leaffliction/augmentations.py — Augmentations](#augmentations)
8. [leaffliction/transformations.py — Création de Tensors](#transformations)
9. [leaffliction/model.py — Modèle PyTorch](#model)
10. [leaffliction/train_pipeline.py — Pipeline d'entraînement](#train-pipeline)
11. [leaffliction/predict_pipeline.py — Pipeline de prédiction](#predict-pipeline)

---

<a id="vue-densemble"></a>
## 1. Vue d'ensemble : PyTorch avec Transformations

### Architecture Unique

Cette approche combine :
- ✅ **Transformations manuelles** (Grayscale, Canny, etc.) comme **canaux d'entrée**
- ✅ **CNN simple** (PyTorch) pour apprendre des patterns sur ces transformations
- ✅ **Pas de features manuelles** (histogrammes, stats)

### Concept Clé

```
Image RGB (H, W, 3)
     ↓
Appliquer 6 transformations
     ↓
Créer tensor (6, H, W)  ← 6 canaux au lieu de 3 RGB
     ↓
CNN PyTorch
     ↓
Classification
```

### Avantages

✅ **Plus performant** que features manuelles (histogrammes)
✅ **Plus simple** qu'un CNN complet (pas besoin de millions d'images)
✅ **Interprétable** : On sait quels canaux sont utilisés
✅ **Rapide** : Entraînement en quelques minutes
✅ **Flexible** : Architecture PyTorch modifiable

---

<a id="pipeline-pytorch"></a>
## 2. Pipeline PyTorch

### Schéma Complet

```
(1) Dataset brut
     ↓
(2) Scan + Split train/valid
     ↓
(3) Augmentation du train set (images physiques)
     ↓ Crée plus d'images sur disque
     ↓
(4) Transformation en tensors PyTorch
     ↓ Applique 6 transformations par image
     ↓ Crée tensor (n, 6, H, W)
     ↓
(5) Création DataLoaders PyTorch
     ↓
(6) Entraînement CNN (backpropagation)
     ↓
(7) Évaluation (accuracy > 90%)
     ↓
(8) Sauvegarde (model.pth, labels.json)
     ↓
(9) Packaging (learnings.zip)
```

### Les 6 Transformations comme Canaux

```python
Canal 0: Grayscale
Canal 1: Canny Edges
Canal 2: Histogram Equalisation
Canal 3: Sharpen
Canal 4: Threshold
Canal 5: Morphology

Tensor final: (batch, 6, 224, 224)
```

---

<a id="cli"></a>
## 3. leaffliction/cli.py — Parsers d'arguments

**Statut** : ✅ Déjà implémenté, pas de changement nécessaire.

---

<a id="utils"></a>
## 4. leaffliction/utils.py — Utilitaires

**Statut** : ✅ Déjà implémenté, pas de changement nécessaire.

---

<a id="dataset"></a>
## 5. leaffliction/dataset.py — Gestion du dataset

### Changements par rapport à ML traditionnel

**À SUPPRIMER** :
- ❌ `TFDataConfig` (pas besoin de tf.data)
- ❌ `TFDatasetBuilder` (pas besoin de pipeline TensorFlow)

**À GARDER** :
- ✅ `DatasetIndex`
- ✅ `DatasetScanner`
- ✅ `DatasetSplitter`

---

### **Classe : DatasetScanner**

**Méthode : scan(root)**

**Ce qu'elle doit faire** :

**Étape 1 : Lister les sous-dossiers**
- Recevoir un chemin vers le dossier racine (Path)
- Lister TOUS les sous-dossiers directs
- Filtrer pour ne garder que les dossiers (pas les fichiers)
- Trier alphabétiquement

**Étape 2 : Extraire les noms de classes**
- Pour chaque dossier, extraire son nom
- Ces noms deviennent `class_names`
- L'ordre détermine les `class_id` (0, 1, 2, ...)

**Étape 3 : Scanner chaque classe**
- Pour chaque dossier (avec son index comme class_id) :
  - Utiliser `PathManager.ft_iter_images()` pour lister les images
  - Compter le nombre d'images
  - Pour chaque image :
    - Créer un tuple `(chemin_image, class_id)`
    - Ajouter à la liste `items`
  - Stocker le compte dans `counts`

**Étape 4 : Retourner**
- Créer un `DatasetIndex` avec toutes ces informations
- Retourner cet objet

---

### **Classe : DatasetSplitter**

**Statut** : ✅ Déjà implémenté (split stratifié)

Pas de changement nécessaire.

---

<a id="plotting"></a>
## 6. leaffliction/plotting.py — Visualisations

**Statut** : ✅ Déjà implémenté, pas de changement nécessaire.

---

<a id="augmentations"></a>
## 7. leaffliction/augmentations.py — Augmentations

### Rôle dans PyTorch

**Augmentations** = Créer des **images physiques** sur disque (TRAIN ONLY)

**Différence avec CNN classique** :
- **CNN classique** : Augmentations à la volée (dans le DataLoader)
- **Notre approche** : Augmentations créent des fichiers AVANT le training

**Pourquoi** : Simplifie le pipeline et permet de visualiser les augmentations.

---

### **Classe : AugmentationEngine**

#### **Méthode : augment_dataset(train_items, output_dir, augmentations_per_image)**

**Ce qu'elle doit faire** :

**Paramètres** :
- `train_items` : Liste de tuples `(Path, class_id)`
- `output_dir` : Dossier où sauvegarder les images augmentées
- `augmentations_per_image` : Nombre d'augmentations par image (ex: 3)

**Retour** :
- Liste étendue : originales + augmentées

---

**Étape 1 : Créer le dossier de sortie**
- S'assurer que `output_dir` existe
- Créer les sous-dossiers par classe si nécessaire

**Étape 2 : Initialiser la liste de retour**
- Créer une liste vide `augmented_items`

**Étape 3 : Pour chaque image du train set**

**Sous-étape 3.1 : Garder l'originale**
- Ajouter `(img_path, label)` à `augmented_items`

**Sous-étape 3.2 : Charger l'image**
- Utiliser OpenCV : `cv2.imread(str(img_path))`
- Convertir BGR → RGB : `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`

**Sous-étape 3.3 : Créer N versions augmentées**
- Pour `i` de 0 à `augmentations_per_image - 1` :
  - Appliquer `apply_random(img, n=2)` (2 augmentations aléatoires)
  - Générer un nom de fichier : `{stem}_aug{i}{ext}`
  - Créer le chemin complet : `output_dir / class_name / filename`
  - Créer les dossiers parents si nécessaire
  - Convertir RGB → BGR
  - Sauvegarder avec `cv2.imwrite()`
  - Ajouter `(aug_path, label)` à `augmented_items`

**Étape 4 : Retourner**
- Retourner `augmented_items` (liste étendue)

---

### **Les 6 Augmentations**

Toutes travaillent avec **NumPy arrays** et **OpenCV**.

#### **FlipHorizontalAug**
```python
def apply(self, img: np.ndarray) -> np.ndarray:
    return cv2.flip(img, 1)  # 1 = horizontal
```

#### **FlipVerticalAug**
```python
def apply(self, img: np.ndarray) -> np.ndarray:
    return cv2.flip(img, 0)  # 0 = vertical
```

#### **RotateAug**
```python
def apply(self, img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, self.angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))
```

#### **BrightnessContrastAug**
```python
def apply(self, img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    img = img * (1 + self.contrast) + self.brightness
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)
```

#### **GaussianBlurAug**
```python
def apply(self, img: np.ndarray) -> np.ndarray:
    ksize = int(2 * np.ceil(3 * self.sigma) + 1)
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), self.sigma)
```

#### **RandomCropResizeAug**
```python
def apply(self, img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    new_h = int(h * self.crop_ratio)
    new_w = int(w * self.crop_ratio)
    
    top = random.randint(0, h - new_h)
    left = random.randint(0, w - new_w)
    
    cropped = img[top:top+new_h, left:left+new_w]
    return cv2.resize(cropped, (w, h))
```

---

<a id="transformations"></a>
## 8. leaffliction/transformations.py — Création de Tensors

### Rôle dans PyTorch

**Transformations** = **Création de canaux** pour le CNN

Les transformations ne sont plus pour extraire des features numériques, mais pour créer des **canaux visuels** que le CNN va analyser.

---

### **Classe : TransformationEngine**

#### **Méthode : apply_all_as_tensor(img)**

**Ce qu'elle doit faire** :

**Paramètres** :
- `img` : Image RGB (H, W, 3) en NumPy array

**Retour** :
- `torch.Tensor` de shape `(n_transforms, H, W)`

---

**Étape 1 : Initialiser la liste de canaux**
- Créer une liste vide `channels = []`

**Étape 2 : Pour chaque transformation**
- Appliquer la transformation : `transformed = tf.apply(img)`
- Si l'image est en couleur (3 canaux) :
  - Convertir en grayscale : `cv2.cvtColor(transformed, cv2.COLOR_RGB2GRAY)`
- Normaliser [0, 255] → [0, 1] :
  - `transformed = transformed.astype(np.float32) / 255.0`
- Ajouter à la liste : `channels.append(transformed)`

**Étape 3 : Stack en tensor**
- Utiliser NumPy : `stacked = np.stack(channels, axis=0)`
- Convertir en PyTorch : `tensor = torch.from_numpy(stacked)`
- Shape finale : `(n_transforms, H, W)`

**Étape 4 : Retourner**
- Retourner le tensor

---

#### **Méthode : batch_transform(items, img_size)**

**Ce qu'elle doit faire** :

**Paramètres** :
- `items` : Liste de tuples `(Path, class_id)`
- `img_size` : Tuple `(H, W)` pour redimensionner (ex: (224, 224))

**Retour** :
- `X` : torch.Tensor de shape `(n, n_transforms, H, W)`
- `y` : torch.Tensor de shape `(n,)`

---

**Étape 1 : Initialiser les listes**
- Créer `X_list = []` et `y_list = []`

**Étape 2 : Pour chaque item**

**Sous-étape 2.1 : Charger l'image**
- Utiliser OpenCV : `img = cv2.imread(str(img_path))`
- Vérifier si l'image est chargée (pas None)
- Convertir BGR → RGB : `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`
- Redimensionner : `cv2.resize(img, img_size)`

**Sous-étape 2.2 : Transformer en tensor**
- Appeler `apply_all_as_tensor(img)`
- Ajouter à X_list : `X_list.append(tensor)`
- Ajouter le label à y_list : `y_list.append(label)`

**Étape 3 : Stack en batch**
- `X = torch.stack(X_list)` → shape `(n, n_transforms, H, W)`
- `y = torch.tensor(y_list, dtype=torch.long)` → shape `(n,)`

**Étape 4 : Retourner**
- Retourner `(X, y)`

---

### **Les 6 Transformations**

Identiques aux augmentations, mais appliquées pour créer des canaux.

#### **GrayscaleTf**
```python
def apply(self, img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
```

#### **CannyEdgesTf**
```python
def apply(self, img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.Canny(gray, self.threshold1, self.threshold2)
```

#### **HistogramEqualisationTf**
```python
def apply(self, img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.equalizeHist(gray)
```

#### **SharpenTf**
```python
def apply(self, img: np.ndarray) -> np.ndarray:
    kernel = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]])
    return cv2.filter2D(img, -1, kernel)
```

#### **ThresholdTf**
```python
def apply(self, img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)
    return thresh
```

#### **MorphologyTf**
```python
def apply(self, img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
    
    if self.mode == "erode":
        return cv2.erode(gray, kernel)
    elif self.mode == "dilate":
        return cv2.dilate(gray, kernel)
    # etc.
```

---

<a id="model"></a>
## 9. leaffliction/model.py — Modèle PyTorch

### Architecture : TransformationClassifier

**Input** : `(batch, 6, 224, 224)` - 6 canaux de transformations

**Output** : `(batch, num_classes)` - Logits pour chaque classe

---

### **Classe : TransformationClassifier (nn.Module)**

#### **Architecture**

```
Input: (batch, 6, 224, 224)
     ↓
Conv2D(6→32) + ReLU + MaxPool(2)
     → (batch, 32, 112, 112)
     ↓
Conv2D(32→64) + ReLU + MaxPool(2)
     → (batch, 64, 56, 56)
     ↓
Conv2D(64→128) + ReLU + MaxPool(2)
     → (batch, 128, 28, 28)
     ↓
Conv2D(128→256) + ReLU + MaxPool(2)
     → (batch, 256, 14, 14)
     ↓
GlobalAveragePooling
     → (batch, 256, 1, 1)
     ↓
Flatten
     → (batch, 256)
     ↓
Dense(256→128) + ReLU + Dropout(0.5)
     → (batch, 128)
     ↓
Dense(128→num_classes)
     → (batch, num_classes)
```

---

#### **Méthode : __init__(num_classes, input_channels)**

**Ce qu'elle doit faire** :

**Paramètres** :
- `num_classes` : Nombre de classes (ex: 7)
- `input_channels` : Nombre de transformations (ex: 6)

---

**Étape 1 : Définir les convolutions**
- Créer un `nn.Sequential` avec :
  - Conv2D(input_channels → 32, kernel=3, padding=1)
  - ReLU
  - MaxPool2D(2)
  - Conv2D(32 → 64, kernel=3, padding=1)
  - ReLU
  - MaxPool2D(2)
  - Conv2D(64 → 128, kernel=3, padding=1)
  - ReLU
  - MaxPool2D(2)
  - Conv2D(128 → 256, kernel=3, padding=1)
  - ReLU
  - MaxPool2D(2)

**Étape 2 : Définir le Global Average Pooling**
- `nn.AdaptiveAvgPool2d(1)` - Réduit à (batch, 256, 1, 1)

**Étape 3 : Définir le classifier**
- Créer un `nn.Sequential` avec :
  - Flatten
  - Linear(256 → 128)
  - ReLU
  - Dropout(0.5)
  - Linear(128 → num_classes)

---

#### **Méthode : forward(x)**

**Ce qu'elle doit faire** :

**Paramètres** :
- `x` : Tensor de shape `(batch, 6, 224, 224)`

**Retour** :
- Tensor de shape `(batch, num_classes)`

---

**Étape 1 : Passer par les convolutions**
- `x = self.features(x)` → `(batch, 256, 14, 14)`

**Étape 2 : Global Average Pooling**
- `x = self.gap(x)` → `(batch, 256, 1, 1)`

**Étape 3 : Classifier**
- `x = self.classifier(x)` → `(batch, num_classes)`

**Étape 4 : Retourner**
- Retourner `x` (logits)

---

### **Classe : PyTorchModelFactory**

#### **Méthode : build(cfg)**

**Ce qu'elle doit faire** :

**Paramètres** :
- `cfg` : ModelConfig

**Retour** :
- `TransformationClassifier` non entraîné

---

**Étape 1 : Créer le modèle**
```python
model = TransformationClassifier(
    num_classes=cfg.num_classes,
    input_channels=cfg.input_channels  # 6
)
```

**Étape 2 : Retourner**
- Retourner le modèle

---

### **Classe : PyTorchModelBundle**

#### **Attributs**

**Ce qu'elle doit contenir** :
- `model` : TransformationClassifier entraîné
- `labels` : LabelEncoder (mapping classe ↔ id)
- `transformation_engine` : TransformationEngine (pour créer tensors)
- `cfg` : ModelConfig
- `device` : torch.device (CPU ou GPU)

---

#### **Méthode : save(out_dir)**

**Ce qu'elle doit faire** :

**Étape 1 : Créer le dossier**
- S'assurer que `out_dir` existe

**Étape 2 : Sauvegarder le modèle**
- Utiliser PyTorch : `torch.save(self.model.state_dict(), out_dir / "model.pth")`

**Étape 3 : Sauvegarder les labels**
- Convertir en dict : `labels_dict = self.labels.to_json_dict()`
- Écrire en JSON : `json.dump(labels_dict, open(out_dir / "labels.json", "w"))`

**Étape 4 : Sauvegarder la config**
- Convertir cfg en dict
- Écrire en JSON : `json.dump(config_dict, open(out_dir / "config.json", "w"))`

---

#### **Méthode : load(in_dir)** (classmethod)

**Ce qu'elle doit faire** :

**Étape 1 : Charger la config**
- Lire le JSON
- Créer un ModelConfig

**Étape 2 : Charger les labels**
- Lire le JSON
- Créer un LabelEncoder : `labels = LabelEncoder.from_json_dict(data)`

**Étape 3 : Créer le modèle**
- `model = TransformationClassifier(cfg.num_classes, cfg.input_channels)`
- Charger les poids : `model.load_state_dict(torch.load(in_dir / "model.pth"))`

**Étape 4 : Recréer le TransformationEngine**
- Créer un TransformationEngine avec les 6 transformations

**Étape 5 : Créer et retourner le bundle**
- `return PyTorchModelBundle(model, labels, tf_engine, cfg)`

---

#### **Méthode : predict(tensor)**

**Ce qu'elle doit faire** :

**Paramètres** :
- `tensor` : torch.Tensor de shape `(n_transforms, H, W)` ou `(1, n_transforms, H, W)`

**Retour** :
- `pred_id` : int (ID de la classe prédite)
- `probs` : Dict[str, float] (probabilités par classe)

---

**Étape 1 : Préparer le tensor**
- Si shape `(n_transforms, H, W)` : ajouter batch dimension
  - `tensor = tensor.unsqueeze(0)` → `(1, n_transforms, H, W)`
- Déplacer sur le device : `tensor = tensor.to(self.device)`

**Étape 2 : Mode évaluation**
- `self.model.eval()`

**Étape 3 : Prédire (sans gradient)**
```python
with torch.no_grad():
    outputs = self.model(tensor)  # (1, num_classes)
    probs_tensor = torch.softmax(outputs, dim=1)  # Probabilités
    pred_id = torch.argmax(probs_tensor, dim=1).item()  # ID prédit
```

**Étape 4 : Convertir probs en dict**
```python
probs_np = probs_tensor.cpu().numpy()[0]
probs = {
    self.labels.decode(i): float(probs_np[i])
    for i in range(len(probs_np))
}
```

**Étape 5 : Retourner**
- `return pred_id, probs`

---

<a id="train-pipeline"></a>
## 10. leaffliction/train_pipeline.py — Pipeline d'entraînement

### Pipeline PyTorch

```
1. Scanner dataset
2. Split train/valid
3. Augmenter train set (images physiques)
4. Transformer en tensors PyTorch
5. Créer DataLoaders
6. Entraîner avec backpropagation
7. Évaluer
8. Sauvegarder
```

---

### **Classe : PyTorchTrainer**

#### **Méthode : train(dataset_dir, out_dir, cfg)**

**Ce qu'elle doit faire** :

**Paramètres** :
- `dataset_dir` : Path vers le dataset
- `out_dir` : Path vers le dossier de sortie
- `cfg` : TrainConfig

**Retour** :
- `Metrics` (train_accuracy, valid_accuracy, valid_count)

---

**Étape 1 : Scanner le dataset**
```python
index = self.dataset_scanner.scan(dataset_dir)
```

**Étape 2 : Fitter le LabelEncoder**
```python
self.labels.fit(index.class_names)
```

**Étape 3 : Split train/valid**
```python
train_items, valid_items = self.dataset_splitter.split(
    index.items,
    cfg.valid_ratio,
    cfg.seed,
    stratified=True
)
```

**Étape 4 : Augmenter le train set (optionnel)**
```python
if cfg.augment_train:
    train_items = self.augmentation_engine.augment_dataset(
        train_items,
        out_dir / "augmented",
        cfg.augmentations_per_image
    )
```

**Étape 5 : Transformer en tensors**
```python
X_train, y_train = self.transformation_engine.batch_transform(
    train_items, 
    cfg.img_size
)
X_valid, y_valid = self.transformation_engine.batch_transform(
    valid_items, 
    cfg.img_size
)
```

**Résultat** :
- `X_train` : `(n_train, 6, 224, 224)`
- `y_train` : `(n_train,)`
- `X_valid` : `(n_valid, 6, 224, 224)`
- `y_valid` : `(n_valid,)`

**Étape 6 : Créer DataLoaders**
```python
from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(X_train, y_train)
valid_dataset = TensorDataset(X_valid, y_valid)

train_loader = DataLoader(
    train_dataset, 
    batch_size=cfg.batch_size, 
    shuffle=True
)
valid_loader = DataLoader(
    valid_dataset, 
    batch_size=cfg.batch_size, 
    shuffle=False
)
```

**Étape 7 : Construire le modèle**
```python
model = self.model_factory.build(ModelConfig(
    num_classes=index.num_classes,
    input_channels=6,
    img_size=cfg.img_size
))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
```

**Étape 8 : Définir loss et optimizer**
```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
```

**Étape 9 : Training loop**

Pour chaque epoch :

**Phase Training** :
```python
model.train()
for X_batch, y_batch in train_loader:
    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
    
    # Forward
    optimizer.zero_grad()
    outputs = model(X_batch)
    loss = criterion(outputs, y_batch)
    
    # Backward
    loss.backward()
    optimizer.step()
    
    # Calculer accuracy
    _, predicted = torch.max(outputs, 1)
    train_correct += (predicted == y_batch).sum().item()
```

**Phase Validation** :
```python
model.eval()
with torch.no_grad():
    for X_batch, y_batch in valid_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        valid_correct += (predicted == y_batch).sum().item()
```

**Sauvegarder meilleur modèle** :
```python
if valid_acc > best_valid_acc:
    best_valid_acc = valid_acc
    torch.save(model.state_dict(), out_dir / "best_model.pth")
```

**Étape 10 : Évaluation finale**
- Charger le meilleur modèle
- Calculer train_accuracy et valid_accuracy

**Étape 11 : Créer métriques**
```python
metrics = Metrics(
    train_accuracy=train_acc,
    valid_accuracy=valid_acc,
    valid_count=len(valid_items)
)
```

**Étape 12 : Sauvegarder le bundle**
```python
bundle = PyTorchModelBundle(
    model=model,
    labels=self.labels,
    transformation_engine=self.transformation_engine,
    cfg=model_cfg
)
bundle.save(out_dir / "model")
```

**Étape 13 : Retourner**
- Retourner `metrics`

---

<a id="predict-pipeline"></a>
## 11. leaffliction/predict_pipeline.py — Pipeline de prédiction

### Pipeline PyTorch

```
1. Charger le bundle (model.pth, labels.json)
2. Charger et transformer l'image en tensor
3. Prédire avec le modèle PyTorch
4. Décoder le label
5. (Optionnel) Afficher transformations
```

---

### **Classe : PyTorchPredictor**

#### **Méthode : predict(bundle_zip, image_path, cfg)**

**Ce qu'elle doit faire** :

**Paramètres** :
- `bundle_zip` : Path vers learnings.zip
- `image_path` : Path vers l'image à prédire
- `cfg` : PredictConfig

**Retour** :
- `predicted_label` : str (nom de la classe)
- `probs` : Dict[str, float] (probabilités par classe)
- `transformed` : Dict[str, np.ndarray] (transformations pour visualisation)

---

**Étape 1 : Charger le bundle**
```python
import tempfile

with tempfile.TemporaryDirectory() as temp_dir:
    bundle = self.bundle_loader.load_from_zip(bundle_zip, Path(temp_dir))
```

**Étape 2 : Charger et transformer l'image**

**Sous-étape 2.1 : Charger l'image**
- Utiliser OpenCV : `img = cv2.imread(str(image_path))`
- Vérifier que l'image est chargée (pas None)
- Convertir BGR → RGB : `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`
- Redimensionner : `cv2.resize(img, bundle.cfg.img_size)`

**Sous-étape 2.2 : Créer le tensor**
- Appeler `self.transformation_engine.apply_all_as_tensor(img)`
- Résultat : tensor de shape `(6, 224, 224)`

**Étape 3 : Prédire**
```python
pred_id, probs = bundle.predict(tensor)
```

**Étape 4 : Décoder le label**
```python
predicted_label = bundle.labels.decode(pred_id)
```

**Étape 5 : (Optionnel) Appliquer transformations pour visualisation**
```python
transformed = {}
if cfg.show_transforms:
    transformed = self.transformation_engine.apply_all(img)
```

**Étape 6 : Retourner**
- Retourner `(predicted_label, probs, transformed)`

---

### **Classe : PredictionVisualiser**

#### **Méthode : show(original, transformed, predicted_label)**

**Ce qu'elle doit faire** :

**Paramètres** :
- `original` : Image originale (np.ndarray)
- `transformed` : Dict des transformations `{name: img}`
- `predicted_label` : str (classe prédite)

---

**Étape 1 : Créer le titre**
- `title = f"Prediction: {predicted_label}"`

**Étape 2 : Utiliser GridPlotter**
```python
from leaffliction.plotting import GridPlotter

grid = GridPlotter()
grid.show_grid(title, transformed, original=original)
```

---

## 📚 Ordre d'Implémentation Recommandé

### **Phase 1 : Dataset (Priorité 🔴)**

1. **DatasetScanner.scan()**
   - Lister sous-dossiers
   - Extraire class_names
   - Scanner images
   - Créer DatasetIndex

**Test** : `python Distribution.py ./leaves/images/`

---

### **Phase 2 : Augmentations (Priorité 🟡)**

2. **Les 6 augmentations**
   - FlipHorizontalAug
   - FlipVerticalAug
   - RotateAug
   - BrightnessContrastAug
   - GaussianBlurAug
   - RandomCropResizeAug

3. **AugmentationEngine.default_six()**
   - Factory pour créer les 6 augmentations

4. **AugmentationEngine.apply_random()**
   - Sélectionner n augmentations aléatoires
   - Appliquer séquentiellement

5. **AugmentationEngine.augment_dataset()**
   - Créer images physiques sur disque
   - Retourner liste étendue

**Test** : `python Augmentation.py ./leaves/images/Apple_healthy/image\ \(1\).JPG`

---

### **Phase 3 : Transformations (Priorité 🔴)**

6. **Les 6 transformations**
   - GrayscaleTf
   - CannyEdgesTf
   - HistogramEqualisationTf
   - SharpenTf
   - ThresholdTf
   - MorphologyTf

7. **TransformationEngine.default_six()**
   - Factory pour créer les 6 transformations

8. **TransformationEngine.apply_all()**
   - Pour visualisation
   - Retourne Dict[str, np.ndarray]

9. **TransformationEngine.apply_all_as_tensor()**
   - Créer tensor PyTorch
   - Shape : (6, H, W)

10. **TransformationEngine.batch_transform()**
    - Transformer batch d'images
    - Retourne (X, y) tensors

**Test** : `python Transformation.py ./leaves/images/Apple_healthy/image\ \(1\).JPG`

---

### **Phase 4 : Modèle (Priorité 🔴)**

11. **LabelEncoder**
    - fit()
    - encode()
    - decode()
    - to_json_dict()
    - from_json_dict()

12. **TransformationClassifier**
    - __init__() : Définir architecture
    - forward() : Forward pass

13. **PyTorchModelFactory.build()**
    - Créer TransformationClassifier

14. **PyTorchModelBundle**
    - save() : Sauvegarder model.pth, labels.json
    - load() : Charger depuis dossier
    - load_from_zip() : Charger depuis ZIP
    - predict() : Prédire depuis tensor

**Test** : Créer un petit modèle et tester forward pass

---

### **Phase 5 : Training (Priorité 🔴)**

15. **PyTorchTrainer.train()**
    - Scanner dataset
    - Split train/valid
    - Augmenter train set
    - Transformer en tensors
    - Créer DataLoaders
    - Training loop
    - Évaluation
    - Sauvegarder bundle

16. **RequirementsGate.assert_ok()**
    - Vérifier accuracy > 90%
    - Vérifier valid_count >= 100

17. **TrainingPackager**
    - prepare_artifacts_dir()
    - build_zip()

**Test** : `python train.py ./leaves/images/ --epochs 5`

---

### **Phase 6 : Prédiction (Priorité 🟢)**

18. **PyTorchPredictor.predict()**
    - Charger bundle
    - Transformer image
    - Prédire
    - Retourner résultat

19. **PredictionVisualiser.show()**
    - Afficher grille avec transformations

**Test** : `python predict.py learnings.zip ./leaves/images/Apple_healthy/image\ \(1\).JPG`

---

### **Phase 7 : Finalisation (Priorité 🟢)**

20. **Génération signature.txt**
    - Calculer SHA1 de learnings.zip
    - Écrire dans signature.txt

21. **Tests end-to-end**
    - Training complet
    - Prédiction sur plusieurs images
    - Vérification accuracy

---

## 🎯 Points Clés à Retenir

### **1. Transformations = Canaux**
- 6 transformations → 6 canaux d'entrée
- Pas de features manuelles (histogrammes, stats)
- Le CNN apprend directement des transformations

### **2. Augmentations = Images Physiques**
- Créées AVANT le training
- Sauvegardées sur disque
- Pas d'augmentation à la volée

### **3. Pipeline PyTorch**
- DataLoaders pour batching
- Training loop avec backpropagation
- Sauvegarde best model

### **4. Architecture Simple**
- 4 Conv2D + GAP + 2 Dense
- ~1M paramètres
- Entraînement rapide (quelques minutes)

### **5. Défendable**
- Architecture claire et interprétable
- Transformations explicites
- Performance élevée (>90%)

---

## 📖 Utilisation de ce Guide

### **Pour l'Implémentation**

1. **Lire la section** correspondant à la classe
2. **Comprendre les étapes** décrites
3. **Implémenter en Python** en suivant les étapes
4. **Tester** avec des données réelles

### **Pour la Soutenance**

1. **Expliquer l'architecture** : Transformations → Tensors → CNN
2. **Justifier les choix** : Pourquoi 6 transformations ?
3. **Défendre la logique** : Pourquoi cette approche ?
4. **Répondre aux questions** : Utiliser les explications du guide

---

## 🎉 Conclusion

Ce guide explique **littéralement** ce que chaque classe doit faire, **sans code**, pour implémenter le projet Leaffliction avec PyTorch.

**Points forts** :
- ✅ Architecture unique et performante
- ✅ Explications détaillées étape par étape
- ✅ Ordre d'implémentation recommandé
- ✅ Tests pour chaque phase

**Prochaine étape** : Commencer l'implémentation en suivant l'ordre recommandé !

**Bon courage ! 🚀**
