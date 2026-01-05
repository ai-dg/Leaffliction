# 📖 Guide Conceptuel d'Implémentation — Leaffliction (ML Traditionnel)

> **Objectif** : Expliquer **littéralement** ce que chaque classe doit faire, **sans code**, pour une approche **Machine Learning traditionnelle** (SVM, Random Forest, KNN).

---

## 📑 Table des matières

1. [Vue d'ensemble : ML Traditionnel vs Deep Learning](#vue-densemble)
2. [Pipeline ML Traditionnel](#pipeline-ml-traditionnel)
3. [leaffliction/cli.py — Parsers d'arguments](#cli)
4. [leaffliction/utils.py — Utilitaires](#utils)
5. [leaffliction/dataset.py — Gestion du dataset](#dataset)
6. [leaffliction/plotting.py — Visualisations](#plotting)
7. [leaffliction/augmentations.py — Augmentations](#augmentations)
8. [leaffliction/transformations.py — Extraction de Features](#transformations)
9. [leaffliction/model.py — Modèle ML](#model)
10. [leaffliction/train_pipeline.py — Pipeline d'entraînement](#train-pipeline)
11. [leaffliction/predict_pipeline.py — Pipeline de prédiction](#predict-pipeline)

---

<a id="vue-densemble"></a>
## 1. Vue d'ensemble : ML Traditionnel vs Deep Learning

### Différences Fondamentales

| Aspect | Deep Learning (CNN) | ML Traditionnel |
|--------|-------------------|-----------------|
| **Modèle** | Réseau de neurones | SVM, Random Forest, KNN |
| **Features** | Apprises automatiquement | Extraites manuellement |
| **Données** | Beaucoup (milliers) | Moins (centaines) |
| **Training** | Lent (GPU) | Rapide (CPU) |
| **Interprétabilité** | Faible (boîte noire) | Élevée (features explicites) |

### Pourquoi ML Traditionnel ?

**Avantages** :
- ✅ Plus simple à comprendre
- ✅ Plus rapide à entraîner
- ✅ Moins de données nécessaires
- ✅ Pas besoin de GPU
- ✅ Features interprétables

**Inconvénients** :
- ⚠️ Accuracy potentiellement plus faible
- ⚠️ Nécessite une bonne extraction de features
- ⚠️ Moins flexible

---

<a id="pipeline-ml-traditionnel"></a>
## 2. Pipeline ML Traditionnel

### Schéma Complet

```
(1) Dataset brut
     ↓
(2) Scan + Split train/valid
     ↓
(3) Augmentation du train set (images physiques)
     ↓ Crée plus d'images sur disque
     ↓
(4) Extraction de features (train + valid)
     ↓ Transformations → vecteurs numériques
     ↓ Exemple: histogrammes, textures, contours
     ↓
(5) Normalisation (StandardScaler)
     ↓ Mean=0, Std=1
     ↓
(6) Entraînement modèle ML (SVM, Random Forest, KNN)
     ↓
(7) Évaluation (accuracy > 90%)
     ↓
(8) Sauvegarde (model.pkl, scaler.pkl, labels.json)
     ↓
(9) Packaging (learnings.zip)
```

### Différence Clé avec CNN

**CNN** :
```
Image → CNN → Prédiction
(Le CNN apprend les features automatiquement)
```

**ML Traditionnel** :
```
Image → Extraction Features → Modèle ML → Prédiction
(On extrait manuellement les features)
```

---

<a id="cli"></a>
## 3. leaffliction/cli.py — Parsers d'arguments

**Statut** : ✅ Déjà implémenté, pas de changement nécessaire.

Les parsers restent identiques pour les deux approches.

---

<a id="utils"></a>
## 4. leaffliction/utils.py — Utilitaires

**Statut** : ✅ Déjà implémenté, pas de changement nécessaire.

Les utilitaires (PathManager, Hasher, ZipPackager) sont identiques.

---

<a id="dataset"></a>
## 5. leaffliction/dataset.py — Gestion du dataset

### Changements par rapport à CNN

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

**Exemple de structure** :
```
root/
  Apple_Black_rot/     ← class_id = 0
    image (1).JPG
    image (2).JPG
  Apple_healthy/       ← class_id = 1
    image (1).JPG
```

---

### **Classe : DatasetSplitter**

**Statut** : ✅ Déjà implémenté (split stratifié)

Pas de changement nécessaire, fonctionne pour les deux approches.

---

<a id="plotting"></a>
## 6. leaffliction/plotting.py — Visualisations

**Statut** : ✅ Déjà implémenté, pas de changement nécessaire.

Les visualisations sont identiques pour les deux approches.

---

<a id="augmentations"></a>
## 7. leaffliction/augmentations.py — Augmentations

### Changements par rapport à CNN

**À SUPPRIMER** :
- ❌ `KerasAugmentationsFactory` (pas de Keras layers)

**À MODIFIER** :
- ✅ `AugmentationEngine.augment_dataset()` - Devient la méthode principale
- ✅ Les augmentations travaillent avec NumPy/OpenCV (pas TensorFlow)

---

### **Utilisation dans ML Traditionnel**

**Différence clé** :
- **CNN** : Augmentations à la volée pendant le training (dans le pipeline tf.data)
- **ML Traditionnel** : Augmentations créent des images PHYSIQUES sur disque AVANT le training

**Pourquoi** : Les modèles ML traditionnels ne peuvent pas faire d'augmentation à la volée. On doit créer les images augmentées une fois, puis extraire leurs features.

---

### **Classe : AugmentationEngine**

#### **Méthode : augment_dataset(train_items, output_dir, augmentations_per_image)**

**Ce qu'elle doit faire** :

**Paramètres** :
- `train_items` : Liste de tuples `(Path, class_id)`
- `output_dir` : Dossier où sauvegarder les images augmentées
- `augmentations_per_image` : Nombre d'augmentations par image (ex: 3)

---

**Étape 1 : Créer le dossier de sortie**
- S'assurer que `output_dir` existe
- Créer les sous-dossiers par classe si nécessaire

---

**Étape 2 : Initialiser la liste de retour**
- Créer une liste vide `augmented_items`

---

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

---

**Étape 4 : Retourner**
- Retourner `augmented_items` (liste étendue : originales + augmentées)

---

**Exemple** :
```
Input:
  train_items = [
    (Path("Apple_healthy/img1.jpg"), 1),  # 1 image
  ]
  augmentations_per_image = 3

Output:
  augmented_items = [
    (Path("Apple_healthy/img1.jpg"), 1),           # Originale
    (Path("augmented/Apple_healthy/img1_aug0.jpg"), 1),  # Aug 1
    (Path("augmented/Apple_healthy/img1_aug1.jpg"), 1),  # Aug 2
    (Path("augmented/Apple_healthy/img1_aug2.jpg"), 1),  # Aug 3
  ]
  # Total: 4 images (1 originale + 3 augmentées)
```

---

#### **Méthode : apply_random(img, n)**

**Ce qu'elle doit faire** :

**Paramètres** :
- `img` : Image NumPy array
- `n` : Nombre d'augmentations à appliquer (ex: 2)

---

**Étape 1 : Sélectionner n augmentations aléatoires**
- Utiliser `random.sample(self.augs, n)`
- Cela choisit n augmentations différentes au hasard

**Étape 2 : Appliquer séquentiellement**
- Copier l'image : `result = img.copy()`
- Pour chaque augmentation sélectionnée :
  - Appliquer : `result = aug.apply(result)`

**Étape 3 : Retourner**
- Retourner l'image augmentée

---

### **Les 6 Augmentations**

Toutes travaillent avec **NumPy arrays** et **OpenCV** (pas TensorFlow).

#### **FlipHorizontalAug**

```python
def apply(self, img: np.ndarray) -> np.ndarray:
    """Flip horizontal avec OpenCV"""
    return cv2.flip(img, 1)  # 1 = horizontal
```

#### **FlipVerticalAug**

```python
def apply(self, img: np.ndarray) -> np.ndarray:
    """Flip vertical avec OpenCV"""
    return cv2.flip(img, 0)  # 0 = vertical
```

#### **RotateAug**

```python
def apply(self, img: np.ndarray) -> np.ndarray:
    """Rotation avec OpenCV"""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, self.angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))
```

#### **BrightnessContrastAug**

```python
def apply(self, img: np.ndarray) -> np.ndarray:
    """Ajuste brightness et contrast"""
    img = img.astype(np.float32)
    img = img * (1 + self.contrast) + self.brightness
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)
```

#### **GaussianBlurAug**

```python
def apply(self, img: np.ndarray) -> np.ndarray:
    """Gaussian blur avec OpenCV"""
    ksize = int(2 * np.ceil(3 * self.sigma) + 1)
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), self.sigma)
```

#### **RandomCropResizeAug**

```python
def apply(self, img: np.ndarray) -> np.ndarray:
    """Random crop puis resize"""
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
## 8. leaffliction/transformations.py — Extraction de Features

### Rôle dans ML Traditionnel

**Transformations** = **Extraction de Features**

Les transformations ne sont plus juste pour la visualisation, elles sont **essentielles** pour extraire des caractéristiques numériques des images.

---

### **Classe : FeatureExtractor**

**Responsabilité** : Extraire un vecteur de features numériques depuis une image.

---

#### **Méthode : extract_features(img_path)**

**Ce qu'elle doit faire** :

**Paramètres** :
- `img_path` : Path vers une image

**Retour** :
- `np.ndarray` de shape `(n_features,)` - Vecteur de features

---

**Étape 1 : Charger l'image**
- Utiliser OpenCV : `cv2.imread(str(img_path))`
- Convertir BGR → RGB

---

**Étape 2 : Extraire features couleur**

**2.1 : Histogramme RGB**
- Pour chaque canal (R, G, B) :
  - Calculer l'histogramme : `np.histogram(img[:,:,channel], bins=256, range=(0, 256))`
  - Normaliser : `hist = hist / hist.sum()`
  - Ajouter les 256 valeurs à la liste de features
- Total : 256 × 3 = 768 features

**2.2 : Statistiques RGB**
- Pour chaque canal :
  - Mean : `img[:,:,channel].mean()`
  - Std : `img[:,:,channel].std()`
  - Min : `img[:,:,channel].min()`
  - Max : `img[:,:,channel].max()`
- Total : 4 × 3 = 12 features

---

**Étape 3 : Appliquer les transformations et extraire stats**

- Pour chaque transformation dans `self.transformations` :
  - Appliquer la transformation : `transformed = tf.apply(img)`
  - Extraire statistiques :
    - Mean
    - Std
    - Min
    - Max
  - Ajouter à la liste de features
- Total : 4 stats × 6 transformations = 24 features

---

**Étape 4 : (Optionnel) Features de texture**

**Haralick Features** (avec mahotas ou skimage) :
- Convertir en grayscale
- Calculer la matrice de co-occurrence (GLCM)
- Extraire 13 features de Haralick
- Total : 13 features

---

**Étape 5 : (Optionnel) Features de forme**

**Moments de Hu** :
- Convertir en grayscale
- Binariser (threshold)
- Calculer les moments
- Total : 7 features

---

**Étape 6 : Concaténer et retourner**
- Concaténer toutes les features en un seul vecteur
- Convertir en `np.ndarray` de type `float32`
- Retourner

**Total de features** : ~800-1000 features

---

#### **Méthode : extract_batch(items)**

**Ce qu'elle doit faire** :

**Paramètres** :
- `items` : Liste de tuples `(Path, class_id)`

**Retour** :
- `X` : np.ndarray de shape `(n_samples, n_features)`
- `y` : np.ndarray de shape `(n_samples,)`

---

**Étape 1 : Initialiser les listes**
- Créer `X = []` et `y = []`

**Étape 2 : Pour chaque item**
- Extraire les features : `features = self.extract_features(img_path)`
- Ajouter à X : `X.append(features)`
- Ajouter le label à y : `y.append(label)`

**Étape 3 : Convertir en arrays**
- `X = np.array(X)` → shape `(n_samples, n_features)`
- `y = np.array(y)` → shape `(n_samples,)`

**Étape 4 : Retourner**
- Retourner `(X, y)`

---

### **Les 6 Transformations**

Identiques à la version CNN, mais travaillent avec NumPy/OpenCV.

---

<a id="model"></a>
## 9. leaffliction/model.py — Modèle ML

### Changements par rapport à CNN

**À SUPPRIMER** :
- ❌ Tout ce qui concerne Keras/TensorFlow
- ❌ `ModelFactory` qui construit un CNN

**À AJOUTER** :
- ✅ `MLModelFactory` qui construit un modèle sklearn
- ✅ `MLModelBundle` qui sauvegarde avec joblib

---

### **Classe : MLModelFactory**

**Responsabilité** : Construire un modèle ML traditionnel (sklearn).

---

#### **Méthode : build(cfg, model_type)**

**Ce qu'elle doit faire** :

**Paramètres** :
- `cfg` : ModelConfig
- `model_type` : String ("svm", "random_forest", "knn")

**Retour** :
- Modèle sklearn non entraîné

---

**Si model_type == "svm"** :
```python
from sklearn.svm import SVC

return SVC(
    kernel='rbf',        # Radial Basis Function
    C=1.0,               # Régularisation
    gamma='scale',       # Coefficient du kernel
    probability=True,    # Pour avoir des probabilités
    random_state=cfg.seed
)
```

**Si model_type == "random_forest"** :
```python
from sklearn.ensemble import RandomForestClassifier

return RandomForestClassifier(
    n_estimators=100,    # Nombre d'arbres
    max_depth=None,      # Profondeur max (None = illimitée)
    random_state=cfg.seed,
    n_jobs=-1            # Utiliser tous les CPU
)
```

**Si model_type == "knn"** :
```python
from sklearn.neighbors import KNeighborsClassifier

return KNeighborsClassifier(
    n_neighbors=5,       # Nombre de voisins
    weights='distance',  # Pondération par distance
    n_jobs=-1
)
```

---

### **Classe : MLModelBundle**

**Responsabilité** : Encapsuler tout ce qui est nécessaire pour sauvegarder/charger un modèle ML.

---

#### **Attributs**

**Ce qu'elle doit contenir** :
- `model` : Modèle sklearn entraîné
- `scaler` : StandardScaler (pour normaliser les features)
- `labels` : LabelEncoder (mapping classe ↔ id)
- `feature_extractor` : FeatureExtractor (pour extraire features)
- `cfg` : ModelConfig

---

#### **Méthode : save(out_dir)**

**Ce qu'elle doit faire** :

**Étape 1 : Créer le dossier**
- S'assurer que `out_dir` existe

**Étape 2 : Sauvegarder le modèle**
- Utiliser joblib : `joblib.dump(self.model, out_dir / "model.pkl")`

**Étape 3 : Sauvegarder le scaler**
- Utiliser joblib : `joblib.dump(self.scaler, out_dir / "scaler.pkl")`

**Étape 4 : Sauvegarder les labels**
- Convertir en dict : `labels_dict = self.labels.to_json_dict()`
- Écrire en JSON : `json.dump(labels_dict, open(out_dir / "labels.json", "w"))`

**Étape 5 : Sauvegarder la config**
- Convertir cfg en dict
- Écrire en JSON : `json.dump(config_dict, open(out_dir / "config.json", "w"))`

**Étape 6 : Sauvegarder la config des features**
- Informations sur les transformations utilisées
- Écrire en JSON : `json.dump(feature_config, open(out_dir / "feature_config.json", "w"))`

---

#### **Méthode : load(in_dir)** (classmethod)

**Ce qu'elle doit faire** :

**Étape 1 : Charger le modèle**
- `model = joblib.load(in_dir / "model.pkl")`

**Étape 2 : Charger le scaler**
- `scaler = joblib.load(in_dir / "scaler.pkl")`

**Étape 3 : Charger les labels**
- Lire le JSON
- Créer un LabelEncoder : `labels = LabelEncoder.from_json_dict(data)`

**Étape 4 : Charger la config**
- Lire le JSON
- Créer un ModelConfig

**Étape 5 : Recréer le FeatureExtractor**
- Créer un TransformationEngine avec les 6 transformations
- Créer un FeatureExtractor avec ce moteur

**Étape 6 : Créer et retourner le bundle**
- `return MLModelBundle(model, scaler, labels, feature_extractor, cfg)`

---

#### **Méthode : predict(features)**

**Ce qu'elle doit faire** :

**Paramètres** :
- `features` : np.ndarray de shape `(n_features,)` ou `(1, n_features)`

**Retour** :
- `pred_id` : int (ID de la classe prédite)
- `probs` : Dict[str, float] (probabilités par classe)

---

**Étape 1 : Reshape si nécessaire**
- Si shape `(n_features,)` : reshape en `(1, n_features)`

**Étape 2 : Normaliser**
- `features_scaled = self.scaler.transform(features)`

**Étape 3 : Prédire**
- `pred_id = self.model.predict(features_scaled)[0]`

**Étape 4 : Obtenir les probabilités**
- Si le modèle supporte `predict_proba` :
  - `probs_array = self.model.predict_proba(features_scaled)[0]`
  - Créer un dict : `{self.labels.decode(i): float(p) for i, p in enumerate(probs_array)}`
- Sinon :
  - `probs = {self.labels.decode(pred_id): 1.0}`

**Étape 5 : Retourner**
- `return pred_id, probs`

---

<a id="train-pipeline"></a>
## 10. leaffliction/train_pipeline.py — Pipeline d'entraînement

### Changements par rapport à CNN

**Pipeline ML Traditionnel** :
```
1. Scanner dataset
2. Split train/valid
3. Augmenter train set (images physiques)
4. Extraire features (train + valid)
5. Normaliser features
6. Entraîner modèle ML
7. Évaluer
8. Sauvegarder
```

---

### **Classe : MLTrainer**

**Responsabilité** : Orchestrer tout le processus d'entraînement ML.

---

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
scanner = DatasetScanner()
index = scanner.scan(dataset_dir)
```

---

**Étape 2 : Fitter le LabelEncoder**
```python
labels = LabelEncoder()
labels.fit(index.class_names)
```

---

**Étape 3 : Split train/valid**
```python
splitter = DatasetSplitter()
train_items, valid_items = splitter.split(
    index.items,
    cfg.valid_ratio,
    cfg.seed,
    stratified=True
)
```

---

**Étape 4 : Augmenter le train set (optionnel)**
```python
if cfg.augment_train:
    aug_engine = AugmentationEngine.default_six()
    train_items = aug_engine.augment_dataset(
        train_items,
        out_dir / "augmented",
        augmentations_per_image=3
    )
```

**Résultat** : `train_items` contient maintenant les originales + les augmentées.

---

**Étape 5 : Extraire les features**
```python
feature_extractor = FeatureExtractor(
    TransformationEngine.default_six().tfs
)

print("Extracting train features...")
X_train, y_train = feature_extractor.extract_batch(train_items)

print("Extracting validation features...")
X_valid, y_valid = feature_extractor.extract_batch(valid_items)
```

**Résultat** :
- `X_train` : shape `(n_train, n_features)`
- `y_train` : shape `(n_train,)`
- `X_valid` : shape `(n_valid, n_features)`
- `y_valid` : shape `(n_valid,)`

---

**Étape 6 : Normaliser les features**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
```

**Important** : `fit_transform` sur train, `transform` sur valid (pas de data leakage).

---

**Étape 7 : Construire le modèle**
```python
model_factory = MLModelFactory()
model = model_factory.build(cfg, model_type="svm")
```

---

**Étape 8 : Entraîner**
```python
print("Training model...")
model.fit(X_train_scaled, y_train)
```

---

**Étape 9 : Évaluer**
```python
train_acc = model.score(X_train_scaled, y_train)
valid_acc = model.score(X_valid_scaled, y_valid)

metrics = Metrics(
    train_accuracy=train_acc,
    valid_accuracy=valid_acc,
    valid_count=len(valid_items)
)
```

---

**Étape 10 : Sauvegarder le bundle**
```python
bundle = MLModelBundle(
    model=model,
    scaler=scaler,
    labels=labels,
    feature_extractor=feature_extractor,
    cfg=ModelConfig(num_classes=index.num_classes, seed=cfg.seed)
)
bundle.save(out_dir / "model")
```

---

**Étape 11 : Retourner les métriques**
```python
return metrics
```

---

### **Classe : RequirementsGate**

**Identique à la version CNN**, pas de changement.

---

### **Classe : TrainingPackager**

**Identique à la version CNN**, pas de changement.

---

<a id="predict-pipeline"></a>
## 11. leaffliction/predict_pipeline.py — Pipeline de prédiction

### Changements par rapport à CNN

**Pipeline ML Traditionnel** :
```
1. Charger le bundle (model.pkl, scaler.pkl, labels.json)
2. Extraire features de l'image
3. Normaliser features
4. Prédire avec le modèle ML
5. Décoder le label
6. (Optionnel) Afficher transformations
```

---

### **Classe : MLPredictor**

**Responsabilité** : Charger le modèle et prédire sur une image.

---

#### **Méthode : predict(bundle_zip, image_path, cfg)**

**Ce qu'elle doit faire** :

**Paramètres** :
- `bundle_zip` : Path vers learnings.zip
- `image_path` : Path vers l'image à prédire
- `cfg` : PredictConfig

**Retour** :
- `predicted_label` : str (nom de la classe)
- `probs` : Dict[str, float] (probabilités par classe)

---

**Étape 1 : Charger le bundle**
```python
import tempfile

with tempfile.Temporary
