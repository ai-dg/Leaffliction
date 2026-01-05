# 🍃 Leaffliction — Guide Complet (ML Traditionnel)

> **Objectif de ce document**  
> Ce guide est un **manuel personnel de développement** pour le projet **Leaffliction**.  
> Il explique l'approche **Machine Learning traditionnelle** avec extraction de features,  
> les **formules mathématiques**, et la **défendabilité à l'oral**.

---

## 📑 Table des matières

1. [Vue d'ensemble du projet](#vue-densemble-du-projet)
2. [Architecture globale](#architecture-globale)
3. [ML Traditionnel vs Deep Learning](#ml-traditionnel-vs-deep-learning)
4. [Partie 1 : Analyse du Dataset](#partie-1--analyse-du-dataset)
5. [Partie 2 : Augmentation de données](#partie-2--augmentation-de-données)
6. [Partie 3 : Transformations et Features](#partie-3--transformations-et-features)
7. [Partie 4 : Classification ML](#partie-4--classification-ml)
8. [Module leaffliction/](#module-leaffliction)
9. [Pipeline ML Traditionnel détaillé](#pipeline-ml-traditionnel-détaillé)
10. [Mathématiques et formules](#mathématiques-et-formules)
11. [Contraintes du sujet](#contraintes-du-sujet)
12. [Génération de signature.txt](#génération-de-signaturetxt)
13. [Checklist finale](#checklist-finale)
14. [Conseils pour la soutenance](#conseils-pour-la-soutenance)

---

<a id="vue-densemble-du-projet"></a>
## 1. Vue d'ensemble du projet

**Leaffliction** est un projet de **computer vision** visant à classifier des maladies de feuilles à partir d'images en utilisant une approche **Machine Learning traditionnelle**.

### Objectifs principaux

1. **Analyser** la distribution des données
2. **Augmenter** les données (images physiques sur disque)
3. **Extraire** des features numériques des images
4. **Entraîner** un modèle ML (SVM, Random Forest, KNN)
5. **Prédire** la maladie d'une feuille

### Technologies utilisées

- **scikit-learn** : modèles ML (SVM, Random Forest, KNN)
- **OpenCV** : manipulation d'images et extraction de features
- **NumPy** : calculs numériques
- **Python 3.x** : langage principal
- **Matplotlib** : visualisation

---

<a id="architecture-globale"></a>
## 2. Architecture globale

```
Leaffliction/
│
├── Distribution.py          # Partie 1: Analyse distribution
├── Augmentation.py          # Partie 2: Visualisation augmentations
├── Transformation.py        # Partie 3: Visualisation transformations
├── train.py                 # Partie 4: Entraînement modèle ML
├── predict.py               # Partie 4: Prédiction
├── signature.txt            # Hash SHA1 du learnings.zip
├── README.md
│
└── leaffliction/            # Package Python
    ├── cli.py               # ✅ Parsers argparse
    ├── utils.py             # ✅ PathManager, Hasher, ZipPackager
    ├── plotting.py          # ✅ Visualisations
    ├── dataset.py           # Scanner, Splitter
    ├── augmentations.py     # Augmentations (images physiques)
    ├── transformations.py   # Transformations + FeatureExtractor ⭐
    ├── model.py             # MLModelFactory, MLModelBundle
    ├── train_pipeline.py    # MLTrainer
    └── predict_pipeline.py  # MLPredictor
```

### Principe de séparation

**Scripts racine** : Parsing + Instanciation + Appel
**Package leaffliction/** : Toute la logique métier

---

<a id="ml-traditionnel-vs-deep-learning"></a>
## 3. ML Traditionnel vs Deep Learning

### Comparaison

| Aspect | Deep Learning (CNN) | ML Traditionnel |
|--------|-------------------|-----------------|
| **Modèle** | Réseau de neurones | SVM, Random Forest, KNN |
| **Features** | Apprises automatiquement | Extraites manuellement |
| **Données** | Beaucoup (milliers) | Moins (centaines) |
| **Training** | Lent (GPU, heures) | Rapide (CPU, minutes) |
| **Interprétabilité** | Faible | Élevée |
| **Complexité** | Haute | Moyenne |

### Pipeline Visuel

**Deep Learning** :
```
Image → CNN → Prédiction
```

**ML Traditionnel** :
```
Image → Extraction Features → Modèle ML → Prédiction
       (Histogrammes, textures, contours)
```

### Pourquoi ML Traditionnel ?

**Avantages** :
- ✅ Plus simple à comprendre et expliquer
- ✅ Plus rapide à entraîner (minutes vs heures)
- ✅ Pas besoin de GPU
- ✅ Features interprétables (on sait ce qu'on mesure)
- ✅ Bon pour la soutenance (facile à justifier)

**Inconvénients** :
- ⚠️ Accuracy potentiellement plus faible que CNN
- ⚠️ Nécessite une bonne extraction de features
- ⚠️ Moins flexible pour des images très complexes

---

<a id="partie-1--analyse-du-dataset"></a>
## 4. Partie 1 : Analyse du Dataset

### Objectif

Analyser la distribution des classes dans le dataset pour détecter les déséquilibres.

### Utilisation

```bash
python Distribution.py ./leaves/images/
```

### Implémentation

```python
# Distribution.py
from pathlib import Path
from leaffliction.cli import CLIBuilder
from leaffliction.dataset import DatasetScanner
from leaffliction.plotting import DistributionPlotter

def main() -> None:
    parser = CLIBuilder().build_distribution_parser()
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    
    # Scanner le dataset
    scanner = DatasetScanner()
    index = scanner.scan(dataset_dir)
    
    # Afficher les graphiques
    title = f"Dataset distribution: {index.root.name}"
    plotter = DistributionPlotter()
    plotter.plot_pie(index.counts, title=title)
    plotter.plot_bar(index.counts, title=title)
```

### Sortie attendue

- **Pie chart** : Proportions de chaque classe
- **Bar chart** : Nombre d'images par classe

### Pourquoi c'est important

- Détection du déséquilibre de classes
- Justification des augmentations
- Compréhension du dataset

---

<a id="partie-2--augmentation-de-données"></a>
## 5. Partie 2 : Augmentation de données

### Objectif

Créer des **images physiques augmentées** sur disque pour équilibrer le dataset.

### Différence avec Deep Learning

**Deep Learning** : Augmentations à la volée pendant le training (dans le pipeline)
**ML Traditionnel** : Augmentations créent des fichiers AVANT le training

### Les 6 Augmentations

| Augmentation | Description | Paramètre |
|-------------|-------------|-----------|
| **FlipH** | Miroir horizontal | - |
| **FlipV** | Miroir vertical | - |
| **Rotate** | Rotation | angle=15° |
| **Brightness** | Luminosité | factor=20 |
| **Blur** | Flou gaussien | sigma=1.5 |
| **Crop** | Recadrage + resize | ratio=0.85 |

### Utilisation (Visualisation)

```bash
python Augmentation.py "./leaves/images/Apple_healthy/image (1).JPG"
```

**Sortie** :
- Affichage grille (original + 6 augmentations)
- Sauvegarde 6 fichiers avec suffixes

### Utilisation (Training)

Dans `train.py`, les augmentations sont appliquées automatiquement :

```python
# Augmenter le train set
if cfg.augment_train:
    aug_engine = AugmentationEngine.default_six()
    train_items = aug_engine.augment_dataset(
        train_items,
        out_dir / "augmented",
        augmentations_per_image=3  # 3 versions par image
    )
```

**Résultat** :
- 400 images originales → 400 + 1200 augmentées = 1600 images
- Toutes sauvegardées sur disque dans `augmented/`

---

<a id="partie-3--transformations-et-features"></a>
## 6. Partie 3 : Transformations et Features

### Rôle dans ML Traditionnel

**Transformations** = **Extraction de Features**

Les transformations ne sont plus juste pour la visualisation, elles sont **essentielles** pour extraire des caractéristiques numériques.

### Les 6 Transformations

| Transformation | Description | Features extraites |
|---------------|-------------|-------------------|
| **Grayscale** | Niveaux de gris | Histogramme, stats |
| **Canny** | Détection contours | Nombre, densité |
| **HistEq** | Égalisation histogramme | Contraste amélioré |
| **Sharpen** | Accentuation | Détails renforcés |
| **Threshold** | Seuillage binaire | Segmentation |
| **Morphology** | Érosion/dilatation | Formes nettoyées |

### Extraction de Features

**FeatureExtractor** extrait ~800-1000 features numériques par image :

1. **Histogrammes RGB** : 256 bins × 3 channels = 768 features
2. **Statistiques RGB** : mean, std, min, max × 3 = 12 features
3. **Stats des transformations** : 4 stats × 6 = 24 features
4. **Textures** (optionnel) : Haralick = 13 features
5. **Formes** (optionnel) : Moments de Hu = 7 features

**Total** : ~800-1000 features par image

### Exemple de Features

```
Image: Apple_healthy/image1.jpg

Features extraites:
[
  # Histogramme R
  0.012, 0.015, 0.018, ..., 0.003,  # 256 valeurs
  
  # Histogramme G
  0.010, 0.013, 0.020, ..., 0.005,  # 256 valeurs
  
  # Histogramme B
  0.008, 0.011, 0.016, ..., 0.004,  # 256 valeurs
  
  # Stats RGB
  120.5, 45.2, 0, 255,  # R: mean, std, min, max
  115.3, 42.1, 0, 255,  # G: mean, std, min, max
  110.8, 40.5, 0, 255,  # B: mean, std, min, max
  
  # Stats Grayscale
  115.2, 43.5, 0, 255,
  
  # Stats Canny
  0.15, 0.08, 0, 1,
  
  # ... autres transformations
]

→ Vecteur de 824 features
```

### Utilisation (Visualisation)

```bash
python Transformation.py "./leaves/images/Apple_healthy/image (1).JPG"
```

### Utilisation (Training)

```python
# Extraire features
feature_extractor = FeatureExtractor(
    TransformationEngine.default_six().tfs
)

X_train, y_train = feature_extractor.extract_batch(train_items)
# X_train shape: (n_images, 824)
# y_train shape: (n_images,)
```

---

<a id="partie-4--classification-ml"></a>
## 7. Partie 4 : Classification ML

### Pipeline Complet

```
1. Scanner dataset
2. Split train/valid (80/20, stratifié)
3. Augmenter train set (images physiques)
4. Extraire features (train + valid)
5. Normaliser features (StandardScaler)
6. Entraîner modèle ML (SVM/Random Forest/KNN)
7. Évaluer (accuracy > 90%)
8. Sauvegarder (model.pkl, scaler.pkl, labels.json)
9. Zipper (learnings.zip)
```

### Modèles Disponibles

#### **SVM (Support Vector Machine)**
```python
from sklearn.svm import SVC

model = SVC(
    kernel='rbf',        # Radial Basis Function
    C=1.0,               # Régularisation
    gamma='scale',
    probability=True,    # Pour avoir des probabilités
    random_state=42
)
```

**Avantages** : Performant, robuste
**Inconvénients** : Lent sur gros datasets

#### **Random Forest**
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,    # 100 arbres
    max_depth=None,
    random_state=42,
    n_jobs=-1            # Tous les CPU
)
```

**Avantages** : Rapide, robuste, interprétable
**Inconvénients** : Peut overfitter

#### **KNN (K-Nearest Neighbors)**
```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(
    n_neighbors=5,
    weights='distance',
    n_jobs=-1
)
```

**Avantages** : Simple, pas de training
**Inconvénients** : Lent en prédiction, sensible au bruit

### Training

```bash
python train.py ./leaves/images/ --epochs 10 --model_type svm
```

**Sortie** :
```
Scanning dataset...
Found 8 classes, 3424 images

Splitting dataset...
Train: 2739 images
Valid: 685 images

Augmenting train set...
Created 8217 augmented images

Extracting features...
Train features: (11956, 824)
Valid features: (685, 824)

Normalizing features...
StandardScaler fitted

Training SVM...
Training completed in 45.2s

Evaluating...
Train accuracy: 98.5%
Valid accuracy: 92.3% ✅
Valid count: 685 ✅

Saving model...
Model saved to artifacts/model/

Creating learnings.zip...
✅ Training completed!
```

### Prédiction

```bash
python predict.py learnings.zip "./leaves/images/Apple_Black_rot/image (1).JPG"
```

**Sortie** :
```
Loading model...
Extracting features...
Predicting...

Predicted class: Apple_Black_rot
Confidence: 95.7%

Top 3 predictions:
1. Apple_Black_rot: 95.7%
2. Apple_scab: 3.2%
3. Grape_Black_rot: 1.1%
```

---

<a id="module-leaffliction"></a>
## 8. Module leaffliction/

### Structure

```
leaffliction/
├── cli.py                   # ✅ Parsers argparse
├── utils.py                 # ✅ PathManager, Hasher, ZipPackager
├── plotting.py              # ✅ DistributionPlotter, GridPlotter
├── dataset.py               # DatasetScanner, DatasetSplitter
├── augmentations.py         # AugmentationEngine (images physiques)
├── transformations.py       # TransformationEngine + FeatureExtractor ⭐
├── model.py                 # MLModelFactory, MLModelBundle
├── train_pipeline.py        # MLTrainer
└── predict_pipeline.py      # MLPredictor
```

### Fichiers Clés

#### **transformations.py** ⭐ **CRUCIAL**

**FeatureExtractor** : Classe centrale pour ML traditionnel

```python
class FeatureExtractor:
    def extract_features(self, img_path: Path) -> np.ndarray:
        """
        Extrait ~800-1000 features numériques depuis une image
        
        Returns:
            np.ndarray de shape (n_features,)
        """
        # 1. Charger image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        features = []
        
        # 2. Histogrammes RGB
        for channel in range(3):
            hist, _ = np.histogram(img[:,:,channel], bins=256)
            hist = hist / hist.sum()
            features.extend(hist)
        
        # 3. Statistiques RGB
        for channel in range(3):
            features.append(img[:,:,channel].mean())
            features.append(img[:,:,channel].std())
            features.append(img[:,:,channel].min())
            features.append(img[:,:,channel].max())
        
        # 4. Appliquer transformations et extraire stats
        for tf in self.transformations:
            transformed = tf.apply(img)
            features.append(transformed.mean())
            features.append(transformed.std())
            features.append(transformed.min())
            features.append(transformed.max())
        
        return np.array(features, dtype=np.float32)
```

#### **model.py**

**MLModelFactory** : Construit des modèles sklearn

```python
class MLModelFactory:
    def build(self, cfg: ModelConfig, model_type: str = "svm"):
        if model_type == "svm":
            return SVC(kernel='rbf', C=1.0, probability=True)
        elif model_type == "random_forest":
            return RandomForestClassifier(n_estimators=100)
        elif model_type == "knn":
            return KNeighborsClassifier(n_neighbors=5)
```

**MLModelBundle** : Sauvegarde/charge le modèle

```python
class MLModelBundle:
    def save(self, out_dir: Path):
        """
        Sauvegarde:
        - model.pkl (modèle sklearn)
        - scaler.pkl (StandardScaler)
        - labels.json
        - config.json
        """
        joblib.dump(self.model, out_dir / "model.pkl")
        joblib.dump(self.scaler, out_dir / "scaler.pkl")
        # ... labels et config en JSON
```

#### **train_pipeline.py**

**MLTrainer** : Orchestrateur complet

```python
class MLTrainer:
    def train(self, dataset_dir, out_dir, cfg) -> Metrics:
        # 1. Scanner
        index = self.dataset_scanner.scan(dataset_dir)
        
        # 2. Split
        train_items, valid_items = self.dataset_splitter.split(...)
        
        # 3. Augmenter (optionnel)
        if cfg.augment_train:
            train_items = aug_engine.augment_dataset(...)
        
        # 4. Extraire features
        X_train, y_train = feature_extractor.extract_batch(train_items)
        X_valid, y_valid = feature_extractor.extract_batch(valid_items)
        
        # 5. Normaliser
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_valid_scaled = scaler.transform(X_valid)
        
        # 6. Entraîner
        model = self.model_factory.build(cfg, model_type="svm")
        model.fit(X_train_scaled, y_train)
        
        # 7. Évaluer
        valid_acc = model.score(X_valid_scaled, y_valid)
        
        # 8. Sauvegarder
        bundle = MLModelBundle(model, scaler, labels, ...)
        bundle.save(out_dir / "model")
        
        return Metrics(...)
```

---

<a id="pipeline-ml-traditionnel-détaillé"></a>
## 9. Pipeline ML Traditionnel détaillé

### Schéma Complet

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING (train.py)                      │
└─────────────────────────────────────────────────────────────┘

1. Dataset brut (leaves/images/)
   ↓
2. DatasetScanner.scan()
   → DatasetIndex (class_names, items, counts)
   ↓
3. DatasetSplitter.split() (stratifié)
   → train_items (80%), valid_items (20%)
   ↓
4. AugmentationEngine.augment_dataset() [OPTIONNEL]
   → Crée images physiques sur disque
   → train_items étendu (originales + augmentées)
   ↓
5. FeatureExtractor.extract_batch(train_items)
   → X_train (n_train, 824), y_train (n_train,)
   ↓
6. FeatureExtractor.extract_batch(valid_items)
   → X_valid (n_valid, 824), y_valid (n_valid,)
   ↓
7. StandardScaler
   → fit_transform(X_train) → X_train_scaled
   → transform(X_valid) → X_valid_scaled
   ↓
8. MLModelFactory.build(model_type="svm")
   → model sklearn
   ↓
9. model.fit(X_train_scaled, y_train)
   → Entraînement
   ↓
10. model.score(X_valid_scaled, y_valid)
    → Accuracy validation > 90% ✅
    ↓
11. MLModelBundle.save()
    → model.pkl, scaler.pkl, labels.json
    ↓
12. TrainingPackager.build_zip()
    → learnings.zip


┌─────────────────────────────────────────────────────────────┐
│                  PRÉDICTION (predict.py)                    │
└─────────────────────────────────────────────────────────────┘

1. Image test
   ↓
2. MLModelBundle.load_from_zip(learnings.zip)
   → model, scaler, labels, feature_extractor
   ↓
3. FeatureExtractor.extract_features(image_path)
   → features (824,)
   ↓
4. scaler.transform(features)
   → features_scaled (824,)
   ↓
5. model.predict(features_scaled)
   → class_id
   ↓
6. model.predict_proba(features_scaled)
   → probabilités
   ↓
7. LabelEncoder.decode(class_id)
   → nom de la classe
   ↓
8. Affichage résultat
```

---

<a id="mathématiques-et-formules"></a>
## 10. Mathématiques et formules

### 🔹 StandardScaler (Normalisation)

**Formule** :
```
x_scaled = (x - mean) / std
```

**Pourquoi** : Met toutes les features sur la même échelle (mean=0, std=1)

**Exemple** :
```
Feature 1: [100, 200, 300] → mean=200, std=81.6
Feature 2: [0.1, 0.2, 0.3] → mean=0.2, std=0.08

Après normalisation:
Feature 1: [-1.22, 0, 1.22]
Feature 2: [-1.22, 0, 1.22]

→ Même échelle !
```

### 🔹 SVM (Support Vector Machine)

**Objectif** : Trouver l'hyperplan qui sépare au mieux les classes

**Formule du kernel RBF** :
```
K(x, x') = exp(-γ ||x - x'||²)
```

Où :
- γ = gamma (contrôle la "portée" du kernel)
- ||x - x'|| = distance euclidienne

**Décision** :
```
f(x) = sign(Σ αᵢ yᵢ K(xᵢ, x) + b)
```

### 🔹 Random Forest

**Principe** : Ensemble d'arbres de décision

**Prédiction** :
```
ŷ = mode{tree₁(x), tree₂(x), ..., tree_n(x)}
```

**Probabilité** :
```
P(classe_k | x) = (nombre d'arbres prédisant k) / n_arbres
```

### 🔹 KNN (K-Nearest Neighbors)

**Principe** : Voter parmi les K voisins les plus proches

**Distance euclidienne** :
```
d(x, x') = √(Σᵢ (xᵢ - x'ᵢ)²)
```

**Prédiction** :
```
ŷ = mode{y₁, y₂, ..., y_k}
```

Où y₁, ..., y_k sont les labels des K voisins les plus proches.

### 🔹 Accuracy

```
Accuracy = (Nombre de prédictions correctes) / (Nombre total)
```

**Contrainte du sujet** : Accuracy > 90%

### 🔹 Histogramme

**Formule** :
```
hist[i] = nombre de pixels avec valeur dans [i, i+1)
hist_normalized[i] = hist[i] / Σ hist[j]
```

**Exemple** :
```
Image 100×100 = 10000 pixels
Valeurs entre 0-255

hist[120] = 150  → 150 pixels ont valeur ~120
hist_normalized[120] = 150/10000 = 0.015 = 1.5%
```

---

<a id="contraintes-du-sujet"></a>
## 11. Contraintes du sujet

### ✅ Contraintes Obligatoires

| Contrainte | Valeur | Vérification |
|-----------|--------|--------------|
| **Validation accuracy** | > 90% | `RequirementsGate.assert_ok()` |
| **Validation set size** | ≥ 100 images | `metrics.valid_count >= 100` |
| **Augmentations** | 6 types | `AugmentationEngine.default_six()` |
| **Transformations** | 6 types | `TransformationEngine.default_six()` |
| **Dataset dans repo** | ❌ INTERDIT | Seulement `signature.txt` |

### 📦 Structure du learnings.zip

```
learnings.zip/
├── model.pkl              # Modèle sklearn (SVM/RF/KNN)
├── scaler.pkl             # StandardScaler
├── labels.json            # {"Apple_Black_rot": 0, ...}
├── config.json            # {"num_classes": 8, ...}
└── feature_config.json    # Config des features
```

---

<a id="génération-de-signaturetxt"></a>
## 12. Génération de signature.txt

### Commandes

```bash
# Linux/macOS
sha1sum learnings.zip > signature.txt

# Windows
certUtil -hashfile learnings.zip sha1 > signature.txt
```

### Automatisation

```python
from leaffliction.utils import Hasher

hasher = Hasher()
sha1_hash = hasher.ft_sha1_file(Path("learnings.zip"))

with open("signature.txt", "w") as f:
    f.write(sha1_hash + "\n")
```

### ⚠️ IMPORTANT

- ❌ Ne JAMAIS modifier `learnings.zip` après avoir généré `signature.txt`
- ❌ Ne JAMAIS commit `learnings.zip` dans git
- ✅ Seulement commit `signature.txt`

---

<a id="checklist-finale"></a>
## 13. Checklist finale

### 📋 Code

- [ ] `Distribution.py` fonctionne
- [ ] `Augmentation.py` affiche et sauvegarde 6 augmentations
- [ ] `Transformation.py` affiche 6 transformations
- [ ] `train.py` entraîne le modèle ML
- [ ] `predict.py` prédit correctement
- [ ] Séparation logique/entrypoint respectée

### 📊 Dataset

- [ ] Dataset équilibré (ou augmenté)
- [ ] Split stratifié
- [ ] Validation ≥ 100 images
- [ ] Pas de data leakage

### 🎓 Modèle

- [ ] Accuracy validation > 90%
- [ ] Modèle reproductible (seed fixé)
- [ ] Features bien extraites
- [ ] Normalisation correcte

### 📦 Packaging

- [ ] `learnings.zip` contient tout
- [ ] `signature.txt` correct
- [ ] SHA1 vérifié
- [ ] Pas de fichiers inutiles

---

<a id="conseils-pour-la-soutenance"></a>
## 14. Conseils pour la soutenance

### 🎯 Points Forts de l'Approche ML

**À mettre en avant** :
1. **Simplicité** : "J'ai choisi ML traditionnel car plus simple à comprendre et expliquer"
2. **Rapidité** : "Training en 2 minutes vs 2 heures pour CNN"
3. **Interprétabilité** : "Je peux montrer exactement quelles features sont importantes"
4. **Efficacité** : "Pas besoin de GPU, fonctionne sur n'importe quel ordinateur"

### 📊 Démonstration

**Script de démo** :
```bash
# 1. Distribution
python Distribution.py ./leaves/images/
# → Montrer le déséquilibre

# 2. Augmentation
python Augmentation.py "./leaves/images/Apple_healthy/image (1).JPG"
# → Montrer les 6 augmentations

# 3. Transformation
python Transformation.py "./leaves/images/Apple_healthy/image (1).JPG"
# → Montrer les 6 transformations

# 4. Training
python train.py ./leaves/images/ --model_type svm
# → Montrer les logs, accuracy > 90%

# 5. Prediction
python predict.py learnings.zip "./test_image.jpg"
# → Montrer la prédiction
```

### 🗣️ Questions Probables

**Q: Pourquoi ML traditionnel et pas CNN ?**
R: "ML traditionnel est plus
