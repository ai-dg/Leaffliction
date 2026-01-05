# 🧠 Leaffliction — Guide personnel complet (ML Traditionnel)

> **Objectif de ce document**  
> Ce guide est un **manuel personnel de développement** pour le projet **Leaffliction**.  
> Il explique l'approche **Machine Learning traditionnelle** avec extraction de features,  
> les **formules mathématiques**, et la **défendabilité à l'oral**.

---

## 📑 Table des matières

1. [Vue d'ensemble du projet](#vue-densemble-du-projet)
2. [Architecture globale](#architecture-globale)
3. [ML Traditionnel vs Deep Learning](#ml-traditionnel-vs-deep-learning)
4. [Pipeline de données](#pipeline-de-données)
5. [Scripts racine (entrypoints)](#scripts-racine-entrypoints)
6. [Dossier `leaffliction/` (cœur du projet)](#dossier-leaffliction-cœur-du-projet)
7. [Mathématiques et formules essentielles](#mathématiques-et-formules-essentielles)
8. [Contraintes du sujet & validation](#contraintes-du-sujet--validation)
9. [Checklist finale avant rendu](#checklist-finale-avant-rendu)
10. [Conseils pour la soutenance](#conseils-pour-la-soutenance)

---

## Vue d'ensemble du projet

**Leaffliction** est un projet de **computer vision** visant à classifier des maladies de feuilles en utilisant une approche **Machine Learning traditionnelle** (SVM, Random Forest, KNN).

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

### Pipeline général

```
(1) Dataset brut
     ↓
(2) Analyse distribution (Distribution.py)
     ↓
(3) Augmentations visuelles (Augmentation.py) - pour comprendre
     ↓
(4) Transformations visuelles (Transformation.py) - pour comprendre
     ↓
(5) Training (train.py)
     ├─ Split train/valid
     ├─ Augmentations (images physiques)
     ├─ Extraction features
     ├─ Normalisation (StandardScaler)
     ├─ Modèle ML (SVM/RF/KNN)
     └─ Sauvegarde learnings.zip
     ↓
(6) Prédiction (predict.py)
     ├─ Charge learnings.zip
     ├─ Extrait features de l'image
     ├─ Prédit la classe
     └─ (Optionnel) Affiche transformations
```

---

## Architecture globale

```
.
├── Distribution.py          # Partie 1: Analyse distribution
├── Augmentation.py          # Partie 2: Visualisation augmentations
├── Transformation.py        # Partie 3: Visualisation transformations
├── train.py                 # Partie 4: Entraînement ML
├── predict.py               # Partie 4: Prédiction ML
├── signature.txt            # SHA1 du learnings.zip
│
├── docs/                    # Documentation
│   ├── GUIDE.md                              # Guide complet ML
│   ├── GUIDE_IMPLEMENTATION.md               # Guide d'implémentation
│   ├── ETAT_PROJET.md                        # État du projet
│   ├── ARCHITECTURE_ML_TRADITIONNELLE.md     # Architecture ML
│   └── en.subject.pdf                        # Sujet original
│
└── leaffliction/            # Package principal
    ├── cli.py              # ✅ Finalisé - Parsers argparse
    ├── utils.py            # ✅ Finalisé - Utilitaires (paths, zip, hash)
    ├── plotting.py         # ✅ Finalisé - Visualisations
    ├── dataset.py          # 🔄 En cours - Scanner, Splitter
    ├── augmentations.py    # 📝 Squelette - Augmentations (NumPy/OpenCV)
    ├── transformations.py  # 📝 Squelette - Transformations + FeatureExtractor ⭐
    ├── model.py            # 📝 Squelette - MLModelFactory, MLModelBundle
    ├── train_pipeline.py   # 📝 Squelette - MLTrainer
    └── predict_pipeline.py # 📝 Squelette - MLPredictor
```

---

## ML Traditionnel vs Deep Learning

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

## Pipeline de données

### Schéma détaillé

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING (train.py)                      │
└─────────────────────────────────────────────────────────────┘

1. DatasetScanner.scan()
   └─ Lit leaves/images/
      └─ Retourne DatasetIndex (class_names, items, counts)

2. DatasetSplitter.split()
   └─ Split stratifié train/valid (80/20)
      └─ Retourne (train_items, valid_items)

3. AugmentationEngine.augment_dataset() [OPTIONNEL]
   └─ Crée des images augmentées PHYSIQUES
      └─ Sauvegarde dans artifacts/augmented/
      └─ Retourne liste étendue (originales + augmentées)

4. FeatureExtractor.extract_batch(train_items)
   └─ Extrait features de toutes les images train
      └─ Retourne X_train (n_train, 824), y_train (n_train,)

5. FeatureExtractor.extract_batch(valid_items)
   └─ Extrait features de toutes les images valid
      └─ Retourne X_valid (n_valid, 824), y_valid (n_valid,)

6. StandardScaler
   └─ fit_transform(X_train) → X_train_scaled
   └─ transform(X_valid) → X_valid_scaled

7. MLModelFactory.build(model_type="svm")
   └─ Construit modèle sklearn (SVM/RF/KNN)

8. model.fit(X_train_scaled, y_train)
   └─ Entraînement

9. model.score(X_valid_scaled, y_valid)
   └─ Accuracy validation > 90% ✅

10. MLModelBundle.save()
    └─ Sauvegarde model.pkl + scaler.pkl + labels.json

11. TrainingPackager.build_zip()
    └─ Crée learnings.zip


┌─────────────────────────────────────────────────────────────┐
│                  PRÉDICTION (predict.py)                    │
└─────────────────────────────────────────────────────────────┘

1. MLModelBundle.load_from_zip()
   └─ Extrait et charge model.pkl, scaler.pkl, labels.json

2. FeatureExtractor.extract_features(image_path)
   └─ Extrait features de l'image (824,)

3. scaler.transform(features)
   └─ Normalise les features

4. model.predict(features_scaled)
   └─ Prédiction

5. model.predict_proba(features_scaled)
   └─ Probabilités

6. LabelEncoder.decode(class_id)
   └─ Convertit ID → nom classe

7. (Optionnel) TransformationEngine.apply_all()
   └─ Visualisation transformations
```

---

## Scripts racine (entrypoints)

### Règle fondamentale

> **Aucune logique métier dans les scripts racine.**  
> Ils doivent uniquement :
> 1. Parser les arguments (via `cli.py`)
> 2. Instancier des classes (depuis `leaffliction/`)
> 3. Appeler des méthodes

---

### `Distribution.py` ✅

**But** : Analyser la répartition des données (Partie 1 du sujet)

**Pipeline** :
```python
1. Parser arguments (dataset_dir)
2. DatasetScanner.scan(dataset_dir)
3. DistributionPlotter.plot_pie() + plot_bar()
```

**Utilisation** :
```bash
python Distribution.py ./leaves/images/
```

**Pourquoi c'est important ?**
- Détection de déséquilibre de classes
- Justification des augmentations
- Compréhension du dataset

---

### `Augmentation.py` ✅

**But** : Visualiser les 6 augmentations sur UNE image (Partie 2 du sujet)

**Pipeline** :
```python
1. Parser arguments (image_path)
2. Charger l'image avec OpenCV
3. Convertir BGR → RGB
4. AugmentationEngine.default_six()
5. engine.apply_all(img)
6. GridPlotter.show_grid()
7. AugmentationSaver.save_all()
```

**Utilisation** :
```bash
python Augmentation.py "./leaves/images/Apple_healthy/image (1).JPG"
```

**Résultat** : 6 images sauvegardées avec suffixes
- `image (1)_FlipH.JPG`
- `image (1)_FlipV.JPG`
- `image (1)_Rotate.JPG`
- `image (1)_BrightContrast.JPG`
- `image (1)_Blur.JPG`
- `image (1)_CropResize.JPG`

---

### `Transformation.py` ✅

**But** : Visualiser les 6 transformations sur UNE image (Partie 3 du sujet)

**Pipeline** :
```python
1. Parser arguments (image_path ou -src/-dst)
2. TransformationEngine.default_six()
3. Mode single: 
   - Charger image avec OpenCV
   - engine.apply_all() 
   - GridPlotter.show_grid()
4. Mode batch: 
   - BatchTransformer.run()
```

**Utilisation** :
```bash
# Mode single
python Transformation.py "./leaves/images/Apple_healthy/image (1).JPG"

# Mode batch
python Transformation.py -src ./leaves/images/ -dst ./transformed/
```

**Transformations** :
- Grayscale
- Canny (contours)
- Histogram Equalisation
- Sharpen
- Threshold
- Morphology

---

### `train.py` ✅

**But** : Entraîner le modèle ML (Partie 4 du sujet)

**Pipeline** :
```python
1. Parser arguments
2. Scanner dataset
3. Split train/valid (stratifié)
4. (Optionnel) Augmenter train set (images physiques)
5. Extraire features (train + valid)
6. Normaliser features (StandardScaler)
7. Construire modèle ML (SVM/RF/KNN)
8. Entraîner
9. Évaluer (accuracy > 90%)
10. Sauvegarder bundle (model.pkl, scaler.pkl, labels.json)
11. Créer learnings.zip
12. Générer signature.txt
```

**Utilisation** :
```bash
python train.py ./leaves/images/ --model_type svm --augment
```

**Options** :
- `--model_type` : svm, random_forest, knn (défaut: svm)
- `--augment` : Activer augmentations (défaut: True)
- `--aug_per_image` : Nombre d'augmentations par image (défaut: 3)
- `--valid_ratio` : Ratio de validation (défaut: 0.2)
- `--seed` : Seed pour reproductibilité (défaut: 42)

**Contraintes** :
- ✅ Valid accuracy > 90%
- ✅ Valid count ≥ 100 images
- ✅ Sauvegarder tout dans learnings.zip

---

### `predict.py` ✅

**But** : Prédire la classe d'une image (Partie 4 du sujet)

**Pipeline** :
```python
1. Parser arguments (bundle_zip, image_path)
2. Charger MLModelBundle
3. Extraire features de l'image
4. Normaliser features
5. Prédire avec le modèle ML
6. Afficher résultat + top K prédictions
7. (Optionnel) Afficher transformations
```

**Utilisation** :
```bash
python predict.py learnings.zip "./leaves/images/Apple_Black_rot/image (1).JPG"
```

**Sortie** :
```
🔍 Predicting disease for: image (1).JPG
📦 Using model from: learnings.zip

============================================================
✅ PREDICTION RESULT
============================================================
🍃 Predicted class: Apple_Black_rot
📊 Confidence: 95.7%

Top 3 predictions:
   1. Apple_Black_rot: 95.7%
   2. Apple_scab: 3.2%
   3. Grape_Black_rot: 1.1%

============================================================
```

---

## Dossier `leaffliction/` (cœur du projet)

### `cli.py` ✅ Finalisé

**Responsabilités** :
- Centralisation de tous les parsers `argparse`
- `build_distribution_parser()`
- `build_augmentation_parser()`
- `build_transformation_parser()`
- `build_train_parser()`
- `build_predict_parser()`

---

### `utils.py` ✅ Finalisé

**Responsabilités** :
- `PathManager`: gestion chemins, itération images
- `ZipPackager`: création/extraction zip
- `Hasher`: calcul SHA1

---

### `plotting.py` ✅ Finalisé

**Responsabilités** :
- `DistributionPlotter`: pie chart + bar chart
- `GridPlotter`: grilles d'images (augmentations/transformations)

---

### `dataset.py` 🔄 En cours

**Classes** :

#### `DatasetIndex`
```python
@dataclass
class DatasetIndex:
    root: Path
    class_names: List[str]  # Triés alphabétiquement
    items: List[Tuple[Path, int]]  # (chemin, class_id)
    counts: Dict[str, int]  # {class_name: count}
```

#### `DatasetScanner`
```python
def scan(self, root: Path) -> DatasetIndex:
    """
    Scan récursif du dossier:
    root/
      Apple_healthy/
        image1.jpg
      Apple_scab/
        image2.jpg
    
    Retourne: DatasetIndex
    """
```

#### `DatasetSplitter` ✅ Implémenté
```python
def split(
    self,
    items: List[Tuple[Path, int]],
    valid_ratio: float,
    seed: int,
    stratified: bool = True
) -> Tuple[List, List]:
    """
    Split stratifié pour conserver proportions de classes.
    """
```

---

### `augmentations.py` 📝 Squelette

**Différence clé avec CNN** :
- CNN : Augmentations à la volée pendant training
- ML : Augmentations créent des images PHYSIQUES sur disque

**Classes d'augmentation** (NumPy/OpenCV) :
- `FlipHorizontalAug` : `cv2.flip(img, 1)`
- `FlipVerticalAug` : `cv2.flip(img, 0)`
- `RotateAug` : `cv2.warpAffine()`
- `BrightnessContrastAug` : Ajustement pixel values
- `GaussianBlurAug` : `cv2.GaussianBlur()`
- `RandomCropResizeAug` : Crop + `cv2.resize()`

**Moteur** :
```python
class AugmentationEngine:
    def default_six(cls) -> "AugmentationEngine":
        """Factory des 6 augmentations"""
    
    def apply_all(self, img) -> Dict[str, np.ndarray]:
        """Pour visualisation (Augmentation.py)"""
    
    def apply_random(self, img, n=2) -> np.ndarray:
        """Applique n augmentations aléatoires"""
    
    def augment_dataset(self, train_items, output_dir, n=3):
        """
        Crée images augmentées PHYSIQUES (pour training)
        
        Input:  400 images
        Output: 400 + 1200 = 1600 images (sauvegardées sur disque)
        """
```

---

### `transformations.py` 📝 Squelette ⭐

**Rôle dans ML Traditionnel** :
- Transformations = Extraction de Features
- Essentielles pour le modèle ML

**Classes de transformation** (NumPy/OpenCV) :
- `GrayscaleTf` : `cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)`
- `CannyEdgesTf` : `cv2.Canny()`
- `HistogramEqualisationTf` : `cv2.equalizeHist()`
- `SharpenTf` : Convolution avec kernel
- `ThresholdTf` : Seuillage binaire
- `MorphologyTf` : `cv2.erode()`, `cv2.dilate()`

**Moteur** :
```python
class TransformationEngine:
    def default_six(cls) -> "TransformationEngine":
        """Factory des 6 transformations"""
    
    def apply_all(self, img) -> Dict[str, np.ndarray]:
        """Applique toutes les transformations"""
```

**FeatureExtractor** ⭐ **CLASSE CENTRALE** :
```python
class FeatureExtractor:
    def extract_features(self, img_path: Path) -> np.ndarray:
        """
        Extrait ~800-1000 features numériques depuis une image
        
        Features extraites:
        1. Histogrammes RGB : 256 bins × 3 = 768 features
        2. Statistiques RGB : mean, std, min, max × 3 = 12 features
        3. Stats des transformations : 4 stats × 6 = 24 features
        4. (Optionnel) Textures Haralick : 13 features
        5. (Optionnel) Moments de Hu : 7 features
        
        Returns:
            np.ndarray de shape (n_features,)  # ~824 features
        """
    
    def extract_batch(self, items) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extrait features de plusieurs images
        
        Returns:
            X: np.ndarray de shape (n_samples, n_features)
            y: np.ndarray de shape (n_samples,)
        """
```

---

### `model.py` 📝 Squelette

**Classes** :

#### `LabelEncoder`
```python
def fit(self, class_names: List[str]):
    """Crée mapping class_name ↔ id"""

def encode(self, class_name: str) -> int:
    """class_name → id"""

def decode(self, class_id: int) -> str:
    """id → class_name"""

def to_json_dict() -> Dict:
    """Pour sauvegarde"""

@classmethod
def from_json_dict(cls, data: Dict) -> "LabelEncoder":
    """Pour chargement"""
```

#### `MLModelFactory`
```python
def build(self, cfg: ModelConfig, model_type: str = "svm"):
    """
    Construit un modèle sklearn:
    
    - "svm": SVC(kernel='rbf', C=1.0, probability=True)
    - "random_forest": RandomForestClassifier(n_estimators=100)
    - "knn": KNeighborsClassifier(n_neighbors=5)
    """
```

#### `MLModelBundle`
```python
def save(self, out_dir: Path):
    """
    Sauvegarde:
    - model.pkl (modèle sklearn)
    - scaler.pkl (StandardScaler)
    - labels.json
    - config.json
    - feature_config.json
    """

@classmethod
def load_from_zip(cls, zip_path: Path):
    """Charge depuis learnings.zip"""

def predict(self, features: np.ndarray):
    """
    Prédit la classe
    
    Returns:
        pred_id: int
        probs: Dict[str, float]
    """
```

---

### `train_pipeline.py` 📝 Squelette

**Classes** :

#### `MLTrainer`
```python
def train(self, dataset_dir, out_dir, cfg) -> Metrics:
    """
    Pipeline complet ML:
    1. Scanner dataset
    2. Split train/valid (stratifié)
    3. (Optionnel) Augmenter train set (images physiques)
    4. Extraire features (train + valid)
    5. Normaliser features (StandardScaler)
    6. Entraîner modèle ML (SVM/RF/KNN)
    7. Évaluer (accuracy > 90%)
    8. Sauvegarder bundle
    """
```

#### `RequirementsGate`
```python
def assert_ok(self, metrics: Metrics):
    """
    Vérifie:
    - valid_accuracy > 0.90
    - valid_count >= 100
    
    Lève ValueError si non conforme
    """
```

#### `TrainingPackager`
```python
def build_zip(self, artifacts_dir, out_zip):
    """Crée learnings.zip"""
```

---

### `predict_pipeline.py` 📝 Squelette

**Classes** :

#### `MLPredictor`
```python
def predict(self, bundle_zip, image_path, cfg):
    """
    1. Charge bundle (model.pkl, scaler.pkl)
    2. Extrait features de l'image
    3. Normalise features
    4. Prédit avec modèle ML
    5. Retourne (label, probs, transformed)
    """
```

#### `PredictionVisualiser`
```python
def show(self, original, transformed, predicted_label):
    """Affiche grille avec résultat"""
```

---

## Mathématiques et formules essentielles

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

---

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

---

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

---

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

---

### 🔹 Accuracy

```
Accuracy = (Nombre de prédictions correctes) / (Nombre total)
```

**Contrainte du sujet** : Accuracy > 90%

---

### 🔹 Histogramme

**Formule** :
```
hist[i] = nombre de pixels avec valeur dans [i, i+1)
hist_normalized[i] = hist[i] / Σ hist[j]
```

---

## Contraintes du sujet & validation

### ✅ Contraintes obligatoires

1. **Accuracy validation > 90%**
   - Vérifiée par `RequirementsGate`
   - Minimum 100 images de validation

2. **Dataset interdit dans le repo**
   - Seulement `signature.txt` (SHA1 du zip)
   - Vérification pendant la défense

3. **Structure du zip**
   ```
   learnings.zip/
     model.pkl              # Modèle sklearn
     scaler.pkl             # StandardScaler
     labels.json            # Mapping classes
     config.json            # Configuration
     feature_config.json    # Config features
   ```

4. **Signature SHA1**
   ```bash
   sha1sum learnings.zip > signature.txt
   ```

---

## Checklist finale avant rendu

### Code

- [ ] Tous les scripts racine fonctionnels
- [ ] Code modulaire et propre
- [ ] Pas de code mort
- [ ] Imports corrects
- [ ] Type hints présents

### Dataset

- [ ] Dataset **NON** dans le repo
- [ ] `signature.txt` présent
- [ ] SHA1 correct

### Training

- [ ] Accuracy > 90% sur validation
- [ ] ≥ 100 images de validation
- [ ] `learnings.zip` reproductible

### Documentation

- [ ] README.md à jour
- [ ] GUIDE.md complet
- [ ] Commentaires dans le code

---

## Conseils pour la soutenance

### 1. Expliquer l'architecture ML

**Soyez capable de dessiner au tableau** :
```
Dataset → Scanner → Split → Augment (images physiques)
                                ↓
                        Extract Features
                                ↓
                        Normalize (StandardScaler)
                                ↓
                        Train ML Model (SVM/RF/KNN)
                                ↓
                        Validation (accuracy > 90%)
```

### 2. Justifier le choix ML traditionnel

**Points forts** :
- ✅ Plus simple à comprendre et expliquer
- ✅ Plus rapide à entraîner (minutes vs heures)
- ✅ Pas besoin de GPU
- ✅ Features interprétables
- ✅ Bon pour la soutenance

**Quand utiliser CNN** :
- Dataset très large (>10k images)
- Images très complexes
- Besoin d'accuracy maximale

### 3. Expliquer l'extraction de features

**Être capable d'expliquer** :
- Histogrammes RGB : Distribution des couleurs
- Statistiques : Caractéristiques globales
- Transformations : Contours, textures, formes
- Total : ~800-1000 features par image

### 4. Maîtriser les formules

**Être capable d'expliquer** :
- StandardScaler : Normalisation des features
- SVM : Séparation par hyperplan
- Random Forest : Vote d'arbres
- KNN : Vote des voisins

### 5. Démontrer la reproductibilité

**Montrer** :
```bash
# 1. Training
python train.py leaves/images/ --model_type svm

# 2. Vérification SHA1
sha1sum learnings.zip
cat signature.txt

# 3. Prédiction
python predict.py learnings.zip test_image.jpg
```

### 6. Anticiper les questions

**Questions fréquentes** :
- "Pourquoi ML traditionnel et pas CNN ?"
  → Plus simple, plus rapide, features interprétables
  
- "Comment extrayez-vous les features ?"
  → Histogrammes, stats, transformations (détailler)
  
- "Quelle est votre accuracy finale ?"
  → Montrer les résultats (>90%)
  
- "Combien de temps pour entraîner ?"
  → Quelques minutes (vs heures pour CNN)
  
- "Pourquoi SVM plutôt que Random Forest ?"
  →
