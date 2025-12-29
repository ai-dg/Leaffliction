# 🍃 Leaffliction — Guide Complet de Développement

> **Objectif de ce document**  
> Ce guide est un **manuel personnel de développement** pour le projet **Leaffliction**.  
> Il explique **quoi faire, pourquoi le faire, et comment le faire**, fichier par fichier,  
> avec un accent fort sur les **formules mathématiques**, la **logique ML**, et la **défendabilité à l'oral**.

---

## 📑 Table des matières

1. [Vue d'ensemble du projet](#vue-densemble-du-projet)
2. [Architecture globale](#architecture-globale)
3. [Partie 1 : Analyse du Dataset (Distribution.py)](#partie-1--analyse-du-dataset-distributionpy)
4. [Partie 2 : Augmentation de données (Augmentation.py)](#partie-2--augmentation-de-données-augmentationpy)
5. [Partie 3 : Transformations d'images (Transformation.py)](#partie-3--transformations-dimages-transformationpy)
6. [Partie 4 : Classification (train.py & predict.py)](#partie-4--classification-trainpy--predictpy)
7. [Module leaffliction/ — Cœur du projet](#module-leaffliction--cœur-du-projet)
8. [Pipeline Machine Learning détaillé](#pipeline-machine-learning-détaillé)
9. [Mathématiques et formules essentielles](#mathématiques-et-formules-essentielles)
10. [Contraintes du sujet & validation](#contraintes-du-sujet--validation)
11. [Génération de signature.txt](#génération-de-signaturetxt)
12. [Checklist finale avant rendu](#checklist-finale-avant-rendu)
13. [Conseils pour la soutenance](#conseils-pour-la-soutenance)

---

<a id="vue-densemble-du-projet"></a>
## 1. Vue d'ensemble du projet

---

## 1. Vue d'ensemble du projet

**Leaffliction** est un projet de **computer vision** visant à classifier des maladies de feuilles à partir d'images.

### Objectifs principaux

1. **Analyser** la distribution des données (déséquilibre de classes)
2. **Augmenter** les données pour équilibrer le dataset
3. **Transformer** les images pour extraire des caractéristiques
4. **Entraîner** un modèle de classification (CNN)
5. **Prédire** la maladie d'une feuille à partir d'une image

### Technologies utilisées

- **TensorFlow/Keras** : framework de deep learning
- **Python 3.x** : langage principal
- **OpenCV/PIL** : manipulation d'images
- **Matplotlib** : visualisation
- **NumPy** : calculs numériques

---

<a id="architecture-globale"></a>
## 2. Architecture globale

```
Leaffliction/
│
├── Distribution.py          # Entrypoint : analyse distribution
├── Augmentation.py          # Entrypoint : visualisation augmentations
├── Transformation.py        # Entrypoint : transformations d'images
├── train.py                 # Entrypoint : entraînement du modèle
├── predict.py               # Entrypoint : prédiction sur une image
├── signature.txt            # Hash SHA1 du dataset + modèle
├── README.md                # Documentation utilisateur
│
└── leaffliction/            # Package Python (logique métier)
    ├── __init__.py
    ├── cli.py               # Parsers argparse centralisés
    ├── utils.py             # Utilitaires (paths, hash, zip)
    ├── dataset.py           # Scan, split, tf.data.Dataset
    ├── plotting.py          # Visualisations (pie, bar, grids)
    ├── augmentations.py     # Augmentations (Flip, Rotate, etc.)
    ├── transformations.py   # Transformations (Canny, Threshold, etc.)
    ├── model.py             # Architecture CNN, LabelEncoder, ModelBundle
    ├── train_pipeline.py    # Orchestration training
    └── predict_pipeline.py  # Orchestration prédiction
```

### Principe de séparation

**Scripts racine** (Distribution.py, train.py, etc.) :
- ✅ Parsing des arguments
- ✅ Instanciation des classes
- ✅ Appel des méthodes
- ❌ **AUCUNE logique métier**

**Package leaffliction/** :
- ✅ Toute la logique métier
- ✅ Classes réutilisables
- ✅ Testable unitairement

---

<a id="partie-1--analyse-du-dataset-distributionpy"></a>
## 3. Partie 1 : Analyse du Dataset (Distribution.py)

### 📋 Objectif du sujet

> "Write a program named Distribution.[extension] that takes as arguments a directory and fetches images in its subdirectories. This program must extract and analyze/understand the data set from the images and prompt pie charts and bar charts for each plant type."

### 🎯 Ce que fait Distribution.py

1. **Scanner** le dossier dataset (ex: `./leaves/images/`)
2. **Compter** le nombre d'images par classe (sous-dossier)
3. **Afficher** un **pie chart** (camembert)
4. **Afficher** un **bar chart** (histogramme)

### 📊 Exemple d'utilisation

```bash
python Distribution.py ./leaves/images/
```

**Sortie attendue** :
- Pie chart montrant la proportion de chaque classe
- Bar chart montrant le nombre d'images par classe

### 🔧 Implémentation

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
    
    # 1. Scanner le dataset
    scanner = DatasetScanner()
    index = scanner.ft_scan(dataset_dir)
    
    # 2. Titre du graphique
    title = f"Dataset distribution: {index.root.name}"
    
    # 3. Afficher les graphiques
    plotter = DistributionPlotter()
    plotter.plot_pie(index.counts, title=title)
    plotter.plot_bar(index.counts, title=title)

if __name__ == "__main__":
    main()
```

### 🧠 Logique métier (leaffliction/dataset.py)

**DatasetScanner.ft_scan()** :
```python
def ft_scan(self, root: Path) -> DatasetIndex:
    """
    Structure attendue:
    root/
      Apple_Black_rot/
        image (1).JPG
        image (2).JPG
      Apple_healthy/
        image (1).JPG
    
    Retourne:
      DatasetIndex(
        root=root,
        class_names=['Apple_Black_rot', 'Apple_healthy'],
        items=[(Path, class_id), ...],
        counts={'Apple_Black_rot': 252, 'Apple_healthy': 150}
      )
    """
    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    class_names = [d.name for d in class_dirs]
    
    items = []
    counts = {}
    
    for class_id, class_dir in enumerate(class_dirs):
        class_name = class_dir.name
        images = list(class_dir.glob("*.JPG")) + list(class_dir.glob("*.jpg"))
        counts[class_name] = len(images)
        
        for img_path in images:
            items.append((img_path, class_id))
    
    return DatasetIndex(
        root=root,
        class_names=class_names,
        items=items,
        counts=counts
    )
```

### 📈 Visualisation (leaffliction/plotting.py)

**DistributionPlotter** :
```python
import matplotlib.pyplot as plt

class DistributionPlotter:
    def plot_pie(self, counts: Dict[str, int], title: str, save_to: Optional[Path] = None):
        labels = list(counts.keys())
        sizes = list(counts.values())
        
        plt.figure(figsize=(10, 8))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.title(title)
        plt.axis('equal')
        
        if save_to:
            plt.savefig(save_to)
        plt.show()
    
    def plot_bar(self, counts: Dict[str, int], title: str, save_to: Optional[Path] = None):
        labels = list(counts.keys())
        values = list(counts.values())
        
        plt.figure(figsize=(12, 6))
        plt.bar(labels, values, color='skyblue')
        plt.xlabel('Classes')
        plt.ylabel('Number of images')
        plt.title(title)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_to:
            plt.savefig(save_to)
        plt.show()
```

### 💡 Pourquoi c'est important ?

- **Détection du déséquilibre** : si une classe a 500 images et une autre 50, le modèle sera biaisé
- **Justification des augmentations** : on augmente les classes minoritaires
- **Compréhension du dataset** : première étape de tout projet ML

---

<a id="partie-2--augmentation-de-données-augmentationpy"></a>
## 4. Partie 2 : Augmentation de données (Augmentation.py)

### 📋 Objectif du sujet

> "You must display 6 types of data augmentation for each image given to your program, they must also be saved with the original file name followed by the name of the type of augmentation."

### 🎯 Ce que fait Augmentation.py

1. **Charger** une image
2. **Appliquer** 6 augmentations différentes
3. **Afficher** une grille (original + 6 variantes)
4. **Sauvegarder** les 6 images avec suffixes

### 📊 Exemple d'utilisation

```bash
python Augmentation.py "./leaves/images/Apple_healthy/image (1).JPG"
```

**Sortie attendue** :
- Affichage d'une grille 3x3 (original + 6 augmentations)
- Sauvegarde de 6 fichiers :
  - `image (1)_Flip.JPG`
  - `image (1)_Rotate.JPG`
  - `image (1)_Brightness.JPG`
  - `image (1)_Blur.JPG`
  - `image (1)_Crop.JPG`
  - `image (1)_Contrast.JPG`

### 🔧 Les 6 augmentations obligatoires

| Augmentation | Description | Paramètres typiques |
|-------------|-------------|---------------------|
| **Flip** | Miroir horizontal/vertical | axis='horizontal' |
| **Rotate** | Rotation | angle=15° |
| **Brightness** | Luminosité | factor=1.3 |
| **Blur** | Flou gaussien | sigma=2.0 |
| **Crop** | Recadrage + resize | crop_ratio=0.8 |
| **Contrast** | Contraste | factor=1.5 |

### 🔧 Implémentation

```python
# Augmentation.py
from pathlib import Path
from leaffliction.cli import CLIBuilder
from leaffliction.utils import PathManager
from leaffliction.augmentations import AugmentationEngine, AugmentationSaver
from leaffliction.plotting import GridPlotter
import tensorflow as tf

def main() -> None:
    parser = CLIBuilder().build_augmentation_parser()
    args = parser.parse_args()
    
    image_path = Path(args.image_path)
    
    # 1. Charger l'image
    img = tf.io.read_file(str(image_path))
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32) / 255.0  # Normalisation [0, 1]
    
    # 2. Appliquer les 6 augmentations
    engine = AugmentationEngine.default_six()
    results = engine.apply_all(img)
    
    # 3. Afficher la grille
    grid = GridPlotter()
    grid.show_grid("Augmentations", results, original=img)
    
    # 4. Sauvegarder les images
    pm = PathManager()
    saver = AugmentationSaver(pm)
    saver.save_all(image_path, results)

if __name__ == "__main__":
    main()
```

### 🧠 Logique métier (leaffliction/augmentations.py)

**AugmentationEngine** :
```python
class AugmentationEngine:
    def __init__(self, augs: List[Augmentation]):
        self.augs = augs
    
    @classmethod
    def default_six(cls) -> "AugmentationEngine":
        """Factory pour les 6 augmentations obligatoires"""
        return cls([
            FlipHorizontalAug(),
            RotateAug(angle=15.0),
            BrightnessContrastAug(brightness=0.3, contrast=0.0),
            GaussianBlurAug(sigma=2.0),
            RandomCropResizeAug(crop_ratio=0.8),
            BrightnessContrastAug(brightness=0.0, contrast=0.5),
        ])
    
    def apply_all(self, img: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Applique toutes les augmentations"""
        results = {}
        for aug in self.augs:
            results[aug.name] = aug.apply(img)
        return results
```

**Exemple d'augmentation : FlipHorizontalAug** :
```python
@dataclass
class FlipHorizontalAug:
    name: str = "Flip"
    
    def apply(self, img: tf.Tensor) -> tf.Tensor:
        return tf.image.flip_left_right(img)
```

**Exemple d'augmentation : RotateAug** :
```python
import math

@dataclass
class RotateAug:
    angle: float
    name: str = "Rotate"
    
    def apply(self, img: tf.Tensor) -> tf.Tensor:
        # Conversion degrés -> radians
        angle_rad = self.angle * math.pi / 180.0
        
        # Rotation avec interpolation bilinéaire
        import tensorflow_addons as tfa
        return tfa.image.rotate(img, angle_rad, interpolation='bilinear')
```

### 📐 Formule mathématique : Rotation

Pour une rotation d'angle θ autour du centre de l'image :

```
x' = (x - cx) * cos(θ) - (y - cy) * sin(θ) + cx
y' = (x - cx) * sin(θ) + (y - cy) * sin(θ) + cy
```

Où (cx, cy) est le centre de l'image.

### 💾 Sauvegarde (AugmentationSaver)

```python
class AugmentationSaver:
    def __init__(self, path_manager: PathManager):
        self.path_manager = path_manager
    
    def save_all(self, image_path: Path, results: Dict[str, tf.Tensor]) -> List[Path]:
        """
        Sauvegarde avec suffixes :
        image (1).JPG -> image (1)_Flip.JPG
        """
        saved_paths = []
        
        for aug_name, img_tensor in results.items():
            # Générer le chemin avec suffixe
            out_path = self.path_manager.ft_make_suffixed_path(image_path, aug_name)
            
            # Convertir tensor -> uint8 [0, 255]
            img_uint8 = tf.cast(img_tensor * 255.0, tf.uint8)
            
            # Encoder en JPEG
            encoded = tf.image.encode_jpeg(img_uint8, quality=95)
            
            # Écrire sur disque
            tf.io.write_file(str(out_path), encoded)
            
            saved_paths.append(out_path)
        
        return saved_paths
```

### 🛠️ PathManager.ft_make_suffixed_path()

```python
def ft_make_suffixed_path(self, image_path: Path, suffix: str) -> Path:
    """
    Exemple:
    input:  /a/b/image (1).JPG  + suffix="Flip"
    output: /a/b/image (1)_Flip.JPG
    """
    stem = image_path.stem  # "image (1)"
    ext = image_path.suffix  # ".JPG"
    parent = image_path.parent
    
    new_name = f"{stem}_{suffix}{ext}"
    return parent / new_name
```

### 💡 Différence Augmentation vs Transformation

| Aspect | Augmentation | Transformation |
|--------|-------------|----------------|
| **But** | Augmenter la taille du dataset | Extraire des caractéristiques |
| **Utilisation** | Training (data augmentation) | Analyse / visualisation |
| **Nature** | Aléatoire (stochastique) | Déterministe |
| **Préserve la classe** | ✅ Oui | ❌ Non (change la représentation) |
| **Exemples** | Flip, Rotate, Brightness | Canny, Grayscale, Threshold |

---

<a id="partie-3--transformations-dimages-transformationpy"></a>
## 5. Partie 3 : Transformations d'images (Transformation.py)

### 📋 Objectif du sujet

> "Different methods of direct extraction of characteristics from an image of a leaf need to be implemented. You must display at least 6 image transformations."

### 🎯 Ce que fait Transformation.py

1. **Mode single** : affiche les transformations d'une image
2. **Mode batch** : transforme toutes les images d'un dossier

### 📊 Exemple d'utilisation

**Mode single** :
```bash
python Transformation.py "./leaves/images/Apple_healthy/image (1).JPG"
```

**Mode batch** :
```bash
python Transformation.py -src ./leaves/images/Apple_healthy/ -dst ./transformed/ -mask
```

### 🔧 Les 6 transformations obligatoires

| Transformation | Description | Utilité |
|---------------|-------------|---------|
| **Grayscale** | Conversion en niveaux de gris | Réduction de dimensionnalité |
| **Canny** | Détection de contours | Extraction de formes |
| **Histogram Equalisation** | Égalisation d'histogramme | Amélioration du contraste |
| **Sharpen** | Accentuation | Renforcement des détails |
| **Threshold** | Seuillage binaire | Segmentation |
| **Morphology** | Opérations morphologiques | Nettoyage du bruit |

### 🔧 Implémentation

```python
# Transformation.py
from pathlib import Path
from leaffliction.cli import CLIBuilder
from leaffliction.utils import PathManager
from leaffliction.transformations import TransformationEngine, BatchTransformer
from leaffliction.plotting import GridPlotter
import tensorflow as tf

def main() -> None:
    parser = CLIBuilder().build_transformation_parser()
    args = parser.parse_args()
    
    engine = TransformationEngine.default_six()
    grid = GridPlotter()
    pm = PathManager()
    
    # Mode batch: -src / -dst
    if getattr(args, "src", None) and getattr(args, "dst", None):
        src = Path(args.src)
        dst = Path(args.dst)
        batch = BatchTransformer(engine=engine, path_manager=pm)
        batch.run(src=src, dst=dst, recursive=getattr(args, "recursive", True))
        return
    
    # Mode single image
    image_path = Path(args.image_path)
    
    # Charger l'image
    img = tf.io.read_file(str(image_path))
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32) / 255.0
    
    # Appliquer les transformations
    results = engine.apply_all(img)
    
    # Afficher la grille
    grid.show_grid("Transformations", results, original=img)

if __name__ == "__main__":
    main()
```

### 🧠 Logique métier (leaffliction/transformations.py)

**TransformationEngine** :
```python
class TransformationEngine:
    def __init__(self, tfs: List[Transformation]):
        self.tfs = tfs
    
    @classmethod
    def default_six(cls) -> "TransformationEngine":
        return cls([
            GrayscaleTf(),
            CannyEdgesTf(),
            HistogramEqualisationTf(),
            SharpenTf(),
            ThresholdTf(),
            MorphologyTf(mode='erode'),
        ])
    
    def apply_all(self, img: tf.Tensor) -> Dict[str, tf.Tensor]:
        results = {}
        for tf_obj in self.tfs:
            results[tf_obj.name] = tf_obj.apply(img)
        return results
```

### 📐 Formules mathématiques des transformations

#### 1. Grayscale (conversion RGB → Gray)

```
Gray = 0.299 * R + 0.587 * G + 0.114 * B
```

Implémentation :
```python
@dataclass
class GrayscaleTf:
    name: str = "Grayscale"
    
    def apply(self, img: tf.Tensor) -> tf.Tensor:
        return tf.image.rgb_to_grayscale(img)
```

#### 2. Canny Edge Detection

**Étapes** :
1. Flou gaussien (réduction du bruit)
2. Calcul du gradient (Sobel)
3. Suppression des non-maxima
4. Seuillage par hystérésis

**Gradient de Sobel** :
```
Gx = [[-1, 0, 1],      Gy = [[-1, -2, -1],
      [-2, 0, 2],            [ 0,  0,  0],
      [-1, 0, 1]]            [ 1,  2,  1]]

Magnitude = √(Gx² + Gy²)
Direction = arctan(Gy / Gx)
```

Implémentation (avec OpenCV via tf.py_function) :
```python
import cv2

@dataclass
class CannyEdgesTf:
    name: str = "Canny"
    low_threshold: int = 50
    high_threshold: int = 150
    
    def apply(self, img: tf.Tensor) -> tf.Tensor:
        def _canny(img_np):
            # Conversion en uint8
            img_uint8 = (img_np * 255).astype(np.uint8)
            
            # Grayscale si RGB
            if len(img_uint8.shape) == 3:
                gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_uint8
            
            # Canny
            edges = cv2.Canny(gray, self.low_threshold, self.high_threshold)
            
            # Retour en float32 [0, 1]
            return edges.astype(np.float32) / 255.0
        
        result = tf.py_function(_canny, [img], tf.float32)
        result.set_shape(img.shape[:2] + (1,))  # (H, W, 1)
        return result
```

#### 3. Histogram Equalisation

**Formule** :
```
h(i) = nombre de pixels de valeur i
cdf(i) = Σ(j=0 to i) h(j)  (fonction de répartition cumulative)

new_value(i) = round((cdf(i) - cdf_min) / (total_pixels - cdf_min) * (L - 1))
```

Où L = 256 (niveaux de gris).

Implémentation :
```python
@dataclass
class HistogramEqualisationTf:
    name: str = "HistEq"
    
    def apply(self, img: tf.Tensor) -> tf.Tensor:
        def _hist_eq(img_np):
            img_uint8 = (img_np * 255).astype(np.uint8)
            
            if len(img_uint8.shape) == 3:
                gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_uint8
            
            equalized = cv2.equalizeHist(gray)
            return equalized.astype(np.float32) / 255.0
        
        result = tf.py_function(_hist_eq, [img], tf.float32)
        result.set_shape(img.shape[:2] + (1,))
        return result
```

#### 4. Sharpen (accentuation)

**Noyau de convolution** :
```
Kernel = [[ 0, -1,  0],
          [-1,  5, -1],
          [ 0, -1,  0]]
```

Implémentation :
```python
@dataclass
class SharpenTf:
    name: str = "Sharpen"
    
    def apply(self, img: tf.Tensor) -> tf.Tensor:
        kernel = tf.constant([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=tf.float32)
        
        kernel = tf.reshape(kernel, [3, 3, 1, 1])
        
        # Convolution sur chaque canal
        if len(img.shape) == 3 and img.shape[-1] == 3:
            channels = []
            for i in range(3):
                channel = img[:, :, i:i+1]
                channel = tf.expand_dims(channel, 0)  # (1, H, W, 1)
                sharpened = tf.nn.conv2d(channel, kernel, strides=1, padding='SAME')
                channels.append(tf.squeeze(sharpened, 0))
            return tf.concat(channels, axis=-1)
        else:
            img_4d = tf.expand_dims(img, 0)
            sharpened = tf.nn.conv2d(img_4d, kernel, strides=1, padding='SAME')
            return tf.squeeze(sharpened, 0)
```

#### 5. Threshold (seuillage binaire)

**Formule** :
```
output(x, y) = {
    1   si input(x, y) > threshold
    0   sinon
}
```

Implémentation :
```python
@dataclass
class ThresholdTf:
    name: str = "Threshold"
    threshold: float = 0.5
    
    def apply(self, img: tf.Tensor) -> tf.Tensor:
        # Conversion en grayscale si RGB
        if len(img.shape) == 3 and img.shape[-1] == 3:
            gray = tf.image.rgb_to_grayscale(img)
        else:
            gray = img
        
        # Seuillage
        binary = tf.cast(gray > self.threshold, tf.float32)
        return binary
```

#### 6. Morphology (érosion/dilatation)

**Érosion** :
```
output(x, y) = min{input(x+i, y+j) | (i,j) ∈ structuring_element}
```

**Dilatation** :
```
output(x, y) = max{input(x+i, y+j) | (i,j) ∈ structuring_element}
```

Implémentation :
```python
@dataclass
class MorphologyTf:
    mode: str = "erode"  # ou "dilate", "open", "close"
    name: str = "Morphology"
    kernel_size: int = 5
    
    def apply(self, img: tf.Tensor) -> tf.Tensor:
        def _morph(img_np):
            img_uint8 = (img_np * 255).astype(np.uint8)
            
            if len(img_uint8.shape) == 3:
                gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_uint8
            
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, 
                (self.kernel_size, self.kernel_size)
            )
            
            if self.mode == "erode":
                result = cv2.erode(gray, kernel, iterations=1)
            elif self.mode == "dilate":
                result = cv2.dilate(gray, kernel, iterations=1)
            elif self.mode == "open":
                result = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
            elif self.mode == "close":
                result = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            else:
                result = gray
            
            return result.astype(np.float32) / 255.0
        
        result = tf.py_function(_morph, [img], tf.float32)
        result.set_shape(img.shape[:2] + (1,))
        return result
```

---

<a id="partie-4--classification-trainpy--predictpy"></a>
## 6. Partie 4 : Classification (train.py & predict.py)

### 📋 Objectif du sujet

> "You must write a program named train.[extension] that takes as parameter a directory and fetches images in its subdirectories. It must then increase/modify those images in order to learn the characteristics of the diseases. Those learnings must be saved and returned in a .zip."

> "Write a second program that retrieves your learnings. It must take as arguments a path to an image, displays the original image and the transformed image, and gives the type of disease."

### 🎯 Ce que fait train.py

1. **Scanner** le dataset
2. **Split** train/validation (80/20)
3. **Construire** tf.data.Dataset avec augmentations
4. **Entraîner** un modèle CNN
5. **Évaluer** sur validation (accuracy > 90%)
6. **Sauvegarder** le modèle + métadonnées
7. **Zipper** le tout

### 📊 Exemple d'utilisation

```bash
python train.py ./leaves/images/ --epochs 20 --batch_size 32 --lr 0.001
```

**Sortie attendue** :
- Logs d'entraînement (epoch, loss, accuracy)
- Fichier `learnings.zip` contenant :
  - `model.keras` (modèle entraîné)
  - `labels.json` (mapping classe ↔ id)
  - `config.json` (configuration)
  - `preprocess.json` (paramètres de prétraitement)

### 🔧 Implémentation train.py

```python
# train.py
from pathlib import Path
from leaffliction.cli import CLIBuilder
from leaffliction.dataset import DatasetScanner, DatasetSplitter, TFDatasetBuilder, TFDataConfig
from leaffliction.augmentations import KerasAugmentationsFactory
from leaffliction.model import ModelFactory, LabelEncoder
from leaffliction.train_pipeline import Trainer, TrainConfig, RequirementsGate, TrainingPackager
from leaffliction.utils import ZipPackager

def main() -> None:
    parser = CLIBuilder().build_train_parser()
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(getattr(args, "out_dir", "artifacts"))
    out_zip = Path(getattr(args, "out_zip", "learnings.zip"))
    
    cfg = TrainConfig(
        epochs=getattr(args, "epochs", 10),
        batch_size=getattr(args, "batch_size", 32),
        lr=getattr(args, "lr", 1e-3),
        valid_ratio=getattr(args, "valid_ratio", 0.2),
        seed=getattr(args, "seed", 42),
        img_size=(getattr(args, "img_h", 224), getattr(args, "img_w", 224)),
        augment_in_train=getattr(args, "augment", True),
        export_increased_images=getattr(args, "export_images", True),
    )
    
    # Instanciation des composants
    scanner = DatasetScanner()
    splitter = DatasetSplitter()
    labels = LabelEncoder()
    model_factory = ModelFactory()
    
    trainer = Trainer(
        dataset_scanner=scanner,
        dataset_splitter=splitter,
        model_factory=model_factory,
        labels=labels,
    )
    
    # Entraînement
    metrics = trainer.train(dataset_dir=dataset_dir, out_dir=out_dir, cfg=cfg)
    
    # Validation des contraintes (accuracy > 90%, validation_count >= 100)
    gate = RequirementsGate()
    gate.assert_ok(metrics)
    
    # Packaging final
    packager = TrainingPackager(zip_packager=ZipPackager())
    artifacts_dir = packager.prepare_artifacts_dir(tmp_dir=out_dir)
    packager.build_zip(artifacts_dir=artifacts_dir, out_zip=out_zip)
    
    print(f"✅ Training completed! Accuracy: {metrics.valid_accuracy:.2%}")
    print(f"📦 Model saved to: {out_zip}")

if __name__ == "__main__":
    main()
```

### 🎯 Ce que fait predict.py

1. **Charger** le modèle depuis le zip
2. **Prétraiter** l'image
3. **Prédire** la classe
4. **Afficher** le résultat + transformations (optionnel)

### 📊 Exemple d'utilisation

```bash
python predict.py learnings.zip "./leaves/images/Apple_Black_rot/image (1).JPG"
```

**Sortie attendue** :
```
Predicted class: Apple_Black_rot
Confidence: 98.5%
```

### 🔧 Implémentation predict.py

```python
# predict.py
from pathlib import Path
from leaffliction.cli import CLIBuilder
from leaffliction.predict_pipeline import Predictor, PredictConfig, PredictionVisualiser
from leaffliction.transformations import TransformationEngine
from leaffliction.model import ModelBundle

def main() -> None:
    parser = CLIBuilder().build_predict_parser()
    args = parser.parse_args()
    
    bundle_zip = Path(args.bundle_zip)
    image_path = Path(args.image_path)
    
    cfg = PredictConfig(
        show_transforms=getattr(args, "show_transforms", True),
        top_k=getattr(args, "top_k", 1),
    )
    
    engine = TransformationEngine.default_six()
    
    predictor = Predictor(bundle_loader=ModelBundle, transformations_engine=engine)
    
    label, probs = predictor.predict(bundle_zip=bundle_zip, image_path=image_path, cfg=cfg)
    
    # Affichage résultat
    print(f"Predicted class: {label}")
    print(f"Confidence: {max(probs.values()):.1%}")
    
    # Affichage visuel (optionnel)
    if cfg.show_transforms:
        vis = PredictionVisualiser()
        vis.show(original=None, transformed={}, predicted_label=label)

if __name__ == "__main__":
    main()
```

---

<a id="module-leaffliction--cœur-du-projet"></a>
## 7. Module leaffliction/ — Cœur du projet

### 📁 Structure détaillée

```
leaffliction/
├── __init__.py              # Package marker
├── cli.py                   # Parsers argparse centralisés
├── utils.py                 # PathManager, Hasher, ZipPackager
├── dataset.py               # DatasetScanner, DatasetSplitter, TFDatasetBuilder
├── plotting.py              # DistributionPlotter, GridPlotter
├── augmentations.py         # AugmentationEngine, 6 augmentations
├── transformations.py       # TransformationEngine, 6 transformations
├── model.py                 # ModelFactory, LabelEncoder, ModelBundle
├── train_pipeline.py        # Trainer, TrainingPackager, RequirementsGate
└── predict_pipeline.py      # Predictor, PredictionVisualiser
```

### 🔧 cli.py — Parsers centralisés

```python
# leaffliction/cli.py
import argparse

class CLIBuilder:
    def build_distribution_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Analyze dataset distribution")
        parser.add_argument("dataset_dir", type=str, help="Path to dataset directory")
        return parser
    
    def build_augmentation_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Apply augmentations to an image")
        parser.add_argument("image_path", type=str, help="Path to image")
        return parser
    
    def build_transformation_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Apply transformations to images")
        parser.add_argument("image_path", type=str, nargs="?", help="Path to single image")
        parser.add_argument("-src", type=str, help="Source directory (batch mode)")
        parser.add_argument("-dst", type=str, help="Destination directory (batch mode)")
        parser.add_argument("-mask", action="store_true", help="Apply mask transformations")
        parser.add_argument("-recursive", action="store_true", default=True)
        parser.add_argument("-h", "--help", action="help")
        return parser
    
    def build_train_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Train classification model")
        parser.add_argument("dataset_dir", type=str, help="Path to dataset")
        parser.add_argument("--epochs", type=int, default=10)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--valid_ratio", type=float, default=0.2)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--img_h", type=int, default=224)
        parser.add_argument("--img_w", type=int, default=224)
        parser.add_argument("--augment", action="store_true", default=True)
        parser.add_argument("--export_images", action="store_true", default=True)
        parser.add_argument("--out_dir", type=str, default="artifacts")
        parser.add_argument("--out_zip", type=str, default="learnings.zip")
        return parser
    
    def build_predict_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Predict disease from image")
        parser.add_argument("bundle_zip", type=str, help="Path to learnings.zip")
        parser.add_argument("image_path", type=str, help="Path to image")
        parser.add_argument("--show_transforms", action="store_true", default=True)
        parser.add_argument("--top_k", type=int, default=1)
        return parser
```


### 🔧 utils.py — Utilitaires

```python
# leaffliction/utils.py
from pathlib import Path
from typing import List
import hashlib
import zipfile

class PathManager:
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    
    def ft_ensure_dir(self, path: Path) -> Path:
        """Crée le dossier si absent"""
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def ft_make_suffixed_path(self, image_path: Path, suffix: str) -> Path:
        """
        Exemple: image (1).JPG + "Flip" -> image (1)_Flip.JPG
        """
        stem = image_path.stem
        ext = image_path.suffix
        parent = image_path.parent
        new_name = f"{stem}_{suffix}{ext}"
        return parent / new_name
    
    def ft_iter_images(self, root: Path, recursive: bool = False) -> List[Path]:
        """Liste les images dans root"""
        images = []
        pattern = "**/*" if recursive else "*"
        
        for ext in self.IMAGE_EXTS:
            images.extend(root.glob(f"{pattern}{ext}"))
            images.extend(root.glob(f"{pattern}{ext.upper()}"))
        
        return sorted(images)

class Hasher:
    def ft_sha1_file(self, path: Path, chunk_size: int = 1024 * 1024) -> str:
        """Calcule le SHA1 d'un fichier"""
        sha1 = hashlib.sha1()
        
        with open(path, 'rb') as f:
            while chunk := f.read(chunk_size):
                sha1.update(chunk)
        
        return sha1.hexdigest()

class ZipPackager:
    def ft_zip_dir(self, src_dir: Path, out_zip: Path) -> None:
        """Zip tout le contenu de src_dir"""
        with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in src_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(src_dir)
                    zipf.write(file_path, arcname)
```

---

<a id="pipeline-machine-learning-détaillé"></a>
## 8. Pipeline Machine Learning détaillé

### 🔄 Flux complet d'entraînement

```
1. Dataset Scanning
   └─> DatasetIndex (class_names, items, counts)

2. Train/Valid Split (stratified)
   └─> train_items, valid_items

3. TF Dataset Construction
   ├─> load_image()
   ├─> decode_jpeg()
   ├─> resize()
   ├─> normalize [0, 1]
   └─> augmentations (train only)

4. Model Building
   ├─> Backbone (MobileNetV2 / EfficientNet)
   ├─> GlobalAveragePooling2D
   ├─> Dense(num_classes, activation='softmax')
   └─> compile(optimizer='adam', loss='sparse_categorical_crossentropy')

5. Training
   ├─> model.fit(train_ds, validation_data=valid_ds)
   ├─> callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
   └─> history

6. Evaluation
   └─> valid_accuracy > 90% ✅

7. Saving
   ├─> model.keras
   ├─> labels.json
   ├─> config.json
   └─> preprocess.json

8. Packaging
   └─> learnings.zip
```

### 🏗️ Architecture du modèle

```python
# leaffliction/model.py
import tensorflow as tf
import keras
from keras import layers

class ModelFactory:
    def build(self, cfg: ModelConfig) -> keras.Model:
        """
        Architecture:
        Input (224, 224, 3)
          ↓
        MobileNetV2 (pretrained, frozen)
          ↓
        GlobalAveragePooling2D
          ↓
        Dropout(0.2)
          ↓
        Dense(num_classes, softmax)
        """
        # Backbone pré-entraîné
        backbone = keras.applications.MobileNetV2(
            input_shape=(*cfg.img_size, 3),
            include_top=False,
            weights='imagenet'
        )
        backbone.trainable = False  # Freeze pour transfer learning
        
        # Construction du modèle
        inputs = layers.Input(shape=(*cfg.img_size, 3))
        x = backbone(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(cfg.num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs)
        
        return model
```

### 📊 Construction du tf.data.Dataset

```python
# leaffliction/dataset.py
import tensorflow as tf

class TFDatasetBuilder:
    def __init__(self, cfg: TFDataConfig, augmentor=None):
        self.cfg = cfg
        self.augmentor = augmentor
    
    def build(self, items: List[Tuple[Path, int]], training: bool) -> tf.data.Dataset:
        """
        Pipeline:
        1. from_tensor_slices (paths, labels)
        2. map(load_and_preprocess)
        3. augmentations (si training)
        4. batch
        5. prefetch
        """
        paths = [str(p) for p, _ in items]
        labels = [label_id for _, label_id in items]
        
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        
        # Shuffle si training
        if training and self.cfg.shuffle:
            ds = ds.shuffle(buffer_size=len(items), seed=self.cfg.seed)
        
        # Load + preprocess
        ds = ds.map(
            lambda path, label: self._load_and_preprocess(path, label),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Augmentations (training only)
        if training and self.augmentor is not None:
            ds = ds.map(
                lambda img, label: (self.augmentor(img, training=True), label),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        
        # Batch
        ds = ds.batch(self.cfg.batch_size)
        
        # Cache (optionnel)
        if self.cfg.cache:
            ds = ds.cache()
        
        # Prefetch
        if self.cfg.prefetch:
            ds = ds.prefetch(tf.data.AUTOTUNE)
        
        return ds
    
    def _load_and_preprocess(self, path: tf.Tensor, label: tf.Tensor):
        """Charge et prétraite une image"""
        # Lecture
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        
        # Resize
        img = tf.image.resize(img, self.cfg.img_size)
        
        # Normalisation [0, 255] -> [0, 1]
        img = tf.cast(img, tf.float32) / 255.0
        
        return img, label
```

### 🎓 Entraînement avec callbacks

```python
# leaffliction/train_pipeline.py
import keras

class KerasCallbacksFactory:
    def build(self, out_dir: Path) -> List[keras.callbacks.Callback]:
        """Callbacks pour améliorer l'entraînement"""
        callbacks = [
            # Arrêt anticipé si pas d'amélioration
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Sauvegarde du meilleur modèle
            keras.callbacks.ModelCheckpoint(
                filepath=str(out_dir / "best_model.keras"),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            
            # Réduction du learning rate si plateau
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
        ]
        
        return callbacks
```

---

<a id="mathématiques-et-formules-essentielles"></a>
## 9. Mathématiques et formules essentielles

### 🔹 Softmax (couche de sortie)

**Formule** :
```
softmax(z_i) = exp(z_i) / Σ_j exp(z_j)
```

**Propriétés** :
- Σ softmax(z_i) = 1 (somme = 100%)
- softmax(z_i) ∈ [0, 1] (probabilités)

**Exemple** :
```
Logits: [2.0, 1.0, 0.1]

exp(2.0) = 7.39
exp(1.0) = 2.72
exp(0.1) = 1.11
Σ = 11.22

softmax = [7.39/11.22, 2.72/11.22, 1.11/11.22]
        = [0.659, 0.242, 0.099]
        = [65.9%, 24.2%, 9.9%]
```

### 🔹 Cross-Entropy Loss (fonction de coût)

**Formule** :
```
L = -Σ_i y_i * log(ŷ_i)
```

Où :
- y_i = vérité terrain (one-hot encoded)
- ŷ_i = prédiction (softmax)

**Exemple** :
```
Vérité: classe 0 → y = [1, 0, 0]
Prédiction: ŷ = [0.7, 0.2, 0.1]

L = -(1*log(0.7) + 0*log(0.2) + 0*log(0.1))
  = -log(0.7)
  = 0.357
```

**Sparse Categorical Cross-Entropy** :
```
L = -log(ŷ_true_class)
```

Plus efficace quand les labels sont des entiers (pas one-hot).

### 🔹 Gradient Descent (optimisation)

**Formule** :
```
θ_new = θ_old - α * ∂L/∂θ
```

Où :
- θ = paramètres du modèle (poids)
- α = learning rate
- ∂L/∂θ = gradient de la loss par rapport aux poids

**Adam Optimizer** (variante avancée) :
```
m_t = β1 * m_{t-1} + (1 - β1) * g_t        (momentum)
v_t = β2 * v_{t-1} + (1 - β2) * g_t²       (variance)

m̂_t = m_t / (1 - β1^t)                     (bias correction)
v̂_t = v_t / (1 - β2^t)

θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)
```

Paramètres typiques : β1=0.9, β2=0.999, ε=1e-7

### 🔹 Accuracy (métrique)

**Formule** :
```
Accuracy = (Nombre de prédictions correctes) / (Nombre total de prédictions)
```

**Exemple** :
```
100 images de validation
92 prédictions correctes
→ Accuracy = 92/100 = 0.92 = 92% ✅
```

### 🔹 Convolution 2D

**Formule** :
```
Output(i, j) = Σ_m Σ_n Input(i+m, j+n) * Kernel(m, n)
```

**Exemple avec kernel 3x3** :
```
Input:          Kernel:         Output:
[1 2 3]         [1 0 -1]        
[4 5 6]    *    [1 0 -1]    =   [résultat]
[7 8 9]         [1 0 -1]
```

### 🔹 Pooling (réduction de dimensionnalité)

**Max Pooling** :
```
Output(i, j) = max{Input(2i+m, 2j+n) | m,n ∈ {0,1}}
```

**Average Pooling** :
```
Output(i, j) = mean{Input(2i+m, 2j+n) | m,n ∈ {0,1}}
```

**Global Average Pooling** :
```
Output = mean(Input over spatial dimensions)
(H, W, C) → (C,)
```

---

<a id="contraintes-du-sujet--validation"></a>
## 10. Contraintes du sujet & validation

### ✅ Contraintes obligatoires

| Contrainte | Valeur | Vérification |
|-----------|--------|--------------|
| **Validation accuracy** | > 90% | `RequirementsGate.assert_ok()` |
| **Validation set size** | ≥ 100 images | `metrics.valid_count >= 100` |
| **Augmentations** | 6 types | `AugmentationEngine.default_six()` |
| **Transformations** | 6 types | `TransformationEngine.default_six()` |
| **Dataset dans repo** | ❌ INTERDIT | Seulement `signature.txt` |
| **Fichiers à rendre** | Scripts + `signature.txt` | Pas de `.zip` dans git |

### 🔒 RequirementsGate

```python
# leaffliction/train_pipeline.py
class RequirementsGate:
    def assert_ok(self, metrics: Metrics) -> None:
        """Valide les contraintes du sujet"""
        # Contrainte 1: accuracy > 90%
        if metrics.valid_accuracy < 0.90:
            raise ValueError(
                f"Validation accuracy {metrics.valid_accuracy:.2%} < 90%. "
                f"Training failed to meet requirements."
            )
        
        # Contrainte 2: validation set >= 100 images
        if metrics.valid_count < 100:
            raise ValueError(
                f"Validation set has {metrics.valid_count} images < 100. "
                f"Increase dataset size or reduce valid_ratio."
            )
        
        print(f"✅ Requirements met:")
        print(f"   - Validation accuracy: {metrics.valid_accuracy:.2%}")
        print(f"   - Validation set size: {metrics.valid_count}")
```

### 📦 Structure du learnings.zip

```
learnings.zip
├── model.keras              # Modèle Keras complet
├── labels.json              # {"Apple_Black_rot": 0, "Apple_healthy": 1, ...}
├── config.json              # {"img_size": [224, 224], "num_classes": 8, ...}
└── preprocess.json          # {"normalize": true, "mean": [0.485, ...], ...}
```

---

<a id="génération-de-signaturetxt"></a>
## 11. Génération de signature.txt

### 📋 Objectif du sujet

> "You must also deposit a signature.txt file at the root of your Git repository. This file must be a hash of your DB which includes everything you need to run your programs correctly, i.e. your dataset in sha1 format and the training of your model."

### 🔧 Génération du hash SHA1

```bash
# Linux
sha1sum learnings.zip

# macOS
shasum learnings.zip

# Windows
certUtil -hashfile learnings.zip sha1
```

**Exemple de sortie** :
```
7a18a838d2203cc7d6e8c4c521fdd4dd214aa560  learnings.zip
```

### 📝 Création de signature.txt

```bash
echo "7a18a838d2203cc7d6e8c4c521fdd4dd214aa560" > signature.txt
```

### 🔒 Vérification lors de l'évaluation

L'évaluateur va :
1. Calculer le SHA1 de votre `learnings.zip`
2. Comparer avec le contenu de `signature.txt`
3. Si différent → **note = 0** ❌

**Important** :
- ❌ Ne JAMAIS modifier `learnings.zip` après avoir généré `signature.txt`
- ❌ Ne JAMAIS commit `learnings.zip` dans git
- ✅ Seulement commit `signature.txt`

### 🛠️ Automatisation

```python
# leaffliction/utils.py
class Hasher:
    def ft_sha1_file(self, path: Path, chunk_size: int = 1024 * 1024) -> str:
        """Calcule le SHA1 d'un fichier"""
        sha1 = hashlib.sha1()
        
        with open(path, 'rb') as f:
            while chunk := f.read(chunk_size):
                sha1.update(chunk)
        
        return sha1.hexdigest()

# Utilisation dans train.py
hasher = Hasher()
sha1_hash = hasher.ft_sha1_file(out_zip)

signature_file = Path("signature.txt")
signature_file.write_text(sha1_hash + "\n")

print(f"📝 Signature saved to: {signature_file}")
print(f"   SHA1: {sha1_hash}")
```

---

<a id="checklist-finale-avant-rendu"></a>
## 12. Checklist finale avant rendu

### 📋 Code

- [ ] **Distribution.py** fonctionne avec n'importe quel dossier
- [ ] **Augmentation.py** affiche et sauvegarde 6 augmentations
- [ ] **Transformation.py** affiche 6 transformations + mode batch
- [ ] **train.py** entraîne le modèle et génère `learnings.zip`
- [ ] **predict.py** charge le modèle et prédit correctement
- [ ] Tous les scripts respectent la **séparation logique/entrypoint**
- [ ] Code **modulaire** et **réutilisable**
- [ ] Pas de **hardcoded paths** (tout en arguments)

### 📊 Dataset

- [ ] Dataset **équilibré** (ou augmenté pour équilibrer)
- [ ] Split train/valid **stratifié** (proportions conservées)
- [ ] Validation set ≥ **100 images**
- [ ] Pas de **data leakage** (validation jamais vu en training)

### 🎓 Modèle

- [ ] Accuracy validation > **90%**
- [ ] Modèle **reproductible** (seed fixé)
- [ ] Architecture **défendable** (transfer learning justifié)
- [ ] Callbacks **appropriés** (EarlyStopping, etc.)

### 📦 Packaging

- [ ] `learnings.zip` contient tout le nécessaire
- [ ] `signature.txt` généré correctement
- [ ] SHA1 **vérifié** manuellement
- [ ] Aucun fichier inutile dans le repo

### 📝 Documentation

- [ ] `README.md` explique comment utiliser les scripts
- [ ] Usage `-h` fonctionnel pour tous les scripts
- [ ] Code **commenté** aux endroits clés
- [ ] Formules mathématiques **comprises**

### 🧪 Tests

- [ ] Testé sur **plusieurs images**
- [ ] Testé sur **plusieurs classes**
- [ ] Testé avec **différents paramètres**
- [ ] Pas de **crash** ni **erreur silencieuse**

---

<a id="conseils-pour-la-soutenance"></a>
## 13. Conseils pour la soutenance

### 🎯 Préparation mentale

**Tu dois être capable de** :
1. Expliquer **chaque ligne de code** que tu as écrite
2. Justifier **chaque choix d'architecture**
3. Dessiner le **pipeline complet** au tableau
4. Expliquer les **formules mathématiques**
5. Débugger en **live** si nécessaire

### 📊 Structure de présentation

#### 1. Introduction (2 min)
- Objectif du projet
- Technologies utilisées
- Architecture globale

#### 2. Démonstration (5 min)
```bash
# 1. Distribution
python Distribution.py ./leaves/images/

# 2. Augmentation
python Augmentation.py "./leaves/images/Apple_healthy/image (1).JPG"

# 3. Transformation
python Transformation.py "./leaves/images/Apple_healthy/image (1).JPG"

# 4. Training (montrer les logs)
python train.py ./leaves/images/ --epochs 10

# 5. Prediction
python predict.py learnings.zip "./leaves/images/Apple_Black_rot/image (1).JPG"
```

#### 3. Explication technique (10 min)

**Partie 1 : Architecture**
- Séparation entrypoints / logique métier
- Avantages de cette approche
- Réutilisabilité des composants

**Partie 2 : Machine Learning**
- Pipeline de données (tf.data)
- Architecture du modèle (transfer learning)
- Fonction de coût (cross-entropy)
- Optimiseur (Adam)

**Partie 3 : Mathématiques**
- Softmax : transformation logits → probabilités
- Cross-entropy : mesure de l'erreur de classification
- Gradient descent : optimisation des poids
- Convolution : extraction de caractéristiques

**Partie 4 : Différences Augmentation vs Transformation**
- Augmentation : préservation de la classe, stochasticité
- Transformation : modification de représentation, déterminisme

#### 4. Questions techniques (8 min)

**Questions probables** :
- Pourquoi avoir choisi MobileNetV2 ? (rapide, léger, efficace)
- Comment éviter l'overfitting ? (dropout, data augmentation, early stopping)
- Comment s'assurer que le dataset n'est pas dans le repo ? (signature.txt)
- Qu'est-ce que le transfer learning ? (reprendre un modèle pré-entraîné)

**Questions difficiles** :
- Expliquer la backpropagation sur un exemple simple
- Calculer un softmax manuellement
- Expliquer pourquoi il faut normaliser les images
- Différencier loss fonction vs métrique

#### 5. Cas limites & Debugging (3 min)

**Scénarios** :
- Dataset mal formaté (que se passe-t-il ?)
- Modèle qui ne converge pas (diagnostic)
- Accuracy < 90% (stratégies d'amélioration)
- Erreur de chargement d'image (gestion d'erreur)

### 🗣️ Scripts de présentation

#### Demo Script 1 : Distribution
```bash
# Montrer le déséquilibre
python Distribution.py ./leaves/images/

# Expliquer pourquoi on a besoin d'augmentation
echo "On voit que Apple_Black_rot a 252 images contre 150 pour Apple_healthy"
echo "→ Déséquilibre de 68% vs 32%"
```

#### Demo Script 2 : Augmentation
```bash
# Montrer les 6 augmentations
python Augmentation.py "./leaves/images/Apple_healthy/image (1).JPG"

# Expliquer l'utilité
echo "Ces augmentations permettent de créer plus de données d'entraînement"
echo "tout en préservant la classe de l'image"
```

#### Demo Script 3 : Transformation
```bash
# Montrer les 6 transformations
python Transformation.py "./leaves/images/Apple_healthy/image (1).JPG"

# Expliquer l'utilité
echo "Ces transformations extraient des caractéristiques visuelles"
echo "utiles pour comprendre les maladies des feuilles"
```

#### Demo Script 4 : Training
```bash
# Training avec logs
python train.py ./leaves/images/ --epochs 15 --batch_size 32

# Montrer la progression
echo "Epoch 15/15 - loss: 0.1234 - accuracy: 0.9523 - val_accuracy: 0.9234"
echo "✅ Requirement met: accuracy > 90% and validation >= 100 images"
```

#### Demo Script 5 : Prediction
```bash
# Prédiction avec résultat
python predict.py learnings.zip "./leaves/images/Apple_Black_rot/image (1).JPG"

# Sortie attendue
echo "Predicted class: Apple_Black_rot"
echo "Confidence: 98.7%"
```

### 🎯 Points de défense essentiels

#### Architecture (À RETENIR PAR CŒUR)

```
Scripts Racine (entrypoints)
├── Distribution.py     → parsing + DatasetScanner + DistributionPlotter
├── Augmentation.py     → parsing + AugmentationEngine + GridPlotter
├── Transformation.py   → parsing + TransformationEngine + GridPlotter
├── train.py           → parsing + Trainer + RequirementsGate + ZipPackager
└── predict.py         → parsing + Predictor + ModelBundle

Package leaffliction/ (logique métier)
├── cli.py             → tous les parsers argparse
├── utils.py           → PathManager, Hasher, ZipPackager
├── dataset.py         → DatasetScanner, DatasetSplitter, TFDatasetBuilder
├── plotting.py        → DistributionPlotter, GridPlotter
├── augmentations.py   → 6 augmentations + AugmentationEngine
├── transformations.py → 6 transformations + TransformationEngine
├── model.py           → ModelFactory, LabelEncoder, ModelBundle
├── train_pipeline.py  → Trainer, RequirementsGate, TrainingPackager
└── predict_pipeline.py → Predictor, PredictionVisualiser
```

#### Mathématiques (FORMULES À SAVOIR)

**Softmax** :
```
p_i = exp(z_i) / Σ_j exp(z_j)
```

**Cross-Entropy** :
```
L = -log(p_true_class)
```

**Gradient Descent** :
```
θ = θ - α * ∂L/∂θ
```

**Convolution** :
```
y[i,j] = Σ_m Σ_n x[i+m,j+n] * w[m,n]
```

#### Différences fondamentales

| Concept | Augmentation | Transformation |
|---------|-------------|----------------|
| **Moment** | Training | Analyse |
| **Nature** | Stochastique | Déterministe |
| **Préserve classe** | ✅ | ❌ |
| **But** | Augmenter données | Extraire features |
| **Exemples** | Flip, Rotate | Canny, Threshold |

---

## 🎯 Conclusion personnelle

### 🧠 Ce que ce guide t'apporte

**Si tu comprends ce guide** :
- Tu maîtriseras chaque ligne de ton code
- Tu pourras expliquer chaque choix technique
- Tu seras capable de justifier tes formules mathématiques
- Tu pourras débugger efficacement
- Tu seras confiant en soutenance

### 📚 La force de l'architecture

**Pourquoi cette séparation est géniale** :
- **Testabilité** : chaque composant testable séparément
- **Réutilisabilité** : classes utilisées par plusieurs scripts
- **Maintenabilité** : logique centralisée
- **Clarté** : pas de mélange entrypoint/logic
- **Extensibilité** : facile d'ajouter de nouvelles fonctionnalités

### 🔑 Les clés du succès

1. **Comprendre avant de coder** : lis toujours le sujet en détail
2. **Séparer les responsabilités** : jamais de logique dans les entrypoints
3. **Comprendre les formules** : pas seulement les appliquer
4. **Tester beaucoup** : plusieurs images, plusieurs cas
5. **Préparer la défense** : être capable d'expliquer au tableau

### 🎓 Conseils pour la réussite

**Si tu es bloqué** :
1. Reviens au sujet, relis les contraintes
2. Dessine le pipeline au brouillon
3. Teste chaque composant séparément
4. Demande de l'aide (mais en posant des questions précises)

**Pour la soutenance** :
1. Prépare ta démo (scripts prêts, images testées)
2. Révise les formules (peux-tu les expliquer ?)
3. Anticipe les questions difficiles
4. Reste calme et explique ta logique

### 🌟 Mot de fin

Ce projet n'est pas juste "faire fonctionner un code". C'est :
- **Apprendre la computer vision**
- **Comprendre le machine learning**
- **Maîtriser l'architecture logicielle**
- **Savoir expliquer ses choix**

Si tu arrives à expliquer ce guide à quelqu'un d'autre,
alors tu maîtrises vraiment ton projet.

**Bon courage pour la réussite ! 🚀**

---

## 📖 Annexes

### Annexe A : Configuration TensorFlow

```python
# Optimisation des performances
import tensorflow as tf

# Configuration GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Optimisation CPU
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)
```

### Annexe B : Monitoring TensorBoard

```bash
# Lancement de TensorBoard
tensorboard --logdir=./artifacts/tensorboard --port=6006

# URL locale
# http://localhost:6006
```

### Annexe C : Commandes utiles

```bash
# Vérification GPU
nvidia-smi

# Test TensorFlow
python -c "import tensorflow as tf; print(tf.__version__)"

# Taille du dataset
du -sh ./leaves/images/

# Nombre d'images par classe
find ./leaves/images/ -name "*.JPG" | wc -l

# Test d'un script
python -c "from leaffliction.cli import CLIBuilder; print('✅ CLI OK')"
```

### Annexe D : Debugging

```python
# Debug des shapes
print(f"Image shape: {img.shape}")  # (224, 224, 3)
print(f"Batch shape: {batch.shape}")  # (32, 224, 224, 3)

# Debug des types
print(f"Image dtype: {img.dtype}")  # float32
print(f"Label dtype: {label.dtype}")  # int64

# Debug des valeurs
print(f"Image range: [{img.numpy().min():.3f}, {img.numpy().max():.3f}]")
```

---

**Fin du guide** 🎉

> **Dernière mise à jour** : Version 1.0  
> **Auteur** : Assistant IA - Guide personnel de développement  
> **Usage** : Manuel personnel pour le projet Leaffliction

