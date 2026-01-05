# 📖 Guide Conceptuel d'Implémentation — Leaffliction

> **Objectif** : Expliquer **littéralement** ce que chaque classe doit faire, **sans code**, pour que vous puissiez implémenter vous-même.

---

## 📑 Table des matières

1. [leaffliction/cli.py — Parsers d'arguments](#cli)
2. [leaffliction/utils.py — Utilitaires](#utils)
3. [leaffliction/dataset.py — Gestion du dataset](#dataset)
4. [leaffliction/plotting.py — Visualisations](#plotting)
5. [leaffliction/augmentations.py — Augmentations](#augmentations)
6. [leaffliction/transformations.py — Transformations](#transformations)
7. [leaffliction/model.py — Modèle et encodage](#model)
8. [leaffliction/train_pipeline.py — Pipeline d'entraînement](#train-pipeline)
9. [leaffliction/predict_pipeline.py — Pipeline de prédiction](#predict-pipeline)

---


## 🔄 Ordre de réalisation et dépendances

### 📊 Graphe de dépendances

```
Phase 1 (Fondations - Aucune dépendance)
├── cli.py          [Personne A] ⏱️ 1-2h
└── utils.py        [Personne B] ⏱️ 2-3h

Phase 2 (Dataset - Dépend de utils.py)
├── dataset.py      [Personne A] ⏱️ 4-5h (dépend: utils.py)
└── plotting.py     [Personne B] ⏱️ 2-3h (indépendant)

Phase 3 (Transformations - Peuvent être parallèles)
├── augmentations.py      [Personne A] ⏱️ 3-4h (dépend: utils.py)
└── transformations.py    [Personne B] ⏱️ 3-4h (indépendant)

Phase 4 (Modèle - Dépend de dataset.py)
└── model.py        [Personne A ou B] ⏱️ 3-4h (dépend: dataset.py)

Phase 5 (Pipelines - Dépend de tout)
├── train_pipeline.py     [Personne A] ⏱️ 5-6h (dépend: dataset, model, augmentations)
└── predict_pipeline.py   [Personne B] ⏱️ 2-3h (dépend: model, transformations)

Phase 6 (Scripts racine - Dépend de tout)
├── Distribution.py       [Personne A] ⏱️ 30min (dépend: cli, dataset, plotting)
├── Augmentation.py       [Personne B] ⏱️ 30min (dépend: cli, augmentations, plotting)
├── Transformation.py     [Personne A] ⏱️ 30min (dépend: cli, transformations, plotting)
├── train.py             [Personne B] ⏱️ 1h (dépend: cli, train_pipeline, utils)
└── predict.py           [Personne A] ⏱️ 30min (dépend: cli, predict_pipeline)
```

---

### 🎯 Stratégie de travail en équipe (2 personnes)

#### **Option 1 : Division par couches (Recommandé)**

**Personne A : Backend/ML**
- Phase 1 : `cli.py`
- Phase 2 : `dataset.py`
- Phase 3 : `augmentations.py`
- Phase 4 : `model.py`
- Phase 5 : `train_pipeline.py`
- Phase 6 : `Distribution.py`, `Transformation.py`, `predict.py`

**Personne B : Visualisation/Transformations**
- Phase 1 : `utils.py`
- Phase 2 : `plotting.py`
- Phase 3 : `transformations.py`
- Phase 4 : Aide sur `model.py` ou tests
- Phase 5 : `predict_pipeline.py`
- Phase 6 : `Augmentation.py`, `train.py`

**Avantages** :
- ✅ Séparation claire des responsabilités
- ✅ Peu de conflits de merge
- ✅ Chacun devient expert de sa partie

---

#### **Option 2 : Division par fonctionnalités**

**Personne A : Partie 1-2-3 du sujet**
- `cli.py` (parsers Distribution, Augmentation, Transformation)
- `utils.py`
- `dataset.py`
- `plotting.py`
- `augmentations.py`
- `transformations.py`
- Scripts : `Distribution.py`, `Augmentation.py`, `Transformation.py`

**Personne B : Partie 4 du sujet (Classification)**
- `cli.py` (parsers Train, Predict)
- `model.py`
- `train_pipeline.py`
- `predict_pipeline.py`
- Scripts : `train.py`, `predict.py`

**Avantages** :
- ✅ Chacun peut tester sa partie indépendamment
- ✅ Correspond à la structure du sujet
- ⚠️ Nécessite de bien coordonner `cli.py`

---

### 📋 Tableau de dépendances détaillé

| Fichier | Dépend de | Peut être fait en parallèle avec | Temps estimé |
|---------|-----------|----------------------------------|--------------|
| **cli.py** | Rien | utils.py | 1-2h |
| **utils.py** | Rien | cli.py | 2-3h |
| **dataset.py** | utils.py | plotting.py | 4-5h |
| **plotting.py** | Rien | dataset.py, augmentations.py, transformations.py | 2-3h |
| **augmentations.py** | utils.py | transformations.py, plotting.py | 3-4h |
| **transformations.py** | Rien | augmentations.py, plotting.py, dataset.py | 3-4h |
| **model.py** | dataset.py (pour LabelEncoder) | Rien (bloquant pour pipelines) | 3-4h |
| **train_pipeline.py** | dataset.py, model.py, augmentations.py | predict_pipeline.py | 5-6h |
| **predict_pipeline.py** | model.py, transformations.py | train_pipeline.py | 2-3h |
| **Distribution.py** | cli.py, dataset.py, plotting.py | Autres scripts | 30min |
| **Augmentation.py** | cli.py, augmentations.py, plotting.py | Autres scripts | 30min |
| **Transformation.py** | cli.py, transformations.py, plotting.py | Autres scripts | 30min |
| **train.py** | cli.py, train_pipeline.py, utils.py | predict.py | 1h |
| **predict.py** | cli.py, predict_pipeline.py | train.py | 30min |

---

### ⚡ Fichiers qui PEUVENT être faits en parallèle

**Groupe 1 (Phase 1 - Aucune dépendance)** :
- `cli.py` ⚡ `utils.py`

**Groupe 2 (Phase 2)** :
- `plotting.py` ⚡ `dataset.py` (si utils.py est terminé)

**Groupe 3 (Phase 3 - Maximum de parallélisme)** :
- `augmentations.py` ⚡ `transformations.py` ⚡ `plotting.py` (si pas encore fait)

**Groupe 4 (Phase 5)** :
- `train_pipeline.py` ⚡ `predict_pipeline.py` (si model.py est terminé)

**Groupe 5 (Phase 6 - Tous les scripts racine)** :
- `Distribution.py` ⚡ `Augmentation.py` ⚡ `Transformation.py` ⚡ `train.py` ⚡ `predict.py`

---

### 🚨 Fichiers BLOQUANTS (à faire en priorité)

Ces fichiers bloquent beaucoup d'autres :

1. **utils.py** (bloque : dataset.py, augmentations.py)
2. **dataset.py** (bloque : model.py, train_pipeline.py)
3. **model.py** (bloque : train_pipeline.py, predict_pipeline.py)

**Stratégie** : Commencer par ces 3 fichiers dans l'ordre !

---

### 📅 Planning suggéré pour 2 personnes (sur 3-4 jours)

#### **Jour 1 : Fondations (6-8h)**
- **Matin** :
  - Personne A : `cli.py` (2h)
  - Personne B : `utils.py` (3h)
- **Après-midi** :
  - Personne A : `dataset.py` (4h)
  - Personne B : `plotting.py` (2h) + début `transformations.py` (2h)

#### **Jour 2 : Transformations et Modèle (6-8h)**
- **Matin** :
  - Personne A : Finir `dataset.py` + début `model.py` (4h)
  - Personne B : Finir `transformations.py` (2h) + `augmentations.py` (3h)
- **Après-midi** :
  - Personne A : Finir `model.py` (2h)
  - Personne B : Tests des transformations (2h)

#### **Jour 3 : Pipelines (6-8h)**
- **Matin** :
  - Personne A : `train_pipeline.py` (5h)
  - Personne B : `predict_pipeline.py` (3h)
- **Après-midi** :
  - Personne A : Finir `train_pipeline.py` (2h)
  - Personne B : Tests de prédiction (2h)

#### **Jour 4 : Scripts racine et tests (4-6h)**
- **Matin** :
  - Personne A : `Distribution.py`, `Transformation.py`, `predict.py` (2h)
  - Personne B : `Augmentation.py`, `train.py` (2h)
- **Après-midi** :
  - Les deux : Tests complets, debugging, génération signature.txt (2-4h)

---

### 💡 Conseils pour le travail en équipe

**Communication** :
- 📱 Utiliser un canal de communication rapide (Discord, Slack, etc.)
- 📝 Documenter les interfaces entre fichiers (signatures de fonctions)
- 🔄 Faire des points réguliers (matin et soir)

**Git** :
- 🌿 Créer une branche par fichier : `feature/cli`, `feature/utils`, etc.
- 🔀 Merger régulièrement pour éviter les gros conflits
- ✅ Faire des commits atomiques avec des messages clairs

**Tests** :
- 🧪 Tester chaque fichier individuellement avant de merger
- 📊 Créer des données de test minimales
- 🔍 Valider les interfaces entre fichiers

**Répartition des tâches** :
- 📋 Utiliser un Trello/Notion pour suivre l'avancement
- ⏰ Estimer le temps pour chaque tâche
- 🎯 Prioriser les fichiers bloquants

---

### 🎓 Points de synchronisation obligatoires

**Sync 1 : Après Phase 1** (cli.py + utils.py)
- Valider les signatures des fonctions
- S'assurer que PathManager fonctionne
- Tester les parsers

**Sync 2 : Après Phase 2** (dataset.py + plotting.py)
- Valider DatasetIndex
- Tester le scan d'un petit dataset
- Vérifier les visualisations

**Sync 3 : Après Phase 4** (model.py)
- Valider LabelEncoder
- Tester la construction du modèle
- S'assurer que ModelBundle fonctionne

**Sync 4 : Avant Phase 6** (tous les pipelines)
- Test end-to-end du training
- Test end-to-end de la prédiction
- Validation des contraintes (accuracy > 90%)

---

<a id="cli"></a>
## 1. leaffliction/cli.py — Parsers d'arguments

### **Classe : CLIBuilder**

**Responsabilité** : Centraliser la création de tous les parsers d'arguments pour éviter la duplication.

---

#### **Méthode : build_distribution_parser()**

**Ce qu'elle doit faire** :
1. Créer un parser argparse avec une description claire
2. Ajouter UN argument positionnel obligatoire nommé "dataset_dir"
3. Cet argument doit accepter un string (chemin vers le dossier)
4. Ajouter un message d'aide pour expliquer à quoi sert cet argument
5. Retourner le parser configuré

**Utilisation attendue** : `python Distribution.py ./leaves/images/`

---

#### **Méthode : build_augmentation_parser()**

**Ce qu'elle doit faire** :
1. Créer un parser argparse
2. Ajouter UN argument positionnel obligatoire nommé "image_path"
3. Cet argument doit accepter un string (chemin vers une image)
4. Retourner le parser

**Utilisation attendue** : `python Augmentation.py "./leaves/images/Apple_healthy/image (1).JPG"`

---

#### **Méthode : build_transformation_parser()**

**Ce qu'elle doit faire** :
1. Créer un parser argparse
2. Ajouter un argument positionnel OPTIONNEL "image_path" (nargs="?")
   - Cet argument est utilisé pour le mode single image
3. Ajouter un argument optionnel "-src" (type string)
   - Pour spécifier le dossier source en mode batch
4. Ajouter un argument optionnel "-dst" (type string)
   - Pour spécifier le dossier destination en mode batch
5. Ajouter un flag "-mask" (action="store_true")
   - Booléen pour appliquer des transformations de masque
6. Ajouter un flag "-recursive" (action="store_true", default=True)
   - Pour traiter les sous-dossiers récursivement
7. Retourner le parser

**Utilisation attendue** :
- Mode single : `python Transformation.py "image.jpg"`
- Mode batch : `python Transformation.py -src ./dossier/ -dst ./sortie/`

---

#### **Méthode : build_train_parser()**

**Ce qu'elle doit faire** :
1. Créer un parser argparse
2. Ajouter UN argument positionnel obligatoire "dataset_dir"
3. Ajouter TOUS ces arguments optionnels avec leurs valeurs par défaut :
   - `--epochs` : entier, défaut 10
   - `--batch_size` : entier, défaut 32
   - `--lr` : float, défaut 0.001
   - `--valid_ratio` : float, défaut 0.2
   - `--seed` : entier, défaut 42
   - `--img_h` : entier, défaut 224
   - `--img_w` : entier, défaut 224
   - `--augment` : booléen (store_true), défaut True
   - `--export_images` : booléen (store_true), défaut True
   - `--out_dir` : string, défaut "artifacts"
   - `--out_zip` : string, défaut "learnings.zip"
4. Retourner le parser

**Utilisation attendue** : `python train.py ./leaves/images/ --epochs 20 --batch_size 32`

---

#### **Méthode : build_predict_parser()**

**Ce qu'elle doit faire** :
1. Créer un parser argparse
2. Ajouter DEUX arguments positionnels obligatoires :
   - "bundle_zip" : chemin vers learnings.zip
   - "image_path" : chemin vers l'image à prédire
3. Ajouter des arguments optionnels :
   - `--show_transforms` : booléen (store_true), défaut True
   - `--top_k` : entier, défaut 1
4. Retourner le parser

**Utilisation attendue** : `python predict.py learnings.zip "./image.jpg"`

---

**Pourquoi centraliser** : Si tu dois changer un argument (ex: renommer, changer la valeur par défaut), tu le changes à UN SEUL endroit au lieu de modifier 5 fichiers différents.

---

<a id="utils"></a>
## 2. leaffliction/utils.py — Utilitaires

### **Classe 1 : PathManager**

**Responsabilité** : Gérer toutes les opérations liées aux chemins de fichiers et aux conventions de nommage.

---

#### **Attribut de classe : IMAGE_EXTS**

**Ce qu'il doit contenir** :
- Un ensemble (set) de toutes les extensions d'images supportées
- Inclure les versions minuscules ET majuscules
- Exemples : ".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", ".bmp", ".tif", ".tiff", ".webp"

**Pourquoi un set** : Recherche O(1) pour vérifier si une extension est supportée.

---

#### **Méthode : ft_ensure_dir(path)**

**Ce qu'elle doit faire** :
1. Recevoir un objet Path (chemin vers un dossier)
2. Vérifier si ce dossier existe
3. Si le dossier n'existe pas :
   - Le créer
   - Créer TOUS les dossiers parents nécessaires dans le chemin
4. Si le dossier existe déjà :
   - Ne rien faire, ne pas lever d'erreur
5. Retourner le même objet Path (pour permettre le chaînage de méthodes)

**Exemple** : Si tu appelles `ft_ensure_dir(Path("a/b/c/d"))` et que seul "a" existe, la méthode doit créer "b", "c", et "d".

**Pourquoi retourner path** : Permet d'écrire `path = pm.ft_ensure_dir(Path("artifacts")).resolve()`

---

#### **Méthode : ft_make_suffixed_path(image_path, suffix)**

**Ce qu'elle doit faire** :
1. Recevoir un chemin d'image (Path) et un suffixe (string)
2. Extraire le nom du fichier SANS l'extension (appelé "stem")
   - Exemple : `image (1).JPG` → stem = `"image (1)"`
3. Extraire l'extension du fichier
   - Exemple : `image (1).JPG` → ext = `".JPG"`
4. Extraire le dossier parent
   - Exemple : `/path/to/image (1).JPG` → parent = `/path/to/`
5. Construire un nouveau nom de fichier : `{stem}_{suffix}{extension}`
   - Exemple : `"image (1)"` + `"_Flip"` + `".JPG"` = `"image (1)_Flip.JPG"`
6. Combiner le parent avec le nouveau nom
7. Retourner le nouveau chemin complet

**Cas d'usage** : Sauvegarder les augmentations avec des suffixes comme "_Flip", "_Rotate", etc.

---

#### **Méthode : ft_iter_images(root, recursive)**

**Ce qu'elle doit faire** :
1. Recevoir un dossier racine (Path) et un flag recursive (booléen)
2. Créer une liste vide pour stocker les chemins d'images
3. Déterminer le pattern de recherche :
   - Si recursive=True : chercher dans tous les sous-dossiers (`"**/*"`)
   - Si recursive=False : chercher seulement au niveau racine (`"*"`)
4. Pour CHAQUE extension dans IMAGE_EXTS :
   - Utiliser glob pour trouver tous les fichiers avec cette extension
   - Ajouter tous les chemins trouvés à la liste
5. Trier la liste alphabétiquement
6. Retourner la liste triée

**Pourquoi trier** : Garantit un ordre reproductible entre différentes exécutions.

---

### **Classe 2 : Hasher**

**Responsabilité** : Calculer le hash SHA1 de fichiers pour générer signature.txt.

---

#### **Méthode : ft_sha1_file(path, chunk_size)**

**Ce qu'elle doit faire** :
1. Recevoir un chemin de fichier (Path) et une taille de chunk (défaut 1MB = 1024*1024 bytes)
2. Initialiser un objet de hachage SHA1
3. Ouvrir le fichier en mode binaire lecture
4. Lire le fichier par morceaux (chunks) :
   - Lire chunk_size bytes
   - Mettre à jour le hash avec ces bytes
   - Répéter jusqu'à la fin du fichier
5. Fermer le fichier automatiquement
6. Obtenir le digest final du hash
7. Convertir le digest en format hexadécimal (string de 40 caractères)
8. Retourner cette string

**Pourquoi par chunks** : Un fichier ZIP peut faire plusieurs GB. Lire par chunks évite de saturer la RAM en chargeant tout le fichier d'un coup.

**Exemple de sortie** : `"7a18a838d2203cc7d6e8c4c521fdd4dd214aa560"`

---

### **Classe 3 : ZipPackager**

**Responsabilité** : Compresser des dossiers en fichiers ZIP et décompresser des ZIP.

---

#### **Méthode : ft_zip_dir(src_dir, out_zip)**

**Ce qu'elle doit faire** :
1. Recevoir un dossier source (Path) et un chemin de sortie pour le ZIP (Path)
2. **VÉRIFICATION CRITIQUE** : S'assurer que out_zip n'est PAS à l'intérieur de src_dir
   - Sinon, boucle infinie (le ZIP essaie de se compresser lui-même)
3. Créer un fichier ZIP en mode écriture avec compression DEFLATED
4. Parcourir RÉCURSIVEMENT tous les fichiers dans src_dir :
   - Pour chaque fichier trouvé :
     - Calculer son chemin RELATIF par rapport à src_dir
     - Ajouter le fichier au ZIP avec ce chemin relatif (pas absolu)
5. Fermer le ZIP automatiquement

**Pourquoi chemin relatif** : Si tu utilises des chemins absolus, le ZIP contiendra des chemins comme `/home/user/project/file.txt` qui ne fonctionneront pas sur une autre machine.

**Exemple** :
- src_dir = `/home/user/artifacts/`
- Fichier = `/home/user/artifacts/models/model.keras`
- Chemin relatif dans le ZIP = `models/model.keras`

---

#### **Méthode : ft_unzip(zip_path, extract_dir)** (optionnel mais utile)

**Ce qu'elle doit faire** :
1. Recevoir un chemin de ZIP (Path) et un dossier de destination (Path)
2. Ouvrir le ZIP en mode lecture
3. Extraire TOUT le contenu dans extract_dir
4. Préserver la structure des dossiers du ZIP
5. Fermer le ZIP automatiquement

---

<a id="dataset"></a>
## 3. leaffliction/dataset.py — Gestion du dataset

### **Classe 1 : DatasetIndex (dataclass)**

**Responsabilité** : Représenter l'index complet du dataset après scan. C'est une structure de données.

---

#### **Attributs obligatoires**

**Ce qu'elle doit contenir** :
1. `root` : Path vers le dossier racine du dataset
2. `class_names` : Liste des noms de classes (strings), triée alphabétiquement
3. `items` : Liste de tuples `(chemin_image: Path, class_id: int)`
4. `counts` : Dictionnaire `{nom_classe: string → nombre_images: int}`

**Exemple** :
```
DatasetIndex(
    root=Path("./leaves/images/"),
    class_names=["Apple_Black_rot", "Apple_healthy", "Grape_Black_rot"],
    items=[
        (Path("./leaves/images/Apple_Black_rot/image1.JPG"), 0),
        (Path("./leaves/images/Apple_Black_rot/image2.JPG"), 0),
        (Path("./leaves/images/Apple_healthy/image1.JPG"), 1),
        ...
    ],
    counts={
        "Apple_Black_rot": 252,
        "Apple_healthy": 150,
        "Grape_Black_rot": 180
    }
)
```

---

#### **Propriétés calculées**

1. **num_classes**
   - Retourner la longueur de class_names
   - Exemple : 3 classes → retourne 3

2. **size**
   - Retourner la longueur de items (nombre total d'images)
   - Exemple : 582 images → retourne 582

**Pourquoi des propriétés** : Évite de stocker des valeurs redondantes. Elles sont calculées à la demande.

---

### **Classe 2 : DatasetScanner**

**Responsabilité** : Scanner un dossier organisé en sous-dossiers (un par classe) et construire un DatasetIndex.

---

#### **Méthode : ft_scan(root)**

**Ce qu'elle doit faire** :

**Étape 1 : Lister les sous-dossiers**
- Recevoir un chemin vers le dossier racine (Path)
- Lister TOUS les sous-dossiers directs (pas récursif)
- Filtrer pour ne garder que les dossiers (pas les fichiers)

**Étape 2 : Trier les dossiers**
- Trier la liste des dossiers alphabétiquement
- Cet ordre détermine les class_id (0, 1, 2, ...)

**Étape 3 : Extraire les noms de classes**
- Pour chaque dossier, extraire son nom
- Ces noms deviennent class_names

**Étape 4 : Scanner chaque classe**
- Pour chaque dossier (avec son index comme class_id) :
  - Lister toutes les images dans ce dossier
  - Pour chaque extension supportée, chercher les fichiers
  - Compter le nombre total d'images trouvées
  - Pour chaque image trouvée :
    - Créer un tuple (chemin_image, class_id)
    - Ajouter ce tuple à la liste items
  - Stocker le compte dans le dictionnaire counts

**Étape 5 : Construire et retourner**
- Créer un objet DatasetIndex avec toutes ces informations
- Retourner cet objet

**Structure attendue du dataset** :
```
root/
  Apple_Black_rot/     ← class_id = 0
    image (1).JPG
    image (2).JPG
  Apple_healthy/       ← class_id = 1
    image (1).JPG
  Grape_Black_rot/     ← class_id = 2
    image (1).JPG
```

**Pourquoi trier** : L'ordre alphabétique garantit que `Apple_Black_rot` aura toujours class_id=0, même si tu relances le programme.

---

### **Classe 3 : DatasetSplitter**

**Responsabilité** : Diviser les données en ensembles train et validation de manière stratifiée.

---

#### **Méthode : ft_split(items, valid_ratio, seed, stratified)**

**Ce qu'elle doit faire** :

**Paramètres** :
- `items` : Liste de tuples (Path, class_id)
- `valid_ratio` : Float entre 0 et 1 (ex: 0.2 = 20% en validation)
- `seed` : Entier pour la reproductibilité
- `stratified` : Booléen (True recommandé)

**Étape 0 : Fixer le seed**
- Fixer le seed du générateur aléatoire pour reproductibilité

---

**Si stratified=False** (simple, NON recommandé) :

**Étape 1** : Copier la liste items
**Étape 2** : Mélanger aléatoirement toute la liste
**Étape 3** : Calculer l'index de séparation
- `split_idx = int(len(items) * (1 - valid_ratio))`
**Étape 4** : Séparer
- train = items[:split_idx]
- valid = items[split_idx:]
**Étape 5** : Retourner (train, valid)

**Problème** : Si une classe a peu d'images, elle peut être absente de la validation.

---

**Si stratified=True** (RECOMMANDÉ) :

**Étape 1 : Grouper par classe**
- Créer un dictionnaire vide : `{class_id: []}`
- Pour chaque item dans items :
  - Extraire le class_id
  - Ajouter l'item à la liste correspondante dans le dictionnaire
- Résultat : `{0: [items de classe 0], 1: [items de classe 1], ...}`

**Étape 2 : Splitter chaque classe séparément**
- Créer deux listes vides : train_items et valid_items
- Pour CHAQUE classe dans le dictionnaire :
  - Récupérer la liste des items de cette classe
  - Copier cette liste
  - Mélanger aléatoirement cette copie
  - Calculer combien d'items vont en validation : `n_valid = int(len(liste) * valid_ratio)`
  - Séparer : 
    - Les n_valid derniers items → valid
    - Le reste → train
  - Ajouter les items train de cette classe à train_items
  - Ajouter les items valid de cette classe à valid_items

**Étape 3 : Mélanger les listes finales**
- Mélanger train_items (pour que les classes soient mélangées)
- Mélanger valid_items

**Étape 4 : Retourner**
- Retourner (train_items, valid_items)

**Pourquoi stratifié** : Si une classe représente 10% du dataset, elle représentera aussi ~10% du train ET ~10% du valid. Cela garantit que toutes les classes sont présentes dans les deux ensembles.

**Exemple** :
- Classe A : 100 images → 80 train, 20 valid
- Classe B : 50 images → 40 train, 10 valid
- Classe C : 200 images → 160 train, 40 valid

---

### **Classe 4 : TFDataConfig (dataclass)**

**Responsabilité** : Stocker la configuration pour construire un tf.data.Dataset.

---

#### **Attributs**

**Ce qu'elle doit contenir** :
- `img_size` : Tuple (hauteur, largeur) pour redimensionner les images (ex: (224, 224))
- `batch_size` : Entier, taille des batchs (ex: 32)
- `shuffle` : Booléen, mélanger ou non les données
- `seed` : Entier, seed pour reproductibilité
- `cache` : Booléen, mettre en cache les données en RAM
- `prefetch` : Booléen, précharger les données pendant le training

**Pourquoi une dataclass** : Regroupe tous les paramètres de configuration dans un seul objet facile à passer.

---

### **Classe 5 : TFDatasetBuilder**

**Responsabilité** : Construire un tf.data.Dataset optimisé à partir d'une liste d'items.

---

#### **Méthode : __init__(cfg, augmentor)**

**Ce qu'elle doit faire** :
1. Recevoir une configuration (TFDataConfig)
2. Recevoir un augmenteur optionnel (peut être None)
3. Stocker ces deux objets comme attributs

---

#### **Méthode : build(items, training)**

**Ce qu'elle doit faire** :

**Paramètres** :
- `items` : Liste de tuples (Path, class_id)
- `training` : Booléen (True pour train, False pour valid)

---

**Étape 1 : Extraire paths et labels**
- Créer une liste `paths` : extraire tous les chemins, convertir en strings
- Créer une liste `labels` : extraire tous les class_id

---

**Étape 2 : Créer le dataset TensorFlow**
- Utiliser `tf.data.Dataset.from_tensor_slices((paths, labels))`
- Cela crée un dataset qui yield des tuples (path_string, label_int)

---

**Étape 3 : Shuffle (seulement si training=True)**
- Si training ET cfg.shuffle :
  - Appliquer shuffle avec buffer_size = nombre total d'items
  - Utiliser cfg.seed pour reproductibilité

**Pourquoi buffer_size = len(items)** : Garantit un mélange complet.

---

**Étape 4 : Map load_and_preprocess**
- Appliquer la méthode `_load_and_preprocess` à chaque élément
- Utiliser `num_parallel_calls=AUTOTUNE` pour paralléliser le chargement
- Cela transforme (path_string, label) → (image_tensor, label)

---

**Étape 5 : Map augmentations (seulement si training=True ET augmentor existe)**
- Si training ET self.augmentor n'est pas None :
  - Appliquer l'augmenteur à chaque image
  - Utiliser `num_parallel_calls=AUTOTUNE`
  - Cela transforme (image, label) → (image_augmentée, label)

---

**Étape 6 : Batch**
- Grouper les éléments par paquets de cfg.batch_size
- Cela transforme des éléments individuels en batchs
- Exemple : (image, label) → (batch_images[32, 224, 224, 3], batch_labels[32])

---

**Étape 7 : Cache (si cfg.cache=True)**
- Mettre en cache les données en RAM
- Évite de recharger les images à chaque epoch
- **Attention** : Utiliser seulement si le dataset tient en RAM

---

**Étape 8 : Prefetch**
- Si cfg.prefetch=True :
  - Précharger les données pendant que le GPU travaille
  - Utiliser `AUTOTUNE` pour optimisation automatique

**Pourquoi prefetch** : Pendant que le GPU entraîne sur le batch N, le CPU prépare le batch N+1.

---

**Étape 9 : Retourner**
- Retourner le dataset final configuré

---

#### **Méthode : _load_and_preprocess(path, label)**

**Ce qu'elle doit faire** :

**Paramètres** :
- `path` : Tensor string (chemin vers l'image)
- `label` : Tensor int (class_id)

---

**Étape 1 : Lire le fichier**
- Utiliser `tf.io.read_file(path)` pour lire les bytes du fichier
- Retourne un tensor de bytes

---

**Étape 2 : Décoder l'image**
- Utiliser `tf.image.decode_jpeg` avec channels=3
- Cela décode les bytes JPEG en tensor RGB
- Résultat : tensor de shape (H_original, W_original, 3) avec valeurs [0, 255]

---

**Étape 3 : Redimensionner**
- Utiliser `tf.image.resize` pour redimensionner à cfg.img_size
- Exemple : (1024, 768, 3) → (224, 224, 3)

---

**Étape 4 : Normaliser**
- Convertir en float32
- Diviser par 255.0
- Résultat : valeurs entre [0, 1]

**Pourquoi normaliser** : Les réseaux de neurones fonctionnent mieux avec des valeurs entre 0 et 1.

---

**Étape 5 : Retourner**
- Retourner le tuple (image_normalisée, label)

---

**Pourquoi ce pipeline** : C'est le pipeline standard TensorFlow optimisé pour les performances. Chaque étape a un rôle précis et l'ordre est important.

---

<a id="plotting"></a>
## 4. leaffliction/plotting.py — Visualisations

### **Classe 1 : DistributionPlotter**

**Responsabilité** : Afficher des graphiques pour visualiser la distribution des classes dans le dataset.

---

#### **Méthode : plot_pie(counts, title, save_to)**

**Ce qu'elle doit faire** :

**Paramètres** :
- `counts` : Dictionnaire {nom_classe: nombre_images}
- `title` : String, titre du graphique
- `save_to` : Path optionnel pour sauvegarder

---

**Étape 1 : Extraire les données**
- Créer une liste `labels` avec les clés du dictionnaire (noms de classes)
- Créer une liste `sizes` avec les valeurs du dictionnaire (nombres)

---

**Étape 2 : Créer la figure**
- Créer une nouvelle figure matplotlib
- Taille recommandée : 10x8 pouces

---

**Étape 3 : Dessiner le pie chart**
- Utiliser `plt.pie` avec :
  - `sizes` comme valeurs (détermine la taille des parts)
  - `labels` comme étiquettes
  - `autopct='%1.1f%%'` pour afficher les pourcentages (ex: "25.5%")
  - `startangle=90` pour commencer à 12h (haut)
  - Couleurs distinctes pour chaque classe (utiliser un colormap)

---

**Étape 4 : Ajouter le titre**
- Utiliser `plt.title` avec le titre fourni
- Style recommandé : fontsize=14, fontweight='bold'

---

**Étape 5 : Rendre le cercle parfait**
- Utiliser `plt.axis('equal')` pour que le pie chart soit un cercle et pas une ellipse

---

**Étape 6 : Sauvegarder si demandé**
- Si save_to n'est pas None :
  - Sauvegarder la figure en haute résolution (dpi=300)
  - Utiliser bbox_inches='tight' pour éviter de couper les labels

---

**Étape 7 : Afficher**
- Utiliser `plt.show()` pour afficher la figure

---

#### **Méthode : plot_bar(counts, title, save_to)**

**Ce qu'elle doit faire** :

**Paramètres** : Identiques à plot_pie

---

**Étape 1 : Extraire les données**
- Identique à plot_pie

---

**Étape 2 : Créer la figure**
- Taille recommandée : 12x6 pouces (plus large pour les barres)

---

**Étape 3 : Dessiner les barres**
- Utiliser `plt.bar` avec :
  - `labels` comme positions X
  - `values` comme hauteurs
  - Couleur : 'skyblue' ou autre couleur agréable
  - Bordure : 'navy' pour contraste
  - Transparence : alpha=0.7

---

**Étape 4 : Ajouter les valeurs au-dessus des barres**
- Pour chaque barre :
  - Récupérer sa hauteur
  - Afficher le nombre exact au-dessus de la barre
  - Centrer le texte horizontalement
  - Positionner verticalement juste au-dessus

**Pourquoi** : Permet de lire les valeurs exactes sans deviner.

---

**Étape 5 : Configurer les axes**
- Label axe X : "Classes" (fontsize=12)
- Label axe Y : "Number of images" (fontsize=12)
- Rotation des labels X : 45° si nombreux, horizontal sinon
- Alignement : ha='right' si rotation
- Grille horizontale : alpha=0.3 pour faciliter la lecture

---

**Étape 6 : Ajouter le titre**
- Identique à plot_pie

---

**Étape 7 : Ajuster le layout**
- Utiliser `plt.tight_layout()` pour éviter que les labels se chevauchent

---

**Étape 8 : Sauvegarder et afficher**
- Identique à plot_pie

---

**Pourquoi deux graphiques** :
- **Pie chart** : Montre les proportions relatives (25% vs 30%)
- **Bar chart** : Montre les valeurs absolues (250 images vs 300 images)
- Les deux sont complémentaires et donnent des insights différents

---

### **Classe 2 : GridPlotter**

**Responsabilité** : Afficher une grille d'images (original + variantes) de manière organisée.

---

#### **Méthode : show_grid(title, images, original, save_to, max_cols)**

**Ce qu'elle doit faire** :

**Paramètres** :
- `title` : String, titre de la figure
- `images` : Dictionnaire {nom_variante: image_tensor}
- `original` : Image originale (optionnel, peut être None)
- `save_to` : Path optionnel pour sauvegarder
- `max_cols` : Entier, nombre max de colonnes (défaut 3)

---

**Étape 1 : Calculer le nombre total d'images**
- Compter les images dans le dictionnaire
- Ajouter 1 si original existe
- Exemple : 6 variantes + 1 original =
7 images

---

**Étape 2 : Calculer le layout (rows, cols)**
- cols = minimum entre max_cols et total
  - Exemple : si total=7 et max_cols=3, alors cols=3
- rows = arrondi supérieur de (total / cols)
  - Exemple : 7 images / 3 cols = 2.33 → 3 rows

---

**Étape 3 : Créer la figure avec subplots**
- Créer une grille de subplots (rows x cols)
- Taille adaptée : cols * 4 pouces de large, rows * 4 pouces de haut
  - Exemple : 3 cols × 3 rows = figure de 12x12 pouces

---

**Étape 4 : Ajouter le titre principal**
- Utiliser `fig.suptitle` avec le titre fourni
- Style : fontsize=16, fontweight='bold'

---

**Étape 5 : Aplatir les axes pour itération facile**
- Si une seule image : axes devient une liste avec un élément
- Si plusieurs : aplatir la grille 2D en liste 1D
- Cela permet d'itérer facilement avec un index

---

**Étape 6 : Afficher l'original en premier (si existe)**
- Si original n'est pas None :
  - Utiliser `_show_image` sur le premier axe (index 0)
  - Titre : "Original"
  - Incrémenter l'index

---

**Étape 7 : Afficher toutes les variantes**
- Pour chaque (nom, image) dans le dictionnaire images :
  - Si l'index est encore valide (< nombre d'axes) :
    - Utiliser `_show_image` sur l'axe courant
    - Titre : le nom de la variante
    - Incrémenter l'index

---

**Étape 8 : Désactiver les axes inutilisés**
- Si rows * cols > total d'images :
  - Pour chaque axe restant (de index jusqu'à la fin) :
    - Désactiver complètement cet axe (axis('off'))

**Pourquoi** : Évite d'avoir des cases vides avec des axes visibles.

---

**Étape 9 : Ajuster le layout**
- Utiliser `plt.tight_layout()` pour espacer correctement les subplots

---

**Étape 10 : Sauvegarder et afficher**
- Si save_to fourni : sauvegarder en haute résolution
- Afficher avec `plt.show()`

---

#### **Méthode : _show_image(ax, img, title)**

**Ce qu'elle doit faire** :

**Paramètres** :
- `ax` : Un axe matplotlib (subplot)
- `img` : Image (tensor ou numpy array)
- `title` : String, titre pour cette image

---

**Étape 1 : Convertir le tensor en numpy si nécessaire**
- Si l'image a une méthode `.numpy()` : l'appeler
- Sinon : supposer que c'est déjà un numpy array

---

**Étape 2 : Clipper les valeurs entre [0, 1]**
- Utiliser `np.clip(img_np, 0, 1)`
- **Critique** : Évite les erreurs d'affichage si des valeurs sortent de [0, 1]

---

**Étape 3 : Déterminer le type d'image**
- Vérifier la shape de l'image :
  - Si 2D (H, W) : grayscale
  - Si 3D avec shape[-1] == 1 : grayscale avec channel
  - Si 3D avec shape[-1] == 3 : RGB

---

**Étape 4 : Afficher avec le bon colormap**
- Si grayscale (2D) :
  - Utiliser `ax.imshow(img_np, cmap='gray')`
- Si grayscale avec channel (3D, 1 channel) :
  - Extraire le channel : `img_np[:, :, 0]`
  - Utiliser `ax.imshow(img_np[:, :, 0], cmap='gray')`
- Si RGB (3D, 3 channels) :
  - Utiliser `ax.imshow(img_np)` sans cmap

---

**Étape 5 : Ajouter le titre**
- Utiliser `ax.set_title(title)` avec fontsize=12

---

**Étape 6 : Désactiver les axes**
- Utiliser `ax.axis('off')` pour enlever les ticks et les bordures
- Rend l'affichage plus propre

---

**Pourquoi cette structure** : Elle permet d'afficher n'importe quel nombre d'images de manière flexible, avec gestion automatique du layout et support de différents types d'images.

---

<a id="augmentations"></a>
## 5. leaffliction/augmentations.py — Augmentations

### **Classe 1 : KerasAugmentationsFactory**

**Responsabilité** : Créer un Sequential Keras contenant des layers d'augmentation pour le training.

---

#### **Méthode : build()**

**Ce qu'elle doit faire** :
1. Créer un objet `keras.Sequential`
2. Ajouter des layers d'augmentation Keras :
   - `RandomFlip("horizontal")` : flip horizontal aléatoire
   - `RandomRotation(0.1)` : rotation aléatoire ±10%
   - `RandomZoom(0.1)` : zoom aléatoire ±10%
   - `RandomContrast(0.1)` : contraste aléatoire
   - `RandomBrightness(0.1)` : luminosité aléatoire
3. Retourner ce Sequential

**Utilisation** : Ce Sequential sera appliqué PENDANT le training dans le pipeline tf.data.

**Pourquoi Keras layers** : Ils s'exécutent sur GPU, sont intégrés au graphe TensorFlow, et sont automatiquement désactivés en mode validation.

---

### **Classe 2 : AugmentationEngine**

**Responsabilité** : Appliquer des augmentations DÉTERMINISTES pour visualisation et sauvegarde.

---

#### **Méthode : __init__(augs)**

**Ce qu'elle doit faire** :
1. Recevoir une liste d'objets Augmentation
2. Stocker cette liste comme attribut

---

#### **Méthode : default_six()**

**Ce qu'elle doit faire** :
1. Créer une liste contenant exactement 6 augmentations :
   - FlipHorizontalAug()
   - RotateAug(angle=15.0)
   - BrightnessContrastAug(brightness=0.3, contrast=0.0)
   - GaussianBlurAug(sigma=2.0)
   - RandomCropResizeAug(crop_ratio=0.8)
   - BrightnessContrastAug(brightness=0.0, contrast=0.5)
2. Créer et retourner un AugmentationEngine avec cette liste

**Pourquoi 6** : Le sujet demande exactement 6 types d'augmentations.

---

#### **Méthode : apply_all(img)**

**Ce qu'elle doit faire** :
1. Recevoir une image (tensor)
2. Créer un dictionnaire vide pour les résultats
3. Pour chaque augmentation dans self.augs :
   - Appliquer l'augmentation à l'image
   - Stocker le résultat dans le dictionnaire avec le nom de l'augmentation comme clé
4. Retourner le dictionnaire {nom: image_augmentée}

**Exemple de retour** :
```
{
    "Flip": tensor_flippé,
    "Rotate": tensor_tourné,
    "Brightness": tensor_lumineux,
    ...
}
```

---

### **Les 6 Augmentations (classes individuelles)**

Chaque augmentation doit avoir :
- Un attribut `name` (string) pour identifier l'augmentation
- Une méthode `apply(img)` qui prend une image et retourne l'image augmentée

---

#### **Classe : FlipHorizontalAug**

**Attributs** :
- `name` = "Flip"

**Méthode apply(img)** :
1. Recevoir une image tensor
2. Appliquer un flip horizontal (miroir gauche-droite)
3. Utiliser `tf.image.flip_left_right(img)`
4. Retourner l'image flippée

**Effet** : L'image est inversée horizontalement comme dans un miroir.

---

#### **Classe : RotateAug**

**Attributs** :
- `angle` : Float, angle de rotation en degrés (ex: 15.0)
- `name` = "Rotate"

**Méthode apply(img)** :
1. Recevoir une image tensor
2. Convertir l'angle de degrés en radians : `angle_rad = angle * π / 180`
3. Appliquer une rotation avec interpolation bilinéaire
4. Utiliser `tfa.image.rotate` (TensorFlow Addons) ou équivalent
5. Retourner l'image tournée

**Effet** : L'image est tournée de X degrés dans le sens horaire ou anti-horaire.

---

#### **Classe : BrightnessContrastAug**

**Attributs** :
- `brightness` : Float, facteur de luminosité (ex: 0.3 = +30%)
- `contrast` : Float, facteur de contraste (ex: 0.5 = +50%)
- `name` = "Brightness" ou "Contrast" selon ce qui est modifié

**Méthode apply(img)** :
1. Recevoir une image tensor
2. Si brightness != 0 :
   - Ajuster la luminosité : `img = tf.image.adjust_brightness(img, brightness)`
3. Si contrast != 0 :
   - Ajuster le contraste : `img = tf.image.adjust_contrast(img, 1 + contrast)`
4. Clipper les valeurs entre [0, 1]
5. Retourner l'image modifiée

**Effet** : L'image devient plus claire/sombre (brightness) ou plus/moins contrastée.

---

#### **Classe : GaussianBlurAug**

**Attributs** :
- `sigma` : Float, écart-type du flou gaussien (ex: 2.0)
- `name` = "Blur"

**Méthode apply(img)** :
1. Recevoir une image tensor
2. Créer un noyau gaussien avec le sigma donné
3. Appliquer une convolution 2D avec ce noyau
4. Utiliser `tfa.image.gaussian_filter2d` ou implémenter manuellement
5. Retourner l'image floutée

**Effet** : L'image devient floue, les détails sont atténués.

**Formule du noyau gaussien** :
```
G(x, y) = (1 / 2πσ²) * exp(-(x² + y²) / 2σ²)
```

---

#### **Classe : RandomCropResizeAug**

**Attributs** :
- `crop_ratio` : Float entre 0 et 1 (ex: 0.8 = garder 80% de l'image)
- `name` = "Crop"

**Méthode apply(img)** :
1. Recevoir une image tensor de shape (H, W, C)
2. Calculer la nouvelle taille après crop :
   - `new_h = int(H * crop_ratio)`
   - `new_w = int(W * crop_ratio)`
3. Calculer les offsets pour centrer le crop :
   - `offset_h = (H - new_h) // 2`
   - `offset_w = (W - new_w) // 2`
4. Extraire la région centrale : `img[offset_h:offset_h+new_h, offset_w:offset_w+new_w]`
5. Redimensionner à la taille originale (H, W)
6. Utiliser `tf.image.resize` avec interpolation bilinéaire
7. Retourner l'image croppée et resizée

**Effet** : Zoom sur le centre de l'image.

---

### **Classe 3 : AugmentationSaver**

**Responsabilité** : Sauvegarder les images augmentées sur disque avec les bons noms.

---

#### **Méthode : __init__(path_manager)**

**Ce qu'elle doit faire** :
1. Recevoir un objet PathManager
2. Stocker comme attribut

---

#### **Méthode : save_all(image_path, results)**

**Ce qu'elle doit faire** :

**Paramètres** :
- `image_path` : Path vers l'image originale
- `results` : Dictionnaire {nom_aug: image_tensor}

---

**Étape 1 : Créer une liste pour les chemins sauvegardés**

---

**Étape 2 : Pour chaque augmentation dans results**
- Extraire le nom de l'augmentation et l'image tensor

**Étape 3 : Générer le chemin de sortie**
- Utiliser `path_manager.ft_make_suffixed_path(image_path, nom_aug)`
- Exemple : `image (1).JPG` + "Flip" → `image (1)_Flip.JPG`

**Étape 4 : Convertir le tensor en uint8**
- Multiplier par 255.0 : `img_tensor * 255.0`
- Caster en uint8 : `tf.cast(..., tf.uint8)`
- Résultat : valeurs entre [0, 255]

**Étape 5 : Encoder en JPEG**
- Utiliser `tf.image.encode_jpeg(img_uint8, quality=95)`
- Retourne des bytes JPEG

**Étape 6 : Écrire sur disque**
- Utiliser `tf.io.write_file(str(out_path), encoded_jpeg)`

**Étape 7 : Ajouter le chemin à la liste**

---

**Étape 8 : Retourner la liste des chemins sauvegardés**

---

**Pourquoi quality=95** : Bon compromis entre qualité et taille de fichier.

---

<a id="transformations"></a>
## 6. leaffliction/transformations.py — Transformations

### **Classe 1 : TransformationEngine**

**Responsabilité** : Appliquer des transformations déterministes pour extraire des caractéristiques.

---

#### **Méthode : __init__(tfs)**

**Ce qu'elle doit faire** :
1. Recevoir une liste d'objets Transformation
2. Stocker cette liste comme attribut

---

#### **Méthode : default_six()**

**Ce qu'elle doit faire** :
1. Créer une liste contenant exactement 6 transformations :
   - GrayscaleTf()
   - CannyEdgesTf()
   - HistogramEqualisationTf()
   - SharpenTf()
   - ThresholdTf()
   - MorphologyTf(mode='erode')
2. Créer et retourner un TransformationEngine avec cette liste

---

#### **Méthode : apply_all(img)**

**Ce qu'elle doit faire** :
1. Recevoir une image (tensor)
2. Créer un dictionnaire vide pour les résultats
3. Pour chaque transformation dans self.tfs :
   - Appliquer la transformation à l'image
   - Stocker le résultat dans le dictionnaire avec le nom comme clé
4. Retourner le dictionnaire {nom: image_transformée}

---

### **Les 6 Transformations (classes individuelles)**

Chaque transformation doit avoir :
- Un attribut `name` (string)
- Une méthode `apply(img)` qui prend une image et retourne l'image transformée

---

#### **Classe : GrayscaleTf**

**Attributs** :
- `name` = "Grayscale"

**Méthode apply(img)** :
1. Recevoir une image RGB tensor (H, W, 3)
2. Convertir en niveaux de gris
3. Utiliser `tf.image.rgb_to_grayscale(img)`
4. Résultat : tensor (H, W, 1)
5. Retourner l'image en grayscale

**Formule** :
```
Gray = 0.299 * R + 0.587 * G + 0.114 * B
```

**Effet** : Supprime les informations de couleur, garde seulement l'intensité lumineuse.

---

#### **Classe : CannyEdgesTf**

**Attributs** :
- `low_threshold` : Entier (ex: 50)
- `high_threshold` : Entier (ex: 150)
- `name` = "Canny"

**Méthode apply(img)** :

**Étape 1 : Définir une fonction Python**
- Créer une fonction qui prend un numpy array
- Convertir en uint8 [0, 255]
- Si RGB : convertir en grayscale avec OpenCV
- Appliquer `cv2.Canny(gray, low_threshold, high_threshold)`
- Retourner en float32 [0, 1]

**Étape 2 : Wrapper avec tf.py_function**
- Utiliser `tf.py_function` pour appeler la fonction Python
- Spécifier le type de sortie : tf.float32
- Définir la shape de sortie : (H, W, 1)

**Étape 3 : Retourner le résultat**

**Effet** : Détecte les contours dans l'image. Résultat binaire (blanc = contour, noir = fond).

**Algorithme Canny** :
1. Flou gaussien (réduction du bruit)
2. Calcul du gradient (Sobel)
3. Suppression des non-maxima
4. Seuillage par hystérésis (low et high thresholds)

---

#### **Classe : HistogramEqualisationTf**

**Attributs** :
- `name` = "HistEq"

**Méthode apply(img)** :

**Étape 1 : Définir une fonction Python**
- Convertir en uint8 [0, 255]
- Si RGB : convertir en grayscale
- Appliquer `cv2.equalizeHist(gray)`
- Retourner en float32 [0, 1]

**Étape 2 : Wrapper avec tf.py_function**

**Étape 3 : Retourner le résultat**

**Effet** : Améliore le contraste en redistribuant les intensités de pixels.

**Formule** :
```
Pour chaque intensité i :
  cdf(i) = somme cumulative de l'histogramme jusqu'à i
  nouvelle_valeur(i) = (cdf(i) - cdf_min) / (total_pixels - cdf_min) * 255
```

---

#### **Classe : SharpenTf**

**Attributs** :
- `name` = "Sharpen"

**Méthode apply(img)** :

**Étape 1 : Définir le noyau de convolution**
```
kernel = [[ 0, -1,  0],
          [-1,  5, -1],
          [ 0, -1,  0]]
```

**Étape 2 : Convertir en tensor TensorFlow**
- Shape : (3, 3, 1, 1) pour convolution 2D

**Étape 3 : Appliquer la convolution**
- Si RGB : appliquer sur chaque canal séparément
- Utiliser `tf.nn.conv2d` avec padding='SAME'

**Étape 4 : Retourner le résultat**

**Effet** : Accentue les détails et les contours de l'image.

**Principe** : Le noyau amplifie les différences entre pixels voisins.

---

#### **Classe : ThresholdTf**

**Attributs** :
- `threshold` : Float entre 0 et 1 (ex: 0.5)
- `name` = "Threshold"

**Méthode apply(img)** :

**Étape 1 : Convertir en grayscale si RGB**
- Si shape[-1] == 3 : utiliser `tf.image.rgb_to_grayscale`

**Étape 2 : Appliquer le seuillage**
- Comparer chaque pixel au threshold
- Si pixel > threshold : 1.0 (blanc)
- Sinon : 0.0 (noir)
- Utiliser : `tf.cast(gray > threshold, tf.float32)`

**Étape 3 : Retourner l'image binaire**

**Effet** : Segmentation binaire de l'image.

**Formule** :
```
output(x, y) = 1 si input(x, y) > threshold, sinon 0
```

---

#### **Classe : MorphologyTf**

**Attributs** :
- `mode` : String ("erode", "dilate", "open", "close")
- `kernel_size` : Entier (ex: 5)
- `name` = "Morphology"

**Méthode apply(img)** :

**Étape 1 : Définir une fonction Python**
- Convertir en uint8 [0, 255]
- Si RGB : convertir en grayscale
- Créer un élément structurant (kernel) :
  - Utiliser `cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))`
- Selon le mode :
  - "erode" : `cv2.erode(gray, kernel)`
  - "dilate" : `cv2.dilate(gray, kernel)`
  - "open" : `cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)`
  - "close" : `cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)`
- Retourner en float32 [0, 1]

**Étape 2 : Wrapper avec tf.py_function**

**Étape 3 : Retourner le résultat**

**Effet** :
- **Érosion** : Réduit les objets blancs, élimine le bruit
- **Dilatation** : Agrandit les objets blancs, comble les trous
- **Opening** : Érosion puis dilatation (enlève le bruit)
- **Closing** : Dilatation puis érosion (comble les trous)

**Formules** :
```
Érosion : output(x,y) = min{input(x+i, y+j) | (i,j) ∈ kernel}
Dilatation : output(x,y) = max{input(x+i, y+j) | (i,j) ∈ kernel}
```

---

### **Classe 2 : BatchTransformer**

**Responsabilité** : Appliquer des transformations à tout un dossier d'images.

---

#### **Méthode : __init__(engine, path_manager)**

**Ce qu'elle doit faire** :
1. Recevoir un TransformationEngine
2. Recevoir un PathManager
3. Stocker comme attributs

---

#### **Méthode : run(src, dst, recursive)**

**Ce qu'elle doit faire** :

**Paramètres** :
- `src` : Path vers le dossier source
- `dst` : Path vers le dossier destination
- `recursive` : Booléen

---

**Étape 1 : Créer le dossier destination**
- Utiliser `path_manager.ft_ensure_dir(dst)`

---

**Étape 2 : Lister toutes les images**
- Utiliser `path_manager.ft_iter_images(src, recursive)`

---

**Étape 3 : Pour chaque image**

**Sous-étape 3.1 : Charger l'image**
- Lire le fichier avec `tf.io.read_file`
- Décoder avec `tf.image.decode_jpeg`
- Normaliser [0, 1]

**Sous-étape 3.2 : Appliquer toutes les transformations**
- Utiliser `engine.apply_all(img)`
- Résultat : dictionnaire {nom: image_transformée}

**Sous-étape 3.3 : Sauvegarder chaque transformation**
- Pour chaque (nom, img_tf) dans le dictionnaire :
  - Calculer le chemin relatif de l'image par rapport à src
  - Créer le même chemin relatif dans dst
  - Ajouter le suffixe du nom de transformation
  - Créer les dossiers nécessaires
  - Sauvegarder l'image transformée

---

**Étape 4 : Afficher un message de progression**
- Optionnel : afficher combien d'images ont été traitées

---

<a id="model"></a>
## 7. leaffliction/model.py — Modèle et encodage

### **Classe 1 : ModelConfig (dataclass)**

**Responsabilité** : Stocker la configuration du modèle.

---

#### **Attributs**

**Ce qu'elle doit contenir** :
- `img_size` : Tuple (hauteur, largeur) (ex: (224, 224))
- `num_classes` : Entier, nombre de classes
- `seed` : Entier pour reproductibilité
- `framework` : String, toujours "tf" pour TensorFlow
- `extra` : Dictionnaire pour paramètres additionnels

---

### **Classe 2 : ModelPaths (dataclass)**

**Responsabilité** : Définir les noms de fichiers pour sauvegarder le modèle.

---

#### **Attributs**

**Ce qu'elle doit contenir** :
- `model_file` : String, nom du fichier modèle (ex: "model.keras")
- `labels_file` : String, nom du fichier labels (ex: "labels.json")
- `config_file` : String, nom du fichier config (ex: "config.json")
- `preprocess_file` : String, nom du fichier preprocess (ex: "preprocess.json")

---

### **Classe 3 : LabelEncoder**

**Responsabilité** : Gérer le mapping bidirectionnel entre noms de classes et IDs.

---

#### **Méthode : __init__()**

**Ce qu'elle doit faire** :
1. Créer un dictionnaire vide `class_to_id` : {nom_classe → id}
2. Créer un dictionnaire vide `id_to_class` : {id → nom_classe}

---

#### **Méthode : fit(class_names)**

**Ce qu'elle doit faire** :
1. Recevoir une liste de noms de classes (triée)
2. Pour chaque nom avec son index :
   - Ajouter au dictionnaire `class_to_id` : {nom → index}
   - Ajouter au dictionnaire `id_to_class` : {index → nom}

**Exemple** :
```
class_names = ["Apple_Black_rot", "Apple_healthy", "Grape_Black_rot"]

Résultat :
class_to_id = {
    "Apple_Black_rot": 0,
    "Apple_healthy": 1,
    "Grape_Black_rot": 2
}
id_to_class = {
    0: "Apple_Black_rot",
    1: "Apple_healthy",
    2: "Grape_Black_rot"
}
```

---

#### **Méthode : encode(class_name)**

**Ce qu'elle doit faire** :
1. Recevoir un nom de classe (string)
2. Chercher dans `class_to_id`
3. Retourner l'ID correspondant
4. Si non trouvé : lever une erreur

---

#### **Méthode : decode(class_id)**

**Ce qu'elle doit faire** :
1. Recevoir un ID (entier)
2. Chercher dans `id_to_class`
3. Retourner le nom de classe correspondant
4. Si non trouvé : lever une erreur

---

#### **Méthode : to_json_dict()**

**Ce qu'elle doit faire** :
1. Créer un dictionnaire avec :
   - "class_to_id" : self.class_to_id
   - "id_to_class" : self.id_to_class (convertir les clés int en string)
2. Retourner ce dictionnaire

**Pourquoi convertir les clés** : JSON n'accepte que des clés string.

---

#### **Méthode : from_json_dict(data)** (classmethod)

**Ce qu'elle doit faire** :
1. Créer une nouvelle instance de LabelEncoder
2. Charger `class_to_id` depuis data
3. Charger `id_to_class` depuis data (convertir les clés string en int)
4. Retourner l'instance

---

### **Classe 4 : ModelFactory**

**Responsabilité** : Construire l'architecture du modèle CNN.

---

#### **Méthode : build(cfg)**

**Ce qu'elle doit faire** :

**Paramètres** :
- `cfg` : ModelConfig

---

**Étape 1 : Charger le backbone pré-entraîné**
- Utiliser `keras.applications.MobileNetV2` (ou EfficientNet, ResNet, etc.)
- Paramètres :
  - `input_shape` : (*cfg.img_size, 3)
  - `include_top` : False (on ne veut pas la couche de classification)
  - `weights` : 'imagenet' (poids pré-entraînés)
- Figer les poids : `backbone.trainable = False`

**Pourquoi figer** : Transfer learning - on utilise les features apprises sur ImageNet.

---

**Étape 2 : Construire le modèle complet**

**Sous-étape 2.1 : Créer l'input**
- `inputs = layers.Input(shape=(*cfg.img_size, 3))`

**Sous-étape 2.2 : Passer par le backbone**
- `x = backbone(inputs, training=False)`
- Résultat : features maps (7, 7, 1280) pour MobileNetV2

**Sous-étape 2.3 : Global Average Pooling**
- `x = layers.GlobalAveragePooling2D()(x)`
- Résultat : vecteur (1280,)

**Pourquoi GAP** : Réduit les dimensions spatiales en une se
ule valeur par channel.

**Sous-étape 2.4 : Dropout**
- `x = layers.Dropout(0.2)(x)`
- Résultat : vecteur (1280,) avec dropout

**Pourquoi Dropout** : Régularisation pour éviter l'overfitting.

**Sous-étape 2.5 : Couche de classification**
- `outputs = layers.Dense(cfg.num_classes, activation='softmax')(x)`
- Résultat : vecteur (num_classes,) avec probabilités

**Pourquoi softmax** : Transforme les logits en probabilités qui somment à 1.

---

**Étape 3 : Créer le modèle Keras**
- `model = keras.Model(inputs, outputs)`

---

**Étape 4 : Compiler le modèle**
- Optimizer : Adam avec learning rate de cfg
- Loss : sparse_categorical_crossentropy (labels sont des entiers)
- Metrics : accuracy

---

**Étape 5 : Retourner le modèle**

---

**Architecture complète** :
```
Input (224, 224, 3)
    ↓
MobileNetV2 (frozen)
    ↓
GlobalAveragePooling2D
    ↓
Dropout(0.2)
    ↓
Dense(num_classes, softmax)
```

---

### **Classe 5 : ModelBundle**

**Responsabilité** : Encapsuler tout ce qui est nécessaire pour sauvegarder et charger un modèle.

---

#### **Méthode : __init__(model, labels, cfg, preprocess, paths)**

**Ce qu'elle doit faire** :
1. Recevoir un modèle Keras
2. Recevoir un LabelEncoder
3. Recevoir une ModelConfig
4. Recevoir un dictionnaire preprocess (optionnel)
5. Recevoir un ModelPaths (optionnel, créer par défaut)
6. Stocker tous ces objets comme attributs

---

#### **Méthode : save(out_dir)**

**Ce qu'elle doit faire** :

**Paramètres** :
- `out_dir` : Path vers le dossier de sortie

---

**Étape 1 : Créer le dossier de sortie**
- S'assurer que out_dir existe

---

**Étape 2 : Sauvegarder le modèle**
- Utiliser `model.save(out_dir / self.paths.model_file)`
- Format : .keras (nouveau format Keras 3)

---

**Étape 3 : Sauvegarder les labels**
- Convertir labels en dictionnaire avec `labels.to_json_dict()`
- Écrire en JSON dans out_dir / self.paths.labels_file

---

**Étape 4 : Sauvegarder la config**
- Convertir cfg en dictionnaire
- Écrire en JSON dans out_dir / self.paths.config_file

---

**Étape 5 : Sauvegarder preprocess**
- Écrire self.preprocess en JSON dans out_dir / self.paths.preprocess_file

---

#### **Méthode : load(in_dir)** (classmethod)

**Ce qu'elle doit faire** :

**Paramètres** :
- `in_dir` : Path vers le dossier contenant les fichiers

---

**Étape 1 : Charger le modèle**
- Utiliser `keras.models.load_model(in_dir / "model.keras")`

---

**Étape 2 : Charger les labels**
- Lire le JSON
- Créer un LabelEncoder avec `LabelEncoder.from_json_dict(data)`

---

**Étape 3 : Charger la config**
- Lire le JSON
- Créer un ModelConfig à partir du dictionnaire

---

**Étape 4 : Charger preprocess**
- Lire le JSON

---

**Étape 5 : Créer et retourner un ModelBundle**
- Avec tous les objets chargés

---

#### **Méthode : load_from_zip(zip_path, extract_dir)** (classmethod)

**Ce qu'elle doit faire** :
1. Extraire le ZIP dans extract_dir
2. Appeler `load(extract_dir)`
3. Retourner le ModelBundle

---

<a id="train-pipeline"></a>
## 8. leaffliction/train_pipeline.py — Pipeline d'entraînement

### **Classe 1 : TrainConfig (dataclass)**

**Responsabilité** : Stocker tous les hyperparamètres d'entraînement.

---

#### **Attributs**

**Ce qu'elle doit contenir** :
- `epochs` : Entier, nombre d'epochs
- `batch_size` : Entier, taille des batchs
- `lr` : Float, learning rate
- `valid_ratio` : Float, ratio de validation (ex: 0.2)
- `seed` : Entier, seed pour reproductibilité
- `img_size` : Tuple (H, W)
- `augment_in_train` : Booléen, activer les augmentations
- `export_increased_images` : Booléen, exporter les images augmentées
- `extra` : Dictionnaire pour paramètres additionnels

---

### **Classe 2 : Metrics (dataclass)**

**Responsabilité** : Stocker les métriques d'entraînement.

---

#### **Attributs**

**Ce qu'elle doit contenir** :
- `train_accuracy` : Float, accuracy sur train
- `valid_accuracy` : Float, accuracy sur validation
- `valid_count` : Entier, nombre d'images de validation
- `notes` : Dictionnaire pour informations additionnelles

---

### **Classe 3 : Trainer**

**Responsabilité** : Orchestrer tout le processus d'entraînement.

---

#### **Méthode : __init__(dataset_scanner, dataset_splitter, model_factory, labels)**

**Ce qu'elle doit faire** :
1. Recevoir un DatasetScanner
2. Recevoir un DatasetSplitter
3. Recevoir un ModelFactory
4. Recevoir un LabelEncoder
5. Stocker comme attributs

---

#### **Méthode : train(dataset_dir, out_dir, cfg)**

**Ce qu'elle doit faire** :

**Paramètres** :
- `dataset_dir` : Path vers le dataset
- `out_dir` : Path vers le dossier de sortie
- `cfg` : TrainConfig

---

**Étape 1 : Scanner le dataset**
- Utiliser `dataset_scanner.ft_scan(dataset_dir)`
- Résultat : DatasetIndex

---

**Étape 2 : Fitter le LabelEncoder**
- Utiliser `labels.fit(index.class_names)`

---

**Étape 3 : Splitter train/valid**
- Utiliser `dataset_splitter.ft_split(index.items, cfg.valid_ratio, cfg.seed, stratified=True)`
- Résultat : (train_items, valid_items)

---

**Étape 4 : Créer l'augmenteur (si demandé)**
- Si cfg.augment_in_train :
  - Créer un KerasAugmentationsFactory
  - Appeler `factory.build()`
  - Résultat : keras.Sequential d'augmentations
- Sinon : None

---

**Étape 5 : Construire les tf.data.Dataset**
- Créer une TFDataConfig avec les paramètres de cfg
- Créer un TFDatasetBuilder avec la config et l'augmenteur
- Construire train_ds : `builder.build(train_items, training=True)`
- Construire valid_ds : `builder.build(valid_items, training=False)`

---

**Étape 6 : Construire le modèle**
- Créer une ModelConfig avec img_size et num_classes
- Utiliser `model_factory.build(model_cfg)`
- Résultat : modèle Keras compilé

---

**Étape 7 : Créer les callbacks**
- Utiliser KerasCallbacksFactory pour créer :
  - EarlyStopping
  - ModelCheckpoint
  - ReduceLROnPlateau
  - TensorBoard (optionnel)

---

**Étape 8 : Entraîner le modèle**
- Appeler `model.fit(train_ds, validation_data=valid_ds, epochs=cfg.epochs, callbacks=callbacks)`
- Résultat : history

---

**Étape 9 : Évaluer sur validation**
- Appeler `model.evaluate(valid_ds)`
- Extraire la loss et l'accuracy

---

**Étape 10 : Créer les métriques**
- Créer un objet Metrics avec :
  - train_accuracy : depuis history
  - valid_accuracy : depuis evaluate
  - valid_count : len(valid_items)

---

**Étape 11 : Sauvegarder le ModelBundle**
- Créer un ModelBundle avec le modèle, labels, config
- Appeler `bundle.save(out_dir / "model")`

---

**Étape 12 : Exporter les images augmentées (si demandé)**
- Si cfg.export_increased_images :
  - Pour un échantillon d'images :
    - Appliquer les augmentations
    - Sauvegarder dans out_dir / "augmented"

---

**Étape 13 : Retourner les métriques**

---

### **Classe 4 : RequirementsGate**

**Responsabilité** : Valider que les contraintes du sujet sont respectées.

---

#### **Méthode : assert_ok(metrics)**

**Ce qu'elle doit faire** :

**Paramètres** :
- `metrics` : Objet Metrics

---

**Vérification 1 : Accuracy > 90%**
- Si metrics.valid_accuracy < 0.90 :
  - Lever une ValueError avec un message clair
  - Exemple : "Validation accuracy 87.5% < 90%. Training failed."

---

**Vérification 2 : Validation set >= 100 images**
- Si metrics.valid_count < 100 :
  - Lever une ValueError
  - Exemple : "Validation set has 85 images < 100."

---

**Si tout est OK**
- Afficher un message de succès
- Retourner (ou ne rien faire)

---

### **Classe 5 : TrainingPackager**

**Responsabilité** : Préparer les artefacts et créer le ZIP final.

---

#### **Méthode : __init__(zip_packager)**

**Ce qu'elle doit faire** :
1. Recevoir un ZipPackager
2. Stocker comme attribut

---

#### **Méthode : prepare_artifacts_dir(tmp_dir)**

**Ce qu'elle doit faire** :

**Paramètres** :
- `tmp_dir` : Path vers le dossier temporaire

---

**Étape 1 : Créer un dossier artifacts**
- Créer tmp_dir / "artifacts" s'il n'existe pas

---

**Étape 2 : Copier les fichiers nécessaires**
- Copier model/ vers artifacts/model/
- Copier augmented/ vers artifacts/augmented/ (si existe)

---

**Étape 3 : Retourner le chemin vers artifacts**

---

#### **Méthode : build_zip(artifacts_dir, out_zip)**

**Ce qu'elle doit faire** :
1. Utiliser `zip_packager.ft_zip_dir(artifacts_dir, out_zip)`
2. Afficher un message de succès

---

### **Classe 6 : KerasCallbacksFactory**

**Responsabilité** : Créer les callbacks Keras pour améliorer l'entraînement.

---

#### **Méthode : build(out_dir)**

**Ce qu'elle doit faire** :

**Paramètres** :
- `out_dir` : Path pour sauvegarder les checkpoints

---

**Callback 1 : EarlyStopping**
- Monitor : 'val_accuracy'
- Patience : 5 epochs
- Restore best weights : True
- Verbose : 1

**Effet** : Arrête l'entraînement si pas d'amélioration pendant 5 epochs.

---

**Callback 2 : ModelCheckpoint**
- Filepath : out_dir / "best_model.keras"
- Monitor : 'val_accuracy'
- Save best only : True
- Verbose : 1

**Effet** : Sauvegarde le meilleur modèle automatiquement.

---

**Callback 3 : ReduceLROnPlateau**
- Monitor : 'val_loss'
- Factor : 0.5 (divise le LR par 2)
- Patience : 3 epochs
- Min LR : 1e-7
- Verbose : 1

**Effet** : Réduit le learning rate si plateau détecté.

---

**Callback 4 : TensorBoard (optionnel)**
- Log dir : out_dir / "tensorboard"

**Effet** : Permet de visualiser l'entraînement avec TensorBoard.

---

**Retourner la liste des callbacks**

---

<a id="predict-pipeline"></a>
## 9. leaffliction/predict_pipeline.py — Pipeline de prédiction

### **Classe 1 : PredictConfig (dataclass)**

**Responsabilité** : Stocker la configuration pour la prédiction.

---

#### **Attributs**

**Ce qu'elle doit contenir** :
- `show_transforms` : Booléen, afficher les transformations
- `top_k` : Entier, nombre de prédictions à afficher
- `extra` : Dictionnaire pour paramètres additionnels

---

### **Classe 2 : Predictor**

**Responsabilité** : Charger le modèle et prédire sur une image.

---

#### **Méthode : __init__(bundle_loader, transformations_engine)**

**Ce qu'elle doit faire** :
1. Recevoir une classe bundle_loader (ModelBundle)
2. Recevoir un TransformationEngine
3. Stocker comme attributs

---

#### **Méthode : predict(bundle_zip, image_path, cfg)**

**Ce qu'elle doit faire** :

**Paramètres** :
- `bundle_zip` : Path vers learnings.zip
- `image_path` : Path vers l'image à prédire
- `cfg` : PredictConfig

---

**Étape 1 : Charger le ModelBundle**
- Créer un dossier temporaire pour extraction
- Utiliser `bundle_loader.load_from_zip(bundle_zip, temp_dir)`
- Résultat : bundle avec modèle, labels, config

---

**Étape 2 : Charger l'image**
- Lire le fichier avec `tf.io.read_file`
- Décoder avec `tf.image.decode_jpeg`
- Resize à bundle.cfg.img_size
- Normaliser [0, 1]

---

**Étape 3 : Appliquer les transformations (si demandé)**
- Si cfg.show_transforms :
  - Utiliser `transformations_engine.apply_all(img)`
  - Stocker les résultats pour affichage

---

**Étape 4 : Préparer l'image pour prédiction**
- Ajouter une dimension batch : `img_batch = tf.expand_dims(img, 0)`
- Résultat : (1, H, W, 3)

---

**Étape 5 : Prédire**
- Appeler `bundle.model.predict(img_batch)`
- Résultat : probabilités (1, num_classes)

---

**Étape 6 : Extraire les probabilités**
- Squeeze pour enlever la dimension batch : `probs = predictions[0]`

---

**Étape 7 : Trouver la classe prédite**
- Trouver l'index du maximum : `class_id = np.argmax(probs)`
- Décoder avec `bundle.labels.decode(class_id)`
- Résultat : nom de la classe

---

**Étape 8 : Créer le dictionnaire de probabilités**
- Pour chaque classe :
  - Créer {nom_classe: probabilité}
- Trier par probabilité décroissante
- Garder seulement les top_k

---

**Étape 9 : Retourner**
- Retourner (predicted_label, probs_dict)

---

### **Classe 3 : PredictionVisualiser**

**Responsabilité** : Afficher visuellement le résultat de la prédiction.

---

#### **Méthode : show(original, transformed, predicted_label)**

**Ce qu'elle doit faire** :

**Paramètres** :
- `original` : Image originale
- `transformed` : Dictionnaire {nom: image_transformée}
- `predicted_label` : String, classe prédite

---

**Étape 1 : Calculer le layout**
- Nombre total d'images : 1 (original) + len(transformed)
- Calculer rows et cols

---

**Étape 2 : Créer la figure**
- Créer une grille de subplots

---

**Étape 3 : Afficher l'original**
- Titre : f"Original - Predicted: {predicted_label}"
- Utiliser un cadre vert pour indiquer la prédiction

---

**Étape 4 : Afficher les transformations**
- Pour chaque transformation :
  - Afficher avec son nom comme titre

---

**Étape 5 : Ajuster et afficher**
- tight_layout()
- show()

---

**Pourquoi afficher les transformations** : Permet de voir comment le modèle "voit" l'image après différentes transformations, utile pour le debugging.

---

## 🎯 Conclusion du guide d'implémentation

### Ce que vous avez maintenant

**Un guide complet qui explique** :
1. ✅ **Quoi faire** : Chaque classe et méthode est décrite
2. ✅ **Pourquoi le faire** : Les justifications sont données
3. ✅ **Comment le faire** : Les étapes sont détaillées
4. ✅ **Sans code** : Vous pouvez implémenter vous-même

### Comment utiliser ce guide

**Pour chaque fichier à implémenter** :
1. Lire la section correspondante dans ce guide
2. Comprendre la responsabilité de chaque classe
3. Suivre les étapes décrites pour chaque méthode
4. Implémenter dans votre propre style
5. Tester votre implémentation

### Ordre d'implémentation recommandé

**Phase 1 : Fondations**
1. `utils.py` (PathManager, Hasher, ZipPackager)
2. `cli.py` (tous les parsers)

**Phase 2 : Dataset**
3. `dataset.py` (DatasetIndex, Scanner, Splitter, TFDatasetBuilder)
4. `plotting.py` (DistributionPlotter, GridPlotter)

**Phase 3 : Transformations**
5. `augmentations.py` (6 augmentations + Engine + Saver)
6. `transformations.py` (6 transformations + Engine + BatchTransformer)

**Phase 4 : Modèle**
7. `model.py` (LabelEncoder, ModelFactory, ModelBundle)

**Phase 5 : Pipelines**
8. `train_pipeline.py` (Trainer, RequirementsGate, Packager)
9. `predict_pipeline.py` (Predictor, Visualiser)

**Phase 6 : Scripts racine**
10. `Distribution.py`
11. `Augmentation.py`
12. `Transformation.py`
13. `train.py`
14. `predict.py`

### Points critiques à ne pas oublier

**Reproductibilité** :
- Toujours fixer les seeds (numpy, tensorflow, random)
- Utiliser le même ordre de tri partout

**Validation** :
- Split stratifié obligatoire
- Accuracy > 90% obligatoire
- Validation set >= 100 images obligatoire

**Sauvegarde** :
- Chemins relatifs dans les ZIP
- SHA1 correct pour signature.txt
- Tous les fichiers nécessaires dans le bundle

**Performance** :
- Utiliser AUTOTUNE pour tf.data
- Prefetch pour ne pas bloquer le GPU
- Cache si le dataset tient en RAM

### Ressources complémentaires

**Documentation officielle** :
- TensorFlow : https://www.tensorflow.org/api_docs
- Keras : https://keras.io/api/
- OpenCV : https://docs.opencv.org/

**Pour la soutenance** :
- Référez-vous au GUIDE.md pour les formules mathématiques
- Préparez des exemples de résultats
- Soyez capable d'expliquer chaque choix d'architecture

---

**Bon courage pour l'implémentation ! 🚀**

---

> **Note finale** : Ce guide est votre feuille de route. Chaque section est conçue pour que vous puissiez implémenter le code vous-même en comprenant exactement ce que chaque partie doit faire. Si vous suivez ce guide étape par étape, vous aurez un projet Leaffliction complet et fonctionnel.
