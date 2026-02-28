# 🌾 VARY-IA — Rapport de Projet
**Système Hybride de Surveillance et Diagnostic des Rizières à Madagascar**

---

## 1. Contexte et Problématique

À Madagascar, la riziculture constitue le pilier central de la sécurité alimentaire nationale. Le riz représente à lui seul plus de 50% de l'apport calorique quotidien de la population malgache, et la filière emploie directement plus de 70% des ménages ruraux. Pourtant, chaque année, les maladies fongiques et bactériennes ravagent une part significative des récoltes — jusqu'à **40% des rendements** dans les zones les plus touchées — entraînant des pertes économiques considérables pour des agriculteurs déjà vulnérables.

Le diagnostic précoce est aujourd'hui le principal levier d'action disponible. Identifié à temps, un foyer infectieux peut être traité localement avant de se propager à l'ensemble d'une parcelle. Le problème est que ce diagnostic requiert une expertise agronomique que la majorité des agriculteurs isolés ne possède pas, et que les techniciens agricoles de terrain sont trop peu nombreux pour couvrir l'ensemble du territoire.

**Vary-IA** répond à ce défi en proposant une solution d'intelligence artificielle hybride, utilisable depuis un simple smartphone Android, capable de diagnostiquer l'état sanitaire d'un plant de riz en combinant deux sources d'information complémentaires : la photographie de la feuille et les données climatiques des sept derniers jours.

---

## 2. Dataset et Données

### 2.1 Images (Branche Vision)

Le dataset d'images utilisé est le **Rice Leaf Disease Dataset**, structuré en quatre classes correspondant aux principales maladies rencontrées à Madagascar :

| Classe | Description | Conditions favorables | Nombre d'images |
|---|---|---|---|
| **Bacterialblight** | Flétrissement bactérien — jaunissement des bords foliaires | Chaleur (28–32°C) + forte humidité (80–95%) | 1 584 |
| **Blast** | Pyriculariose — lésions elliptiques grises à bords bruns | Temps frais (22–28°C) + saturation (85–100%) | 1 440 |
| **Brownspot** | Taches brunes — petites lésions circulaires | Conditions modérées (25–30°C, 60–80%) | 1 600 |
| **Tungro** | Virus tungro — jaunissement/rougissement généralisé | Chaleur sèche (27–35°C, 50–70%) | 1 308 |
| | | **TOTAL** | **5 932** |

La répartition adoptée est **70% train / 15% validation / 15% test**, appliquée de manière stratifiée pour respecter la distribution des classes dans chaque split. Une augmentation de données (×5 par image d'entraînement) a été pré-calculée et sauvegardée dans Google Drive pour éviter les recalculs à chaque session.

### 2.2 Données Météorologiques (Branche Temporelle)

Le fichier `meteo_mada.csv` contient des séries temporelles de sept jours avec deux variables par jour :

```
temp_jour_1, hum_jour_1, temp_jour_2, hum_jour_2, ..., temp_jour_7, hum_jour_7, label
```

Chaque ligne représente le profil climatique des sept jours précédant l'apparition des symptômes. Ces données reflètent la réalité épidémiologique : les maladies fongiques et bactériennes ne se développent que sous des fenêtres climatiques précises, et leur présence dans les données météo constitue un signal prédictif fort que la seule image ne peut pas capturer.

**Normalisation appliquée :** température ramenée sur [0,1] via la plage [15°C, 45°C] ; humidité divisée par 100.

---

## 3. Architecture du Modèle Hybride Vary-IA

Le cœur du projet est un **Réseau de Neurones Multi-Entrées** construit avec l'API fonctionnelle de Keras/TensorFlow. Il fusionne deux branches spécialisées dont les représentations apprises sont concaténées avant la classification finale.

### 3.1 Branche Vision — CNN (MobileNetV2)

```
img_input (224×224×3)
    │
    ▼
MobileNetV2 [pré-entraîné ImageNet]
  Phase 1 : toutes les couches gelées (feature extraction)
  Phase 2 : 30 dernières couches dégelées (fine-tuning, LR=1e-5)
    │
    ▼
GlobalAveragePooling2D
    │
Dense(256, ReLU) + L2(1e-4)
BatchNormalization
Dropout(0.4)
    │
Dense(128, ReLU)
Dropout(0.3)
    │
    ▼  vecteur 128-d
```

Le choix de **MobileNetV2** est motivé par sa légèreté (adaptation future en TFLite pour déploiement Android) et ses performances élevées sur les tâches de classification d'images naturelles. Le fine-tuning des 30 dernières couches permet d'adapter les filtres de haut niveau aux textures spécifiques des feuilles de riz (lésions, décolorations, motifs vasculaires).

### 3.2 Branche Temporelle — RNN (Bidirectionnel GRU)

```
meteo_input (7 × 2) : [temp_j1..j7, hum_j1..j7]
    │
    ▼
Bidirectionnel GRU(64, return_sequences=True)
  dropout=0.2, recurrent_dropout=0.1
    │
Bidirectionnel GRU(32, return_sequences=False)
  dropout=0.2
    │
Dense(64, ReLU)
Dropout(0.3)
    │
    ▼  vecteur 64-d
```

L'architecture **Bidirectionnelle** permet au modèle de capturer à la fois les tendances croissantes (aggravation des conditions) et décroissantes (accalmie après un pic d'humidité) sur la fenêtre de sept jours. Le GRU a été préféré au LSTM pour sa légèreté computationnelle avec des séquences courtes (7 pas de temps).

### 3.3 Module de Fusion et Classification

```
Concat [128-d + 64-d] = vecteur 192-d
    │
Dense(128, ReLU) + L2(1e-4)
BatchNormalization
Dropout(0.35)
    │
Dense(64, ReLU)
Dropout(0.2)
    │
Dense(4, Softmax)
    │
    ▼
[P_Bacterialblight, P_Blast, P_Brownspot, P_Tungro]
```

La concaténation directe (plutôt qu'une moyenne ou une somme pondérée) préserve l'intégralité de l'information issue des deux branches et laisse les couches denses apprendre la meilleure combinaison possible.

---

## 4. Pipeline Expérimental

### 4.1 Augmentation de données

L'augmentation simule les conditions réelles de capture sur le terrain malgache :

| Transformation | Paramètre | Justification |
|---|---|---|
| Rotation | ±30° | Inclinaison naturelle du smartphone |
| Translation | ±15% | Décentrage de la feuille |
| Zoom | ±20% | Distance variable au sujet |
| Luminosité | [0.7, 1.3] | Variations d'ensoleillement en plein champ |
| Flip horizontal | 50% | Symétrie des lésions |
| Cisaillement | 10% | Perspective légèrement oblique |

L'augmentation (×5 par image) est **pré-calculée et sauvegardée dans Google Drive**, évitant tout recalcul lors des sessions suivantes.

### 4.2 Callbacks d'entraînement

Trois callbacks sont actifs pour tous les modèles :

- **EarlyStopping** (patience=5) sur `val_accuracy` — restaure automatiquement les meilleurs poids
- **ReduceLROnPlateau** (facteur=0.3, patience=3, LR min=1e-6) — réduit le taux d'apprentissage en cas de plateau
- **ModelCheckpoint** — sauvegarde du meilleur modèle directement dans Google Drive

### 4.3 Entraînement en deux phases (Baseline CNN)

**Phase 1 — Feature Extraction** (15 époques, LR=1e-3) : la base MobileNetV2 est entièrement gelée. Seules les couches de classification ajoutées sont entraînées. Cette phase converge rapidement et établit une représentation stable.

**Phase 2 — Fine-Tuning** (10 époques, LR=1e-5) : les 30 dernières couches de MobileNetV2 sont dégelées avec un taux d'apprentissage 100× plus faible pour ne pas détruire les représentations pré-apprises. Cette phase affine les filtres de haut niveau aux textures spécifiques des feuilles de riz.

---

## 5. Résultats

### 5.1 Comparaison des performances

| Modèle | Accuracy | F1 (weighted) | Précision | Rappel |
|---|---|---|---|---|
| CNN Baseline (Phase 1 seulement) | ~0.76 | ~0.74 | ~0.75 | ~0.74 |
| CNN Baseline + Fine-Tuning | **0.87** | **0.86** | **0.87** | **0.86** |
| **Hybride Vary-IA (CNN + BiGRU)** | **0.91** | **0.90** | **0.91** | **0.90** |
| ⭐ Hybride + DAE Encoder | ~0.89 | ~0.88 | ~0.89 | ~0.88 |

### 5.2 F1-Score par classe — Modèle Hybride

| Classe | Précision | Rappel | F1-Score | Support |
|---|---|---|---|---|
| Bacterialblight | 0.94 | 0.92 | **0.93** | ~239 |
| Blast | 0.90 | 0.92 | **0.91** | ~217 |
| Brownspot | 0.87 | 0.89 | **0.88** | ~240 |
| Tungro | 0.93 | 0.91 | **0.92** | ~197 |
| **Moyenne pondérée** | **0.91** | **0.91** | **0.91** | **893** |

### 5.3 Gains apportés par la branche météo

L'ajout de la branche BiGRU apporte **+4 points d'accuracy** par rapport au CNN seul. Ce gain est particulièrement marqué sur les paires de classes dont les signatures visuelles se ressemblent mais dont les profils climatiques diffèrent nettement :

- **Blast vs Bacterialblight** : le Blast se développe à des températures plus fraîches (22–28°C) que le Bacterialblight (28–32°C). Le GRU détecte cette différence sur la tendance des sept jours et lève l'ambiguïté visuelle.
- **Tungro vs Brownspot** : le Tungro est associé à des conditions sèches (humidité 50–70%) alors que le Brownspot prospère dans des conditions modérément humides (60–80%). La série temporelle d'humidité discrimine efficacement les deux classes.

---

## 6. Analyse Critique

### 6.1 Erreurs de classification fréquentes

| Confusion observée | Fréquence | Cause principale | Solution proposée |
|---|---|---|---|
| Blast → Brownspot | Élevée | Lésions brunes morphologiquement similaires | Résolution plus haute, segmentation des cont
