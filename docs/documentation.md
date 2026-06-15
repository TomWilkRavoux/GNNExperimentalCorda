# Documentation — Expérimentations GNN sur Cora

Documentation **fonctionnelle** et **technique** des notebooks du projet et des
différents entraînements de Graph Neural Networks (GNN) explorés. Ce document
explique, pour chaque architecture, *comment* le modèle est construit et
*pourquoi* il est construit ainsi, puis dresse un comparatif des modèles à partir
des résultats réellement mesurés et en tire des conclusions.

> Documents associés (fiches détaillées par modèle) : [`gcn.md`](gcn.md),
> [`gat.md`](gat.md), [`graphsage.md`](graphsage.md), [`gcn3.md`](gcn3.md),
> [`comparaison.md`](comparaison.md).

---

## 1. Partie fonctionnelle

### 1.1 Objectif

Classifier automatiquement **2708 articles scientifiques** en **7 catégories**
thématiques, à partir :

- de leur **contenu** (vecteur de mots-clés), et
- de leur **réseau de citations** (qui cite qui).

L'hypothèse métier est qu'un article est mieux classé si l'on tient compte de
**ses voisins** dans le graphe de citations : deux articles qui se citent traitent
souvent du même sujet. C'est exactement ce qu'un GNN sait exploiter, là où un
classifieur classique (qui ne verrait que le texte) ignorerait la structure du
réseau.

### 1.2 Le dataset Cora

| Propriété | Valeur |
|---|---|
| Nœuds (articles) | 2708 |
| Arêtes (citations) | 5429 (~10556 en non-orienté) |
| Features par nœud | 1433 (bag-of-words binaire) |
| Classes | 7 |
| Nœuds d'entraînement | 140 (20 par classe) |
| Nœuds de validation | 500 |
| Nœuds de test | 1000 |

Les 7 classes : `Case_Based`, `Genetic_Alg`, `Neural_Nets`, `Probabilistic`,
`Reinf_Learn`, `Rule_Learn`, `Theory`.

**Point clé** : l'entraînement n'utilise que **140 nœuds labellisés** (≈5 % du
graphe). C'est un cas de **classification de nœuds semi-supervisée
transductive** — tout le graphe (features + arêtes) est visible pendant
l'entraînement, mais seuls quelques labels le sont. La capacité à propager
l'information le long des arêtes est donc déterminante.

Cora est un graphe **homophile** : les nœuds reliés partagent majoritairement la
même classe. C'est le terrain de jeu idéal pour les GNN à agrégation de voisinage.

### 1.3 Les notebooks

Le projet s'articule autour de deux notebooks **Marimo** (notebooks réactifs en
Python pur, versionnables comme du code).

| Notebook | Rôle fonctionnel |
|---|---|
| [`notebook/analyzeCora.py`](../notebook/analyzeCora.py) | Notebook principal : exploration du dataset, entraînement et évaluation des **4 modèles** (GCN, GAT, GraphSAGE, GCN3), visualisation interactive du graphe. |
| [`notebook/testExplainer.py`](../notebook/testExplainer.py) | **Explicabilité** : recharge le GCN entraîné et explique ses prédictions (quelles arêtes/citations comptent le plus) via GNNExplainer / PGExplainer. |

Le notebook principal suit un déroulé pédagogique : on part du modèle de référence
(GCN) puis on explore des variantes (V2 GAT, V3 GraphSAGE, V4 GCN3), chacune
introduisant **une idée architecturale** supplémentaire.

---

## 2. Partie technique

### 2.1 Stack

| Brique | Usage |
|---|---|
| PyTorch + **PyTorch Geometric** (PyG) | Définition et entraînement des GNN |
| Marimo | Notebooks réactifs |
| scikit-learn | `classification_report`, `confusion_matrix` |
| matplotlib / seaborn | Courbes de loss/accuracy, heatmaps |
| pyvis | Visualisation interactive du graphe (HTML) |

Dépendances déclarées dans [`pyproject.toml`](../pyproject.toml), gérées par `uv`
(roue PyTorch CUDA 13.0). Le code bascule automatiquement sur GPU si disponible
(`torch.device('cuda' if torch.cuda.is_available() else 'cpu')`).

### 2.2 Le principe commun : le message passing

Tous les modèles reposent sur le même mécanisme — le **message passing**. À chaque
couche, un nœud :

1. **collecte** les représentations de ses voisins,
2. les **agrège** (somme pondérée, moyenne, attention…),
3. **combine** le résultat avec sa propre représentation via une transformation
   linéaire apprenable + non-linéarité.

Empiler *k* couches revient à donner à chaque nœud une vue de son voisinage à
**distance *k***. Ce qui distingue les 4 architectures, c'est **la façon
d'agréger** les voisins — et c'est tout l'intérêt de la comparaison.

### 2.3 Pipeline d'entraînement (identique pour les 4 modèles)

Pour rendre la comparaison équitable, les 4 modèles partagent exactement la même
boucle d'entraînement et les mêmes mécanismes de régularisation :

| Élément | Valeur | Pourquoi |
|---|---|---|
| Optimiseur | Adam | Standard, robuste, peu de réglage |
| Loss | `cross_entropy` sur le `train_mask` | Classification multi-classes ; n'utilise que les 140 nœuds labellisés |
| Epochs max | 500 | Plafond large, la convergence est gérée par l'early stopping |
| **Early stopping** | patience 30 (100 pour SAGE) | Restaure les **meilleurs poids** (best val accuracy) → évite l'overfitting |
| **LR scheduler** | `ReduceLROnPlateau` (factor 0.5, patience 10) | Divise le LR par 2 quand la val stagne → réglage fin en fin d'entraînement |
| Weight decay | 5e-4 | Régularisation L2 |

Schéma de la boucle (commun) :

```
pour chaque epoch :
    train()                          # forward + cross_entropy + backward sur train_mask
    train_acc = evaluate(train_mask)
    val_acc   = evaluate(val_mask)
    scheduler.step(val_acc)
    si val_acc > meilleur :          # on mémorise le meilleur état
        sauvegarde des poids
    sinon : patience += 1            # early stopping
charge les meilleurs poids
test_acc = evaluate(test_mask)       # score final reporté
```

**Évaluation** : pour chaque modèle, on produit les **courbes loss/accuracy**, un
`classification_report` (precision/recall/F1 par classe) et une **matrice de
confusion** (heatmap). Les poids entraînés sont sauvegardés dans
[`models/`](../models) (`gcn_cora.pth`, `gat_cora.pth`, `sage_cora.pth`,
`gcn3_cora.pth`).

---

## 3. Les modèles : construction et justification

### 3.1 V1 — GCN (Graph Convolutional Network) · *baseline*

**Construction**

```python
class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels=64):
        self.conv1 = GCNConv(num_features, hidden_channels)   # 1433 → 64
        self.conv2 = GCNConv(hidden_channels, num_classes)    # 64 → 7

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x
```

Chaque couche `GCNConv` calcule `H' = σ(D̃^-1/2 Ã D̃^-1/2 H W)` : agrégation des
voisins **pondérée par le degré** (Kipf & Welling, 2017), puis projection linéaire.

**Pourquoi ainsi**

- **2 couches** : c'est le sweet spot sur Cora. Le diamètre informationnel utile
  est court, et au-delà de 2-3 couches le GCN souffre d'**over-smoothing** (toutes
  les représentations convergent vers la même valeur).
- **Dropout 0.5** : fort, car seulement 140 exemples d'entraînement → risque
  d'overfitting élevé.
- **hidden=64, lr=0.005** : un LR un peu plus bas que le 0.01 « par défaut » pour
  stabiliser la convergence avec une couche cachée plus large.
- **Rôle** : c'est le **point de référence** contre lequel on juge les variantes.

### 3.2 V2 — GAT (Graph Attention Network)

**Construction**

```python
class GAT(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels=16, heads=8):
        self.conv1 = GATConv(num_features, hidden_channels, heads=8, dropout=0.6)
        self.conv2 = GATConv(hidden_channels*8, num_classes, heads=1,
                             concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x
```

**Pourquoi ainsi**

- **Attention** : au lieu de pondérer les voisins par le degré (GCN), GAT
  **apprend** un coefficient d'attention par arête → il peut donner plus de poids
  aux citations « pertinentes » et ignorer le bruit.
- **8 têtes** d'attention en couche 1 (`16×8 = 128` dimensions concaténées) : on
  apprend 8 vues d'attention indépendantes, ce qui stabilise l'apprentissage ; en
  couche 2 on revient à 1 tête avec `concat=False` (moyenne) pour produire la
  sortie 7 classes.
- **dropout 0.6** (plus élevé) et **ELU** : conformément au papier d'origine
  (Veličković et al., 2018) ; GAT a beaucoup plus de paramètres, donc on
  régularise davantage.
- **Hypothèse** : l'attention devrait aider… mais sur un graphe homophile et avec
  140 labels, le surcroît de paramètres peut au contraire **nuire** (voir §4).

### 3.3 V3 — GraphSAGE (Sample and Aggregate)

**Construction**

```python
class GraphSAGE(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels=128):
        self.conv1 = SAGEConv(num_features, hidden_channels)   # 1433 → 128
        self.conv2 = SAGEConv(hidden_channels, num_classes)    # 128 → 7

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x
```

**Pourquoi ainsi**

- **Agrégation séparée** : `SAGEConv` concatène la représentation propre du nœud
  et la **moyenne** de ses voisins, avant projection. Le nœud conserve donc
  toujours sa propre information (contrairement au GCN qui la mélange).
- **Inductif** : SAGE est conçu pour **généraliser à des nœuds jamais vus** (via
  échantillonnage de voisinage) → pertinent pour des graphes dynamiques / en
  production, là où le GCN est transductif.
- **hidden=128** (plus large) et **lr=0.01** (plus haut) : SAGE supporte une
  capacité plus grande sans diverger.
- **patience early-stopping = 100** : SAGE converge plus lentement/irrégulièrement,
  on lui laisse plus de marge avant d'arrêter.

### 3.4 V4 — GCN3 (3 couches + skip connection)

**Construction**

```python
class GCN3(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels=64):
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        h = F.relu(self.conv2(x, edge_index))
        h = F.dropout(h, p=0.5, training=self.training)
        x = x + h                 # ← skip connection (résiduelle)
        x = self.conv3(x, edge_index)
        return x
```

**Pourquoi ainsi**

- **3 couches** = portée à distance 3 dans le graphe → aide les nœuds mal
  connectés localement.
- **Skip connection** (inspirée de ResNet) : `x = x + h` réinjecte la sortie de la
  couche 1 dans la couche 3, ce qui **contre l'over-smoothing** qui condamnerait un
  GCN à 3 couches « naïf ». Sans elle, ajouter une couche ferait *baisser* le
  score.
- **Hypothèse** : capter des relations plus longues devrait aider, mais Cora est
  assez dense → le gain attendu est marginal.

---

## 4. Comparatif des modèles

### 4.1 Configurations

| | GCN | GAT | GraphSAGE | GCN3 |
|---|---|---|---|---|
| Couches | 2 | 2 | 2 | 3 |
| Hidden | 64 | 16 × 8 têtes | 128 | 64 |
| Idée clé | conv. pondérée par degré | **attention** apprise | moyenne + concat de soi | **skip connection** |
| Activation | ReLU | ELU | ReLU | ReLU |
| Dropout | 0.5 | 0.6 | 0.5 | 0.5 |
| Learning rate | 0.005 | 0.005 | 0.01 | 0.005 |
| Early stopping | 30 | 30 | 100 | 30 |

### 4.2 Résultats mesurés (test set, 1000 nœuds)

Résultats issus de la dernière exécution du notebook (session Marimo) :

| Modèle | Accuracy test | Macro F1 | Weighted F1 |
|---|---|---|---|
| **GCN** | **0.806** | 0.80 | 0.81 |
| GAT | 0.779 | 0.78 | 0.78 |
| GraphSAGE | 0.790 | 0.79 | 0.79 |
| **GCN3** | **0.806** | 0.80 | 0.81 |

> ⚠️ Ces chiffres mesurés sont **en dessous** des valeurs « cibles » (~84-85 %)
> souvent citées et présentes dans les anciennes fiches. Surtout, **GAT n'est pas
> le meilleur ici** — il est même le plus faible. Voir l'analyse §4.4.

### 4.3 Analyse par classe (F1-score)

| Classe (support) | GCN | GAT | SAGE | GCN3 |
|---|---|---|---|---|
| Case_Based (130) | 0.73 | 0.71 | 0.71 | 0.72 |
| Genetic_Alg (91) | 0.85 | 0.81 | 0.79 | 0.79 |
| Neural_Nets (144) | 0.87 | 0.88 | **0.89** | **0.89** |
| Probabilistic (319) | 0.83 | 0.77 | 0.79 | 0.82 |
| Reinf_Learn (149) | 0.79 | 0.78 | 0.79 | 0.81 |
| Rule_Learn (103) | 0.77 | 0.77 | 0.79 | 0.78 |
| Theory (64) | 0.74 | 0.71 | 0.74 | 0.79 |

Lectures transverses :

- **`Neural_Nets`** est la classe la mieux reconnue partout (recall jusqu'à 0.94-0.97)
  : classe nombreuse et thématiquement bien séparée.
- **`Probabilistic`** (319 nœuds, la plus grosse) a une **forte precision mais un
  recall plus faible** (0.65-0.75) : le modèle est prudent dessus et en « rate » une
  partie qu'il redistribue vers les classes voisines (Theory, Case_Based).
- **`Theory`** (64 nœuds, la plus petite) et **`Case_Based`** sont les plus
  confondues — peu d'exemples, frontières thématiques floues. GCN3 est le seul à
  bien remonter `Theory` (F1 0.79) grâce à sa portée à 3 sauts.

### 4.4 Interprétation

- **GAT sous-performe** alors qu'il est théoriquement le plus expressif. Causes
  probables : (1) Cora est **homophile** → pondérer les voisins par degré (GCN)
  est déjà quasi-optimal, l'attention apporte peu ; (2) seulement **140 labels**
  d'entraînement pour un modèle à **beaucoup plus de paramètres** (8 têtes) →
  régime de sur-paramétrisation, l'attention est mal estimée et le modèle
  généralise moins bien. GAT brille surtout sur graphes **hétérophiles** ou riches
  en labels.
- **GCN et GCN3 sont à égalité (0.806)** : sur Cora, dense et homophile, ajouter
  une 3ᵉ couche n'apporte rien en accuracy globale — le gain est **marginal et
  qualitatif** (GCN3 répartit mieux les classes minoritaires, cf. `Theory`).
- **GraphSAGE (0.790)** est légèrement sous GCN : son vrai atout (inductif,
  scalable) ne se voit pas sur un petit graphe transductif. Sur Cora, il n'y a pas
  de nouveaux nœuds à généraliser, donc son avantage architectural ne s'exprime
  pas.
- **Le baseline gagne** : l'expérience illustre un résultat classique de la
  littérature GNN — sur Cora, **un GCN 2 couches bien régularisé est une référence
  difficile à battre**, et la complexité supplémentaire ne se traduit pas
  mécaniquement en gain.

---

## 5. Explicabilité (notebook `testExplainer.py`)

Au-delà de la performance, le second notebook répond à *« pourquoi le modèle
prédit-il cela ? »*. Il recharge le GCN entraîné (`models/gcn_cora.pth`) et utilise
le module `torch_geometric.explain`.

- **GNNExplainer** (présent en commentaire) : explique un nœud en cherchant le
  sous-graphe / les features minimaux qui préservent la prédiction.
- **PGExplainer** (actif) : entraîne un explainer paramétré sur les nœuds
  d'entraînement (phase 1), puis attribue à **chaque arête** une importance moyenne
  sur tout le graphe (phase 2).

Le résultat (`edge_importance`) est visualisé avec pyvis (`explanation_full.html`)
: les arêtes sont colorées par rang d'importance (dégradé rouge = citations les
plus influentes pour les décisions du modèle, gris = peu influentes). Cela permet
de **valider que le modèle s'appuie bien sur les citations pertinentes**, et non
sur du bruit structurel.

---

## 6. Conclusions

1. **Le modèle le plus simple est le plus efficace ici.** GCN (2 couches) et GCN3
   atteignent 0.806, devant GraphSAGE (0.790) et GAT (0.779). Sur un graphe
   homophile et peu labellisé comme Cora, la sophistication architecturale n'achète
   pas de performance.

2. **L'attention (GAT) n'est pas une amélioration universelle.** Son surcoût en
   paramètres se retourne contre elle dans un régime à 140 labels. Elle reste
   pertinente pour des graphes hétérophiles ou des datasets plus fournis.

3. **La profondeur n'aide qu'avec des skip connections — et ici à la marge.** GCN3
   égale GCN en accuracy globale mais **rééquilibre les classes minoritaires**
   (`Theory` : F1 0.74 → 0.79), ce qui peut justifier son usage si le recall sur
   les petites classes importe.

4. **Le choix dépend du contexte, pas du score brut :**
   - Prototypage rapide / meilleure base → **GCN**
   - Graphe dynamique, nouveaux nœuds, mise en production → **GraphSAGE**
   - Besoin de portée longue / classes minoritaires → **GCN3**
   - Graphe hétérophile, beaucoup de labels → **GAT**

5. **Pistes d'amélioration :** fixer une `seed` pour des comparaisons
   reproductibles ; moyenner sur plusieurs runs (l'écart GCN/GAT/SAGE est dans le
   bruit de quelques points) ; tester un GAT plus léger (moins de têtes, plus de
   dropout) ; exploiter l'explicabilité PGExplainer pour élaguer les arêtes peu
   informatives.

> **À corriger dans le README** : il mentionne `analyzeNasa.py` et un GCN
> `hidden=16 / lr=0.01 / 200 epochs`, ce qui ne correspond plus au notebook actuel
> (`analyzeCora.py`, `hidden=64 / lr=0.005 / 500 epochs + early stopping`).
