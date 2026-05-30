# GCN3 — GCN 3 couches avec Skip Connections

## Principe

Le GCN3 est une extension du GCN classique à **3 couches** avec une **skip connection** (connexion résiduelle) entre la 1ère et la 2ème couche. L'idée est d'augmenter la profondeur du réseau (et donc sa capacité à capturer des structures complexes) tout en contrant le problème d'**over-smoothing**.

### Le problème de l'over-smoothing

Dans un GCN standard, chaque couche mélange les features d'un nœud avec celles de ses voisins. Avec 2 couches, chaque nœud "voit" ses voisins à distance 2. Avec 3 couches, il voit ses voisins à distance 3. Le problème : plus on empile de couches, plus les représentations de tous les nœuds convergent vers la même valeur. C'est l'over-smoothing.

### La solution : skip connections

Inspirée de ResNet (He et al., 2015), la skip connection ajoute directement la sortie de la couche 1 à la sortie de la couche 2 :

```
h = conv2(x) + x   ← skip connection
output = conv3(h)
```

Cela permet au réseau de conserver l'information des couches précédentes même si les couches plus profondes "lissent" trop les features.

## Pourquoi ce modèle sur Cora

Le GCN 2 couches est limité par sa profondeur : il ne peut capturer que des relations à distance 2 dans le graphe. Le GCN3 étend cette portée à distance 3, ce qui peut aider pour les nœuds qui sont mal connectés localement mais qui partagent des voisins distants avec leur classe.

La skip connection permet de bénéficier de cette profondeur supplémentaire sans payer le prix de l'over-smoothing.

## Architecture

| Paramètre | Valeur |
|---|---|
| Couches | 3 (GCNConv) |
| Couche 1 | input → 64 |
| Couche 2 | 64 → 64 (+ skip connection) |
| Couche 3 | 64 → 7 classes |
| Activation | ReLU (entre couches) |
| Dropout | 0.5 (après chaque couche cachée) |

### Détail du forward pass

```
x = relu(conv1(input))        # [N, 64]
x = dropout(x, 0.5)
h = relu(conv2(x))            # [N, 64]
h = dropout(h, 0.5)
x = x + h                     # skip connection
output = conv3(x)              # [N, 7]
```

## Hyperparamètres d'entraînement

| Paramètre | Valeur | Justification |
|---|---|---|
| Learning rate | 0.005 | Comme le GCN, stabilité avec des hidden channels larges |
| Weight decay | 5e-4 | Régularisation standard |
| Epochs max | 500 | Avec early stopping |
| Early stopping patience | 30 | |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=10) | |

## Limites

- **Une seule skip connection** : on pourrait en ajouter d'autres (dense connections comme DenseGCN) pour des architectures encore plus profondes.
- **Même limitation d'attention que le GCN** : pas de pondération des voisins.
- **Overhead marginal** : une couche supplémentaire signifie plus de paramètres et un temps d'entraînement légèrement plus long.

## Accuracy cible

~84-85% sur le test set de Cora (similaire au GCN mais plus stable grâce à la skip connection).
