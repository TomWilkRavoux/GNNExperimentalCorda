# GCN — Graph Convolutional Network

## Principe

Le GCN (Kipf & Welling, 2017) applique des convolutions spectrales simplifiées sur un graphe. Chaque couche agrège les features des voisins d'un nœud en les pondérant par le degré du nœud, puis applique une transformation linéaire.

La formule d'une couche :

```
H(l+1) = σ(D̃⁻¹/² Ã D̃⁻¹/² H(l) W(l))
```

Où `Ã = A + I` (matrice d'adjacence + self-loops), `D̃` est la matrice de degré de `Ã`, et `W` les poids apprenables.

## Pourquoi ce modèle sur Cora

Le GCN est le modèle de référence pour la classification de nœuds sur des graphes de citations. Cora est un graphe homophile (les nœuds connectés tendent à partager la même classe), ce qui est exactement le cas où l'agrégation par voisinage du GCN est efficace.

C'est aussi le **baseline** contre lequel on compare les autres architectures.

## Architecture

| Paramètre | Valeur |
|---|---|
| Couches | 2 (GCNConv) |
| Hidden channels | 64 |
| Activation | ReLU (entre couches) |
| Dropout | 0.5 |
| Sortie | 7 classes (log-softmax implicite via cross-entropy) |

## Hyperparamètres d'entraînement

| Paramètre | Valeur | Justification |
|---|---|---|
| Learning rate | 0.005 | Plus bas que le 0.01 standard pour stabiliser la convergence avec des hidden channels plus grands |
| Weight decay | 5e-4 | Régularisation L2 standard pour éviter l'overfitting |
| Epochs max | 500 | Suffisant pour converger avec early stopping |
| Early stopping patience | 30 | Arrête si la validation ne s'améliore plus pendant 30 epochs |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=10) | Réduit le lr de moitié quand la validation stagne |

## Limites

- **Over-smoothing** : avec plus de 2 couches, les représentations des nœuds convergent toutes vers la même valeur, ce qui dégrade les performances. C'est pourquoi on est limité à 2 couches.
- **Poids fixes** : tous les voisins sont traités de la même façon (pondérés uniquement par le degré). Pas de mécanisme d'attention pour distinguer les voisins importants.

## Accuracy cible

~84-85% sur le test set de Cora.
