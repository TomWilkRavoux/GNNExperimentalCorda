# GraphSAGE — Graph Sample and Aggregate

## Principe

GraphSAGE (Hamilton et al., 2017) utilise une approche **inductive** : au lieu d'apprendre des embeddings fixes pour chaque nœud (transductif), il apprend une **fonction d'agrégation** applicable à n'importe quel nœud, y compris des nœuds jamais vus pendant l'entraînement.

Chaque couche :
1. **Sample** : échantillonne un sous-ensemble de voisins (dans PyG, tous les voisins sont utilisés par défaut)
2. **Aggregate** : agrège les features des voisins (mean, LSTM, ou max pooling)
3. **Combine** : concatène l'agrégation avec les features du nœud lui-même, puis applique une transformation linéaire

```
h_v = σ(W · CONCAT(h_v, AGG({h_u : u ∈ N(v)})))
```

## Pourquoi ce modèle sur Cora

GraphSAGE offre un avantage clé : la **scalabilité**. Bien que Cora soit petit, GraphSAGE permet de tester une architecture qui fonctionnerait aussi sur des graphes beaucoup plus grands grâce à son sampling de voisins.

De plus, la concaténation des features du nœud avec l'agrégation de ses voisins (au lieu de les mélanger comme le GCN) préserve mieux l'information locale du nœud.

## Architecture

| Paramètre | Valeur |
|---|---|
| Couches | 2 (SAGEConv) |
| Hidden channels | 128 |
| Agrégation | Mean (par défaut dans PyG) |
| Activation | ReLU (entre couches) |
| Dropout | 0.5 |

## Hyperparamètres d'entraînement

| Paramètre | Valeur | Justification |
|---|---|---|
| Learning rate | 0.01 | GraphSAGE est plus robuste au lr grâce à la concaténation |
| Weight decay | 5e-3 | Plus élevé que les autres modèles pour compenser les 128 hidden channels |
| Epochs max | 500 | Avec early stopping |
| Early stopping patience | 30 | |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=10) | |

## Pourquoi un weight decay plus élevé

Avec 128 hidden channels, le modèle a beaucoup de paramètres. Un weight decay de 5e-3 (vs 5e-4 pour GCN/GAT) impose une régularisation L2 plus forte pour contrer l'overfitting. C'est un compromis : on augmente la capacité du modèle mais on le contraint davantage.

## Limites

- **Mean aggregation** : la moyenne des voisins peut diluer l'information, surtout si le nœud a beaucoup de voisins avec des features variées.
- **Pas d'attention** : contrairement au GAT, tous les voisins contribuent de façon égale.

## Accuracy cible

~84-85% sur le test set de Cora.
