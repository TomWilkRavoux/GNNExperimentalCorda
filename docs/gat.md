# GAT — Graph Attention Network

## Principe

Le GAT (Veličković et al., 2018) remplace l'agrégation uniforme du GCN par un mécanisme d'**attention multi-head**. Chaque nœud apprend à pondérer différemment ses voisins : certains voisins contribuent plus que d'autres à la représentation finale.

Pour chaque paire de nœuds (i, j) connectés, un coefficient d'attention est calculé :

```
α_ij = softmax(LeakyReLU(a^T [Wh_i || Wh_j]))
```

Où `W` est une transformation linéaire, `||` la concaténation, et `a` un vecteur de poids d'attention appris.

Le **multi-head** consiste à calculer K mécanismes d'attention indépendants en parallèle, puis à concaténer (ou moyenner) les résultats.

## Pourquoi ce modèle sur Cora

Dans Cora, certaines citations sont plus informatives que d'autres pour déterminer la catégorie d'un article. Le GCN traite toutes les citations de la même façon, tandis que le GAT peut apprendre quels voisins sont les plus pertinents.

C'est le modèle qui obtient les **meilleurs résultats** sur Cora parmi les architectures classiques.

## Architecture

| Paramètre | Valeur |
|---|---|
| Couches | 2 (GATConv) |
| Couche 1 | 16 channels × 8 heads = 128 features |
| Couche 2 | 7 classes × 1 head (moyenne, pas concat) |
| Activation | ELU (entre couches) |
| Dropout | 0.6 (sur les features ET sur les coefficients d'attention) |

## Hyperparamètres d'entraînement

| Paramètre | Valeur | Justification |
|---|---|---|
| Learning rate | 0.005 | Le mécanisme d'attention est sensible à un lr trop haut |
| Weight decay | 5e-4 | Régularisation L2 |
| Epochs max | 500 | Avec early stopping |
| Early stopping patience | 30 | |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=10) | |

## Pourquoi le dropout est plus élevé (0.6 vs 0.5)

Le GAT a significativement plus de paramètres que le GCN (8 têtes d'attention indépendantes). Un dropout plus agressif est nécessaire pour éviter l'overfitting, surtout sur un dataset de taille modeste comme Cora (seulement 140 nœuds d'entraînement).

## Limites

- **Coût computationnel** : le calcul des coefficients d'attention pour chaque arête est plus coûteux que l'agrégation du GCN.
- **Attention statique** : les coefficients d'attention ne dépendent que des features des nœuds, pas de la tâche ou du contexte global du graphe.

## Accuracy cible

~85-86% sur le test set de Cora.
