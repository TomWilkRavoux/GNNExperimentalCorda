# Comparaison des modèles GNN sur Cora

## Vue d'ensemble

Ce projet compare 4 architectures de Graph Neural Networks sur le dataset Cora (classification de 2708 articles scientifiques en 7 catégories).

## Tableau comparatif

| | GCN | GAT | GraphSAGE | GCN3 |
|---|---|---|---|---|
| **Couches** | 2 | 2 | 2 | 3 |
| **Hidden** | 64 | 16×8 heads | 128 | 64 |
| **Paramètres clés** | Convolution spectrale | Attention multi-head | Mean aggregation + concat | Skip connections |
| **Activation** | ReLU | ELU | ReLU | ReLU |
| **Dropout** | 0.5 | 0.6 | 0.5 | 0.5 |
| **Learning rate** | 0.005 | 0.005 | 0.01 | 0.005 |
| **Weight decay** | 5e-4 | 5e-4 | 5e-3 | 5e-4 |
| **Accuracy cible** | ~84-85% | ~85-86% | ~84-85% | ~84-85% |

## Forces et faiblesses

### GCN
- **Forces** : Simple, rapide, bon baseline. Peu de paramètres.
- **Faiblesses** : Pas d'attention (tous les voisins sont égaux). Limité à 2 couches avant over-smoothing.
- **Quand l'utiliser** : Premier modèle à tester, ou quand la vitesse d'entraînement est prioritaire.

### GAT
- **Forces** : Meilleure accuracy grâce à l'attention. Capable de distinguer les voisins importants.
- **Faiblesses** : Plus lent à entraîner. Plus de paramètres. Sensible aux hyperparamètres (lr, dropout).
- **Quand l'utiliser** : Quand la performance prime, sur des graphes hétérophiles ou avec des relations complexes.

### GraphSAGE
- **Forces** : Inductif (peut généraliser à de nouveaux nœuds). Scalable grâce au sampling. Préserve l'information locale du nœud.
- **Faiblesses** : Mean aggregation peut diluer l'information. Pas d'attention.
- **Quand l'utiliser** : Sur des graphes dynamiques où de nouveaux nœuds apparaissent, ou sur des très grands graphes.

### GCN3
- **Forces** : Capture des relations à plus longue distance (3 hops). Skip connection prévient l'over-smoothing.
- **Faiblesses** : Plus de paramètres que le GCN 2 couches. Le gain sur Cora est marginal car le graphe est assez dense.
- **Quand l'utiliser** : Sur des graphes épars où les nœuds ont besoin d'information à longue portée.

## Techniques d'entraînement communes

Les 4 modèles utilisent les mêmes techniques d'optimisation :

1. **Early stopping** (patience=30) : arrête l'entraînement quand la validation accuracy ne s'améliore plus, puis restaure les meilleurs poids. Empêche l'overfitting.

2. **LR scheduler** (ReduceLROnPlateau, factor=0.5, patience=10) : divise le learning rate par 2 quand la validation stagne. Permet un apprentissage plus fin en fin d'entraînement.

3. **Matrice de confusion + classification report** : pour chaque modèle, on affiche precision/recall/F1 par classe et une heatmap de la matrice de confusion. Permet d'identifier quelles classes sont les plus difficiles à discriminer.

## Recommandation

Pour Cora spécifiquement, le **GAT** offre les meilleurs résultats. Mais le choix du modèle dépend du contexte :

- **Prototype rapide** → GCN
- **Meilleure accuracy** → GAT
- **Graphes dynamiques / production** → GraphSAGE
- **Relations longue distance** → GCN3
