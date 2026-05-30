# GNN Corda Project

Expérimentation de Graph Neural Networks (GNN) sur le dataset **Cora** via un notebook interactif [Marimo](https://marimo.io/).

## Objectif

Classifier des articles scientifiques en 7 catégories à partir de leur contenu et de leurs citations, en utilisant un réseau GCN (Graph Convolutional Network).

## Dataset : Cora

- **2708 nœuds** (articles), **5429 arêtes** (citations)
- **1433 features** par nœud (bag-of-words)
- **7 classes** : Case_Based, Genetic_Alg, Neural_Nets, Probabilistic, Reinf_Learn, Rule_Learn, Theory

## Structure du projet

```
gnn_corda_project/
├── analyzeNasa.py      # Notebook Marimo principal
├── main.py             # Point d'entrée CLI
├── pyproject.toml      # Dépendances (uv)
└── uv.lock
```

## Installation

```bash
# Installer uv si nécessaire
curl -LsSf https://astral.sh/uv/install.sh | sh

# Installer les dépendances
uv sync
```

## Lancement

```bash
# Ouvrir le notebook interactif
uv run marimo edit analyzeNasa.py
```

Le notebook télécharge automatiquement le dataset Cora au premier lancement.

## Contenu du notebook

1. **Chargement & analyse** — statistiques du graphe, distribution des classes, degré moyen
2. **Modèle GCN** — 2 couches GCNConv (hidden=16), dropout 0.5, Adam lr=0.01
3. **Entraînement** — 200 epochs, courbes loss / accuracy
4. **Visualisation** — rendu interactif du graphe complet avec pyvis (coloré par classe)

## Stack technique

| Lib | Usage |
|---|---|
| PyTorch + PyTorch Geometric | Modèle GCN |
| Marimo | Notebook réactif |
| pyvis / networkx | Visualisation du graphe |
| scikit-learn | Métriques (à venir) |
