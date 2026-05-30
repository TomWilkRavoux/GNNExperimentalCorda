import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch
    from torch_geometric.datasets import Planetoid
    import pandas as pd
    import numpy as np

    return Planetoid, mo, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Chargement du dataset + analyse rapide
    """)
    return


@app.cell
def _(Planetoid):
    dataset = Planetoid(root='./data/Cora', name='Cora')
    data = dataset[0]

    print(f"Nombre de nœuds       : {data.num_nodes}")
    print(f"Nombre d'arêtes       : {data.num_edges}")
    print(f"Features par nœud     : {data.num_node_features}")
    print(f"Nombre de classes     : {dataset.num_classes}")
    print(f"Nœuds d'entraînement  : {data.train_mask.sum().item()}")
    print(f"Nœuds de validation   : {data.val_mask.sum().item()}")
    print(f"Nœuds de test         : {data.test_mask.sum().item()}")
    return data, dataset


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exploration dataset
    """)
    return


@app.cell
def _(data):
    # Distribution des classes
    labels = data.y.numpy()
    classes = ['Case_Based', 'Genetic_Alg', 'Neural_Nets', 'Probabilistic', 'Reinf_Learn', 'Rule_Learn', 'Theory']

    for i, name in enumerate(classes):
        count = (labels == i).sum()
        print(f"  {name:16s} : {count:4d} nœuds ({count/len(labels)*100:.1f}%)")

    # Degré moyen
    from torch_geometric.utils import degree
    deg = degree(data.edge_index[0], num_nodes=data.num_nodes)
    print(f"\nDegré moyen : {deg.mean():.2f}")
    print(f"Degré max   : {deg.max():.0f}")

    return (classes,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Entrainement V1 du model GNN
    """)
    return


@app.cell
def _(dataset, torch):
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv

    class GCN(torch.nn.Module):
        def __init__(self, num_features, num_classes, hidden_channels=16):
            super().__init__()
            self.conv1 = GCNConv(num_features, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, num_classes)

        def forward(self, x, edge_index):
            # Première couche GCN + ReLU + Dropout
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
            # Deuxième couche GCN
            x = self.conv2(x, edge_index)
            return x

    model = GCN(dataset.num_features, dataset.num_classes, hidden_channels=16)
    print(model)
    return F, GCN


@app.cell
def _(F, GCN, data, dataset, torch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modelTest = GCN(dataset.num_features, dataset.num_classes).to(device)
    dataTest = data.to(device)

    optimizer = torch.optim.Adam(modelTest.parameters(), lr=0.01, weight_decay=5e-4)

    def train():
        modelTest.train()
        optimizer.zero_grad()
        out = modelTest(dataTest.x, dataTest.edge_index)
        loss = F.cross_entropy(out[dataTest.train_mask], dataTest.y[dataTest.train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()

    def evaluate(mask):
        modelTest.eval()
        with torch.no_grad():
            out = modelTest(dataTest.x, dataTest.edge_index)
            pred = out.argmax(dim=1)
            correct = (pred[mask] == dataTest.y[mask]).sum().item()
            acc = correct / mask.sum().item()
        return acc

    # Boucle d'entraînement
    history = {'loss': [], 'val_acc': [], 'train_acc': []}

    for epoch in range(1, 201):
        loss = train()
        train_acc = evaluate(dataTest.train_mask)
        val_acc = evaluate(dataTest.val_mask)
        history['loss'].append(loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | "
                  f"Train: {train_acc:.4f} | Val: {val_acc:.4f}")

    test_acc = evaluate(dataTest.test_mask)
    print(f"Accuracy sur le test set : {test_acc:.4f}")
    return (history,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Evaluation + courbe
    """)
    return


@app.cell
def _(history):
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history['loss'], color='#e74c3c')
    ax1.set_title('Loss pendant l\'entraînement')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy Loss')

    ax2.plot(history['train_acc'], label='Train', color='#3498db')
    ax2.plot(history['val_acc'], label='Validation', color='#2ecc71')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mettre les matrixe de confusion + F1, recall, accuracy
    """)
    return


@app.cell
def _(classes, data):
    # from pyvis.network import Network

    # # Créer le réseau interactif
    # net = Network(height="800px", width="100%", bgcolor="#1a1a2e",
    #               font_color="white", notebook=True)

    # # Palette de couleurs par classe
    # palette = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12',
    #            '#9b59b6', '#1abc9c', '#e67e22']

    # # Ajouter un sous-ensemble de nœuds (Cora complet = lent)
    # subset = 500
    # for a in range(subset):
    #     label_id = data.y[a].item()
    #     net.add_node(a,
    #                  label=str(a),
    #                  title=f"Nœud {a} — Classe: {classes[label_id]}",
    #                  color=palette[label_id],
    #                  size=10)

    # # Ajouter les arêtes correspondantes
    # edge_index = data.edge_index.cpu().numpy()
    # for j in range(edge_index.shape[1]):
    #     src, dst = int(edge_index[0, j]), int(edge_index[1, j])
    #     if src < subset and dst < subset:
    #         net.add_edge(src, dst)

    # # Options de physique pour un joli layout
    # net.set_options('''
    # {
    #   "physics": {
    #     "forceAtlas2Based": {
    #       "gravitationalConstant": -50,
    #       "centralGravity": 0.01,
    #       "springLength": 100,
    #       "springConstant": 0.08
    #     },
    #     "solver": "forceAtlas2Based"
    #   }
    # }
    # ''')

    # net.show("cora_interactive.html")

    from pyvis.network import Network
    from torch_geometric.utils import to_networkx
    import networkx as nx

    G = to_networkx(data, to_undirected=True)

    palette = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12',
               '#9b59b6', '#1abc9c', '#e67e22']

    net = Network(height="900px", width="100%", bgcolor="#1a1a2e",
                  font_color="white", notebook=True)

    # Tous les nœuds
    for node in range(data.num_nodes):
        label_id = data.y[node].item()
        net.add_node(node,
                     label="",          # pas de label = plus léger
                     title=f"Nœud {node} — Classe: {classes[label_id]}",
                     color=palette[label_id],
                     size=10)            # petits nœuds pour la lisibilité

    # Toutes les arêtes
    edge_index = data.edge_index.cpu().numpy()
    for j in range(edge_index.shape[1]):
        src, dst = int(edge_index[0, j]), int(edge_index[1, j])
        net.add_edge(src, dst)  # arêtes très transparentes

    net.set_options('''
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -80,
          "centralGravity": 0.005,
          "springLength": 50,
          "springConstant": 0.05
        },
        "solver": "forceAtlas2Based",
        "stabilization": {
          "iterations": 300,
          "fit": true
        },
        "stopSimulationAfterStabilization": true
      },
      "nodes": {
        "shape": "dot",
        "borderWidth": 0
      },
      "edges": {
        "smooth": false,
        "width": 0.3
      },
      "interaction": {
        "hideEdgesOnDrag": true,
        "tooltipDelay": 100
      }
    }
    ''')

    net.show("cora_full.html")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
