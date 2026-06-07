import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from torch_geometric.explain import Explainer, GNNExplainer
    from torch_geometric.datasets import Planetoid

    import torch
    # import sweetviz as sv TODO : à importer quand reseau op 
    return Explainer, GNNExplainer, Planetoid, mo, torch


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


@app.cell
def _(dataset, torch):
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv

    class GCN(torch.nn.Module):
        def __init__(self, num_features, num_classes, hidden_channels=64):
            super().__init__()
            self.conv1 = GCNConv(num_features, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, num_classes)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index)
            return x

    gcn_model = GCN(dataset.num_features, dataset.num_classes)
    print(gcn_model)
    return (GCN,)


@app.cell
def _(GCN, dataset, torch):
    # Cellule 3 — Charger le modèle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GCN(dataset.num_features, dataset.num_classes)  # meme signature qu'à l'entraînement
    model.load_state_dict(torch.load("models/gcn_cora.pth", map_location=device))
    model.to(device)
    model.eval()# important : désactive le dropout
    return (model,)


@app.cell
def _(data, model, torch):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)

    # Accuracy sur chaque split
    train_acc = (pred[data.train_mask] == data.y[data.train_mask]).sum() / data.train_mask.sum()
    val_acc   = (pred[data.val_mask]   == data.y[data.val_mask]).sum()   / data.val_mask.sum()
    test_acc  = (pred[data.test_mask]  == data.y[data.test_mask]).sum()  / data.test_mask.sum()

    print(f"Train accuracy : {train_acc:.4f}")
    print(f"Val   accuracy : {val_acc:.4f}")
    print(f"Test  accuracy : {test_acc:.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Test Explainer
    """)
    return


@app.cell
def _(Explainer, GNNExplainer, data, model):
    explainerGCN = Explainer(
        model= model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model', 
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='probs',
        ),
    )

    explanation = explainerGCN(data.x, data.edge_index, index=10)

    print(explanation.edge_mask)
    return


@app.cell
def _(model):
    print(model)
    return


@app.cell
def _(model):
    import inspect
    print(inspect.getsource(model.forward))
    return


@app.cell
def _():
    from pyvis.network import Network


    return


if __name__ == "__main__":
    app.run()
