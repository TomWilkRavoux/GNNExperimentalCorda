import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from torch_geometric.explain import Explainer, GNNExplainer, PGExplainer
    from torch_geometric.datasets import Planetoid

    import torch
    # import sweetviz as sv TODO : à importer quand reseau op 
    return Planetoid, mo, torch


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


@app.function
def Report(data, model, torch):
    data = data.to(next(model.parameters()).device)   # ← ligne ajoutée

    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)

    train_acc = (pred[data.train_mask] == data.y[data.train_mask]).sum() / data.train_mask.sum()
    val_acc   = (pred[data.val_mask]   == data.y[data.val_mask]).sum()   / data.val_mask.sum()
    test_acc  = (pred[data.test_mask]  == data.y[data.test_mask]).sum()  / data.test_mask.sum()

    print(f"Train accuracy : {train_acc:.4f}")
    print(f"Val   accuracy : {val_acc:.4f}")
    print(f"Test  accuracy : {test_acc:.4f}")
    return data


@app.cell
def _(data, model, torch):
    Report(data, model, torch)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Test Explainer
    """)
    return


@app.function
# def reportExplain(Explainer, GNNExplainer, data, model):
#     import torch

#     num_nodes = data.num_nodes          # ← compter les nœuds
#     print(f"Nombre de nœuds à expliquer : {num_nodes}")

#     explainerGCN = Explainer(
#         model=model,
#         algorithm=GNNExplainer(epochs=200),
#         explanation_type='model',
#         node_mask_type='attributes',
#         edge_mask_type='object',
#         model_config=dict(
#             mode='multiclass_classification',
#             task_level='node',
#             return_type='raw',   # forward() renvoie des logits bruts
#         ),
#     )

#     edge_importance = torch.zeros(data.edge_index.size(1))

#     # ← boucler sur CHAQUE nœud
#     for target_node in range(num_nodes):
#         if target_node % 200 == 0:
#             print(f"  {target_node}/{num_nodes} nœuds expliqués...")
#         explanation = explainerGCN(data.x, data.edge_index, index=target_node)
#         edge_importance += explanation.edge_mask.cpu().detach()

#     edge_importance /= num_nodes        # importance moyenne sur tout le graphe

#     print(f"\nTerminé — {num_nodes} nœuds expliqués")
#     print(f"Importance max        : {edge_importance.max():.4f}")
#     print(f"Arêtes significatives : {(edge_importance > 0.5).sum().item()}")

#     return edge_importance


def reportExplainPG(data, model):
    import torch
    from torch_geometric.explain import Explainer, PGExplainer

    device = next(model.parameters()).device   # ← le device du modèle (cuda)

    explainer = Explainer(
        model=model,
        algorithm=PGExplainer(epochs=30, lr=0.003),
        explanation_type='phenomenon',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='raw',
        ),
    )
    explainer.algorithm = explainer.algorithm.to(device) 

    # Phase 1 : entraînement
    train_idx = data.train_mask.nonzero(as_tuple=True)[0]
    for epoch in range(30):
        for node in train_idx:
            loss = explainer.algorithm.train(
                epoch, model, data.x, data.edge_index,
                target=data.y, index=int(node),
            )
        if epoch % 5 == 0:
            print(f"Epoch {epoch:2d} | loss {loss:.4f}")

    # Phase 2 : explication de tous les nœuds
    edge_importance = torch.zeros(data.edge_index.size(1))
    for node in range(data.num_nodes):
        if node % 500 == 0:
            print(f"  explication {node}/{data.num_nodes}...")
        explanation = explainer(data.x, data.edge_index, target=data.y, index=node)
        edge_importance += explanation.edge_mask.cpu().detach()
    edge_importance /= data.num_nodes

    print(f"\nTerminé — importance max : {edge_importance.max():.4f}")
    return edge_importance


@app.cell
def _(data, model):
    edge_importance = reportExplainPG(data, model)
    return (edge_importance,)


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
def _(torch):
    def visu(data, edge_importance, mo):
        from pyvis.network import Network

        classes = ['Case_Based', 'Genetic_Alg', 'Neural_Nets', 'Probabilistic',
                   'Reinf_Learn', 'Rule_Learn', 'Theory']
        palette = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12',
                   '#9b59b6', '#1abc9c', '#e67e22']

        _edge_index = data.edge_index.cpu()
        _labels     = data.y.cpu()
        _imp        = edge_importance.cpu()
        _imp_norm   = (_imp - _imp.min()) / (_imp.max() - _imp.min() + 1e-9)

        net = Network(height="900px", width="100%", bgcolor="#1a1a2e",
                      font_color="white", notebook=True, cdn_resources="in_line")

        for _n in range(data.num_nodes):
            _cls = _labels[_n].item()
            net.add_node(
                _n, label="",
                title=f"Nœud {_n} — {classes[_cls]}",
                color=palette[_cls],
                size=8, borderWidth=0,
            )

        _imp = edge_importance.cpu()

        # rang relatif de chaque arête (0 = faible, 1 = forte) - étale les couleurs
        _order = _imp.argsort()
        _ranks = torch.empty_like(_imp)
        _ranks[_order] = torch.arange(_imp.numel(), dtype=torch.float)
        _ranks /= (_imp.numel() - 1)

        _s, _d = _edge_index
        for _i in range(_s.size(0)):
            _r = _ranks[_i].item()
            if _r > 0.5:                       # moitié supérieure -> dégradé rouge
                _alpha = 0.15 + 0.85 * (_r - 0.5) / 0.5
                net.add_edge(
                    int(_s[_i]), int(_d[_i]),
                    color=f"rgba(231,76,60,{_alpha:.3f})",
                    width=round(0.4 + 4 * (_r - 0.5) / 0.5, 2),
                    title=f"importance : {_imp[_i].item():.4f} (rang {_r:.2f})",
                )
            else:                              # moitié inférieure -> gris discret
                net.add_edge(
                    int(_s[_i]), int(_d[_i]),
                    color="rgba(255,255,255,0.08)",
                    width=0.4,
                )

        net.set_options('''{
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -80,
              "centralGravity": 0.005,
              "springLength": 50,
              "springConstant": 0.05
            },
            "solver": "forceAtlas2Based",
            "stabilization": {"iterations": 300, "fit": true},
            "stopSimulationAfterStabilization": true
          },
          "nodes": {"shape": "dot", "borderWidth": 0},
          "edges": {"smooth": false},
          "interaction": {"hideEdgesOnDrag": true, "tooltipDelay": 100}
        }''')

        net.save_graph("explanation_full.html")


    return (visu,)


@app.cell
def _(data, edge_importance, mo, visu):
    visu(data, edge_importance, mo)
    return


if __name__ == "__main__":
    app.run()
