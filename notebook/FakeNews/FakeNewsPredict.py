import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, global_max_pool
    from torch_geometric.datasets import UPFD
    from torch_geometric.loader import DataLoader
    from torch_geometric.transforms import ToUndirected
    import pandas as pd

    return (
        DataLoader,
        F,
        GCNConv,
        ToUndirected,
        UPFD,
        global_max_pool,
        pd,
        torch,
    )


@app.cell
def _(F, GCNConv, global_max_pool, torch):
    # reconstruction de l'archi du model a l'indentique 
    class GCNGraph(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, num_classes):
            super().__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.lin   = torch.nn.Linear(hidden_channels, num_classes)

        def forward(self, x, edge_index, batch):
            x = F.relu(self.conv1(x, edge_index))
            x = F.dropout(x, p=0.5, training=self.training)
            x = F.relu(self.conv2(x, edge_index))
            x = global_max_pool(x, batch)
            return self.lin(x)

    return (GCNGraph,)


@app.cell
def _(GCNGraph, torch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load('models/fakenews_gcn.pth', map_location=device, weights_only=False)

    model = GCNGraph(**ckpt['config']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()    # descativation du dropout pour des prédictions stables
    print(f"Modèle chargé : {ckpt['name']} / {ckpt['feature']}")
    return ckpt, device, model


@app.cell
def _(ToUndirected, UPFD, ckpt):
    NAME, FEAT = ckpt['name'], ckpt['feature']
    test_ds = UPFD('./data/UPFD', NAME, FEAT, 'test', transform=ToUndirected())
    print(f"{len(test_ds)} graphes disponibles")
    return (test_ds,)


@app.cell
def _(F, device, model, test_ds, torch):
    def predict_graph(g):
        g = g.to(device)
        # graphe unique => tous les nœuds appartiennent au "graphe 0"
        batch = torch.zeros(g.num_nodes, dtype=torch.long, device=device)
        out  = model(g.x, g.edge_index, batch)
        prob = F.softmax(out, dim=1).squeeze().cpu()
        pred = out.argmax(dim=1).item()
        return pred, ['réel', 'fake'][pred], prob.tolist()

    g = test_ds[33]
    pred, label, prob = predict_graph(g)
    print(f"Prédiction   : {label}  (classe {pred})")
    print(f"Probabilités : réel={prob[0]:.3f} | fake={prob[1]:.3f}")
    print(f"Vrai label   : {['réel', 'fake'][g.y.item()]}")
    return


@app.cell
def _(DataLoader, device, model, test_ds, torch):
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    @torch.no_grad()
    def predict_all(loader):
        preds = []
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            preds.append(out.argmax(dim=1).cpu())
        return torch.cat(preds)

    all_preds = predict_all(test_loader)
    print(f"{all_preds.numel()} prédictions — {(all_preds == 1).sum().item()} classées fake")
    return


@app.cell
def _(DataLoader, F, device, model, pd):
    def predit_detail(ds):
        model.eval()
        loader = DataLoader(ds, batch_size=128, shuffle=False)
        rows, idx = [], 0
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            probs = F.softmax(out, dim=1).cpu()
            preds = out.argmax(dim=1).cpu()
            labels = batch.y.cpu()
            for i in range(batch.num_graphs):
                rows.append({
                    'graph_id':   idx,
                    'prédiction': ['réel', 'fake'][preds[i].item()],
                    'prob_réel':  round(probs[i, 0].item(), 3),
                    'prob_fake':  round(probs[i, 1].item(), 3),
                    'vrai_label': ['réel', 'fake'][labels[i].item()],
                    'correct':    bool(preds[i].item() == labels[i].item()),
                })
                idx += 1
            return pd.DataFrame(rows)

    return (predit_detail,)


@app.cell
def _(predit_detail, test_ds):
    df_pred = predit_detail(test_ds)
    df_pred
    return (df_pred,)


@app.cell
def _(df_pred):
    df_pred.to_csv('./test_pred_FakeNews_test.csv', index=False)
    return


@app.cell
def _(df_pred):
    print(f"Accuracy : {df_pred['correct'].mean():.4f}")
    print(f"Erreurs  : {(~df_pred['correct']).sum()} / {len(df_pred)}\n")

    df_pred[~df_pred['correct']]
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
