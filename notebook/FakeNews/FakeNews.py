import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch
    import pandas as pd
    import networkx as nx
    import matplotlib.pyplot as plt
    from torch_geometric.datasets import UPFD
    from torch_geometric.loader import DataLoader
    from torch_geometric.transforms import ToUndirected
    from torch_geometric.utils import to_networkx
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, global_max_pool
    from sklearn.metrics import f1_score, classification_report, confusion_matrix
    import seaborn as sns
    import numpy as np
    from scipy.stats import mannwhitneyu

    return (
        DataLoader,
        F,
        GCNConv,
        ToUndirected,
        UPFD,
        classification_report,
        confusion_matrix,
        f1_score,
        global_max_pool,
        mannwhitneyu,
        mo,
        np,
        nx,
        pd,
        plt,
        sns,
        to_networkx,
        torch,
    )


@app.cell
def _(ToUndirected, UPFD):
    NAME, FEAT = 'politifact', 'spacy'

    train_ds = UPFD('./data/UPFD', NAME, FEAT, 'train', transform=ToUndirected())
    val_ds   = UPFD('./data/UPFD', NAME, FEAT, 'val',   transform=ToUndirected())
    test_ds  = UPFD('./data/UPFD', NAME, FEAT, 'test',  transform=ToUndirected())

    print(f"Train : {len(train_ds)} graphes")
    print(f"Val   : {len(val_ds)} graphes")
    print(f"Test  : {len(test_ds)} graphes")
    print(f"Features/nœud : {train_ds.num_features}")
    print(f"Classes       : {train_ds.num_classes}  (convention UPFD : 0=réel, 1=fake)")
    return FEAT, NAME, test_ds, train_ds, val_ds


@app.cell
def _(nx, pd, test_ds, to_networkx, train_ds, val_ds):
    def _depth(g):
        # arbre raciné au nœud 0 (la news) → profondeur = plus long chemin depuis la racine
        G = to_networkx(g, to_undirected=True)
        if G.number_of_nodes() <= 1:
            return 0
        return max(nx.single_source_shortest_path_length(G, 0).values())

    def _stats(ds, split):
        rows = []
        for _i, _g in enumerate(ds):
            rows.append({
                'split': split,
                'graph_id': _i,
                'label': int(_g.y.item()),
                'num_nodes': _g.num_nodes,      # = 1 news + nb retweeteurs (la portée)
                'num_edges': _g.num_edges,      # liens de propagation (non-dirigés)
                'depth': _depth(_g),            # profondeur de l'arbre de retweets
            })
        return rows

    df = pd.DataFrame(
        _stats(train_ds, 'train') + _stats(val_ds, 'val') + _stats(test_ds, 'test')
    )
    df['classe'] = df['label'].map({0: 'réel', 1: 'fake'})
    df
    return (df,)


@app.cell
def _(df):
    print("Moyennes par classe :")
    print(df.groupby('classe')[['num_nodes', 'num_edges', 'depth']].mean().round(1))
    print("\nRépartition :")
    print(df['classe'].value_counts())
    print("\nMédianes (plus robustes aux valeurs extrêmes) :")
    print(df.groupby('classe')[['num_nodes', 'depth']].median())
    return


@app.cell
def _(df, plt):
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for _cls, _color in [('réel', '#2ecc71'), ('fake', '#e74c3c')]:
        _sub = df[df['classe'] == _cls]
        _ax1.hist(_sub['num_nodes'], bins=30, alpha=0.6, label=_cls, color=_color)
        _ax2.hist(_sub['depth'], bins=range(0, df['depth'].max() + 2),
                  alpha=0.6, label=_cls, color=_color)

    _ax1.set_title('Portée (nb de nœuds)'); _ax1.set_xlabel('nœuds'); _ax1.legend()
    _ax2.set_title('Profondeur de propagation'); _ax2.set_xlabel('profondeur'); _ax2.legend()
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(df):
    _path = './data/upfd_politifact_stats.csv'
    df.to_csv(_path, index=False)
    print(f"CSV exporté : {_path}  ({len(df)} lignes)")
    df.describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## DataLoader
    """)
    return


@app.cell
def _(DataLoader, test_ds, train_ds, val_ds):
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=128, shuffle=False)

    # un batch = plusieurs petits graphes collés ensemble ; le vecteur `batch`
    # dit à quel graphe appartient chaque nœud (c'est lui qui permet le pooling)
    _b = next(iter(train_loader))
    print(f"Batch : {_b.num_graphs} graphes")
    print(f"  x          : {_b.x.shape}      (tous les nœuds empilés)")
    print(f"  edge_index : {_b.edge_index.shape}")
    print(f"  batch      : {_b.batch.shape}  -> valeurs de 0 à {_b.batch.max().item()}")
    print(f"  y          : {_b.y.shape}      (1 label par graphe)")
    return test_loader, train_loader, val_loader


@app.cell
def _(F, GCNConv, global_max_pool, torch, train_ds):
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
            x = global_max_pool(x, batch)        # [nb_graphes, hidden] : 1 vecteur par graphe
            return self.lin(x)

    _m = GCNGraph(train_ds.num_features, 128, train_ds.num_classes)
    print(_m)
    return (GCNGraph,)


@app.cell
def _(
    F,
    GCNGraph,
    f1_score,
    test_loader,
    torch,
    train_ds,
    train_loader,
    val_loader,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modelGCN = GCNGraph(train_ds.num_features, 128, train_ds.num_classes).to(device)
    optimizer = torch.optim.Adam(modelGCN.parameters(), lr=0.01, weight_decay=1e-3)

    def _train():
        modelGCN.train()
        _total = 0
        for _batch in train_loader:
            _batch = _batch.to(device)
            optimizer.zero_grad()
            _out = modelGCN(_batch.x, _batch.edge_index, _batch.batch)
            _loss = F.cross_entropy(_out, _batch.y)
            _loss.backward()
            optimizer.step()
            _total += _loss.item() * _batch.num_graphs
        return _total / len(train_loader.dataset)

    @torch.no_grad()
    def _evaluate(loader):
        modelGCN.eval()
        _preds, _labels = [], []
        for _batch in loader:
            _batch = _batch.to(device)
            _out = modelGCN(_batch.x, _batch.edge_index, _batch.batch)
            _preds.append(_out.argmax(dim=1).cpu())
            _labels.append(_batch.y.cpu())
        _preds = torch.cat(_preds); _labels = torch.cat(_labels)
        _acc = (_preds == _labels).float().mean().item()
        _f1  = f1_score(_labels.numpy(), _preds.numpy())
        return _acc, _f1

    historyGCN = {'loss': [], 'train_acc': [], 'val_acc': []}
    _best_val, _patience, _best_state = 0, 0, None

    for _epoch in range(1, 101):
        _loss = _train()
        _train_acc, _ = _evaluate(train_loader)
        _val_acc, _val_f1 = _evaluate(val_loader)
        historyGCN['loss'].append(_loss)
        historyGCN['train_acc'].append(_train_acc)
        historyGCN['val_acc'].append(_val_acc)
        if _val_acc > _best_val:
            _best_val, _patience = _val_acc, 0
            _best_state = {k: v.clone() for k, v in modelGCN.state_dict().items()}
        else:
            _patience += 1
            if _patience >= 20:
                print(f"Early stopping epoch {_epoch}")
                break
        if _epoch % 10 == 0:
            print(f"Epoch {_epoch:3d} | Loss: {_loss:.4f} | Train: {_train_acc:.4f} | Val: {_val_acc:.4f} (F1 {_val_f1:.4f})")

    modelGCN.load_state_dict(_best_state)
    _test_acc, _test_f1 = _evaluate(test_loader)
    print(f"\nGCN graph-level — Test accuracy : {_test_acc:.4f} | F1 : {_test_f1:.4f}")
    return device, historyGCN, modelGCN


@app.cell
def _(modelGCN):
    modelGCN.eval()
    return


@app.cell
def _(historyGCN, plt):
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(14, 5))

    _ax1.plot(historyGCN['loss'], color='#e74c3c')
    _ax1.set_title('Loss GCN')
    _ax1.set_xlabel('Epoch')
    _ax1.set_ylabel('Cross-Entropy Loss')

    _ax2.plot(historyGCN['train_acc'], label='Train', color='#3498db')
    _ax2.plot(historyGCN['val_acc'], label='Validation', color='#2ecc71')
    _ax2.set_title('Accuracy GCN')
    _ax2.set_xlabel('Epoch')
    _ax2.set_ylabel('Accuracy')
    _ax2.legend()

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(
    classification_report,
    confusion_matrix,
    device,
    modelGCN,
    plt,
    sns,
    test_loader,
    torch,
):
    fake_classes = ['réel', 'fake']      # label 0 = réel, 1 = fake

    modelGCN.eval()
    _preds, _labels = [], []
    with torch.no_grad():
        for _batch in test_loader:
            _batch = _batch.to(device)
            _out = modelGCN(_batch.x, _batch.edge_index, _batch.batch)
            _preds.append(_out.argmax(dim=1).cpu())
            _labels.append(_batch.y.cpu())

    _y_true = torch.cat(_labels).numpy()
    _y_pred = torch.cat(_preds).numpy()

    print(classification_report(_y_true, _y_pred, target_names=fake_classes))

    _cm = confusion_matrix(_y_true, _y_pred)
    _fig, _ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=fake_classes, yticklabels=fake_classes, ax=_ax)
    _ax.set_ylabel('Vrai label')
    _ax.set_xlabel('Prédiction')
    _ax.set_title('Matrice de confusion — GCN graph-level')
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(FEAT, NAME, modelGCN, torch, train_ds):
    import os
    os.makedirs('models', exist_ok=True)

    checkpoint = {
        'model_state_dict': modelGCN.state_dict(),
        'config': {
            'in_channels': train_ds.num_features,
            'hidden_channels': 128,
            'num_classes': train_ds.num_classes,
        },
        'name': NAME,        # 'politifact'
        'feature': FEAT,     # 'spacy'
    }
    torch.save(checkpoint, 'models/fakenews_gcn.pth')
    print("Modèle sauvegardé : models/fakenews_gcn.pth")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualisation
    """)
    return


@app.function
def visu_propagation(g, filename, mo):
    from pyvis.network import Network
    import networkx as nx
    from torch_geometric.utils import to_networkx

    G = to_networkx(g, to_undirected=True)
    depths = nx.single_source_shortest_path_length(G, 0)   # nœud 0 = la news
    palette = ['#f1c40f', '#e67e22', '#e74c3c', '#9b59b6',
               '#3498db', '#1abc9c', '#2ecc71']

    net = Network(height="700px", width="100%", bgcolor="#1a1a2e",
                  font_color="white", notebook=True, cdn_resources="in_line")

    for _n in range(g.num_nodes):
        _d = depths.get(_n, 0)
        _root = (_n == 0)
        net.add_node(
            _n, label="",
            title=("News (racine)" if _root else f"Utilisateur — profondeur {_d}"),
            color="#ffffff" if _root else palette[min(_d, len(palette) - 1)],
            size=30 if _root else 8, borderWidth=0,
        )

    _ei = g.edge_index.cpu().numpy()
    _seen = set()
    for _j in range(_ei.shape[1]):
        _a, _b = int(_ei[0, _j]), int(_ei[1, _j])
        _key = (min(_a, _b), max(_a, _b))
        if _key in _seen:          # ToUndirected double les arêtes → on dédoublonne
            continue
        _seen.add(_key)
        net.add_edge(_a, _b, color="rgba(255,255,255,0.3)", width=0.5)

    net.set_options('''{
      "physics": {
        "forceAtlas2Based": {"gravitationalConstant": -80, "centralGravity": 0.005,
                             "springLength": 50, "springConstant": 0.05},
        "solver": "forceAtlas2Based",
        "stabilization": {"iterations": 300, "fit": true},
        "stopSimulationAfterStabilization": true
      },
      "nodes": {"shape": "dot", "borderWidth": 0},
      "edges": {"smooth": false},
      "interaction": {"hideEdgesOnDrag": true, "tooltipDelay": 100}
    }''')

    net.save_graph(filename)
    return mo.Html(open(filename).read())


@app.cell
def _(test_ds):
    fake_idx = next(_i for _i, _g in enumerate(test_ds) if _g.y.item() == 1)
    real_idx = next(_i for _i, _g in enumerate(test_ds) if _g.y.item() == 0)
    print(f"Exemple FAKE : graphe #{fake_idx} ({test_ds[fake_idx].num_nodes} nœuds)")
    print(f"Exemple RÉEL : graphe #{real_idx} ({test_ds[real_idx].num_nodes} nœuds)")
    return fake_idx, real_idx


@app.cell
def _(fake_idx, mo, test_ds):
    visu_propagation(test_ds[fake_idx], "prop_fake.html", mo)
    return


@app.cell
def _(mo, real_idx, test_ds):
    visu_propagation(test_ds[real_idx], "prop_real.html", mo)
    return


@app.cell
def _(np, nx, plt, test_ds, to_networkx):
    from collections import defaultdict

    def _radial_pos(G, root=0):
        # place la racine au centre, chaque profondeur sur un cercle plus large
        _depths = nx.single_source_shortest_path_length(G, root)
        _by_depth = defaultdict(list)
        for _n, _d in _depths.items():
            _by_depth[_d].append(_n)
        _pos = {}
        for _d, _nodes in _by_depth.items():
            for _i, _node in enumerate(_nodes):
                _ang = 2 * np.pi * _i / max(len(_nodes), 1)
                _pos[_node] = (_d * np.cos(_ang), _d * np.sin(_ang))
        return _pos

    def _draw_tree(g, ax):
        G = to_networkx(g, to_undirected=True)
        _pos = _radial_pos(G, 0)
        _depth = max(nx.single_source_shortest_path_length(G, 0).values())
        nx.draw_networkx_edges(G, _pos, ax=ax, edge_color='white', alpha=0.25, width=0.4)
        _colors = ['#f1c40f' if _n == 0 else '#3498db' for _n in G.nodes()]
        _sizes  = [70 if _n == 0 else 6 for _n in G.nodes()]
        nx.draw_networkx_nodes(G, _pos, ax=ax, node_color=_colors, node_size=_sizes)
        ax.set_facecolor('#1a1a2e'); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{g.num_nodes}n · prof {_depth}", color='white', fontsize=8)

    N = 6
    _rng = np.random.default_rng(42)
    _fakes = [g for g in test_ds if g.y.item() == 1]
    _reals = [g for g in test_ds if g.y.item() == 0]
    _fake_sample = [_fakes[_i] for _i in _rng.choice(len(_fakes), N, replace=False)]
    _real_sample = [_reals[_i] for _i in _rng.choice(len(_reals), N, replace=False)]

    _fig, _axes = plt.subplots(2, N, figsize=(2.3 * N, 5))
    _fig.patch.set_facecolor('#1a1a2e')
    for _c in range(N):
        _draw_tree(_fake_sample[_c], _axes[0, _c])
        _draw_tree(_real_sample[_c], _axes[1, _c])
    _axes[0, 0].set_ylabel('FAKE', color='#e74c3c', fontsize=14, fontweight='bold')
    _axes[1, 0].set_ylabel('RÉEL', color='#2ecc71', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(df, plt):
    _fig, _axes = plt.subplots(1, 4, figsize=(18, 4))

    for _ax, _col in zip(_axes[:3], ['num_nodes', 'num_edges', 'depth']):
        df.boxplot(column=_col, by='classe', ax=_ax, grid=False)
        _ax.set_title(_col); _ax.set_xlabel('')
    plt.suptitle('')

    # nuage portée vs profondeur
    for _cls, _color in [('réel', '#2ecc71'), ('fake', '#e74c3c')]:
        _sub = df[df['classe'] == _cls]
        _axes[3].scatter(_sub['num_nodes'], _sub['depth'], alpha=0.4,
                         label=_cls, color=_color, s=15)
    _axes[3].set_xlabel('Portée (nœuds)'); _axes[3].set_ylabel('Profondeur')
    _axes[3].set_title('Portée vs profondeur'); _axes[3].legend()
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(df, mannwhitneyu):
    print("Test de Mann-Whitney U (fake vs réel) — la différence est-elle significative ?\n")
    for _col in ['num_nodes', 'num_edges', 'depth']:
        _f = df[df['classe'] == 'fake'][_col]
        _r = df[df['classe'] == 'réel'][_col]
        _stat, _p = mannwhitneyu(_f, _r, alternative='two-sided')
        _verdict = "SIGNIFICATIF ✓" if _p < 0.05 else "non significatif"
        print(f"  {_col:10s} | fake méd={_f.median():6.1f} | réel méd={_r.median():6.1f} "
              f"| p={_p:.2e} → {_verdict}")
    return


@app.cell
def _(mo, np, nx, test_ds, to_networkx):
    def _():
        from collections import defaultdict

        def _radial_pos(G, root=0):
            _depths = nx.single_source_shortest_path_length(G, root)
            _by = defaultdict(list)
            for _n, _d in _depths.items():
                _by[_d].append(_n)
            _pos = {}
            for _d, _nodes in _by.items():
                for _i, _node in enumerate(_nodes):
                    _ang = 2 * np.pi * _i / max(len(_nodes), 1)
                    _pos[_node] = (_d * np.cos(_ang), _d * np.sin(_ang))
            return _pos

        def _pyvis_tree_html(g):
            from pyvis.network import Network
            G = to_networkx(g, to_undirected=True)
            _pos    = _radial_pos(G, 0)
            _depths = nx.single_source_shortest_path_length(G, 0)
            _palette = ['#f1c40f', '#e67e22', '#e74c3c', '#9b59b6', '#3498db', '#1abc9c', '#2ecc71']

            net = Network(height="240px", width="240px", bgcolor="#1a1a2e",
                          font_color="white", notebook=True, cdn_resources="in_line")
            net.toggle_physics(False)

            for _n in range(g.num_nodes):
                _d = _depths.get(_n, 0)
                _root = (_n == 0)
                _x, _y = _pos[_n]
                net.add_node(_n, label="",
                             title=("News" if _root else f"User · prof {_d}"),
                             color="#ffffff" if _root else _palette[min(_d, len(_palette) - 1)],
                             size=16 if _root else 5,
                             x=float(_x * 45), y=float(_y * 45))

            _seen, _ei = set(), g.edge_index.cpu().numpy()
            for _j in range(_ei.shape[1]):
                _a, _b = int(_ei[0, _j]), int(_ei[1, _j])
                _k = (min(_a, _b), max(_a, _b))
                if _k in _seen:
                    continue
                _seen.add(_k)
                net.add_edge(_a, _b, color="rgba(255,255,255,0.25)", width=0.5)

            net.save_graph("_tmptree.html")
            return open("_tmptree.html").read()

        # ré-échantillonnage local (même graine → mêmes graphes que la grille matplotlib)
        _rng   = np.random.default_rng(42)
        _fakes = [g for g in test_ds if g.y.item() == 1]
        _reals = [g for g in test_ds if g.y.item() == 0]
        _fake_sample = [_fakes[_i] for _i in _rng.choice(len(_fakes), 6, replace=False)]
        _real_sample = [_reals[_i] for _i in _rng.choice(len(_reals), 6, replace=False)]

        _fake_if = [mo.iframe(_pyvis_tree_html(g), height="250px", width="250px") for g in _fake_sample]
        _real_if = [mo.iframe(_pyvis_tree_html(g), height="250px", width="250px") for g in _real_sample]
        return mo.vstack([
            mo.md("### 🔴 FAKE"),
            mo.hstack(_fake_if, justify="start", gap=0.4, wrap=True),
            mo.md("### 🟢 RÉEL"),
            mo.hstack(_real_if, justify="start", gap=0.4, wrap=True),
        ])


    _()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
