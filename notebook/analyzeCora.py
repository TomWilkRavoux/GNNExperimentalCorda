import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch
    from torch_geometric.datasets import Planetoid
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns

    return (
        Planetoid,
        classification_report,
        confusion_matrix,
        mo,
        plt,
        sns,
        torch,
    )


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
    _labels = data.y.numpy()
    classes = ['Case_Based', 'Genetic_Alg', 'Neural_Nets', 'Probabilistic', 'Reinf_Learn', 'Rule_Learn', 'Theory']

    for _i, _name in enumerate(classes):
        _count = (_labels == _i).sum()
        print(f"  {_name:16s} : {_count:4d} nœuds ({_count/len(_labels)*100:.1f}%)")

    from torch_geometric.utils import degree as _degree
    _deg = _degree(data.edge_index[0], num_nodes=data.num_nodes)
    print(f"\nDegré moyen : {_deg.mean():.2f}")
    print(f"Degré max   : {_deg.max():.0f}")
    return (classes,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Entrainement V1 du model GNN - Classique GCN
    """)
    return


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

    _model = GCN(dataset.num_features, dataset.num_classes)
    print(_model)
    return F, GCN


@app.cell
def _(F, GCN, data, dataset, torch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modelTest = GCN(dataset.num_features, dataset.num_classes).to(device)
    dataTest = data.to(device)

    _optimizer = torch.optim.Adam(modelTest.parameters(), lr=0.005, weight_decay=5e-4)
    _scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(_optimizer, mode='max', factor=0.5, patience=10)

    def _train():
        modelTest.train()
        _optimizer.zero_grad()
        _out = modelTest(dataTest.x, dataTest.edge_index)
        _loss = F.cross_entropy(_out[dataTest.train_mask], dataTest.y[dataTest.train_mask])
        _loss.backward()
        _optimizer.step()
        return _loss.item()

    def _evaluate(mask):
        modelTest.eval()
        with torch.no_grad():
            _out = modelTest(dataTest.x, dataTest.edge_index)
            _pred = _out.argmax(dim=1)
            _correct = (_pred[mask] == dataTest.y[mask]).sum().item()
        return _correct / mask.sum().item()

    history = {'loss': [], 'val_acc': [], 'train_acc': []}
    _best_val_acc = 0
    _patience_counter = 0
    _best_state = None

    for _epoch in range(1, 501):
        _loss = _train()
        _train_acc = _evaluate(dataTest.train_mask)
        _val_acc = _evaluate(dataTest.val_mask)
        history['loss'].append(_loss)
        history['train_acc'].append(_train_acc)
        history['val_acc'].append(_val_acc)
        _scheduler.step(_val_acc)
        if _val_acc > _best_val_acc:
            _best_val_acc = _val_acc
            _patience_counter = 0
            _best_state = {k: v.clone() for k, v in modelTest.state_dict().items()}
        else:
            _patience_counter += 1
            if _patience_counter >= 30:
                print(f"Early stopping epoch {_epoch}")
                break
        if _epoch % 20 == 0:
            print(f"Epoch {_epoch:3d} | Loss: {_loss:.4f} | "
                  f"Train: {_train_acc:.4f} | Val: {_val_acc:.4f}")

    modelTest.load_state_dict(_best_state)
    _test_acc = _evaluate(dataTest.test_mask)
    print(f"\nAccuracy GCN sur le test set : {_test_acc:.4f}")
    return dataTest, history, modelTest


@app.cell
def _(modelTest, torch):
    torch.save(modelTest.state_dict(), "./models/gcn_cora.pth")
    return


@app.cell
def _(os, torch):
    def save_checkpoint(model, model_name, accuracy, config):
        os.makedirs("models", exist_ok=True)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_name": model_name,
            "accuracy": accuracy,
            "config": config,   # ex: {"hidden": 64, "lr": 0.005, "dropout": 0.5}
        }
        path = f"models/{model_name}_acc{accuracy:.4f}.pth"
        torch.save(checkpoint, path)
        print(f"Modèle sauvegardé : {path}")
        return path

    return (save_checkpoint,)


@app.cell
def _(modelTest, save_checkpoint):
    save_checkpoint(modelTest, "gcn", _test_acc, {"hidden": 64, "lr": 0.005})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Evaluation GCN + courbes
    """)
    return


@app.cell
def _(history, plt):
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(14, 5))

    _ax1.plot(history['loss'], color='#e74c3c')
    _ax1.set_title('Loss GCN')
    _ax1.set_xlabel('Epoch')
    _ax1.set_ylabel('Cross-Entropy Loss')

    _ax2.plot(history['train_acc'], label='Train', color='#3498db')
    _ax2.plot(history['val_acc'], label='Validation', color='#2ecc71')
    _ax2.set_title('Accuracy GCN')
    _ax2.set_xlabel('Epoch')
    _ax2.set_ylabel('Accuracy')
    _ax2.legend()

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Matrice de confusion GCN
    """)
    return


@app.cell
def _(
    classes,
    classification_report,
    confusion_matrix,
    dataTest,
    modelTest,
    plt,
    sns,
    torch,
):
    modelTest.eval()
    with torch.no_grad():
        _out = modelTest(dataTest.x, dataTest.edge_index)
        _pred = _out.argmax(dim=1).cpu().numpy()

    _y_true = dataTest.y[dataTest.test_mask].cpu().numpy()
    _y_pred = _pred[dataTest.test_mask.cpu().numpy()]

    print(classification_report(_y_true, _y_pred, target_names=classes))

    _cm = confusion_matrix(_y_true, _y_pred)
    _fig, _ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=_ax)
    _ax.set_ylabel('Vrai label')
    _ax.set_xlabel('Prédiction')
    _ax.set_title('Matrice de confusion — GCN')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Entrainement V2 du model GNN - Variante GAT (Graph Attention Network)
    """)
    return


@app.cell
def _(F, dataset, torch):
    from torch_geometric.nn import GATConv as _GATConv

    class GAT(torch.nn.Module):
        def __init__(self, num_features, num_classes, hidden_channels=16, heads=8):
            super().__init__()
            self.conv1 = _GATConv(num_features, hidden_channels, heads=heads, dropout=0.6)
            self.conv2 = _GATConv(hidden_channels * heads, num_classes, heads=1, concat=False, dropout=0.6)

        def forward(self, x, edge_index):
            x = F.dropout(x, p=0.6, training=self.training)
            x = F.elu(self.conv1(x, edge_index))
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.conv2(x, edge_index)
            return x

    _modelGAT = GAT(dataset.num_features, dataset.num_classes)
    print(_modelGAT)
    return (GAT,)


@app.cell
def _(F, GAT, data, dataset, torch):
    deviceGAT = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modelTrainGAT = GAT(dataset.num_features, dataset.num_classes).to(deviceGAT)
    dataGAT = data.to(deviceGAT)

    _optimizerGAT = torch.optim.Adam(modelTrainGAT.parameters(), lr=0.005, weight_decay=5e-4)
    _schedulerGAT = torch.optim.lr_scheduler.ReduceLROnPlateau(_optimizerGAT, mode='max', factor=0.5, patience=10)

    def _trainGAT():
        modelTrainGAT.train()
        _optimizerGAT.zero_grad()
        _out = modelTrainGAT(dataGAT.x, dataGAT.edge_index)
        _loss = F.cross_entropy(_out[dataGAT.train_mask], dataGAT.y[dataGAT.train_mask])
        _loss.backward()
        _optimizerGAT.step()
        return _loss.item()

    def _evaluateGAT(mask):
        modelTrainGAT.eval()
        with torch.no_grad():
            _out = modelTrainGAT(dataGAT.x, dataGAT.edge_index)
            _pred = _out.argmax(dim=1)
            _correct = (_pred[mask] == dataGAT.y[mask]).sum().item()
        return _correct / mask.sum().item()

    historyGAT = {'loss': [], 'val_acc': [], 'train_acc': []}
    _best_val_acc_gat = 0
    _patience_counter_gat = 0
    _best_state_gat = None

    for _epochGAT in range(1, 501):
        _lossGAT = _trainGAT()
        _train_accGAT = _evaluateGAT(dataGAT.train_mask)
        _val_accGAT = _evaluateGAT(dataGAT.val_mask)
        historyGAT['loss'].append(_lossGAT)
        historyGAT['train_acc'].append(_train_accGAT)
        historyGAT['val_acc'].append(_val_accGAT)
        _schedulerGAT.step(_val_accGAT)
        if _val_accGAT > _best_val_acc_gat:
            _best_val_acc_gat = _val_accGAT
            _patience_counter_gat = 0
            _best_state_gat = {k: v.clone() for k, v in modelTrainGAT.state_dict().items()}
        else:
            _patience_counter_gat += 1
            if _patience_counter_gat >= 30:
                print(f"Early stopping epoch {_epochGAT}")
                break
        if _epochGAT % 20 == 0:
            print(f"Epoch {_epochGAT:3d} | Loss: {_lossGAT:.4f} | "
                  f"Train: {_train_accGAT:.4f} | Val: {_val_accGAT:.4f}")

    modelTrainGAT.load_state_dict(_best_state_gat)
    _test_accGAT = _evaluateGAT(dataGAT.test_mask)
    print(f"\nAccuracy GAT sur le test set : {_test_accGAT:.4f}")
    return dataGAT, historyGAT, modelTrainGAT


@app.cell
def _():
    # torch.save(modelTrainGAT.state_dict(), "./models/gat_cora.pth")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Evaluation GAT + courbes
    """)
    return


@app.cell
def _(historyGAT, plt):
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(14, 5))

    _ax1.plot(historyGAT['loss'], color='#e74c3c')
    _ax1.set_title('Loss GAT')
    _ax1.set_xlabel('Epoch')
    _ax1.set_ylabel('Cross-Entropy Loss')

    _ax2.plot(historyGAT['train_acc'], label='Train', color='#3498db')
    _ax2.plot(historyGAT['val_acc'], label='Validation', color='#2ecc71')
    _ax2.set_title('Accuracy GAT')
    _ax2.set_xlabel('Epoch')
    _ax2.set_ylabel('Accuracy')
    _ax2.legend()

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Matrice de confusion GAT
    """)
    return


@app.cell
def _(
    classes,
    classification_report,
    confusion_matrix,
    dataGAT,
    modelTrainGAT,
    plt,
    sns,
    torch,
):
    modelTrainGAT.eval()
    with torch.no_grad():
        _out = modelTrainGAT(dataGAT.x, dataGAT.edge_index)
        _pred = _out.argmax(dim=1).cpu().numpy()

    _y_true = dataGAT.y[dataGAT.test_mask].cpu().numpy()
    _y_pred = _pred[dataGAT.test_mask.cpu().numpy()]

    print(classification_report(_y_true, _y_pred, target_names=classes))

    _cm = confusion_matrix(_y_true, _y_pred)
    _fig, _ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(_cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=classes, yticklabels=classes, ax=_ax)
    _ax.set_ylabel('Vrai label')
    _ax.set_xlabel('Prédiction')
    _ax.set_title('Matrice de confusion — GAT')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Entrainement V3 du model GNN - Variante GraphSAGE
    """)
    return


@app.cell
def _(F, dataset, torch):
    from torch_geometric.nn import SAGEConv as _SAGEConv

    class GraphSAGE(torch.nn.Module):
        def __init__(self, num_features, num_classes, hidden_channels=128):
            super().__init__()
            self.conv1 = _SAGEConv(num_features, hidden_channels)
            self.conv2 = _SAGEConv(hidden_channels, num_classes)

        def forward(self, x, edge_index):
            x = F.relu(self.conv1(x, edge_index))
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index)
            return x

    _modelSAGE = GraphSAGE(dataset.num_features, dataset.num_classes)
    print(_modelSAGE)
    return (GraphSAGE,)


@app.cell
def _(F, GraphSAGE, data, dataset, torch):
    deviceSAGE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modelTrainSAGE = GraphSAGE(dataset.num_features, dataset.num_classes).to(deviceSAGE)
    dataSAGE = data.to(deviceSAGE)

    _optimizerSAGE = torch.optim.Adam(modelTrainSAGE.parameters(), lr=0.01, weight_decay=5e-4)
    _schedulerSAGE = torch.optim.lr_scheduler.ReduceLROnPlateau(_optimizerSAGE, mode='max', factor=0.5, patience=10)

    def _trainSAGE():
        modelTrainSAGE.train()
        _optimizerSAGE.zero_grad()
        _out = modelTrainSAGE(dataSAGE.x, dataSAGE.edge_index)
        _loss = F.cross_entropy(_out[dataSAGE.train_mask], dataSAGE.y[dataSAGE.train_mask])
        _loss.backward()
        _optimizerSAGE.step()
        return _loss.item()

    def _evaluateSAGE(mask):
        modelTrainSAGE.eval()
        with torch.no_grad():
            _out = modelTrainSAGE(dataSAGE.x, dataSAGE.edge_index)
            _pred = _out.argmax(dim=1)
            _correct = (_pred[mask] == dataSAGE.y[mask]).sum().item()
        return _correct / mask.sum().item()

    historySAGE = {'loss': [], 'val_acc': [], 'train_acc': []}
    _best_val_acc_sage = 0
    _patience_counter_sage = 0
    _best_state_sage = None

    for _epochSAGE in range(1, 501):
        _lossSAGE = _trainSAGE()
        _train_accSAGE = _evaluateSAGE(dataSAGE.train_mask)
        _val_accSAGE = _evaluateSAGE(dataSAGE.val_mask)
        historySAGE['loss'].append(_lossSAGE)
        historySAGE['train_acc'].append(_train_accSAGE)
        historySAGE['val_acc'].append(_val_accSAGE)
        _schedulerSAGE.step(_val_accSAGE)
        if _val_accSAGE > _best_val_acc_sage:
            _best_val_acc_sage = _val_accSAGE
            _patience_counter_sage = 0
            _best_state_sage = {k: v.clone() for k, v in modelTrainSAGE.state_dict().items()}
        else:
            _patience_counter_sage += 1
            if _patience_counter_sage >= 100:
                print(f"Early stopping epoch {_epochSAGE}")
                break
        if _epochSAGE % 20 == 0:
            print(f"Epoch {_epochSAGE:3d} | Loss: {_lossSAGE:.4f} | "
                  f"Train: {_train_accSAGE:.4f} | Val: {_val_accSAGE:.4f}")

    modelTrainSAGE.load_state_dict(_best_state_sage)
    _test_accSAGE = _evaluateSAGE(dataSAGE.test_mask)
    print(f"\nAccuracy GraphSAGE sur le test set : {_test_accSAGE:.4f}")
    return dataSAGE, historySAGE, modelTrainSAGE


@app.cell
def _():
    # torch.save(modelTrainSAGE.state_dict(), "./models/sage_cora.pth")
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Evaluation GraphSAGE + courbes
    """)
    return


@app.cell
def _(historySAGE, plt):
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(14, 5))

    _ax1.plot(historySAGE['loss'], color='#e74c3c')
    _ax1.set_title('Loss GraphSAGE')
    _ax1.set_xlabel('Epoch')
    _ax1.set_ylabel('Cross-Entropy Loss')

    _ax2.plot(historySAGE['train_acc'], label='Train', color='#3498db')
    _ax2.plot(historySAGE['val_acc'], label='Validation', color='#2ecc71')
    _ax2.set_title('Accuracy GraphSAGE')
    _ax2.set_xlabel('Epoch')
    _ax2.set_ylabel('Accuracy')
    _ax2.legend()

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Matrice de confusion GraphSAGE
    """)
    return


@app.cell
def _(
    classes,
    classification_report,
    confusion_matrix,
    dataSAGE,
    modelTrainSAGE,
    plt,
    sns,
    torch,
):
    modelTrainSAGE.eval()
    with torch.no_grad():
        _out = modelTrainSAGE(dataSAGE.x, dataSAGE.edge_index)
        _pred = _out.argmax(dim=1).cpu().numpy()

    _y_true = dataSAGE.y[dataSAGE.test_mask].cpu().numpy()
    _y_pred = _pred[dataSAGE.test_mask.cpu().numpy()]

    print(classification_report(_y_true, _y_pred, target_names=classes))

    _cm = confusion_matrix(_y_true, _y_pred)
    _fig, _ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(_cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=classes, yticklabels=classes, ax=_ax)
    _ax.set_ylabel('Vrai label')
    _ax.set_xlabel('Prédiction')
    _ax.set_title('Matrice de confusion — GraphSAGE')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Entrainement V4 du model GNN - GCN 3 couches avec Skip Connections
    """)
    return


@app.cell
def _(F, dataset, torch):
    from torch_geometric.nn import GCNConv as _GCNConv3

    class GCN3(torch.nn.Module):
        def __init__(self, num_features, num_classes, hidden_channels=64):
            super().__init__()
            self.conv1 = _GCNConv3(num_features, hidden_channels)
            self.conv2 = _GCNConv3(hidden_channels, hidden_channels)
            self.conv3 = _GCNConv3(hidden_channels, num_classes)

        def forward(self, x, edge_index):
            x = F.relu(self.conv1(x, edge_index))
            x = F.dropout(x, p=0.5, training=self.training)
            h = F.relu(self.conv2(x, edge_index))
            h = F.dropout(h, p=0.5, training=self.training)
            x = x + h
            x = self.conv3(x, edge_index)
            return x

    _modelGCN3 = GCN3(dataset.num_features, dataset.num_classes)
    print(_modelGCN3)
    return (GCN3,)


@app.cell
def _(F, GCN3, data, dataset, torch):
    deviceGCN3 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modelTrainGCN3 = GCN3(dataset.num_features, dataset.num_classes).to(deviceGCN3)
    dataGCN3 = data.to(deviceGCN3)

    _optimizerGCN3 = torch.optim.Adam(modelTrainGCN3.parameters(), lr=0.005, weight_decay=5e-4)
    _schedulerGCN3 = torch.optim.lr_scheduler.ReduceLROnPlateau(_optimizerGCN3, mode='max', factor=0.5, patience=10)

    def _trainGCN3():
        modelTrainGCN3.train()
        _optimizerGCN3.zero_grad()
        _out = modelTrainGCN3(dataGCN3.x, dataGCN3.edge_index)
        _loss = F.cross_entropy(_out[dataGCN3.train_mask], dataGCN3.y[dataGCN3.train_mask])
        _loss.backward()
        _optimizerGCN3.step()
        return _loss.item()

    def _evaluateGCN3(mask):
        modelTrainGCN3.eval()
        with torch.no_grad():
            _out = modelTrainGCN3(dataGCN3.x, dataGCN3.edge_index)
            _pred = _out.argmax(dim=1)
            _correct = (_pred[mask] == dataGCN3.y[mask]).sum().item()
        return _correct / mask.sum().item()

    historyGCN3 = {'loss': [], 'val_acc': [], 'train_acc': []}
    _best_val_acc_gcn3 = 0
    _patience_counter_gcn3 = 0
    _best_state_gcn3 = None

    for _epochGCN3 in range(1, 501):
        _lossGCN3 = _trainGCN3()
        _train_accGCN3 = _evaluateGCN3(dataGCN3.train_mask)
        _val_accGCN3 = _evaluateGCN3(dataGCN3.val_mask)
        historyGCN3['loss'].append(_lossGCN3)
        historyGCN3['train_acc'].append(_train_accGCN3)
        historyGCN3['val_acc'].append(_val_accGCN3)
        _schedulerGCN3.step(_val_accGCN3)
        if _val_accGCN3 > _best_val_acc_gcn3:
            _best_val_acc_gcn3 = _val_accGCN3
            _patience_counter_gcn3 = 0
            _best_state_gcn3 = {k: v.clone() for k, v in modelTrainGCN3.state_dict().items()}
        else:
            _patience_counter_gcn3 += 1
            if _patience_counter_gcn3 >= 30:
                print(f"Early stopping epoch {_epochGCN3}")
                break
        if _epochGCN3 % 20 == 0:
            print(f"Epoch {_epochGCN3:3d} | Loss: {_lossGCN3:.4f} | "
                  f"Train: {_train_accGCN3:.4f} | Val: {_val_accGCN3:.4f}")

    modelTrainGCN3.load_state_dict(_best_state_gcn3)
    _test_accGCN3 = _evaluateGCN3(dataGCN3.test_mask)
    print(f"\nAccuracy GCN3 sur le test set : {_test_accGCN3:.4f}")
    return dataGCN3, historyGCN3, modelTrainGCN3


@app.cell
def _(modelTrainGCN3, torch):
    torch.save(modelTrainGCN3.state_dict(), "./models/gcn3_cora.pth")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Evaluation GCN3 + courbes
    """)
    return


@app.cell
def _(historyGCN3, plt):
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(14, 5))

    _ax1.plot(historyGCN3['loss'], color='#e74c3c')
    _ax1.set_title('Loss GCN3 (Skip Connections)')
    _ax1.set_xlabel('Epoch')
    _ax1.set_ylabel('Cross-Entropy Loss')

    _ax2.plot(historyGCN3['train_acc'], label='Train', color='#3498db')
    _ax2.plot(historyGCN3['val_acc'], label='Validation', color='#2ecc71')
    _ax2.set_title('Accuracy GCN3 (Skip Connections)')
    _ax2.set_xlabel('Epoch')
    _ax2.set_ylabel('Accuracy')
    _ax2.legend()

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Matrice de confusion GCN3
    """)
    return


@app.cell
def _(
    classes,
    classification_report,
    confusion_matrix,
    dataGCN3,
    modelTrainGCN3,
    plt,
    sns,
    torch,
):
    modelTrainGCN3.eval()
    with torch.no_grad():
        _out = modelTrainGCN3(dataGCN3.x, dataGCN3.edge_index)
        _pred = _out.argmax(dim=1).cpu().numpy()

    _y_true = dataGCN3.y[dataGCN3.test_mask].cpu().numpy()
    _y_pred = _pred[dataGCN3.test_mask.cpu().numpy()]

    print(classification_report(_y_true, _y_pred, target_names=classes))

    _cm = confusion_matrix(_y_true, _y_pred)
    _fig, _ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(_cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=classes, yticklabels=classes, ax=_ax)
    _ax.set_ylabel('Vrai label')
    _ax.set_xlabel('Prédiction')
    _ax.set_title('Matrice de confusion — GCN3 (Skip Connections)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualisation interactive du graphe Cora
    """)
    return


@app.cell
def _(classes, data):
    from pyvis.network import Network as _Network

    _palette = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12',
                '#9b59b6', '#1abc9c', '#e67e22']

    _net = _Network(height="900px", width="100%", bgcolor="#1a1a2e",
                    font_color="white", notebook=True)

    for _node in range(data.num_nodes):
        _label_id = data.y[_node].item()
        _net.add_node(_node,
                      label="",
                      title=f"Nœud {_node} — Classe: {classes[_label_id]}",
                      color=_palette[_label_id],
                      size=10)

    _edge_index = data.edge_index.cpu().numpy()
    for _j in range(_edge_index.shape[1]):
        _src, _dst = int(_edge_index[0, _j]), int(_edge_index[1, _j])
        _net.add_edge(_src, _dst)

    _net.set_options('''
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

    _net.show("cora_full.html")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
