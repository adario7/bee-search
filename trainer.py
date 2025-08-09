import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

NORMAL_MODEL_PATH = "build/engines/normal/bee-search"

FEATURES_LEN = 12

class HiveGNN(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=64, num_gnn_layers=3, dropout=0.2, use_gat=True, heads=2):
        super().__init__()
        self.use_gat = use_gat
        self.dropout = nn.Dropout(dropout)

        self.gnn_layers = nn.ModuleList()
        self.projections = nn.ModuleList()
        self.norms = nn.ModuleList()

        in_dim = input_dim
        for i in range(num_gnn_layers):
            # determine out dimension for this layer
            if use_gat:
                # internal layers (except possibly last) may use concat heads
                if i == num_gnn_layers - 1:
                    # last GAT layer: single head, concat=False -> out dim = hidden_dim
                    self.gnn_layers.append(GATConv(in_dim, hidden_dim, heads=1, concat=False, dropout=dropout))
                    out_dim = hidden_dim
                else:
                    self.gnn_layers.append(GATConv(in_dim, hidden_dim, heads=heads, concat=True, dropout=dropout))
                    out_dim = hidden_dim * heads
            else:
                self.gnn_layers.append(GCNConv(in_dim, hidden_dim))
                out_dim = hidden_dim

            # projection to match residual dimension: maps previous feature dim -> out_dim
            self.projections.append(nn.Linear(in_dim, out_dim))
            # per-layer LayerNorm with correct normalized shape
            self.norms.append(nn.LayerNorm(out_dim))

            # next layer's input dim equals this layer's output dim
            in_dim = out_dim

        # head: take mean+max pooled global repr (2 * last out_dim/2 => we keep hidden_dim as internal)
        inter_dim = hidden_dim * 8
        self.mlp_head = nn.Sequential(
            nn.Linear(2 * in_dim, inter_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(inter_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x, edge_index, batch=None):
        for i, layer in enumerate(self.gnn_layers):
            out = layer(x, edge_index)
            out = F.leaky_relu(out)
            out = self.dropout(out)
            out = self.norms[i](out)
            res = self.projections[i](x)
            x = res + out

        if batch is None:
            mean_pool = torch.mean(x, dim=0, keepdim=True)
            max_pool = torch.max(x, dim=0, keepdim=True)[0]
        else:
            mean_pool = global_mean_pool(x, batch)
            max_pool = global_max_pool(x, batch)

        global_repr = torch.cat([mean_pool, max_pool], dim=-1)
        evaluation = self.mlp_head(global_repr)
        return torch.tanh(evaluation) * 1.5


def create_node_features(board_state, device):
    return torch.tensor(board_state, dtype=torch.float32, device=device)

def create_adjacency_matrix(adjacency_matrix, device):
    edges = np.nonzero(adjacency_matrix)
    edge_list = list(zip(edges[0], edges[1]))
    if len(edge_list) == 0:
        num_nodes = adjacency_matrix.shape[0]
        return torch.zeros((2, 0), dtype=torch.long, device=device)
    return torch.tensor(edge_list, dtype=torch.long, device=device).t().contiguous()

def get_static_eval(eval, engine):

    engine.send(f"newgame {eval["position"]}")

    engine.receive(1)

    engine.send("static_eval")
    result = engine.receive(1)  

    if not result or len(result) < 1:
        raise ValueError("Failed to get static evaluation from engine.")


    return int(result[0])



def load_training_data(evals_path, graphs_path, device, dumb_train=False, static_train=False):
    from evaluator import load_results
    evals = load_results(evals_path)
    print(f"Loaded {len(evals)} evaluations.")
    with open(graphs_path, "rb") as f:
        graphs = pickle.load(f)
    graph_dict = {g["position"]: g for g in graphs}
    print(f"Loaded {len(graph_dict)} graphs.")
    data_list = []

    if static_train:
        from arena import Engine
        evals = tqdm(evals, desc="Loading static evaluations")
        engine = Engine(NORMAL_MODEL_PATH)

    for eval in evals[:]:
        position = eval["position"]
        graph = graph_dict.get(position)
        if graph:
            features = graph["graph"][1]
            assert features.shape[1] == FEATURES_LEN, f"Feature length mismatch: {features.shape[1]} != FEATURES_LEN = {FEATURES_LEN}"
            edges = graph["graph"][2]
            if dumb_train:
                evaluation = np.sum(features[:, 2] - features[:, 3])
            elif static_train:
                evaluation = eval['static_eval'] if 'static_eval' in eval.keys() else get_static_eval(eval, engine)
            else:
                evaluation = eval["evaluation"]
            evaluation = np.clip(evaluation, -6000, 6000) / 6000.0
            data_list.append(Data(
                x=create_node_features(features, device),
                edge_index=create_adjacency_matrix(edges, device),
                y=torch.tensor([evaluation], dtype=torch.float32, device=device)
            ))
    print(f"Loaded {len(data_list)} samples")
    return data_list

def load_from_pth(model, model_path, device='cuda'):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def train_model(model, train_loader, val_loader, num_epochs=100, lr=0.001, device='cuda', export_folder='./'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        train_loss = train_mae = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out.squeeze(), batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            train_mae += F.l1_loss(out.squeeze(), batch.y).item()
        model.eval()
        val_loss = val_mae = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out.squeeze(), batch.y)
                val_loss += loss.item()
                val_mae += F.l1_loss(out.squeeze(), batch.y).item()
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        print(f'Epoch {epoch+1:3d}: Train Loss: {avg_train_loss:.4f}, Train MAE: {train_mae/len(train_loader):.2f}, Val Loss: {avg_val_loss:.4f}, Val MAE: {val_mae/len(val_loader):.2f}')
        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(export_folder, 'best_hive_gnn.pth'))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    return train_losses, val_losses

def export_to_onnx(model, sample_data, onnx_path='hive_gnn.onnx'):
    model.eval()
    dummy_x = sample_data.x.to('cpu')
    dummy_edge_index = sample_data.edge_index.to('cpu')
    torch.onnx.export(
        model.to('cpu'),
        (dummy_x, dummy_edge_index),
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['node_features', 'edge_index'],
        output_names=['evaluation'],
        dynamic_axes={'node_features': {0: 'num_nodes'}, 'edge_index': {1: 'num_edges'}}
    )
    print(f"Model exported to {onnx_path}")




def trainer(evals_path, graphs_path, num_epochs=100, lr=0.001, batch_size=32, use_gat=False, model_path=None, device=None, dumb_train=False, static_train=False,
            hidden_dim=64, num_gnn_layers=3, dropout=0.2, export_folder='./'):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    data_list = load_training_data(evals_path, graphs_path, device, dumb_train=dumb_train, static_train=static_train)
    train_data, val_data = train_test_split(data_list, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    model = HiveGNN(input_dim=12, hidden_dim=hidden_dim, num_gnn_layers=num_gnn_layers, dropout=dropout, use_gat=use_gat)
    print("Using GAT model (requires ONNX opset 16+)")
    if model_path:
        model = load_from_pth(model, model_path, device=device)
        print(f"Loaded model from {model_path}")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,} | Trainable: {trainable:,} | Non-trainable: {total-trainable:,}")

    print("Starting training...")
    train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs=num_epochs, lr=lr, device=device, export_folder=export_folder)
    model.load_state_dict(torch.load(os.path.join(export_folder, 'best_hive_gnn.pth')))
    if data_list:
        export_to_onnx(model, data_list[0], onnx_path=os.path.join(export_folder, 'hive_gnn.onnx'))
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training History')
    plt.savefig(os.path.join(export_folder, 'training_history.png'))
    plt.show()
    print("Training completed!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train a GNN model for Hive evaluation.")
    parser.add_argument('--evals-path', type=str, default='logs/evaluations.json',
                        help='Path to the evaluations JSON file.')
    parser.add_argument('--graphs-path', type=str, default='logs/graphs.pkl',
                        help='Path to the graphs pickle file.')
    parser.add_argument('--num-epochs', type=int, default=100,
                        help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for training.')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training.')
    parser.add_argument('--use-gat', action='store_true',
                        help='Use GAT layers instead of GCN.')
    parser.add_argument('--load-from-pth', type=str, default=None,
                        help='Path to a pre-trained model in .pth format.')
    
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--dumb-train', action='store_true',
                        help='Train the model predicting the difference between the current player\'s piece count and the opponent\'s piece count.')
    group.add_argument('--static-train', action='store_true',
                        help='Train the model predicting the static evaluation.')

    parser.add_argument('--export-folder', type=str, default='./',
                        help='Folder to save the exported ONNX and pth model.')
    parser.add_argument('--hidden-dim', type=int, default=64,
                        help='Hidden dimension for the GNN model.')
    parser.add_argument('--num-gnn-layers', type=int, default=3,
                        help='Number of GNN layers in the model.')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate for the GNN model.')
    args = parser.parse_args()

    # Log arguments
    print(f"Arguments: {args}")


    trainer(evals_path=args.evals_path, graphs_path=args.graphs_path, num_epochs=args.num_epochs, lr=args.lr, batch_size=args.batch_size, use_gat=args.use_gat, model_path=args.load_from_pth,
            dumb_train=args.dumb_train, static_train=args.static_train, hidden_dim=args.hidden_dim, num_gnn_layers=args.num_gnn_layers, dropout=args.dropout,
            export_folder=args.export_folder)
    print("Done!")
