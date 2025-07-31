from arena import Engine
import os
import json
import pickle
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import os
import onnx

default_engine_path = "build/debug/bee-search"
FEATURES_LEN = 12

def get_engine_eval(engine_path, positions, depth=5, timeout = 120):

    engine = Engine(engine_path)

    engine.send("newgame Base")
    result = engine.receive(timeout=1)
    if len(result) == 0:
        raise TimeoutError("Engine didn't respond")
    engine.send("bestmove depth 1")
    result = engine.receive(timeout=1)
    if len(result) == 0:
        raise TimeoutError("Engine didn't respond")

    scores = []
    
    for position in tqdm(positions):
        try:
            engine.send(f"newgame {position}")
            result = engine.receive(timeout=1)
            if len(result) == 0:
                raise TimeoutError("Engine didn't respond")
            
            engine.send(f"eval {depth}")
            result = engine.receive(timeout=timeout)
            
            if len(result) != 0:
                result = result[0]

                if len(result) < 10:
                    pos = {}
                    pos["engine"] = engine.name
                    pos["depth"] = depth
                    pos["position"] = position
                    pos["evaluation"] = int(result)
                    scores.append(pos)
                else:
                    print(position)
                    print(result)
        except Exception as e:
            print(position)
            print(e)
            engine.terminate()
            engine = Engine(engine_path)
            engine.send("newgame Base")
            result = engine.receive(timeout=1)
            if len(result) == 0:
                raise TimeoutError("Engine didn't respond")
            engine.send("bestmove depth 1")
            result = engine.receive(timeout=1)
            if len(result) == 0:
                raise TimeoutError("Engine didn't respond")

    return scores 

def load_results(results_file):
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            return json.load(f)
    else:
        return []

def get_positions_from_results(results_paths, engine = Engine(default_engine_path)):
    positions = []

    for path in results_paths:
        results = load_results(path)

        for result in results:
            tmp = result["final_gamestate"].split(';')
            
            gametype = tmp[0]
            moves = tmp[3:]

            engine.send(f"newgame {gametype}")
            position = engine.receive(timeout=1)
            if len(position) == 0:
                raise TimeoutError("Engine didn't respond")
            
            for move in moves:
                engine.send(f"play {move}")
                position = engine.receive(timeout=1)
                if len(position) == 0:
                    raise TimeoutError("Engine didn't respond")
                
                positions.append(position[0])

    return set(positions)

def get_graph_from_positions(positions, engine = Engine(default_engine_path)):

    graphs = []

    print("Generating graphs...")

    for position in tqdm(positions):
        engine.send(f"newgame {position}")
        result = engine.receive(timeout=1)
        if len(result) == 0:
            raise TimeoutError("Engine didn't respond")

        engine.send("graph")
        result = engine.receive(timeout=1)
        if len(result) == 0:
            raise TimeoutError("Engine didn't respond")

        n = int(result[0])
        nodes = np.zeros(n, dtype=np.int16)

        up_to = 1
        for i in range(n):
            nodes[i] = int(result[i + up_to])

        features = np.zeros((n, FEATURES_LEN), dtype=np.float32)
        up_to += n
        for i in range(n):
            features[i] = np.array([float(x) for x in result[up_to + i].split(' ')])

        up_to += n
        edges = np.zeros((n, n), dtype=bool)
        for i in range(n):
            edges[i] = np.array([int(x) for x in result[up_to + i].split(' ')])

        graphs.append({"position": position, "graph": [nodes, features, edges]})

    return graphs



class HiveGNN(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=64, num_gnn_layers=3, dropout=0.2, use_gat=True):
        super(HiveGNN, self).__init__()
        
        self.use_gat = use_gat
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        
        if use_gat:
            # GAT layers (attention-based, better performance but needs opset 16+)
            # First layer
            self.gnn_layers.append(GATConv(input_dim, hidden_dim, heads=4, dropout=dropout, concat=True))
            current_dim = hidden_dim * 4
            
            # Hidden layers
            for _ in range(num_gnn_layers - 2):
                self.gnn_layers.append(GATConv(current_dim, hidden_dim, heads=4, dropout=dropout, concat=True))
            
            # Final GNN layer (single head)
            self.gnn_layers.append(GATConv(current_dim, hidden_dim, heads=1, dropout=dropout, concat=False))
        else:
            # GCN layers (simpler, better ONNX compatibility with opset 11)
            self.gnn_layers.append(GCNConv(input_dim, hidden_dim))
            for _ in range(num_gnn_layers - 1):
                self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Global pooling will create 2 * hidden_dim features (mean + max)
        # MLP head for final evaluation
        self.mlp_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
    def forward(self, x, edge_index, batch=None):
        # Apply GNN layers
        for i, layer in enumerate(self.gnn_layers):
            x = layer(x, edge_index)
            if i < len(self.gnn_layers) - 1:  # Don't apply ReLU after last layer
                x = F.relu(x)
            x = self.dropout(x)
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        # Global pooling (combine mean and max pooling)
        if batch is None:
            # Single graph case
            mean_pool = torch.mean(x, dim=0, keepdim=True)
            max_pool = torch.max(x, dim=0, keepdim=True)[0]
        else:
            # Batch case
            mean_pool = global_mean_pool(x, batch)
            max_pool = global_max_pool(x, batch)
        
        # Concatenate pooled features
        global_repr = torch.cat([mean_pool, max_pool], dim=-1)
        
        # Final MLP to get evaluation score
        evaluation = self.mlp_head(global_repr)
        
        evaluation = torch.tanh(evaluation)
        
        return evaluation

# Alternative GCN-based model for better ONNX compatibility
class HiveGCN(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=64, num_gnn_layers=3, dropout=0.2):
        super(HiveGCN, self).__init__()
        
        # GCN layers (better ONNX compatibility)
        self.gnn_layers = nn.ModuleList()
        self.gnn_layers.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_gnn_layers - 1):
            self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # MLP head for final evaluation
        self.mlp_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
    def forward(self, x, edge_index, batch=None):
        # Apply GCN layers
        for i, layer in enumerate(self.gnn_layers):
            x = layer(x, edge_index)
            if i < len(self.gnn_layers) - 1:
                x = F.relu(x)
            x = self.dropout(x)
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        # Global pooling
        if batch is None:
            mean_pool = torch.mean(x, dim=0, keepdim=True)
            max_pool = torch.max(x, dim=0, keepdim=True)[0]
        else:
            mean_pool = global_mean_pool(x, batch)
            max_pool = global_max_pool(x, batch)
        
        global_repr = torch.cat([mean_pool, max_pool], dim=-1)
        evaluation = self.mlp_head(global_repr)
        evaluation = torch.tanh(evaluation)
        
        return evaluation
    
def create_node_features(board_state):
    """
    Convert board state to torch tensor.
    
    board_state: numpy matrix of shape [num_nodes, 12] with one-hot encoded features
    
    Returns: torch tensor of shape [num_nodes, 12]
    """
    return torch.tensor(board_state, dtype=torch.float32)

def create_adjacency_matrix(adjacency_matrix):
    """
    Convert numpy adjacency matrix to PyTorch Geometric edge_index format.
    
    adjacency_matrix: numpy array of shape [num_nodes, num_nodes] with 0s and 1s
    
    Returns: torch tensor edge_index for PyTorch Geometric
    """
    # Find all edges (non-zero entries)
    edges = np.nonzero(adjacency_matrix)
    edge_list = list(zip(edges[0], edges[1]))
    
    if len(edge_list) == 0:
        # Handle case with no edges (isolated nodes)
        num_nodes = adjacency_matrix.shape[0]
        return torch.zeros((2, 0), dtype=torch.long)
    
    return torch.tensor(edge_list, dtype=torch.long).t().contiguous()


def load_training_data(evals_path, graphs_path):
    
    evals = load_results(evals_path)
    with open(graphs_path, "rb") as f:
        graphs = pickle.load(f)

    # Pair up evaluations with graphs with the same position
    # Create a dictionary for fast lookup: position -> graph
    graph_dict = {g["position"]: g for g in graphs}
    data_list = []
    for eval in evals:
        position = eval["position"]
        graph = graph_dict.get(position)
        if graph:
            features = graph["graph"][1]
            assert features.shape[1] == FEATURES_LEN, f"Feature length mismatch: {features.shape[1]} != FEATURES_LEN = {FEATURES_LEN}"
            edges = graph["graph"][2]
            evaluation = eval["evaluation"]

            # Cap evaluation to [-6000, 6000]
            # This might introduce problems, as the NN will never predict a chekmate
            # but it is probably necessary to avoid training only on extreme values
            evaluation = np.clip(evaluation, -6000, 6000)

            #normalize evaluation to [-1, 1]
            evaluation = evaluation / 6000.0

            data_list.append(Data(
                x=create_node_features(features),
                edge_index=create_adjacency_matrix(edges),
                y=torch.tensor([evaluation], dtype=torch.float32)
            ))
    
    return data_list

def train_model(model, train_loader, val_loader, num_epochs=100, lr=0.001, device='cuda'):
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
        # Training phase
        model.train()
        train_loss = 0
        train_mae = 0
        
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
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_mae = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out.squeeze(), batch.y)
                
                val_loss += loss.item()
                val_mae += F.l1_loss(out.squeeze(), batch.y).item()
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_train_mae = train_mae / len(train_loader)
        avg_val_mae = val_mae / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch {epoch+1:3d}: Train Loss: {avg_train_loss:.4f}, Train MAE: {avg_train_mae:.2f}, '
              f'Val Loss: {avg_val_loss:.4f}, Val MAE: {avg_val_mae:.2f}')
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_hive_gnn.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    return train_losses, val_losses

def export_to_onnx(model, sample_data, onnx_path='hive_gnn.onnx'):
    """Export trained model to ONNX format for Rust inference."""
    model.eval()
    
    # Create dummy input matching your graph structure
    dummy_x = sample_data.x
    dummy_edge_index = sample_data.edge_index
    
    # Export to ONNX
    torch.onnx.export(
        model, 
        (dummy_x, dummy_edge_index),
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['node_features', 'edge_index'],
        output_names=['evaluation'],
        dynamic_axes={
            'node_features': {0: 'num_nodes'},
            'edge_index': {1: 'num_edges'}
        }
    )
    print(f"Model exported to {onnx_path}")

def trainer(evals_path, graphs_path, num_epochs=100, lr=0.001, batch_size=32, use_gat=False):
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load your data
    print("Loading training data...")
    data_list = load_training_data(evals_path, graphs_path) 

    # Split data
    train_data, val_data = train_test_split(data_list, test_size=0.2, random_state=42)
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # Initialize model
    if use_gat:
        model = HiveGNN(input_dim=12, hidden_dim=64, num_gnn_layers=3, dropout=0.2, use_gat=True)
        print("Using GAT model (requires ONNX opset 16+)")
    else:
        model = HiveGCN(input_dim=12, hidden_dim=64, num_gnn_layers=3, dropout=0.2)
        print("Using GCN model (compatible with ONNX opset 11)")
    
    # Train model
    print("Starting training...")
    train_losses, val_losses = train_model(model, train_loader, val_loader, 
                                         num_epochs=num_epochs, lr=lr, device=device)

    # Load best model
    model.load_state_dict(torch.load('best_hive_gnn.pth'))
    
    # Export to ONNX
    if data_list:
        export_to_onnx(model, data_list[0])
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training History')
    plt.savefig('training_history.png')
    plt.show()
    
    print("Training completed!")



if __name__ == "__main__":
    # Args
    import argparse
    parser = argparse.ArgumentParser(description="Train a GNN model for Hive evaluation.")
    parser.add_argument('--evals_path', type=str, default='logs/evaluations.json',
                        help='Path to the evaluations JSON file.')
    parser.add_argument('--graphs_path', type=str, default='logs/graphs.pkl',
                        help='Path to the graphs pickle file.')
    parser.add_argument('--results_paths', type=str, nargs='+', default=['logs/results.json', 'logs/500ms/results.json', 'logs/1000ms/results.json'],
                        help='List of result files.')
    parser.add_argument('--engine_path', type=str, default=default_engine_path,
                        help='Path to the Hive engine executable.')
    parser.add_argument('--compute_evals', type=bool, default=False,
                        help='Whether to compute evaluations from results files.')

    parser.add_argument('--depth', type=int, default=5,
                        help='Depth for engine evaluation.')
    parser.add_argument('--timeout', type=int, default=120,
                        help='Timeout for engine responses in seconds.')
    
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for training.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training.')
    args = parser.parse_args()

    if args.compute_evals:
        # Compute evaluations from results files
        positions = list(get_positions_from_results(args.results_paths, engine=Engine(args.engine_path)))
        evals = get_engine_eval(engine_path=args.engine_path, positions=positions, depth=args.depth, timeout=args.timeout)
        with open(args.evals_path, "w") as f:
            json.dump(evals, f, indent=2)
        print(f"Evaluations saved to {args.evals_path}")
    
        graphs = get_graph_from_positions(positions=positions, engine=Engine(default_engine_path))
        with open("logs/graphs.pkl", "wb") as f:
            pickle.dump(graphs, f)

    # Train the model
    trainer(evals_path=args.evals_path, graphs_path=args.graphs_path, num_epochs=args.num_epochs, lr=args.lr, batch_size=args.batch_size)

    print("Done!")
