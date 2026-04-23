import argparse
import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import random

def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r") as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)

def load_evals(path, clip_val):
    data = load_json(path)
    df = pd.DataFrame(data)
    if "evaluation" not in df.columns:
        raise KeyError("No 'evaluation' in evals")
    df["evaluation"] = pd.to_numeric(df["evaluation"], errors="coerce").fillna(0).astype(int)
    df["evaluation"] = df["evaluation"].clip(-clip_val, clip_val)
    if "winner" not in df.columns:
        raise KeyError("No 'winner' in evals")
    df["winner"] = pd.to_numeric(df["winner"], errors="coerce").fillna(0.0).astype(float)
    df["winner"] = df["winner"].clip(0.0, 1.0)
    return df

def load_graphs(path):
    if not os.path.exists(path):
        return {}
    data = load_json(path)
    res = {}
    if isinstance(data, list):
        for rec in data:
            if "position" in rec and "graph" in rec:
                res[rec["position"]] = rec["graph"]
    else:
        raise ValueError("Unsupported graphs format")
    return res

def load_features(path):
    data = load_json(path)
    assert isinstance(data, list)
    # load only first 72 features as adjecency matrix is a nightmare to invert
    return {rec["position"]: np.array(rec["features"][:72], dtype=float)
        for rec in data if "position" in rec and "features" in rec}

def load_static(path):
    if not os.path.exists(path):
        return {}
    data = load_json(path)
    if isinstance(data, dict):
        return {k: int(v) for k, v in data.items()}
    elif isinstance(data, list):
        return {rec["position"]: int(rec.get("static_eval")) for rec in data if "position" in rec and "static_eval" in rec}
    else:
        raise ValueError("Unsupported static format")

def metrics(y_true, y_pred):
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred))
    }

def estimate_flops_gnn(model, args):
    total_n_ops = 0
    total_m_ops = 0
    constant_ops = 0

    nd = model.node_dim
    qd = model.queen_dim

    if hasattr(model, 'pct_embed') and model.pct_embed is not None:
        # Input embedding addition (per node: node_dim add, queens: queen_dim)
        total_n_ops += nd
        constant_ops += 2 * (qd - nd)  # 2 queen nodes get extra dims

    for layer in model.layers:
        # msg proj (per node): node_dim -> node_dim
        n_ops = nd * nd * 2 + nd

        # queen up/down msg projections
        constant_ops += 2 * (qd * nd * 2 + nd)  # down
        constant_ops += 2 * (nd * qd * 2 + qd)  # up

        # self proj: normal nodes use node_dim x node_dim, queens use queen_dim x queen_dim
        if getattr(layer, 'use_self_proj', False):
            n_ops += nd * nd * 2 + nd  # per normal node
            queen_extra = (qd * qd * 2 + qd) - (nd * nd * 2 + nd)
            constant_ops += 2 * queen_extra  # 2 queens get extra cost

        # relu + residual add (per node: node_dim, queens: queen_dim)
        n_ops += nd
        constant_ops += 2 * (qd - nd)

        m_ops = nd  # scatter_add in node_dim
        if getattr(layer, 'use_pct_proj', False):
            m_ops += nd * nd * 2
        elif getattr(layer, 'use_pct_reweight', False):
            m_ops += nd
        elif getattr(layer, 'use_attention', False):
            m_ops += (nd * 5) + 3
            
        if getattr(layer, 'use_pct_bias', False):
            m_ops += nd
            
        # Edge category bias
        m_ops += nd
            
        total_n_ops += n_ops
        total_m_ops += m_ops

    for layer in model.mlp:
        if isinstance(layer, nn.Linear):
            constant_ops += layer.in_features * layer.out_features * 2
            if layer.bias is not None:
                constant_ops += layer.out_features

    return total_n_ops, total_m_ops, constant_ops

class GraphDataset(Dataset):
    def __init__(self, graphs_list, y_list, w_list, s_list=None, f_list=None):
        self.graphs = graphs_list
        self.y = y_list
        self.w = w_list
        self.s = s_list
        self.f = f_list

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        g = self.graphs[idx]
        nodes = torch.tensor(g["nodes"], dtype=torch.float32)
        edges = torch.tensor(g["edges"], dtype=torch.long)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        w = torch.tensor(self.w[idx], dtype=torch.float32)
        s_val = self.s[idx] if (self.s is not None and self.s[idx] is not None) else 0.0
        s = torch.tensor(s_val, dtype=torch.float32)
        f = torch.tensor(self.f[idx], dtype=torch.float32)
        return nodes, edges, y, w, s, f

def collate_graphs(batch):
    nodes_list = []
    edges_list = []
    y_list = []
    w_list = []
    s_list = []
    f_list = []
    batch_idx_list = []
    
    node_offset = 0
    for i, (nodes, edges, y, w, s, f) in enumerate(batch):
        num_nodes = nodes.size(0)
        nodes_list.append(nodes)
        
        rel_edges = edges.clone()
        rel_edges[:2] += node_offset
        edges_list.append(rel_edges)
        
        y_list.append(y)
        w_list.append(w)
        s_list.append(s)
        f_list.append(f)
        batch_idx_list.append(torch.full((num_nodes,), i, dtype=torch.long))
        node_offset += num_nodes
        
    batched_nodes = torch.cat(nodes_list, dim=0)
    if any(e.size(1) > 0 for e in edges_list):
        batched_edges = torch.cat(edges_list, dim=1)
    else:
        batched_edges = torch.empty((2, 0), dtype=torch.long)
    y = torch.stack(y_list, dim=0)
    w = torch.stack(w_list, dim=0)
    s = torch.stack(s_list, dim=0)
    f = torch.stack(f_list, dim=0)
    batch_idx = torch.cat(batch_idx_list, dim=0)
    
    return batched_nodes, batched_edges, y, w, batch_idx, s, f

class CustomGnnLayer(nn.Module):
    def __init__(self, node_dim, queen_dim, queen_cats=(0, 8), num_categories=16, use_softmax=False, use_attention=False, use_pct_proj=False, use_pct_reweight=False, use_pct_bias=False, use_pct_layers=False, use_self_proj=False):
        super().__init__()
        self.node_dim = node_dim
        self.queen_dim = queen_dim
        self.queen_cats = set(queen_cats)
        self.num_categories = num_categories
        self.use_softmax = use_softmax
        self.use_attention = use_attention
        self.use_pct_proj = use_pct_proj
        self.use_pct_reweight = use_pct_reweight
        self.use_pct_bias = use_pct_bias
        self.use_pct_layers = use_pct_layers
        
        # Message projection operates on node_dim features
        # Queen specific projections to bridge queen_dim and node_dim for msg passing
        self.W_down_queen = nn.Parameter(torch.Tensor(queen_dim, node_dim))
        self.B_down_queen = nn.Parameter(torch.Tensor(node_dim))
        nn.init.xavier_uniform_(self.W_down_queen)
        nn.init.zeros_(self.B_down_queen)
        
        self.W_up_queen = nn.Parameter(torch.Tensor(node_dim, queen_dim))
        self.B_up_queen = nn.Parameter(torch.Tensor(queen_dim))
        nn.init.xavier_uniform_(self.W_up_queen)
        nn.init.zeros_(self.B_up_queen)

        if use_pct_layers:
            self.W_proj = nn.Parameter(torch.Tensor(num_categories, node_dim, node_dim))
            self.B_proj = nn.Parameter(torch.Tensor(num_categories, node_dim))
        else:
            self.W_proj = nn.Parameter(torch.Tensor(node_dim, node_dim))
            self.B_proj = nn.Parameter(torch.Tensor(node_dim))
            
        nn.init.xavier_uniform_(self.W_proj)
        nn.init.zeros_(self.B_proj)

        if use_attention:
            self.a_attn = nn.Parameter(torch.Tensor(num_categories, node_dim * 2, 1))
            nn.init.xavier_uniform_(self.a_attn)
            
        if use_pct_proj:
            self.W_msg = nn.Parameter(torch.Tensor(num_categories, num_categories, node_dim, node_dim))
            nn.init.xavier_uniform_(self.W_msg)
            
        if use_pct_reweight:
            self.V_msg = nn.Parameter(torch.Tensor(num_categories, num_categories, node_dim))
            nn.init.xavier_uniform_(self.V_msg)

        if use_pct_bias:
            self.B_msg = nn.Parameter(torch.Tensor(num_categories, num_categories, node_dim))
            nn.init.zeros_(self.B_msg)

        self.B_edge = nn.Parameter(torch.Tensor(7, node_dim)) # 0-5 directions, 6 underworld
        nn.init.zeros_(self.B_edge)
        
        self.use_self_proj = use_self_proj
        if use_self_proj:
            # Queens: queen_dim -> queen_dim
            self.W_self_queen = nn.Parameter(torch.Tensor(queen_dim, queen_dim))
            self.B_self_queen = nn.Parameter(torch.Tensor(queen_dim))
            nn.init.xavier_uniform_(self.W_self_queen)
            nn.init.zeros_(self.B_self_queen)
            # Normal: node_dim -> node_dim
            self.W_self_normal = nn.Parameter(torch.Tensor(node_dim, node_dim))
            self.B_self_normal = nn.Parameter(torch.Tensor(node_dim))
            nn.init.xavier_uniform_(self.W_self_normal)
            nn.init.zeros_(self.B_self_normal)
        
    def forward(self, x, edges):
        cat = x[:, 0].long()
        feat = x[:, 1:]  # (N, queen_dim)
        N = feat.size(0)
        is_queen = (cat == 0) | (cat == 8)
        normal_mask = ~is_queen
        
        # Message projection: down project queens to node_dim
        feat_msg = torch.zeros(N, self.node_dim, device=x.device)
        if is_queen.any():
            feat_msg[is_queen] = torch.matmul(feat[is_queen], self.W_down_queen) + self.B_down_queen
        if normal_mask.any():
            feat_msg[normal_mask] = feat[normal_mask, :self.node_dim]

        if self.use_pct_layers:
            W = self.W_proj[cat] 
            B = self.B_proj[cat]
            z = torch.bmm(feat_msg.unsqueeze(1), W).squeeze(1) + B
        else:
            z = torch.matmul(feat_msg, self.W_proj) + self.B_proj
        # z is (N, node_dim)
        
        src, dst, edge_cat = edges[0], edges[1], edges[2]
        cat_src = cat[src]
        cat_dst = cat[dst]
        
        z_src = z[src] + self.B_edge[edge_cat]
        
        if self.use_pct_bias:
            z_src = z_src + self.B_msg[cat_dst, cat_src]
        
        if self.use_attention:
            z_dst = z[dst] 
            z_concat = torch.cat([z_src, z_dst], dim=-1)
            a = self.a_attn[cat_dst] 
            e = torch.bmm(z_concat.unsqueeze(1), a).squeeze(1).squeeze(1)
            if self.use_softmax:
                max_alpha = torch.zeros(N, device=x.device).scatter_reduce(
                    0, dst, e, reduce='amax', include_self=False)
                exp_alpha = torch.exp(e - max_alpha[dst])
                sum_exp_alpha = torch.zeros(N, device=x.device).scatter_add(0, dst, exp_alpha)
                alpha = exp_alpha / (sum_exp_alpha[dst] + 1e-16)
            else:
                alpha = torch.relu(e)
            msg = alpha.unsqueeze(-1) * z_src

        elif self.use_pct_proj:
            W_edge = self.W_msg[cat_dst, cat_src]
            msg = torch.bmm(z_src.unsqueeze(1), W_edge).squeeze(1)

        elif self.use_pct_reweight:
            V_edge = self.V_msg[cat_dst, cat_src]
            msg = z_src * V_edge

        else:
            msg = z_src
        
        # Aggregate messages in node_dim, then pad to queen_dim
        msg_sum_nd = torch.zeros(N, self.node_dim, device=x.device)
        msg_sum_nd.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)
        
        # up project queens back to queen_dim
        msg_sum = torch.zeros(N, self.queen_dim, device=x.device)
        if is_queen.any():
            msg_sum[is_queen] = torch.matmul(msg_sum_nd[is_queen], self.W_up_queen) + self.B_up_queen
        if normal_mask.any():
            msg_sum[normal_mask, :self.node_dim] = msg_sum_nd[normal_mask]
            
        # self loop
        if self.use_self_proj:
            is_queen = (cat == 0) | (cat == 8)
            z_self = torch.zeros(N, self.queen_dim, device=x.device)
            if is_queen.any():
                feat_q = feat[is_queen]
                z_self[is_queen] = torch.matmul(feat_q, self.W_self_queen) + self.B_self_queen
            normal_mask = ~is_queen
            if normal_mask.any():
                feat_n = feat[normal_mask, :self.node_dim]
                z_self[normal_mask, :self.node_dim] = torch.matmul(feat_n, self.W_self_normal) + self.B_self_normal
            msg_sum += z_self

        h_prime = torch.relu(feat + msg_sum)
        
        out = torch.cat([cat.unsqueeze(1).float(), h_prime], dim=1)
        return out

class GNNModel(nn.Module):
    def __init__(self, feat_dim, gnn_in_dim=3, num_layers=3, node_dim=4, queen_dim=12, use_softmax=False, use_attention=False, use_pct_proj=False, use_pct_reweight=False, use_pct_bias=False, use_pct_layers=False, use_pct_embeddings=False, use_self_proj=False):
        super().__init__()
        self.use_softmax = use_softmax
        self.use_attention = use_attention
        self.use_pct_proj = use_pct_proj
        self.use_pct_reweight = use_pct_reweight
        self.use_pct_bias = use_pct_bias
        self.use_pct_layers = use_pct_layers
        self.use_pct_embeddings = use_pct_embeddings
        self.use_self_proj = use_self_proj
        self.node_dim = node_dim
        self.queen_dim = queen_dim
        
        if use_pct_embeddings:
            self.pct_embed = nn.Embedding(16, queen_dim)
            with torch.no_grad():
                # Zero out positions beyond node_dim for non-queens
                for c in range(16):
                    if c not in (0, 8):
                        self.pct_embed.weight[c, node_dim:] = 0
        else:
            self.register_parameter('pct_embed', None)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(CustomGnnLayer(node_dim=node_dim, queen_dim=queen_dim, queen_cats=(0, 8), num_categories=16, 
                                              use_softmax=use_softmax, use_attention=use_attention, 
                                              use_pct_proj=use_pct_proj, use_pct_reweight=use_pct_reweight, 
                                              use_pct_bias=use_pct_bias, use_pct_layers=use_pct_layers,
                                              use_self_proj=use_self_proj))
        
        self.pool_dim = 2 * queen_dim + 14 * node_dim
        self.feat_proj = nn.Sequential(nn.Linear(self.pool_dim, feat_dim), nn.LeakyReLU(.1))
        
        self.mlp = nn.Sequential(
            nn.Linear(self.pool_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 12),
            nn.ReLU(),
            nn.Linear(12, 8),
            nn.ReLU(),
            nn.Linear(8, 6),
            nn.ReLU(),
            nn.Linear(6, 2)
        )
        
    def forward(self, nodes, edges, batch_idx, clip):
        cat = nodes[:, 0].long()
        feat = nodes[:, 1:]
        feat_d = feat.size(1)
        is_queen = (cat == 0) | (cat == 8)
        
        # Pad input features to queen_dim
        if feat_d < self.queen_dim:
            feat = torch.cat([feat, torch.zeros(feat.size(0), self.queen_dim - feat_d, device=feat.device)], dim=1)
        
        # Add PCT embeddings
        if self.use_pct_embeddings:
            emb = self.pct_embed(cat)
            feat = feat + emb
        
        h = torch.cat([cat.unsqueeze(1).float(), feat], dim=1)

        for layer in self.layers:
            h = layer(h, edges)
        
        cat = h[:, 0].long()
        feat = h[:, 1:] 
        
        max_batch = batch_idx.max().item() + 1
        flat_idx = batch_idx * 16 + cat 
        
        # pooling adds node embeddings of the same PCT

        pooled_flat = torch.zeros(max_batch * 16, self.queen_dim, device=nodes.device)
        pooled_flat.scatter_add_(0, flat_idx.unsqueeze(-1).expand_as(feat), feat)
        
        pooled_matrix = pooled_flat.view(max_batch, 16, self.queen_dim)
        
        out_features = []
        for c in range(16):
            if c == 0 or c == 8:
                out_features.append(pooled_matrix[:, c, :self.queen_dim])
            else:
                out_features.append(pooled_matrix[:, c, :self.node_dim])
        
        pooled = torch.cat(out_features, dim=1)
        
        out = self.mlp(pooled)
        pred_feat = self.feat_proj(pooled)
        eval_out = torch.nn.functional.softsign(out[:, 0]) * 2 * clip
        winner_out = torch.sigmoid(out[:, 1])
        return eval_out, winner_out, pred_feat

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--evals", default="logs/evaluations.json")
    p.add_argument("--graphs", default="logs/position_graphs.jsonl")
    p.add_argument("--features", default="logs/position_features.json")
    p.add_argument("--static", default="logs/position_static.json")
    p.add_argument("--test-size", type=float, default=0.1)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--clip", type=int, default=6000)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lambda-w", type=float, default=2e-3)
    p.add_argument("--lambda-feat", type=float, default=1e-4)
    p.add_argument("--l2", type=float, default=2e-6)
    p.add_argument("--use-softmax", action="store_true")
    p.add_argument("--use-attention", action="store_true")
    p.add_argument("--use-pct-proj", action="store_true", help="each category lives in its own space, project when passing messages according to PCT")
    p.add_argument("--use-pct-reweight", action="store_true", help="during message passing, reweight according to the two PCTs of the edge")
    p.add_argument("--use-pct-bias", action="store_true", help="sum an edge embedding which dpeends on the two PCTs it connects")
    p.add_argument("--use-pct-layers", action="store_true", help="use different W,B for each category")
    p.add_argument("--no-self-proj", action="store_false", dest="use_self_proj", default=True, help="disable learnable self-projection in GNN update")
    p.add_argument("--no-pct-embeddings", action="store_false", dest="pct_embeddings", default=True, help="disable input embeddings for each PCT")
    p.add_argument("--node-dim", type=int, default=4, help="feature size for normal nodes")
    p.add_argument("--queen-dim", type=int, default=12, help="feature size for queen nodes (cat 0 and 8)")
    p.add_argument("--num-layers", type=int, default=3, help="number of GNN layers")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--no-moves-count", action="store_true")
    p.add_argument("--load-ckpt", default=None, help="Path to checkpoint .pth to load")
    p.add_argument("--verbose", action="store_true", help="Print detailed predictions for one sample after each epoch")
    args = p.parse_args()

    flags_on = sum([args.use_attention, args.use_pct_proj, args.use_pct_reweight])
    if flags_on > 1:
        raise ValueError("--use-attention, --use-pct-proj, and --use-pct-reweight are mutually exclusive")

    evals_df = load_evals(args.evals, clip_val=args.clip)
    print(f"Loaded {len(evals_df)} evals")
    graphs_dict = load_graphs(args.graphs)
    print(f"Loaded {len(graphs_dict)} graphs")
    features_dict = load_features(args.features)
    print(f"Loaded {len(features_dict)} features")
    static_map = load_static(args.static)
    print(f"Loaded {len(static_map)} static evals")

    merged_data = []
    for idx, row in evals_df.iterrows():
        pos = row["position"]
        if pos in graphs_dict and pos in features_dict:
            s = static_map.get(pos)
            g = graphs_dict[pos]
            if args.no_moves_count:
                # Remove last element from each node vector
                new_nodes = [n[:-1] for n in g["nodes"]]
                g = {"nodes": new_nodes, "edges": g["edges"]}
            
            merged_data.append((pos, row["evaluation"], row["winner"], g, s, features_dict[pos]))

    if not merged_data:
        raise ValueError("No matching positions between evals and graphs")

    # Symmetric augmentation approach
    def get_symmetric_graph(g):
        new_nodes = []
        for n in g["nodes"]:
            cat = int(n[0])
            c = cat // 8
            pct = cat % 8
            c_sym = 1 - c
            sym_cat = c_sym * 8 + pct
            new_n = [sym_cat] + n[1:]
            new_nodes.append(new_n)
        return {"nodes": new_nodes, "edges": g["edges"]}

    graphs_list = []
    y_list = []
    w_list = []
    indices = []

    for i, (pos, y, w, g, s, f) in enumerate(merged_data):
        indices.append(i)

    idx_train, idx_test = train_test_split(indices, test_size=args.test_size, random_state=args.random_state)

    def augment(indices_subset):
        g_aug, y_aug, w_aug, s_aug, f_aug = [], [], [], [], []
        for i in indices_subset:
            pos, y, w, g, s, f = merged_data[i]
            N_f = len(f) // 2
            g_aug.append(g)
            y_aug.append(y)
            w_aug.append(w)
            s_aug.append(s)
            f_aug.append(f)
            
            if not args.no_augment:
                g_sym = get_symmetric_graph(g)
                g_aug.append(g_sym)
                y_aug.append(-y)
                w_aug.append(1.0 - w)
                s_aug.append(-s if s is not None else None)
                f_swapped = np.concatenate([f[N_f:], f[:N_f]])
                f_aug.append(f_swapped)
        return g_aug, y_aug, w_aug, s_aug, f_aug

    train_g, train_y, train_w, train_s, train_f = augment(idx_train)
    test_g, test_y, test_w, test_s, test_f = augment(idx_test)

    train_ds = GraphDataset(train_g, train_y, train_w, train_s, train_f)
    test_ds = GraphDataset(test_g, test_y, test_w, test_s, test_f)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_graphs)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_graphs)

    feat_dim = len(merged_data[0][5])
    gnn_in_dim = len(merged_data[0][3]["nodes"][0]) - 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = GNNModel(feat_dim=feat_dim, gnn_in_dim=gnn_in_dim, num_layers=args.num_layers, node_dim=args.node_dim, queen_dim=args.queen_dim, use_softmax=args.use_softmax, use_attention=args.use_attention, use_pct_proj=args.use_pct_proj, use_pct_reweight=args.use_pct_reweight, use_pct_bias=args.use_pct_bias, use_pct_layers=args.use_pct_layers, use_pct_embeddings=args.pct_embeddings, use_self_proj=args.use_self_proj).to(device)

    if args.load_ckpt:
        if os.path.exists(args.load_ckpt):
            ckpt = torch.load(args.load_ckpt, map_location=device)
            model.load_state_dict(ckpt)
            print(f"Loaded checkpoint from {args.load_ckpt}")
        else:
            print(f"Warning: checkpoint not found at {args.load_ckpt}")

    all_params = list(model.parameters())
    print("Total parameters:", sum(p.numel() for p in all_params))

    total_n_ops, total_m_ops, constant_ops = estimate_flops_gnn(model, args)
    
    flops_formula = f"{total_n_ops}*n + {total_m_ops}*m + {constant_ops}"
    n_test, m_test = 15, 40
    flops_test = total_n_ops * n_test + total_m_ops * m_test + constant_ops
    
    print(f"Estimated Inference FLOPs formula: {flops_formula}")
    print(f"Estimated FLOPs for n={n_test}, m={m_test}: {flops_test}")

    optimizer = torch.optim.Adam(all_params, lr=args.lr)
    loss_fn_eval = nn.MSELoss()
    loss_fn_winner = nn.BCELoss()

    best_val_loss = float('inf')
    best_state = None
    best_epoch = -1
    history = {'train_loss': [], 'test_loss': []}

    print("-- Training GNN --")
    for epoch in range(args.epochs):
        model.train()
        running_eval_loss = 0.0
        running_winner_loss = 0.0
        running_feat_loss = 0.0
        running_reg_loss = 0.0
        running_total_loss = 0.0
        total = 0

        for nodes, edges, yb, wb, batch_idx, _, fb in train_loader:
            nodes, edges, yb, wb, batch_idx, fb = nodes.to(device), edges.to(device), yb.to(device), wb.to(device), batch_idx.to(device), fb.to(device)
            optimizer.zero_grad()
            
            pred_eval, pred_winner, pred_feat = model(nodes, edges, batch_idx, args.clip)
            loss_eval = loss_fn_eval(pred_eval/args.clip, yb/args.clip)
            loss_winner = args.lambda_w * loss_fn_winner(pred_winner, wb)
            loss_feat = args.lambda_feat * nn.MSELoss()(pred_feat, fb)
            l2_reg = args.l2 * sum(p.pow(2).sum() for p in all_params)
            
            loss = loss_eval + loss_winner + loss_feat + l2_reg
            loss.backward()
            optimizer.step()
            
            bs = yb.size(0)
            running_eval_loss += loss_eval.item() * bs
            running_winner_loss += loss_winner.item() * bs
            running_feat_loss += loss_feat.item() * bs
            running_reg_loss += l2_reg.item() * bs
            running_total_loss += loss.item() * bs
            total += bs

        train_total_loss = running_total_loss / total
        train_eval_loss = running_eval_loss / total
        train_winner_loss = running_winner_loss / total
        train_feat_loss = running_feat_loss / total
        train_reg_loss = running_reg_loss / total
        
        # Test Validation
        model.eval()
        val_eval_loss = 0.0
        val_winner_loss = 0.0
        val_feat_raw_loss = 0.0
        val_reg_loss = 0.0
        val_eval_rmse = 0.0
        val_winner_mae = 0.0
        val_total = 0
        
        with torch.no_grad():
            for nodes, edges, yb, wb, batch_idx, _, fb in test_loader:
                nodes, edges, yb, wb, batch_idx, fb = nodes.to(device), edges.to(device), yb.to(device), wb.to(device), batch_idx.to(device), fb.to(device)
                pred_eval, pred_winner, pred_feat = model(nodes, edges, batch_idx, args.clip)
                
                loss_eval = loss_fn_eval(pred_eval/args.clip, yb/args.clip)
                loss_w = args.lambda_w * loss_fn_winner(pred_winner, wb)
                raw_feat_mse = nn.MSELoss()(pred_feat, fb)
                loss_f = args.lambda_feat * raw_feat_mse
                l2_r = args.l2 * sum(p.pow(2).sum() for p in all_params)
                
                bs = yb.size(0)
                val_eval_loss += loss_eval.item() * bs
                val_eval_rmse += (pred_eval - yb).pow(2).sum().item()
                val_winner_loss += loss_w.item() * bs
                val_winner_mae += (pred_winner - wb).abs().sum().item()
                val_feat_raw_loss += raw_feat_mse.item() * bs
                val_reg_loss += (l2_r.item() + loss_f.item()) * bs
                val_total += bs

        val_total_loss = (val_eval_loss + val_winner_loss + val_reg_loss) / val_total
        val_eval_rmse = (val_eval_rmse/val_total)**0.5
        val_winner_mae /= val_total
        val_feat_raw_loss /= val_total

        if args.verbose:
            # Pick a random sample from the test dataset for variety
            idx = random.randint(0, len(test_ds) - 1)
            sample = test_ds[idx]
            # Use collate_graphs to prepare the single sample for the model
            nodes_v, edges_v, yv, wv, bidx_v, _, fv = collate_graphs([sample])
            nodes_v, edges_v, yv, wv, bidx_v, fv = nodes_v.to(device), edges_v.to(device), yv.to(device), wv.to(device), bidx_v.to(device), fv.to(device)
            
            with torch.no_grad():
                pred_eval_v, pred_winner_v, pred_feat_v = model(nodes_v, edges_v, bidx_v, args.clip)
            
            num_nodes_v = nodes_v.size(0)
            
            print(f"\n--- Verbose Sample (Epoch {epoch+1}, Test Idx {idx}) ---")
            print(f"Graph with {num_nodes_v} nodes and {edges_v.size(1)} edges:")
            for i, n in enumerate(nodes_v):
                cat = int(n[0].item())
                print(f"  Node {i:2}: cat={cat:2} color={cat//8} type={cat%8} pos=({n[1]:6.1f}, {n[2]:6.1f}, {n[3]:6.1f})")
            print(f"Edges: {edges_v.t().tolist()}")
            
            # Formatting features for readability
            pred_f_list = [round(float(x), 3) for x in pred_feat_v[0].tolist()]
            true_f_list = [round(float(x), 3) for x in fv[0].tolist()]
            print(f"Predicted Features: {pred_f_list}")
            print(f"True Features:      {true_f_list}")
            print(f"Winner (wn)  -> Pred: {pred_winner_v[0].item():.4f}, True: {wv[0].item():.4f}")
            print(f"Eval         -> Pred: {pred_eval_v[0].item():.1f}, True: {yv[0].item():.1f}")
            print("---------------------------------------\n")

        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        history['train_loss'].append(train_total_loss)
        history['test_loss'].append(val_total_loss)

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"TRAIN loss={100*train_total_loss:.4f}={100*train_eval_loss:.4f}+{100*train_winner_loss:.4f}+{100*train_feat_loss:.4f}+{100*train_reg_loss:.4f} | "
            f"TEST  eval={val_eval_rmse:.1f} win={val_winner_mae:.4f} feat={val_feat_raw_loss:.4f} loss={100*val_total_loss:.4f}"
        )

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        save_path = "logs/best_gnn_eval.pth"
        torch.save(best_state, save_path)
        print(f"Best model saved to {save_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Total Loss')
    plt.plot(history['test_loss'], label='Test Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Total Loss over Time (GNN)')
    plt.legend()
    plt.grid(True)
    plt.savefig("logs/training_loss_gnn.png")
    print("Loss plot saved to logs/training_loss_gnn.png")

    # Evaluation Metrics
    model.eval()
    all_preds = []
    all_y = []
    all_static = []
    with torch.no_grad():
        for nodes, edges, yb, wb, batch_idx, sb, _ in test_loader:
            nodes, edges, yb, wb, batch_idx = nodes.to(device), edges.to(device), yb.to(device), wb.to(device), batch_idx.to(device)
            pred_eval, _, _ = model(nodes, edges, batch_idx, args.clip)
            all_preds.append(pred_eval.cpu().numpy())
            all_y.append(yb.cpu().numpy())
            all_static.append(sb.numpy())
    
    y_test_pred = np.concatenate(all_preds)
    y_test_true = np.concatenate(all_y)
    y_test_static = np.concatenate(all_static)
    mask_static = np.array([s is not None for s in test_s])

    # Static eval metrics
    if mask_static.any():
        static_metrics = metrics(y_test_true[mask_static], y_test_static[mask_static])
        print("\nStatic_eval predictor metrics:")
        print(f"  RMSE: {static_metrics['rmse']:.4f}")
        print(f"  MAE:  {static_metrics['mae']:.4f}")
        print(f"  R2:   {static_metrics['r2']:.4f}")
    
    gnn_metrics = metrics(y_test_true, y_test_pred)
    print("\nBest GNN model metrics on test set:")
    print(f"  RMSE: {gnn_metrics['rmse']:.4f}")
    print(f"  MAE:  {gnn_metrics['mae']:.4f}")
    print(f"  R2:   {gnn_metrics['r2']:.4f}")

    rust_out_path = "logs/eval_gnn.rs"
    with open(rust_out_path, "w") as f:
        if hasattr(model, 'pct_embed') and model.pct_embed is not None:
            pct_e = model.pct_embed.weight.detach().cpu().numpy()
            c, d = pct_e.shape
            f.write(f"\tconst PCT_EMBED: [[f32; {d}]; {c}] = [\n")
            for cat in range(c):
                f.write("\t\t[" + ", ".join(f"{float(x):.8e}f32" for x in pct_e[cat]) + "],\n")
            f.write("\t];\n\n")

        for layer_idx, gat_layer in enumerate(model.layers, 1):
            if gat_layer.use_pct_layers:
                w_proj = gat_layer.W_proj.detach().cpu().numpy()
                c, i, o = w_proj.shape
                ident_w = f"GAT{layer_idx}_W"
                f.write(f"\tconst {ident_w}: [[[f32; {o}]; {i}]; {c}] = [\n")
                for cat in range(c):
                    f.write("\t\t[\n")
                    for row in range(i):
                        f.write("\t\t\t[" + ", ".join(f"{float(x):.8e}f32" for x in w_proj[cat, row]) + "],\n")
                    f.write("\t\t],\n")
                f.write("\t];\n\n")
                
                b_proj = gat_layer.B_proj.detach().cpu().numpy()
                ident_b = f"GAT{layer_idx}_B"
                f.write(f"\tconst {ident_b}: [[f32; {o}]; {c}] = [\n")
                for cat in range(c):
                    f.write("\t\t[" + ", ".join(f"{float(x):.8e}f32" for x in b_proj[cat]) + "],\n")
                f.write("\t];\n\n")
            else:
                w_proj = gat_layer.W_proj.detach().cpu().numpy()
                i, o = w_proj.shape
                ident_w = f"GAT{layer_idx}_W"
                f.write(f"\tconst {ident_w}: [[f32; {o}]; {i}] = [\n")
                for row in range(i):
                    f.write("\t\t[" + ", ".join(f"{float(x):.8e}f32" for x in w_proj[row]) + "],\n")
                f.write("\t];\n\n")
                
                b_proj = gat_layer.B_proj.detach().cpu().numpy()
                ident_b = f"GAT{layer_idx}_B"
                f.write(f"\tconst {ident_b}: [f32; {o}] = [" + ", ".join(f"{float(x):.8e}f32" for x in b_proj) + "];\n\n")
            
            if args.use_attention:
                a_attn = gat_layer.a_attn.detach().cpu().numpy().squeeze(-1)
                o2 = a_attn.shape[1]
                ident_a = f"GAT{layer_idx}_A"
                f.write(f"\tconst {ident_a}: [[f32; {o2}]; {c}] = [\n")
                for cat in range(c):
                    f.write("\t\t[" + ", ".join(f"{float(x):.8e}f32" for x in a_attn[cat]) + "],\n")
                f.write("\t];\n\n")
                
            if args.use_pct_proj:
                w_msg = gat_layer.W_msg.detach().cpu().numpy()
                c_dst, c_src, o1, o2 = w_msg.shape
                ident_m = f"GAT{layer_idx}_W_MSG"
                f.write(f"\tconst {ident_m}: [[[[f32; {o2}]; {o1}]; {c_src}]; {c_dst}] = [\n")
                for cd in range(c_dst):
                    f.write("\t\t[\n")
                    for cs in range(c_src):
                        f.write("\t\t\t[\n")
                        for row in range(o1):
                            f.write("\t\t\t\t[" + ", ".join(f"{float(x):.8e}f32" for x in w_msg[cd, cs, row]) + "],\n")
                        f.write("\t\t\t],\n")
                    f.write("\t\t],\n")
                f.write("\t];\n\n")
                
            if args.use_pct_reweight:
                v_msg = gat_layer.V_msg.detach().cpu().numpy()
                c_dst, c_src, out_d = v_msg.shape
                ident_m = f"GAT{layer_idx}_V_MSG"
                f.write(f"\tconst {ident_m}: [[[f32; {out_d}]; {c_src}]; {c_dst}] = [\n")
                for cd in range(c_dst):
                    f.write("\t\t[\n")
                    for cs in range(c_src):
                        f.write("\t\t\t[" + ", ".join(f"{float(x):.8e}f32" for x in v_msg[cd, cs]) + "],\n")
                    f.write("\t\t],\n")
                f.write("\t];\n\n")

            if getattr(args, 'use_pct_bias', False):
                b_msg_val = gat_layer.B_msg.detach().cpu().numpy()
                c_dst, c_src, out_d = b_msg_val.shape
                ident_m = f"GAT{layer_idx}_B_MSG"
                f.write(f"\tconst {ident_m}: [[[f32; {out_d}]; {c_src}]; {c_dst}] = [\n")
                for cd in range(c_dst):
                    f.write("\t\t[\n")
                    for cs in range(c_src):
                        f.write("\t\t\t[" + ", ".join(f"{float(x):.8e}f32" for x in b_msg_val[cd, cs]) + "],\n")
                    f.write("\t\t],\n")
                f.write("\t];\n\n")

            if gat_layer.use_self_proj:
                # Queen self-projection: queen_dim x queen_dim
                w_sq = gat_layer.W_self_queen.detach().cpu().numpy()
                iq, oq = w_sq.shape
                f.write(f"\tconst GAT{layer_idx}_W_SELF_QUEEN: [[f32; {oq}]; {iq}] = [\n")
                for row in range(iq):
                    f.write("\t\t[" + ", ".join(f"{float(x):.8e}f32" for x in w_sq[row]) + "],\n")
                f.write("\t];\n\n")
                b_sq = gat_layer.B_self_queen.detach().cpu().numpy()
                f.write(f"\tconst GAT{layer_idx}_B_SELF_QUEEN: [f32; {oq}] = [" + ", ".join(f"{float(x):.8e}f32" for x in b_sq) + "];\n\n")

                # Normal self-projection: node_dim x node_dim
                w_sn = gat_layer.W_self_normal.detach().cpu().numpy()
                in_, on = w_sn.shape
                f.write(f"\tconst GAT{layer_idx}_W_SELF_NORMAL: [[f32; {on}]; {in_}] = [\n")
                for row in range(in_):
                    f.write("\t\t[" + ", ".join(f"{float(x):.8e}f32" for x in w_sn[row]) + "],\n")
                f.write("\t];\n\n")
                b_sn = gat_layer.B_self_normal.detach().cpu().numpy()
                f.write(f"\tconst GAT{layer_idx}_B_SELF_NORMAL: [f32; {on}] = [" + ", ".join(f"{float(x):.8e}f32" for x in b_sn) + "];\n\n")
            
            b_edge_val = gat_layer.B_edge.detach().cpu().numpy()
            c_edge, out_d = b_edge_val.shape
            ident_e = f"GAT{layer_idx}_E_MSG"
            f.write(f"\tconst {ident_e}: [[f32; {out_d}]; {c_edge}] = [\n")
            for ce in range(c_edge):
                f.write("\t\t[" + ", ".join(f"{float(x):.8e}f32" for x in b_edge_val[ce]) + "],\n")
            f.write("\t];\n\n")

            # Queen up/down projection export
            w_dq = gat_layer.W_down_queen.detach().cpu().numpy()
            iq_d, oq_d = w_dq.shape
            f.write(f"\tconst GAT{layer_idx}_W_DOWN_QUEEN: [[f32; {oq_d}]; {iq_d}] = [\n")
            for row in range(iq_d):
                f.write("\t\t[" + ", ".join(f"{float(x):.8e}f32" for x in w_dq[row]) + "],\n")
            f.write("\t];\n\n")
            b_dq = gat_layer.B_down_queen.detach().cpu().numpy()
            f.write(f"\tconst GAT{layer_idx}_B_DOWN_QUEEN: [f32; {oq_d}] = [" + ", ".join(f"{float(x):.8e}f32" for x in b_dq) + "];\n\n")

            w_uq = gat_layer.W_up_queen.detach().cpu().numpy()
            iq_u, oq_u = w_uq.shape
            f.write(f"\tconst GAT{layer_idx}_W_UP_QUEEN: [[f32; {oq_u}]; {iq_u}] = [\n")
            for row in range(iq_u):
                f.write("\t\t[" + ", ".join(f"{float(x):.8e}f32" for x in w_uq[row]) + "],\n")
            f.write("\t];\n\n")
            b_uq = gat_layer.B_up_queen.detach().cpu().numpy()
            f.write(f"\tconst GAT{layer_idx}_B_UP_QUEEN: [f32; {oq_u}] = [" + ", ".join(f"{float(x):.8e}f32" for x in b_uq) + "];\n\n")
            
        # Removed pool_proj export

        next_layer = 1
        last_bias_layer = None
        for name, p in model.mlp.named_parameters():
            arr = p.detach().cpu().numpy()
            is_final_last_w = name == "4.weight"
            is_final_last_b = name == "4.bias"
            
            if arr.ndim == 2:
                if is_final_last_w and arr.shape[0] == 2:
                    arr_print = arr[0:1, :]
                else:
                    arr_print = arr
                r, c = arr_print.shape
                ident = f"W{next_layer}{next_layer+1}"
                rows = []
                for row in arr_print:
                    rows.append("\t\t[" + ", ".join(f"{float(x):.8e}f32" for x in row) + "]")
                body = "[\n" + ",\n".join(rows) + "\n\t]"
                f.write(f"\tconst {ident}: [[f32; {c}]; {r}] = {body};\n\n")
                last_bias_layer = next_layer + 1
                next_layer += 1
            elif arr.ndim == 1:
                if is_final_last_b and arr.shape[0] == 2:
                    arr_print = arr[:1]
                else:
                    arr_print = arr
                n = arr_print.shape[0]
                layer_idx = last_bias_layer if last_bias_layer is not None else next_layer
                ident = f"B{layer_idx}"
                body = "[" + ", ".join(f"{float(x):.8e}f32" for x in arr_print) + "]"
                f.write(f"\tconst {ident}: [f32; {n}] = {body};\n\n")
        
    print(f"Rust weights written to {rust_out_path}")
