import argparse
import os
import gc
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------
# Binary Record Dtype (Matches Rust BinaryGraphRecord 1:1)
# ---------------------------------------------------------
NUM_TOKENS = 28
NUM_PIECE_TYPES = 16
NUM_EDGE_TYPES = 10  # 0: None, 1..6: NW..W, 7: Stack-Up, 8: Stack-Down, 9: Self
TOKEN_FEAT_DIM = 8

# 28 tokens -> 16 piece types (8 for White, 8 for Black)
TOKEN_TO_PIECE_TYPE = [
    0,          # 0: White Queen (1)
    1, 1, 1,    # 1..3: White Grasshopper (3)
    2, 2,       # 4..5: White Spider (2)
    3, 3, 3,    # 6..8: White Ant (3)
    4, 4,       # 9..10: White Beetle (2)
    5,          # 11: White Mosquito (1)
    6,          # 12: White Ladybug (1)
    7,          # 13: White Pillbug (1)
    8,          # 14: Black Queen (1)
    9, 9, 9,    # 15..17: Black Grasshopper (3)
    10, 10,     # 18..19: Black Spider (2)
    11, 11, 11, # 20..22: Black Ant (3)
    12, 12,     # 23..24: Black Beetle (2)
    13,         # 25: Black Mosquito (1)
    14,         # 26: Black Ladybug (1)
    15,         # 27: Black Pillbug (1)
]

TYPE_TO_TOKENS = [[] for _ in range(NUM_PIECE_TYPES)]
for _tok, _t in enumerate(TOKEN_TO_PIECE_TYPE):
    TYPE_TO_TOKENS[_t].append(_tok)

PIECE_TYPE_NAMES = [
    "White Queen", "White Grasshopper", "White Spider", "White Ant",
    "White Beetle", "White Mosquito", "White Ladybug", "White Pillbug",
    "Black Queen", "Black Grasshopper", "Black Spider", "Black Ant",
    "Black Beetle", "Black Mosquito", "Black Ladybug", "Black Pillbug",
]

FN_DIM = 52
FN2_DIM = 104

SAMPLE_DTYPE = np.dtype([
    ('features', np.float32, (NUM_TOKENS, TOKEN_FEAT_DIM)),
    ('adj', np.uint8, (NUM_TOKENS, NUM_TOKENS)),
    ('fn2', np.float32, (FN2_DIM,)),
    ('eval', np.float32),
    ('winner', np.float32),
    ('static_eval', np.float32),
    ('turn_num', np.uint16),
    ('_pad', np.uint8, 2),
])

def metrics(y_true, y_pred):
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }

# ---------------------------------------------------------
# PyTorch Dataset with Lazy Memory-Mapped Access (Zero-RAM)
# ---------------------------------------------------------
PERM = np.concatenate([np.arange(14, 28), np.arange(0, 14)])

class LazyBinaryGraphDataset(Dataset):
    """
    Zero-RAM Memory-Mapped Dataset.
    Directly indexes disk memory-map without pre-allocating or copying data into RAM.
    Performs on-the-fly 8x symmetric augmentation:
      2 color states (White / Black POV)
      x 4 spatial reflections (Identity, Flip X, Flip Y, Flip Both X & Y).
    """
    def __init__(self, raw_mmap, indices, sample_weights=None, augment=True):
        self.raw_mmap = raw_mmap
        self.indices = np.asarray(indices, dtype=np.int64)
        self.sample_weights = sample_weights
        self.augment = augment
        self.multiplier = 8 if augment else 1

    def __len__(self):
        return len(self.indices) * self.multiplier

    def __getitem__(self, idx):
        if self.augment:
            orig_idx = self.indices[idx // 8]
            aug_type = idx % 8
            is_swapped = (aug_type % 2 == 1)
            flip_x = (aug_type in (2, 3, 6, 7))
            flip_y = (aug_type in (4, 5, 6, 7))
        else:
            orig_idx = self.indices[idx]
            is_swapped = False
            flip_x = False
            flip_y = False

        record = self.raw_mmap[orig_idx]
        features = record['features'].copy()  # (28, 8) float32
        adj = record['adj']                   # (28, 28) uint8
        fn2 = record['fn2']                   # (104,) float32
        eval_val = float(record['eval'])
        winner_val = float(record['winner'])
        static_val = float(record['static_eval'])
        weight_val = float(self.sample_weights[orig_idx]) if self.sample_weights is not None else 1.0

        if is_swapped:
            # Symmetrically swap White (0..13) and Black (14..27) tokens
            features = features[PERM]
            adj = adj[PERM][:, PERM]
            # Symmetrically swap White (0..51) and Black (52..103) FN2 features
            fn2_swapped = np.empty_like(fn2)
            fn2_swapped[:FN_DIM] = fn2[FN_DIM:]
            fn2_swapped[FN_DIM:] = fn2[:FN_DIM]
            fn2 = fn2_swapped
            eval_val = -eval_val
            winner_val = 1.0 - winner_val
            static_val = -static_val

        # Spatial reflections on centered hex coordinates (feat 1: x_centered, feat 2: y_centered)
        if flip_x:
            features[:, 1] = -features[:, 1]
        if flip_y:
            features[:, 2] = -features[:, 2]

        return (
            torch.from_numpy(features),
            torch.from_numpy(adj.copy()),
            torch.from_numpy(fn2.copy()),
            torch.tensor(eval_val, dtype=torch.float32),
            torch.tensor(winner_val, dtype=torch.float32),
            torch.tensor(static_val, dtype=torch.float32),
            torch.tensor(weight_val, dtype=torch.float32),
        )

NUM_EDGE_CASES = 3  # 0: no edge, 1: flat edge, 2: vertical edge
ADJ_TO_EDGE_CASE = [0, 1, 1, 1, 1, 1, 1, 2, 2, 0]

# ---------------------------------------------------------
# Graph Multi-Head Attention (CPU-optimized, no exponentials)
# ---------------------------------------------------------
class GraphMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_tokens=NUM_TOKENS, num_types=NUM_PIECE_TYPES, n_heads=2, num_edge_cases=NUM_EDGE_CASES, use_relu_attn=True, sparse_attention=False, per_piece_type_weights=True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.num_tokens = num_tokens
        self.num_types = num_types
        self.n_heads = n_heads
        self.num_edge_cases = num_edge_cases
        self.d_k = d_model // n_heads
        self.use_relu_attn = use_relu_attn
        self.sparse_attention = sparse_attention
        self.per_piece_type_weights = per_piece_type_weights

        self.register_buffer("token_to_type", torch.tensor(TOKEN_TO_PIECE_TYPE, dtype=torch.long), persistent=False)
        self.register_buffer("adj_map", torch.tensor(ADJ_TO_EDGE_CASE, dtype=torch.long), persistent=False)

        if per_piece_type_weights:
            # 16 piece types have dedicated parameter matrices
            self.W_q = nn.Parameter(torch.Tensor(num_types, d_model, d_model))
            self.W_k = nn.Parameter(torch.Tensor(num_types, d_model, d_model))
            self.W_v = nn.Parameter(torch.Tensor(num_types, d_model, d_model))
            self.W_o = nn.Parameter(torch.Tensor(num_types, d_model, d_model))
            self.b_o = nn.Parameter(torch.zeros(num_types, d_model))
            for w in [self.W_q, self.W_k, self.W_v, self.W_o]:
                nn.init.xavier_uniform_(w)
        else:
            self.W_q = nn.Linear(d_model, d_model, bias=False)
            self.W_k = nn.Linear(d_model, d_model, bias=False)
            self.W_v = nn.Linear(d_model, d_model, bias=False)
            self.W_o = nn.Linear(d_model, d_model, bias=True)

        # Pairwise piece-type edge bias: [piece type 1][piece type 2][3 edge cases][n_heads]
        self.edge_bias = nn.Parameter(torch.zeros(num_types, num_types, num_edge_cases, n_heads))
        nn.init.normal_(self.edge_bias, std=0.02)

    def forward(self, h, adj, on_board_mask=None):
        # h: (B, N, d_model), adj: (B, N, N)
        B, N, _ = h.shape

        if self.per_piece_type_weights:
            w_q_28 = self.W_q[self.token_to_type]
            w_k_28 = self.W_k[self.token_to_type]
            w_v_28 = self.W_v[self.token_to_type]
            q = torch.einsum('bnd,nde->bne', h, w_q_28).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
            k = torch.einsum('bnd,nde->bne', h, w_k_28).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
            v = torch.einsum('bnd,nde->bne', h, w_v_28).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        else:
            q = self.W_q(h).view(B, N, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, N, d_k)
            k = self.W_k(h).view(B, N, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, N, d_k)
            v = self.W_v(h).view(B, N, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, N, d_k)

        # Scaled dot-product logits: (B, H, N, N)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)

        # Map adj (0..9) to 3 cases (0: no edge, 1: flat edge, 2: vertical edge)
        adj_case = self.adj_map[adj.long()]  # (B, N, N)

        # Pairwise piece-type lookup for all 28 tokens: (N, N, 3, n_heads)
        t_i = self.token_to_type.unsqueeze(1)  # (N, 1)
        t_j = self.token_to_type.unsqueeze(0)  # (1, N)
        eb_static = self.edge_bias[t_i, t_j]   # (N, N, 3, n_heads)

        # Gather per-pair edge bias for batch: (B, N, N, n_heads) -> (B, n_heads, N, N)
        tok_i = torch.arange(N, device=h.device).unsqueeze(1)
        tok_j = torch.arange(N, device=h.device).unsqueeze(0)
        e_bias = eb_static[tok_i, tok_j, adj_case].permute(0, 3, 1, 2)
        scores = scores + e_bias

        pair_mask = None
        if on_board_mask is not None:
            # on_board_mask is (B, N, 1) -> (B, 1, N, 1) * (B, 1, 1, N) -> (B, 1, N, N)
            pair_mask = on_board_mask.unsqueeze(1) * on_board_mask.transpose(1, 2).unsqueeze(1)

        if self.sparse_attention:
            # Mask out pairs with no edge (adj_case == 0)
            mask = (adj_case == 0).unsqueeze(1)  # (B, 1, N, N)
            if self.use_relu_attn:
                scores = scores.masked_fill(mask, 0.0)
            else:
                scores = scores.masked_fill(mask, -1e9)

        if self.use_relu_attn:
            # Fast ReLU / Squared-ReLU attention without exponentials
            attn = F.relu(scores) ** 2
            if pair_mask is not None:
                attn = attn * pair_mask
            denom = attn.sum(dim=-1, keepdim=True) + 1e-6
            attn = attn / denom
        else:
            if pair_mask is not None:
                scores = scores.masked_fill(~pair_mask.bool(), -1e9)
            attn = F.softmax(scores, dim=-1)

        out = torch.matmul(attn, v)  # (B, H, N, d_k)
        out = out.transpose(1, 2).contiguous().view(B, N, self.d_model)  # (B, N, d_model)
        if self.per_piece_type_weights:
            out = torch.einsum('bnd,nde->bne', out, self.W_o[self.token_to_type]) + self.b_o[self.token_to_type]
        else:
            out = self.W_o(out)

        if on_board_mask is not None:
            out = out * on_board_mask

        return out

# ---------------------------------------------------------
# Graph Transformer Block
# ---------------------------------------------------------
class GraphTransformerBlock(nn.Module):
    def __init__(self, d_model, num_tokens=NUM_TOKENS, num_types=NUM_PIECE_TYPES, n_heads=2, d_ff=32, use_relu_attn=True, sparse_attention=False, per_piece_type_weights=True):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_tokens = num_tokens
        self.num_types = num_types
        self.per_piece_type_weights = per_piece_type_weights
        self.register_buffer("token_to_type", torch.tensor(TOKEN_TO_PIECE_TYPE, dtype=torch.long), persistent=False)

        self.attn = GraphMultiHeadAttention(
            d_model,
            num_tokens=num_tokens,
            num_types=num_types,
            n_heads=n_heads,
            use_relu_attn=use_relu_attn,
            sparse_attention=sparse_attention,
            per_piece_type_weights=per_piece_type_weights,
        )
        if per_piece_type_weights:
            self.norm1_g = nn.Parameter(torch.ones(num_types, d_model))
            self.norm1_b = nn.Parameter(torch.zeros(num_types, d_model))
            self.W_ff1 = nn.Parameter(torch.Tensor(num_types, d_model, d_ff))
            self.b_ff1 = nn.Parameter(torch.zeros(num_types, d_ff))
            self.W_ff2 = nn.Parameter(torch.Tensor(num_types, d_ff, d_model))
            self.b_ff2 = nn.Parameter(torch.zeros(num_types, d_model))
            self.norm2_g = nn.Parameter(torch.ones(num_types, d_model))
            self.norm2_b = nn.Parameter(torch.zeros(num_types, d_model))
            nn.init.xavier_uniform_(self.W_ff1)
            nn.init.xavier_uniform_(self.W_ff2)
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Linear(d_ff, d_model),
            )
            self.norm2 = nn.LayerNorm(d_model)

    def _layer_norm(self, x, g, b):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + 1e-5) * g + b

    def forward(self, h, adj, on_board_mask=None):
        if self.per_piece_type_weights:
            n1_g = self.norm1_g[self.token_to_type]
            n1_b = self.norm1_b[self.token_to_type]
            h = self._layer_norm(h + self.attn(h, adj, on_board_mask), n1_g, n1_b)
            if on_board_mask is not None:
                h = h * on_board_mask
            ffn1 = F.relu(torch.einsum('bnd,nde->bne', h, self.W_ff1[self.token_to_type]) + self.b_ff1[self.token_to_type])
            ffn2 = torch.einsum('bnd,nde->bne', ffn1, self.W_ff2[self.token_to_type]) + self.b_ff2[self.token_to_type]
            n2_g = self.norm2_g[self.token_to_type]
            n2_b = self.norm2_b[self.token_to_type]
            h = self._layer_norm(h + ffn2, n2_g, n2_b)
            if on_board_mask is not None:
                h = h * on_board_mask
        else:
            h = self.norm1(h + self.attn(h, adj, on_board_mask))
            if on_board_mask is not None:
                h = h * on_board_mask
            h = self.norm2(h + self.ffn(h))
            if on_board_mask is not None:
                h = h * on_board_mask
        return h

# ---------------------------------------------------------
# Complete Graph Transformer Model
# ---------------------------------------------------------
class GraphTransformerEval(nn.Module):
    def __init__(
        self,
        d_in=TOKEN_FEAT_DIM,
        num_tokens=NUM_TOKENS,
        num_types=NUM_PIECE_TYPES,
        d_model=24,
        n_heads=2,
        num_layers=2,
        d_ff=32,
        d_fn2_proj=16,
        mlp_dims=[32, 16, 8],
        pool_type="queen_centric",
        num_pool_queries=2,
        use_relu_attn=True,
        sparse_attention=False,
        per_piece_type_weights=True,
        per_token_weights=None,
        zero_in_hand=True,
        clip=6000.0,
    ):
        super().__init__()
        if per_token_weights is not None:
            per_piece_type_weights = per_token_weights
        self.num_tokens = num_tokens
        self.num_types = num_types
        self.d_model = d_model
        self.d_fn2_proj = d_fn2_proj
        self.zero_in_hand = zero_in_hand
        self.clip = clip
        self.pool_type = pool_type
        self.per_piece_type_weights = per_piece_type_weights

        self.register_buffer("token_to_type", torch.tensor(TOKEN_TO_PIECE_TYPE, dtype=torch.long), persistent=False)

        # Layer 1: Piece-type projections (16 piece types: 8 White, 8 Black)
        if per_piece_type_weights:
            self.W_proj = nn.Parameter(torch.Tensor(num_types, d_in, d_model))
            self.b_proj = nn.Parameter(torch.zeros(num_types, d_model))
            nn.init.xavier_uniform_(self.W_proj)
        else:
            self.W_proj = nn.Linear(d_in, d_model)

        # Attention Layers (1..3)
        self.layers = nn.ModuleList([
            GraphTransformerBlock(
                d_model=d_model,
                num_tokens=num_tokens,
                num_types=num_types,
                n_heads=n_heads,
                d_ff=d_ff,
                use_relu_attn=use_relu_attn,
                sparse_attention=sparse_attention,
                per_piece_type_weights=per_piece_type_weights,
            )
            for _ in range(num_layers)
        ])

        # Pooling Layer Setup
        if pool_type == "flatten":
            in_dim = num_tokens * d_model
        elif pool_type == "diff":
            self.diff_proj = nn.Linear(d_model, d_model)
            in_dim = d_model
        elif pool_type == "queen_centric":
            in_dim = 4 * d_model  # My Queen, Enemy Queen, My Mean, Enemy Mean
        elif pool_type == "attention_pool":
            self.pool_queries = nn.Parameter(torch.randn(num_pool_queries, d_model) * 0.02)
            in_dim = num_pool_queries * d_model
        elif pool_type == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            in_dim = d_model
        else:
            raise ValueError(f"Unknown pool_type: {pool_type}")

        self.in_dim_trans = in_dim

        # FN2 handcrafted features linear projection
        self.fn2_proj = nn.Linear(FN2_DIM, d_fn2_proj)

        # Auxiliary Linear Regression heads (+ tanh * clip) before concat
        # 1. From Graph Transformer representation
        self.trans_aux_linear = nn.Linear(in_dim, 1)
        # 2. From FN2 tabular features
        self.fn2_aux_linear = nn.Linear(FN2_DIM, 1)

        # Concat dimension for Head MLP
        in_dim_fused = in_dim + d_fn2_proj

        head_layers = []
        curr_dim = in_dim_fused
        for out_dim in mlp_dims:
            head_layers.append(nn.Linear(curr_dim, out_dim))
            head_layers.append(nn.ReLU())
            curr_dim = out_dim
        head_layers.append(nn.Linear(curr_dim, 2))  # [eval, winner]
        self.head = nn.Sequential(*head_layers)

    def forward(self, x, adj, fn2=None, return_aux=False):
        # x: (B, 28, 8), adj: (B, 28, 28), fn2: (B, 104)
        B, N, _ = x.shape
        adj = adj.long()

        on_board_mask = (x[:, :, 0:1] > 0.5).float() if self.zero_in_hand else None

        # Piece-type linear projection
        if self.per_piece_type_weights:
            h = torch.einsum('bni,nid->bnd', x, self.W_proj[self.token_to_type]) + self.b_proj[self.token_to_type]
        else:
            h = self.W_proj(x)

        if on_board_mask is not None:
            h = h * on_board_mask

        # Transformer blocks
        for layer in self.layers:
            h = layer(h, adj, on_board_mask)

        # Token Pooling
        if self.pool_type == "flatten":
            pooled = h.reshape(B, -1)
        elif self.pool_type == "diff":
            # My pieces: 0..13, Enemy pieces: 14..27
            my_h = self.diff_proj(h[:, :14, :]).sum(dim=1)
            enemy_h = self.diff_proj(h[:, 14:, :]).sum(dim=1)
            pooled = my_h - enemy_h
        elif self.pool_type == "queen_centric":
            q_my = h[:, 0, :]       # My Queen
            q_enemy = h[:, 14, :]    # Enemy Queen
            my_mean = h[:, :14, :].mean(dim=1)
            enemy_mean = h[:, 14:, :].mean(dim=1)
            pooled = torch.cat([q_my, q_enemy, my_mean, enemy_mean], dim=-1)
        elif self.pool_type == "attention_pool":
            # Q: (num_queries, d_model), K, V: (B, N, d_model)
            q = self.pool_queries.unsqueeze(0).expand(B, -1, -1)  # (B, K, d_model)
            scores = torch.matmul(q, h.transpose(-2, -1)) / (self.d_model ** 0.5)  # (B, K, N)
            attn = F.softmax(scores, dim=-1)
            pooled = torch.matmul(attn, h).reshape(B, -1)  # (B, K * d_model)
        elif self.pool_type == "cls":
            pooled = h[:, 0, :]  # or specialized token
        else:
            pooled = h.reshape(B, -1)

        # Process FN2 features
        if fn2 is not None:
            h_fn2 = F.relu(self.fn2_proj(fn2))
            aux_eval_fn2 = torch.tanh(self.fn2_aux_linear(fn2)[:, 0]) * self.clip
        else:
            h_fn2 = torch.zeros(B, self.d_fn2_proj, device=x.device)
            aux_eval_fn2 = torch.zeros(B, device=x.device)

        # Auxiliary linear regression from Graph Transformer pooled representation
        aux_eval_trans = torch.tanh(self.trans_aux_linear(pooled)[:, 0]) * self.clip

        # Concat transformer features and FN2 features
        fused = torch.cat([pooled, h_fn2], dim=-1)

        # Output head
        out = self.head(fused)
        eval_pred = F.softsign(out[:, 0]) * 1.5 * self.clip
        winner_pred = torch.sigmoid(out[:, 1])

        if return_aux:
            return eval_pred, winner_pred, aux_eval_trans, aux_eval_fn2
        return eval_pred, winner_pred

# ---------------------------------------------------------
# Target Distribution Plotting & Normalization
# ---------------------------------------------------------
def analyze_target_distribution(y, normalize=False, out_prefix="logs/y_distribution"):
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)

    # 1. Original Distribution
    plt.figure(figsize=(10, 5))
    counts, bins, _ = plt.hist(y, bins=100, color='royalblue', edgecolor='black', alpha=0.7)
    plt.title(f"Target (y) Distribution (N={len(y):,})")
    plt.xlabel("Evaluation (Centipawns)")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    orig_path = f"{out_prefix}.png"
    plt.savefig(orig_path, dpi=150)
    plt.close()
    print(f"Saved target distribution plot to {orig_path}")

    sample_weights = None
    if normalize:
        # Sample re-weighting so that the effective binned distribution is flat (horizontal line)
        bin_indices = np.digitize(y, bins) - 1
        bin_indices = np.clip(bin_indices, 0, len(counts) - 1)
        bin_counts = np.maximum(counts[bin_indices], 1.0)
        # Weight is inversely proportional to frequency
        sample_weights = (1.0 / bin_counts)
        sample_weights = (sample_weights / sample_weights.mean()).astype(np.float32)

        # Plot weighted normalized distribution
        plt.figure(figsize=(10, 5))
        plt.hist(y, bins=100, weights=sample_weights, color='forestgreen', edgecolor='black', alpha=0.7)
        plt.title(f"Normalized / Re-weighted Target Distribution (Flat Line Target)")
        plt.xlabel("Evaluation (Centipawns)")
        plt.ylabel("Weighted Density")
        plt.grid(True, alpha=0.3)
        norm_path = f"{out_prefix}_normalized.png"
        plt.savefig(norm_path, dpi=150)
        plt.close()
        print(f"Saved normalized target distribution plot to {norm_path}")

    return sample_weights

# ---------------------------------------------------------
# Export Weights to Rust
# ---------------------------------------------------------
def export_rust_weights(model, args, out_path="logs/eval_gt.rs", epoch=None, train_metrics=None, val_metrics=None):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    total_params = sum(p.numel() for p in model.parameters())
    max_flops, exp_flops, cached_flops = count_transformer_flops(args)

    with open(out_path, "w") as f:
        f.write("// =========================================================\n")
        f.write("// Auto-generated Graph Transformer Weights for Bee-Search\n")
        f.write(f"// Config: d_model={args.d_model}, n_heads={args.n_heads}, num_layers={args.num_layers}, d_ff={args.d_ff}\n")
        f.write(f"// zero_in_hand={args.zero_in_hand}, use_relu_attn={args.use_relu_attn}, sparse_attn={args.sparse_attention}\n")
        f.write(f"// Total Parameters: {total_params:,}\n")
        f.write(f"// FLOPs: Expected (uncached zero-in-hand) = {exp_flops:,.0f} | Cached Tree-Search = {cached_flops:,.0f} | Max = {max_flops:,.0f}\n")
        if epoch is not None:
            f.write(f"// Epochs: {epoch}\n")
        if train_metrics is not None:
            rmse_str = f"{train_metrics['rmse']:.2f}" if 'rmse' in train_metrics else "N/A"
            mae_str = f"{train_metrics['mae']:.2f}" if 'mae' in train_metrics else "N/A"
            r2_str = f"{train_metrics['r2']:.4f}" if 'r2' in train_metrics else "N/A"
            f.write(f"// Train Metrics: RMSE = {rmse_str} cp | MAE = {mae_str} cp | R2 = {r2_str}\n")
        if val_metrics is not None:
            rmse_str = f"{val_metrics['rmse']:.2f}" if 'rmse' in val_metrics else "N/A"
            mae_str = f"{val_metrics['mae']:.2f}" if 'mae' in val_metrics else "N/A"
            r2_str = f"{val_metrics['r2']:.4f}" if 'r2' in val_metrics else "N/A"
            f.write(f"// Validation Metrics: RMSE = {rmse_str} cp | MAE = {mae_str} cp | R2 = {r2_str}\n")
        f.write("// =========================================================\n\n")

        f.write(f"pub const GT_NUM_TOKENS: usize = {NUM_TOKENS};\n")
        f.write(f"pub const GT_FEAT_DIM: usize = {TOKEN_FEAT_DIM};\n")
        f.write(f"pub const GT_D_MODEL: usize = {args.d_model};\n")
        f.write(f"pub const GT_N_HEADS: usize = {args.n_heads};\n")
        f.write(f"pub const GT_D_K: usize = {args.d_model // args.n_heads};\n")
        f.write(f"pub const GT_NUM_LAYERS: usize = {args.num_layers};\n")
        f.write(f"pub const GT_D_FF: usize = {args.d_ff};\n")
        f.write(f"pub const GT_D_FN2_PROJ: usize = {getattr(args, 'd_fn2_proj', 16)};\n")
        f.write(f"pub const GT_ZERO_IN_HAND: bool = {'true' if args.zero_in_hand else 'false'};\n")
        f.write(f"pub const GT_SPARSE_ATTN: bool = {'true' if args.sparse_attention else 'false'};\n")
        f.write(f"pub const GT_POOL_TYPE: &str = \"{args.pool_type}\";\n\n")

        # 1. Piece Projections
        if model.per_piece_type_weights:
            w_proj = model.W_proj[TOKEN_TO_PIECE_TYPE].detach().cpu().numpy()  # (28, 8, d_model)
            b_proj = model.b_proj[TOKEN_TO_PIECE_TYPE].detach().cpu().numpy()  # (28, d_model)
        else:
            w_proj = model.W_proj.detach().cpu().numpy()
            b_proj = model.b_proj.detach().cpu().numpy()

        f.write(f"pub const GT_W_PROJ: [[[f32; {args.d_model}]; {TOKEN_FEAT_DIM}]; {NUM_TOKENS}] = [\n")
        for tok in range(NUM_TOKENS):
            f.write("\t[\n")
            for feat in range(TOKEN_FEAT_DIM):
                row_str = ", ".join(f"{float(v):.8e}f32" for v in w_proj[tok, feat])
                f.write(f"\t\t[{row_str}],\n")
            f.write("\t],\n")
        f.write("];\n\n")

        f.write(f"pub const GT_B_PROJ: [[f32; {args.d_model}]; {NUM_TOKENS}] = [\n")
        for tok in range(NUM_TOKENS):
            row_str = ", ".join(f"{float(v):.8e}f32" for v in b_proj[tok])
            f.write(f"\t[{row_str}],\n")
        f.write("];\n\n")

        # 2. Transformer Layers
        for l_idx, layer in enumerate(model.layers):
            prefix = f"L{l_idx}_"
            # Attention weights
            if model.per_piece_type_weights:
                wq = layer.attn.W_q[TOKEN_TO_PIECE_TYPE].detach().cpu().numpy()  # (NUM_TOKENS, d_model, d_model)
                wk = layer.attn.W_k[TOKEN_TO_PIECE_TYPE].detach().cpu().numpy()
                wv = layer.attn.W_v[TOKEN_TO_PIECE_TYPE].detach().cpu().numpy()
                wo = layer.attn.W_o[TOKEN_TO_PIECE_TYPE].detach().cpu().numpy()
                bo = layer.attn.b_o[TOKEN_TO_PIECE_TYPE].detach().cpu().numpy()  # (NUM_TOKENS, d_model)
                eb = layer.attn.edge_bias[TOKEN_TO_PIECE_TYPE][:, TOKEN_TO_PIECE_TYPE].detach().cpu().numpy()  # (NUM_TOKENS, NUM_TOKENS, NUM_EDGE_CASES, n_heads)

                for name, arr in [("WQ", wq), ("WK", wk), ("WV", wv), ("WO", wo)]:
                    f.write(f"pub const GT_{prefix}{name}: [[[f32; {args.d_model}]; {args.d_model}]; {NUM_TOKENS}] = [\n")
                    for tok_mat in arr:
                        f.write("\t[\n")
                        for row in tok_mat:
                            f.write("\t\t[" + ", ".join(f"{float(v):.8e}f32" for v in row) + "],\n")
                        f.write("\t],\n")
                    f.write("];\n\n")

                f.write(f"pub const GT_{prefix}BO: [[f32; {args.d_model}]; {NUM_TOKENS}] = [\n")
                for row in bo:
                    f.write("\t[" + ", ".join(f"{float(v):.8e}f32" for v in row) + "],\n")
                f.write("];\n\n")

                f.write(f"pub const GT_{prefix}EDGE_BIAS: [[[[f32; {args.n_heads}]; {NUM_EDGE_CASES}]; {NUM_TOKENS}]; {NUM_TOKENS}] = [\n")
                for i_tok in range(NUM_TOKENS):
                    f.write("\t[\n")
                    for j_tok in range(NUM_TOKENS):
                        f.write("\t\t[\n")
                        for case in range(NUM_EDGE_CASES):
                            f.write("\t\t\t[" + ", ".join(f"{float(v):.8e}f32" for v in eb[i_tok, j_tok, case]) + "],\n")
                        f.write("\t\t],\n")
                    f.write("\t],\n")
                f.write("];\n\n")

                # LayerNorm 1
                norm1_g = layer.norm1_g[TOKEN_TO_PIECE_TYPE].detach().cpu().numpy()
                norm1_b = layer.norm1_b[TOKEN_TO_PIECE_TYPE].detach().cpu().numpy()
                f.write(f"pub const GT_{prefix}NORM1_G: [[f32; {args.d_model}]; {NUM_TOKENS}] = [\n")
                for row in norm1_g:
                    f.write("\t[" + ", ".join(f"{float(v):.8e}f32" for v in row) + "],\n")
                f.write("];\n\n")

                f.write(f"pub const GT_{prefix}NORM1_B: [[f32; {args.d_model}]; {NUM_TOKENS}] = [\n")
                for row in norm1_b:
                    f.write("\t[" + ", ".join(f"{float(v):.8e}f32" for v in row) + "],\n")
                f.write("];\n\n")

                # FFN
                w_ff1 = layer.W_ff1[TOKEN_TO_PIECE_TYPE].detach().cpu().numpy()  # (NUM_TOKENS, d_model, d_ff)
                b_ff1 = layer.b_ff1[TOKEN_TO_PIECE_TYPE].detach().cpu().numpy()  # (NUM_TOKENS, d_ff)
                w_ff2 = layer.W_ff2[TOKEN_TO_PIECE_TYPE].detach().cpu().numpy()  # (NUM_TOKENS, d_ff, d_model)
                b_ff2 = layer.b_ff2[TOKEN_TO_PIECE_TYPE].detach().cpu().numpy()  # (NUM_TOKENS, d_model)

                f.write(f"pub const GT_{prefix}FFN1_W: [[[f32; {args.d_ff}]; {args.d_model}]; {NUM_TOKENS}] = [\n")
                for tok_mat in w_ff1:
                    f.write("\t[\n")
                    for row in tok_mat:
                        f.write("\t\t[" + ", ".join(f"{float(v):.8e}f32" for v in row) + "],\n")
                    f.write("\t],\n")
                f.write("];\n\n")

                f.write(f"pub const GT_{prefix}FFN1_B: [[f32; {args.d_ff}]; {NUM_TOKENS}] = [\n")
                for row in b_ff1:
                    f.write("\t[" + ", ".join(f"{float(v):.8e}f32" for v in row) + "],\n")
                f.write("];\n\n")

                f.write(f"pub const GT_{prefix}FFN2_W: [[[f32; {args.d_model}]; {args.d_ff}]; {NUM_TOKENS}] = [\n")
                for tok_mat in w_ff2:
                    f.write("\t[\n")
                    for row in tok_mat:
                        f.write("\t\t[" + ", ".join(f"{float(v):.8e}f32" for v in row) + "],\n")
                    f.write("\t],\n")
                f.write("];\n\n")

                f.write(f"pub const GT_{prefix}FFN2_B: [[f32; {args.d_model}]; {NUM_TOKENS}] = [\n")
                for row in b_ff2:
                    f.write("\t[" + ", ".join(f"{float(v):.8e}f32" for v in row) + "],\n")
                f.write("];\n\n")

                # LayerNorm 2
                norm2_g = layer.norm2_g[TOKEN_TO_PIECE_TYPE].detach().cpu().numpy()
                norm2_b = layer.norm2_b[TOKEN_TO_PIECE_TYPE].detach().cpu().numpy()
                f.write(f"pub const GT_{prefix}NORM2_G: [[f32; {args.d_model}]; {NUM_TOKENS}] = [\n")
                for row in norm2_g:
                    f.write("\t[" + ", ".join(f"{float(v):.8e}f32" for v in row) + "],\n")
                f.write("];\n\n")

                f.write(f"pub const GT_{prefix}NORM2_B: [[f32; {args.d_model}]; {NUM_TOKENS}] = [\n")
                for row in norm2_b:
                    f.write("\t[" + ", ".join(f"{float(v):.8e}f32" for v in row) + "],\n")
                f.write("];\n\n")
            else:
                wq = layer.attn.W_q.weight.detach().cpu().numpy().T  # (d_model, d_model)
                wk = layer.attn.W_k.weight.detach().cpu().numpy().T
                wv = layer.attn.W_v.weight.detach().cpu().numpy().T
                wo = layer.attn.W_o.weight.detach().cpu().numpy().T
                bo = layer.attn.W_o.bias.detach().cpu().numpy()
                eb = layer.attn.edge_bias.detach().cpu().numpy()  # (10, n_heads)

                for name, mat in [("WQ", wq), ("WK", wk), ("WV", wv), ("WO", wo)]:
                    f.write(f"pub const GT_{prefix}{name}: [[f32; {args.d_model}]; {args.d_model}] = [\n")
                    for row in mat:
                        f.write("\t[" + ", ".join(f"{float(v):.8e}f32" for v in row) + "],\n")
                    f.write("];\n\n")

                f.write(f"pub const GT_{prefix}BO: [f32; {args.d_model}] = [" + ", ".join(f"{float(v):.8e}f32" for v in bo) + "];\n\n")

                f.write(f"pub const GT_{prefix}EDGE_BIAS: [[f32; {args.n_heads}]; {NUM_EDGE_TYPES}] = [\n")
                for row in eb:
                    f.write("\t[" + ", ".join(f"{float(v):.8e}f32" for v in row) + "],\n")
                f.write("];\n\n")

                # LayerNorm 1
                gamma1 = layer.norm1.weight.detach().cpu().numpy()
                beta1 = layer.norm1.bias.detach().cpu().numpy()
                f.write(f"pub const GT_{prefix}NORM1_G: [f32; {args.d_model}] = [" + ", ".join(f"{float(v):.8e}f32" for v in gamma1) + "];\n")
                f.write(f"pub const GT_{prefix}NORM1_B: [f32; {args.d_model}] = [" + ", ".join(f"{float(v):.8e}f32" for v in beta1) + "];\n\n")

                # FFN
                w_ffn1 = layer.ffn[0].weight.detach().cpu().numpy().T  # (d_model, d_ff)
                b_ffn1 = layer.ffn[0].bias.detach().cpu().numpy()      # (d_ff)
                w_ffn2 = layer.ffn[2].weight.detach().cpu().numpy().T  # (d_ff, d_model)
                b_ffn2 = layer.ffn[2].bias.detach().cpu().numpy()      # (d_model)

                f.write(f"pub const GT_{prefix}FFN1_W: [[f32; {args.d_ff}]; {args.d_model}] = [\n")
                for row in w_ffn1:
                    f.write("\t[" + ", ".join(f"{float(v):.8e}f32" for v in row) + "],\n")
                f.write("];\n\n")
                f.write(f"pub const GT_{prefix}FFN1_B: [f32; {args.d_ff}] = [" + ", ".join(f"{float(v):.8e}f32" for v in b_ffn1) + "];\n\n")

                f.write(f"pub const GT_{prefix}FFN2_W: [[f32; {args.d_model}]; {args.d_ff}] = [\n")
                for row in w_ffn2:
                    f.write("\t[" + ", ".join(f"{float(v):.8e}f32" for v in row) + "],\n")
                f.write("];\n\n")
                f.write(f"pub const GT_{prefix}FFN2_B: [f32; {args.d_model}] = [" + ", ".join(f"{float(v):.8e}f32" for v in b_ffn2) + "];\n\n")

                # LayerNorm 2
                gamma2 = layer.norm2.weight.detach().cpu().numpy()
                beta2 = layer.norm2.bias.detach().cpu().numpy()
                f.write(f"pub const GT_{prefix}NORM2_G: [f32; {args.d_model}] = [" + ", ".join(f"{float(v):.8e}f32" for v in gamma2) + "];\n")
                f.write(f"pub const GT_{prefix}NORM2_B: [f32; {args.d_model}] = [" + ", ".join(f"{float(v):.8e}f32" for v in beta2) + "];\n\n")

        # 3. FN2 Handcrafted Feature Projection
        w_fn2 = model.fn2_proj.weight.detach().cpu().numpy().T  # (FN2_DIM, d_fn2_proj)
        b_fn2 = model.fn2_proj.bias.detach().cpu().numpy()
        d_fn2 = w_fn2.shape[1]

        f.write(f"pub const GT_FN2_PROJ_W: [[f32; {d_fn2}]; {FN2_DIM}] = [\n")
        for row in w_fn2:
            f.write("\t[" + ", ".join(f"{float(v):.8e}f32" for v in row) + "],\n")
        f.write("];\n\n")

        f.write(f"pub const GT_FN2_PROJ_B: [f32; {d_fn2}] = [" + ", ".join(f"{float(v):.8e}f32" for v in b_fn2) + "];\n\n")

        # 4. Auxiliary Linear Regression Heads
        w_aux_trans = model.trans_aux_linear.weight.detach().cpu().numpy().T  # (in_dim_trans, 1)
        b_aux_trans = float(model.trans_aux_linear.bias[0].item())
        f.write(f"pub const GT_TRANS_AUX_W: [f32; {model.in_dim_trans}] = [" + ", ".join(f"{float(v[0]):.8e}f32" for v in w_aux_trans) + "];\n")
        f.write(f"pub const GT_TRANS_AUX_B: f32 = {b_aux_trans:.8e}f32;\n\n")

        w_aux_fn2 = model.fn2_aux_linear.weight.detach().cpu().numpy().T  # (FN2_DIM, 1)
        b_aux_fn2 = float(model.fn2_aux_linear.bias[0].item())
        f.write(f"pub const GT_FN2_AUX_W: [f32; {FN2_DIM}] = [" + ", ".join(f"{float(v[0]):.8e}f32" for v in w_aux_fn2) + "];\n")
        f.write(f"pub const GT_FN2_AUX_B: f32 = {b_aux_fn2:.8e}f32;\n\n")

        # 5. Readout Head MLP (taking fused transformer + FN2 features)
        head_linear_idx = 1
        for mod in model.head:
            if isinstance(mod, nn.Linear):
                w = mod.weight.detach().cpu().numpy().T
                b = mod.bias.detach().cpu().numpy()
                is_final = (w.shape[1] == 2)
                if is_final:
                    # Keep first column (evaluation) for static eval
                    w_eval = w[:, :1]
                    b_eval = b[:1]
                    f.write(f"pub const GT_HEAD_W{head_linear_idx}: [[f32; 1]; {w_eval.shape[0]}] = [\n")
                    for row in w_eval:
                        f.write("\t[" + ", ".join(f"{float(v):.8e}f32" for v in row) + "],\n")
                    f.write("];\n\n")
                    f.write(f"pub const GT_HEAD_B{head_linear_idx}: [f32; 1] = [{float(b_eval[0]):.8e}f32];\n\n")
                else:
                    r, c = w.shape
                    f.write(f"pub const GT_HEAD_W{head_linear_idx}: [[f32; {c}]; {r}] = [\n")
                    for row in w:
                        f.write("\t[" + ", ".join(f"{float(v):.8e}f32" for v in row) + "],\n")
                    f.write("];\n\n")
                    f.write(f"pub const GT_HEAD_B{head_linear_idx}: [f32; {len(b)}] = [" + ", ".join(f"{float(v):.8e}f32" for v in b) + "];\n\n")
                head_linear_idx += 1

    #print(f"Exported Rust weights to {out_path}")

def count_transformer_flops(args, avg_on_board=12.0):
    """
    Computes theoretical FLOPs for Graph Transformer CPU inference:
    1. Max FLOPs: full uncached inference with all 28 tokens active in Layer 0.
    2. Expected FLOPs: uncached inference skipping in-hand zero tokens in Layer 0.
    3. Cached Tree-Search FLOPs: TokenCache preserving Layer 0 h0 and Layer 0 QKV, recomputing only ~2 dirty tokens.
    """
    d_model = args.d_model
    d_k = d_model // args.n_heads
    d_ff = args.d_ff
    num_layers = args.num_layers
    d_fn2_proj = getattr(args, "d_fn2_proj", 16)
    N = NUM_TOKENS  # 28

    # 1. Layer 0 Initial Projections W_proj (8 -> d_model)
    proj_flops_per_tok = 2 * TOKEN_FEAT_DIM * d_model + d_model

    # 2. Q, K, V Projections (d_model -> d_model for each of Q, K, V)
    qkv_flops_per_tok = 3 * (2 * d_model * d_model)

    # Layer Attention (dot, bias, squared-relu, denom, V-mult, out-proj, LN1)
    attn_dot_flops = args.n_heads * N * N * (2 * d_k)
    attn_score_flops = args.n_heads * N * N * 3  # bias + relu + square
    attn_norm_flops = args.n_heads * N * (2 * N)  # sum + scale
    attn_v_flops = args.n_heads * N * d_k * (2 * N)
    attn_out_flops = N * (2 * d_model * d_model + d_model)
    ln1_flops = N * (4 * d_model)
    layer_attn_flops = attn_dot_flops + attn_score_flops + attn_norm_flops + attn_v_flops + attn_out_flops + ln1_flops

    # Layer FFN (FFN1, FFN2, LN2)
    ffn1_flops = N * (2 * d_model * d_ff + 2 * d_ff)  # linear + relu
    ffn2_flops = N * (2 * d_ff * d_model + d_model)
    ln2_flops = N * (4 * d_model)
    layer_ffn_flops = ffn1_flops + ffn2_flops + ln2_flops

    per_layer_flops = layer_attn_flops + layer_ffn_flops

    # Subsequent Layers Q, K, V Projections (Layers 1..num_layers-1)
    subsequent_qkv_flops = (num_layers - 1) * (N * qkv_flops_per_tok)

    # FN2 Projection FLOPs
    fn2_proj_flops = 2 * FN2_DIM * d_fn2_proj + d_fn2_proj

    # Readout Head MLP FLOPs
    if args.pool_type == "flatten":
        in_dim = N * d_model
    elif args.pool_type == "queen_centric":
        in_dim = 4 * d_model
    elif args.pool_type == "attention_pool":
        in_dim = args.num_pool_queries * d_model
    else:  # diff, cls
        in_dim = d_model

    head_flops = 0
    curr_dim = in_dim + d_fn2_proj
    for out_dim in args.mlp_dims:
        head_flops += 2 * curr_dim * out_dim + out_dim  # Linear + ReLU
        curr_dim = out_dim
    head_flops += 2 * curr_dim * 1 + 1 + 10  # Final linear to 1 + softsign + scale

    # Total Max FLOPs (Uncached, all 28 tokens active in Layer 0)
    max_flops = (
        N * proj_flops_per_tok
        + N * qkv_flops_per_tok
        + (num_layers * per_layer_flops)
        + subsequent_qkv_flops
        + fn2_proj_flops
        + head_flops
    )

    # Expected FLOPs (Uncached, skipping in-hand zero tokens in Layer 0 W_proj & QKV)
    expected_uncached_flops = (
        avg_on_board * proj_flops_per_tok
        + avg_on_board * qkv_flops_per_tok
        + (num_layers * per_layer_flops)
        + subsequent_qkv_flops
        + fn2_proj_flops
        + head_flops
    )

    # Cached Tree-Search FLOPs (TokenCache: only ~2 dirty tokens recomputed for Layer 0 W_proj & QKV)
    dirty_tokens = 2.0
    cached_flops = (
        dirty_tokens * proj_flops_per_tok
        + dirty_tokens * qkv_flops_per_tok
        + (num_layers * per_layer_flops)
        + subsequent_qkv_flops
        + fn2_proj_flops
        + head_flops
    )

    return max_flops, expected_uncached_flops, cached_flops

def load_model_weights_from_ckpt(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    orig_state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    new_state = {}

    for k, v in orig_state.items():
        if k in model.state_dict():
            target_shape = model.state_dict()[k].shape
            if v.shape == target_shape:
                new_state[k] = v
            elif v.ndim >= 1 and v.shape[0] == 28 and target_shape[0] == 16:
                # Average across identical piece types
                shape_tail = list(v.shape[1:])
                new_shape = [16] + shape_tail
                avg_tensor = torch.zeros(new_shape, dtype=v.dtype, device=device)
                for t, idxs in enumerate(TYPE_TO_TOKENS):
                    avg_tensor[t] = v[idxs].mean(dim=0)
                new_state[k] = avg_tensor
                print(f"  Adapted {k:30s}: [28, ...] -> {list(new_shape)} (averaged)")
            elif k.endswith("edge_bias") and v.ndim == 2 and target_shape == (16, 16, 3, v.shape[-1]):
                # Adapt 1D edge bias [10, n_heads] to pairwise [16, 16, 3, n_heads]
                c0 = v[0]
                c1 = v[1:7].mean(dim=0) if v.shape[0] >= 7 else v[0]
                c2 = v[7:9].mean(dim=0) if v.shape[0] >= 9 else v[0]
                case_bias = torch.stack([c0, c1, c2], dim=0)  # (3, n_heads)
                new_state[k] = case_bias.unsqueeze(0).unsqueeze(0).expand(16, 16, 3, v.shape[-1]).clone()
                print(f"  Adapted {k:30s}: {list(v.shape)} -> {list(target_shape)} (expanded across piece types)")
            else:
                new_state[k] = v
        else:
            new_state[k] = v

    # Also average head.0.weight slices if shape is [32, 448]
    if "head.0.weight" in new_state and new_state["head.0.weight"].shape == (32, 448):
        head_w = new_state["head.0.weight"].clone()
        d_model = 448 // 28
        for t, idxs in enumerate(TYPE_TO_TOKENS):
            if len(idxs) > 1:
                slices = torch.stack([head_w[:, i*d_model:(i+1)*d_model] for i in idxs], dim=0)
                mean_slice = slices.mean(dim=0)
                for i in idxs:
                    head_w[:, i*d_model:(i+1)*d_model] = mean_slice
        new_state["head.0.weight"] = head_w
        print("  Adapted head.0.weight: averaged slices for identical piece types")

    model.load_state_dict(new_state)
    return ckpt

# ---------------------------------------------------------
# Main Training Routine
# ---------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Train Graph Transformer for Hive static evaluation")
    p.add_argument("--data", default="logs/position_graphs.bin", help="Path to binary position graphs")
    p.add_argument("--subset", type=int, default=None, help="Train on first N samples for fast iteration / sanity check")
    p.add_argument("--test-size", type=float, default=0.1)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--clip", type=float, default=6000.0)

    # Architecture Hyperparameters
    p.add_argument("--d-model", type=int, default=10, help="Transformer hidden state size")
    p.add_argument("--n-heads", type=int, default=2, help="Number of attention heads")
    p.add_argument("--num-layers", type=int, default=2, help="Number of Graph Transformer blocks (<= 3)")
    p.add_argument("--d-ff", type=int, default=12, help="Feed-forward intermediate size")
    p.add_argument("--d-fn2-proj", type=int, default=28, help="Intermediate dimension for FN2 handcrafted features projection")
    p.add_argument("--mlp-dims", type=int, nargs="+", default=[32, 16, 8], help="Output MLP hidden dimensions")
    p.add_argument("--pool-type", choices=["flatten", "diff", "queen_centric", "attention_pool", "cls"], default="queen_centric", help="Token pooling strategy")
    p.add_argument("--num-pool-queries", type=int, default=2, help="Number of queries for attention_pool")
    p.add_argument("--per-token-weights", action="store_true", default=True, help="Each piece token has its own dedicated matrices (default)")
    p.add_argument("--share-token-weights", action="store_true", help="Share attention and FFN weights across all tokens")
    p.add_argument("--sparse-attention", action="store_true", help="Attend only along graph edges")
    p.add_argument("--use-relu-attn", action="store_true", default=True, help="Use ReLU attention (no exponentials)")
    p.add_argument("--use-softmax-attn", action="store_true", help="Use standard softmax attention instead of ReLU")
    p.add_argument("--zero-in-hand", action="store_true", default=True, help="Zero out unplaced token features for faster inference")
    p.add_argument("--no-zero-in-hand", action="store_true", help="Keep token projections for unplaced pieces")

    # Training Parameters
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--num-workers", type=int, default=2, help="Number of DataLoader worker processes for streaming")
    p.add_argument("--lambda-w", type=float, default=2e-3, help="Weight for winner BCE loss")
    p.add_argument("--lambda-aux", type=float, default=0.05, help="Weight for auxiliary linear regression losses (trans & fn2)")
    p.add_argument("--l2", type=float, default=3e-6, help="L2 regularization")
    p.add_argument("--no-scheduler", action="store_true", help="Disable learning rate cosine annealing scheduler (useful for short training runs)")
    p.add_argument("--no-augment", action="store_true", help="Disable symmetric data augmentation")
    p.add_argument("--normalize-target", action="store_true", help="Equalize target distribution to a flat histogram")
    p.add_argument("--eval-only", action="store_true", help="Run evaluation on the test set and exit")
    p.add_argument("--load-ckpt", default=None, help="Path to checkpoint .pth to resume full training state (epoch, optimizer, scheduler)")
    p.add_argument("--load-model-ckpt", default=None, help="Path to checkpoint .pth to load model weights only (fresh training from epoch 0 with new optimizer/scheduler)")
    p.add_argument("--export-rust", default="logs/eval_gt.rs", help="Path to write Rust const weights")

    args = p.parse_args()

    if args.share_token_weights:
        args.per_token_weights = False
    if args.use_softmax_attn:
        args.use_relu_attn = False
    if args.no_zero_in_hand:
        args.zero_in_hand = False

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Binary graph data not found at {args.data}. Run `cargo run --release --bin extract-graphs` first.")

    print(f"Loading data from {args.data} via mmap...")
    raw_mmap = np.memmap(args.data, dtype=SAMPLE_DTYPE, mode='r')
    total_samples = len(raw_mmap)
    print(f"Total dataset records: {total_samples:,}")

    if args.subset is not None and args.subset < total_samples:
        n_samples = args.subset
        print(f"Using subset of {n_samples:,} records for training.")
        raw_mmap = raw_mmap[:n_samples]
    else:
        n_samples = total_samples

    features_raw = raw_mmap['features']
    adj_raw = raw_mmap['adj']
    y_raw = raw_mmap['eval']
    winner_raw = raw_mmap['winner']
    static_raw = raw_mmap['static_eval']

    # Analyze and plot target distribution
    weights_raw = analyze_target_distribution(
        y_raw, normalize=args.normalize_target, out_prefix="logs/y_distribution"
    )

    # Split train/test indices
    idx = np.arange(n_samples)
    idx_train, idx_test = train_test_split(idx, test_size=args.test_size, random_state=args.random_state)

    augment = not args.no_augment
    train_ds = LazyBinaryGraphDataset(raw_mmap, idx_train, sample_weights=weights_raw, augment=augment)
    test_ds = LazyBinaryGraphDataset(raw_mmap, idx_test, sample_weights=weights_raw, augment=augment)

    print(f"Train samples (augmented): {len(train_ds):,}, Test samples (augmented): {len(test_ds):,}")

    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        pin_memory=pin_mem,
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch,
        shuffle=False,
        pin_memory=pin_mem,
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = GraphTransformerEval(
        d_in=TOKEN_FEAT_DIM,
        num_tokens=NUM_TOKENS,
        num_types=NUM_PIECE_TYPES,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        d_fn2_proj=args.d_fn2_proj,
        mlp_dims=args.mlp_dims,
        pool_type=args.pool_type,
        num_pool_queries=args.num_pool_queries,
        use_relu_attn=args.use_relu_attn,
        sparse_attention=args.sparse_attention,
        per_piece_type_weights=args.per_token_weights,
        zero_in_hand=args.zero_in_hand,
        clip=args.clip,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    sample_n = min(10000, n_samples)
    avg_on_board = float(np.mean(np.sum(features_raw[:sample_n, :, 0] > 0.5, axis=1)))
    max_flops, exp_flops, cached_flops = count_transformer_flops(args, avg_on_board)

    print("\nModel Architecture & FLOPs Summary:")
    print(f"  Graph Transformer Layers: {args.num_layers}")
    print(f"  Hidden State Dimension (d_model): {args.d_model}")
    print(f"  Attention Heads: {args.n_heads}")
    print(f"  FFN Intermediate Dimension: {args.d_ff}")
    print(f"  FN2 Projection Dimension: {args.d_fn2_proj}")
    print(f"  Token Pooling: {args.pool_type}")
    print(f"  Attention Type: {'ReLU Attention (No Exponentials)' if args.use_relu_attn else 'Softmax Attention'}")
    print(f"  Sparse Graph Attention: {args.sparse_attention}")
    print(f"  Zero-in-hand Optimization: {args.zero_in_hand}")
    print(f"  Auxiliary Linear Heads: Graph Trans Aux + FN2 Aux (lambda_aux={args.lambda_aux})")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Average pieces on board: {avg_on_board:.1f} / {NUM_TOKENS} tokens ({(1.0 - avg_on_board / NUM_TOKENS)*100:.1f}% in-hand zeros)")
    print(f"  Max FLOPs (Uncached, 100% active tokens): {max_flops:,.0f}")
    print(f"  Expected FLOPs (Uncached, zero-in-hand): {exp_flops:,.0f} ({(1.0 - exp_flops/max_flops)*100:.1f}% reduction)")
    print(f"  Cached Tree-Search FLOPs (TokenCache Layer-0 QKV): {cached_flops:,.0f} ({(1.0 - cached_flops/max_flops)*100:.1f}% reduction, {max_flops/cached_flops:.2f}x speedup)")
    print(f"  Estimated Single-Core CPU Latency: ~{cached_flops / 1e7 * 1000:.1f} us (assuming 10 GFLOPs/s vector throughput)\n")

    start_epoch = 0
    best_val_loss = float('inf')
    best_epoch = -1
    history = {'train_loss': [], 'test_loss': []}

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 4e-2)
        if not args.no_scheduler
        else None
    )

    if args.load_model_ckpt:
        if os.path.exists(args.load_model_ckpt):
            ckpt = load_model_weights_from_ckpt(model, args.load_model_ckpt, device)
            src_epoch = ckpt.get('epoch', 'N/A') if isinstance(ckpt, dict) else 'N/A'
            print(f"Loaded model weights from {args.load_model_ckpt} (source checkpoint epoch {src_epoch}).")
            print("Starting fresh training run from epoch 0 with new optimizer and scheduler.")
        else:
            print(f"Warning: --load-model-ckpt path {args.load_model_ckpt} does not exist.")

    elif args.load_ckpt:
        if os.path.exists(args.load_ckpt):
            ckpt = load_model_weights_from_ckpt(model, args.load_ckpt, device)
            if isinstance(ckpt, dict):
                if 'optimizer' in ckpt and ckpt['optimizer'] is not None:
                    try:
                        optimizer.load_state_dict(ckpt['optimizer'])
                    except Exception:
                        pass
                if 'scheduler' in ckpt and ckpt['scheduler'] is not None and scheduler is not None:
                    try:
                        scheduler.load_state_dict(ckpt['scheduler'])
                    except Exception:
                        pass
                start_epoch = ckpt.get('epoch', 0)
                best_val_loss = ckpt.get('best_val_loss', float('inf'))
                best_epoch = ckpt.get('best_epoch', start_epoch)
                history = ckpt.get('history', history)
                print(f"Resumed full training state from {args.load_ckpt} at epoch {start_epoch} (best epoch {best_epoch}, best val_loss {best_val_loss:.4f})")
            else:
                print(f"Loaded weights from {args.load_ckpt}")
        else:
            print(f"Warning: --load-ckpt path {args.load_ckpt} does not exist.")

    if args.eval_only:
        print("\n-- Evaluating Model on Test Set (--eval-only) --")
        model.eval()
        all_preds, all_winner, all_aux_t, all_aux_f, all_y, all_wb, all_static = [], [], [], [], [], [], []
        with torch.no_grad():
            for xb, ab, fb, yb, wb, sb, _ in test_loader:
                xb, ab, fb = xb.to(device), ab.to(device), fb.to(device)
                pred_eval, pred_winner, aux_t, aux_f = model(xb, ab, fb, return_aux=True)
                all_preds.append(pred_eval.cpu().numpy())
                all_winner.append(pred_winner.cpu().numpy())
                all_aux_t.append(aux_t.cpu().numpy())
                all_aux_f.append(aux_f.cpu().numpy())
                all_y.append(yb.numpy())
                all_wb.append(wb.numpy())
                all_static.append(sb.numpy())

        y_test_pred = np.concatenate(all_preds)
        winner_test_pred = np.concatenate(all_winner)
        aux_t_test_pred = np.concatenate(all_aux_t)
        aux_f_test_pred = np.concatenate(all_aux_f)
        y_test_true = np.concatenate(all_y)
        winner_test_true = np.concatenate(all_wb)
        y_test_static = np.concatenate(all_static)

        gt_metrics = metrics(y_test_true, y_test_pred)
        aux_t_metrics = metrics(y_test_true, aux_t_test_pred)
        aux_f_metrics = metrics(y_test_true, aux_f_test_pred)
        win_mae = float(np.mean(np.abs(winner_test_pred - winner_test_true)))

        print(f"\nModel on Test Set ({len(y_test_true):,} samples):")
        print(f"  Fused Evaluation RMSE: {gt_metrics['rmse']:.2f}  MAE: {gt_metrics['mae']:.2f}  R2: {gt_metrics['r2']:.4f}")
        print(f"  Trans Aux Linear RMSE: {aux_t_metrics['rmse']:.2f}  MAE: {aux_t_metrics['mae']:.2f}  R2: {aux_t_metrics['r2']:.4f}")
        print(f"  FN2 Aux Linear RMSE:   {aux_f_metrics['rmse']:.2f}  MAE: {aux_f_metrics['mae']:.2f}  R2: {aux_f_metrics['r2']:.4f}")
        print(f"  Winner MAE:            {win_mae:.4f}")

        if (y_test_static != 0).any():
            static_metrics = metrics(y_test_true, y_test_static)
            print(f"\nStatic_eval baseline on same Test Set:")
            print(f"  RMSE: {static_metrics['rmse']:.2f}  MAE: {static_metrics['mae']:.2f}  R2: {static_metrics['r2']:.4f}")

        if args.export_rust:
            export_rust_weights(model, args, args.export_rust, epoch=start_epoch, val_metrics=gt_metrics)
            print(f"\nExported Rust weights to {args.export_rust}")
        return

    loss_fn_eval = nn.MSELoss(reduction='none')
    loss_fn_winner = nn.BCELoss(reduction='none')

    best_state = None

    print(f"-- Training Graph Transformer (Epochs {start_epoch + 1} -> {args.epochs}) --")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_eval_loss = 0.0
        running_eval_rmse = 0.0
        running_aux_t_rmse = 0.0
        running_aux_f_rmse = 0.0
        running_winner_loss = 0.0
        running_winner_mae = 0.0
        running_reg_loss = 0.0
        running_total_loss = 0.0
        total = 0

        for xb, ab, fb, yb, wb, _, wtb in train_loader:
            xb, ab, fb, yb, wb, wtb = (
                xb.to(device),
                ab.to(device),
                fb.to(device),
                yb.to(device),
                wb.to(device),
                wtb.to(device),
            )
            optimizer.zero_grad()

            pred_eval, pred_winner, aux_t, aux_f = model(xb, ab, fb, return_aux=True)

            # Main weighted losses
            loss_e = loss_fn_eval(pred_eval / args.clip, yb / args.clip)
            loss_e = (loss_e * wtb).mean()

            loss_w = loss_fn_winner(pred_winner, wb)
            loss_w = (loss_w * wtb).mean() * args.lambda_w

            # Auxiliary linear regression losses
            loss_aux_t = loss_fn_eval(aux_t / args.clip, yb / args.clip)
            loss_aux_t = (loss_aux_t * wtb).mean() * args.lambda_aux

            loss_aux_f = loss_fn_eval(aux_f / args.clip, yb / args.clip)
            loss_aux_f = (loss_aux_f * wtb).mean() * args.lambda_aux

            l2_reg = args.l2 * sum(p.pow(2).sum() for p in model.parameters())
            loss = loss_e + loss_w + loss_aux_t + loss_aux_f + l2_reg
            loss.backward()
            optimizer.step()

            bs = xb.size(0)
            running_eval_loss += loss_e.item() * bs
            running_eval_rmse += ((pred_eval - yb) ** 2).sum().item()
            running_aux_t_rmse += ((aux_t - yb) ** 2).sum().item()
            running_aux_f_rmse += ((aux_f - yb) ** 2).sum().item()
            running_winner_loss += loss_w.item() * bs
            running_winner_mae += (pred_winner - wb).abs().sum().item()
            running_reg_loss += l2_reg.item() * bs
            running_total_loss += loss.item() * bs
            total += bs

        train_eval_loss = running_eval_loss / total
        train_eval_rmse = (running_eval_rmse / total) ** 0.5
        train_aux_t_rmse = (running_aux_t_rmse / total) ** 0.5
        train_aux_f_rmse = (running_aux_f_rmse / total) ** 0.5
        train_winner_loss = running_winner_loss / total
        train_winner_mae = running_winner_mae / total
        train_reg_loss = running_reg_loss / total
        train_total_loss = running_total_loss / total

        # Validation
        model.eval()
        val_eval_loss = 0.0
        val_eval_rmse = 0.0
        val_aux_t_rmse = 0.0
        val_aux_f_rmse = 0.0
        val_winner_loss = 0.0
        val_winner_mae = 0.0
        val_total = 0

        with torch.no_grad():
            for xb, ab, fb, yb, wb, _, _ in test_loader:
                xb, ab, fb, yb, wb = (
                    xb.to(device),
                    ab.to(device),
                    fb.to(device),
                    yb.to(device),
                    wb.to(device),
                )
                pred_eval, pred_winner, aux_t, aux_f = model(xb, ab, fb, return_aux=True)

                loss_e = ((pred_eval / args.clip - yb / args.clip) ** 2).mean()
                loss_w = F.binary_cross_entropy(pred_winner, wb) * args.lambda_w

                bs = xb.size(0)
                val_eval_loss += loss_e.item() * bs
                val_eval_rmse += ((pred_eval - yb) ** 2).sum().item()
                val_aux_t_rmse += ((aux_t - yb) ** 2).sum().item()
                val_aux_f_rmse += ((aux_f - yb) ** 2).sum().item()
                val_winner_loss += loss_w.item() * bs
                val_winner_mae += (pred_winner - wb).abs().sum().item()
                val_total += bs

        val_eval_loss /= val_total
        val_eval_rmse = (val_eval_rmse / val_total) ** 0.5
        val_aux_t_rmse = (val_aux_t_rmse / val_total) ** 0.5
        val_aux_f_rmse = (val_aux_f_rmse / val_total) ** 0.5
        val_winner_loss /= val_total
        val_winner_mae /= val_total
        val_reg_loss = (args.l2 * sum(p.pow(2).sum() for p in model.parameters())).item()
        val_total_loss = val_eval_loss + val_winner_loss + val_reg_loss

        is_best = val_total_loss < best_val_loss
        if is_best:
            best_val_loss = val_total_loss
            best_epoch = epoch + 1
            best_state = {'model': {k: v.cpu().clone() for k, v in model.state_dict().items()}}

        history['train_loss'].append(train_total_loss)
        history['test_loss'].append(val_total_loss)

        cur_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            scheduler.step()

        # Save latest checkpoint every epoch (resumable)
        os.makedirs("logs", exist_ok=True)
        latest_state = {
            'epoch': epoch + 1,
            'model': {k: v.cpu().clone() for k, v in model.state_dict().items()},
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler is not None else None,
            'val_loss': val_total_loss,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'history': history,
        }
        torch.save(latest_state, "logs/checkpoint_latest.pth")

        star = "*" if is_best else ""
        print(
            f"Epoch {epoch+1:3d}/{args.epochs} lr={cur_lr:.2e} | "
            f"TRAIN eval={train_eval_rmse:6.1f} win={train_winner_mae:.4f} loss={100*train_total_loss:.4f}={100*train_eval_loss:.4f}+{100*train_winner_loss:.4f}+{100*train_reg_loss:.4f} | "
            f"TEST  eval={val_eval_rmse:6.1f} win={val_winner_mae:.4f} loss={100*val_total_loss:.4f} | "
            f"AUX {val_aux_t_rmse:6.1f}, {val_aux_f_rmse:6.1f} {star}"
        )

        # Save best model and export Rust weights immediately if improved
        if is_best:
            torch.save(latest_state, "logs/best_gt_eval.pth")
            if args.export_rust:
                train_m = {'rmse': train_eval_rmse, 'mae': train_winner_mae}
                val_m = {'rmse': val_eval_rmse, 'mae': val_winner_mae}
                export_rust_weights(model, args, args.export_rust, epoch=epoch + 1, train_metrics=train_m, val_metrics=val_m)

        # Plot loss curve dynamically
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['test_loss'], label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Graph Transformer Training Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig("logs/training_loss_gt.png", dpi=150)
        plt.close()

    # Save best checkpoint
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state['model'].items()})
        save_path = "logs/best_gt_eval.pth"
        torch.save(best_state, save_path)
        print(f"\nBest model (Epoch {best_epoch}) saved to {save_path}")

    # Plot training loss history
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Graph Transformer Training Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("logs/training_loss_gt.png", dpi=150)
    print("Saved loss plot to logs/training_loss_gt.png")

    # Final Evaluation on Test Set
    model.eval()
    all_preds, all_aux_t, all_aux_f = [], [], []
    all_y = []
    all_static = []
    with torch.no_grad():
        for xb, ab, fb, yb, _, sb, _ in test_loader:
            xb, ab, fb = xb.to(device), ab.to(device), fb.to(device)
            pred_eval, _, aux_t, aux_f = model(xb, ab, fb, return_aux=True)
            all_preds.append(pred_eval.cpu().numpy())
            all_aux_t.append(aux_t.cpu().numpy())
            all_aux_f.append(aux_f.cpu().numpy())
            all_y.append(yb.numpy())
            all_static.append(sb.numpy())

    y_test_pred = np.concatenate(all_preds)
    y_test_aux_t = np.concatenate(all_aux_t)
    y_test_aux_f = np.concatenate(all_aux_f)
    y_test_true = np.concatenate(all_y)
    y_test_static = np.concatenate(all_static)

    gt_metrics = metrics(y_test_true, y_test_pred)
    aux_t_m = metrics(y_test_true, y_test_aux_t)
    aux_f_m = metrics(y_test_true, y_test_aux_f)

    print(f"\nGraph Transformer on Test Set (Best Epoch {best_epoch}):")
    print(f"  Fused Evaluation: RMSE: {gt_metrics['rmse']:.2f}  MAE: {gt_metrics['mae']:.2f}  R2: {gt_metrics['r2']:.4f}")
    print(f"  Trans Aux Linear: RMSE: {aux_t_m['rmse']:.2f}  MAE: {aux_t_m['mae']:.2f}  R2: {aux_t_m['r2']:.4f}")
    print(f"  FN2 Aux Linear:   RMSE: {aux_f_m['rmse']:.2f}  MAE: {aux_f_m['mae']:.2f}  R2: {aux_f_m['r2']:.4f}")

    if (y_test_static != 0).any():
        static_metrics = metrics(y_test_true, y_test_static)
        print(f"\nStatic_eval baseline on same Test Set:")
        print(f"  RMSE: {static_metrics['rmse']:.2f}  MAE: {static_metrics['mae']:.2f}  R2: {static_metrics['r2']:.4f}")

    # Export Rust Weights
    if args.export_rust:
        export_rust_weights(model, args, args.export_rust, epoch=best_epoch, val_metrics=gt_metrics)

if __name__ == "__main__":
    main()
