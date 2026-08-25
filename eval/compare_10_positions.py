import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import torch
import torch.nn as nn
import numpy as np
from eval.transformer import GraphTransformerEval

# 1. MLP Model Definition (matching best_mlp_eval.pth)
class SingleMLP(nn.Module):
    def __init__(self, in_dim=232, clip=6000.0):
        super().__init__()
        self.clip = clip
        self.net = nn.Sequential(
            nn.Linear(in_dim, 48),
            nn.ReLU(),
            nn.Linear(48, 28),
            nn.ReLU(),
            nn.Linear(28, 14),
            nn.ReLU(),
            nn.Linear(14, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )

    def forward(self, x):
        out = self.net(x)
        eval_out = torch.nn.functional.softsign(out[:, 0]) * 1.5 * self.clip
        winner_out = torch.sigmoid(out[:, 1])
        return eval_out, winner_out

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load 10 sampled positions from evaluations.json
    with open('logs/sampled_10_positions.json', 'r') as f:
        samples = json.load(f)

    # 1. Load MLP Model
    mlp_model = SingleMLP(in_dim=232, clip=6000.0).to(device)
    mlp_ckpt = torch.load('/home/alessandro/local/logs/best_mlp_eval.pth', map_location=device)
    if 'model' in mlp_ckpt:
        mlp_model.load_state_dict(mlp_ckpt['model'])
    else:
        mlp_model.load_state_dict(mlp_ckpt)
    mlp_model.eval()

    # 2. Load GT Model (checkpoint_v1_48epoch.pth)
    gt_model = GraphTransformerEval(
        d_in=8, num_tokens=28, num_types=28, d_model=16, n_heads=2, num_layers=2, d_ff=24,
        mlp_dims=[32, 16, 8], pool_type='flatten', use_relu_attn=True, sparse_attention=False,
        per_piece_type_weights=True, zero_in_hand=True, clip=6000.0
    ).to(device)
    gt_model.token_to_type = torch.arange(28, dtype=torch.long, device=device)
    for l in gt_model.layers:
        l.token_to_type = torch.arange(28, dtype=torch.long, device=device)
        l.attn.token_to_type = torch.arange(28, dtype=torch.long, device=device)

    gt_ckpt = torch.load('logs/checkpoint_v1_48epoch.pth', map_location=device)
    gt_model.load_state_dict(gt_ckpt['model'])
    gt_model.eval()

    print("\n" + "=" * 120)
    print("  COMPARISON: GRAPH TRANSFORMER (GT) vs MLP vs TRUTH (From evaluations.json)")
    print("=" * 120)
    print("| Pos # | Ply | Turn | GT Eval | MLP Eval | Truth Eval | GT Win% | MLP Win% | Truth Win% | Eval Error (GT / MLP) |")
    print("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    with torch.no_grad():
        for i, s in enumerate(samples):
            turn_num = s['turn_num']
            is_white = s['is_white']
            turn_str = "White" if is_white else "Black"
            truth_eval = s['truth_eval']
            truth_win = s['truth_winner']

            # Run MLP
            mlp_x = torch.tensor(s['mlp_features'], dtype=torch.float32).unsqueeze(0).to(device)
            mlp_ev_raw, mlp_win_raw = mlp_model(mlp_x)
            mlp_ev_val = mlp_ev_raw.item()
            mlp_win_val = mlp_win_raw.item()

            # Convert MLP to White perspective if it was Black to move
            mlp_eval_white = mlp_ev_val if is_white else -mlp_ev_val
            mlp_win_white = mlp_win_val if is_white else (1.0 - mlp_win_val)

            # Run GT (already in absolute White perspective: 0..13 White, 14..27 Black)
            gt_feat = torch.tensor(s['gt_features'], dtype=torch.float32).unsqueeze(0).to(device)
            gt_adj = torch.tensor(s['gt_adj'], dtype=torch.uint8).unsqueeze(0).to(device)
            gt_ev_raw, gt_win_raw = gt_model(gt_feat, gt_adj)
            gt_eval_white = gt_ev_raw.item()
            gt_win_white = gt_win_raw.item()

            w_pieces = sum(1 for row in s['gt_features'][:14] if row[0] > 0.5)
            b_pieces = sum(1 for row in s['gt_features'][14:] if row[0] > 0.5)

            stage = "Opening" if turn_num <= 8 else ("Early-Mid" if turn_num <= 16 else ("Midgame" if turn_num <= 30 else "Endgame"))

            print(
                f"| {i+1:5d} | {turn_num:3d} | {stage:<9s} | {turn_str:5s} | {w_pieces:2d}/{b_pieces:2d} | "
                f"{gt_eval_white:+7.0f} cp | {mlp_eval_white:+7.0f} cp | {truth_eval:+7.0f} cp | "
                f"{gt_win_white*100:6.1f}% | {mlp_win_white*100:6.1f}% | {truth_win*100:6.1f}% |"
            )

    print("=" * 120 + "\n")

if __name__ == '__main__':
    main()
