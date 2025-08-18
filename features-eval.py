import argparse
import json
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from itertools import chain

def load_json(path):
	if not os.path.exists(path):
		raise FileNotFoundError(path)
	with open(path, "r") as f:
		return json.load(f)

def load_evals(path, clip_val):
	data = load_json(path)
	df = pd.DataFrame(data)
	if "evaluation" not in df.columns:
		raise KeyError("No 'evaluation' in evals")
	df["evaluation"] = pd.to_numeric(df["evaluation"], errors="coerce").fillna(0).astype(int)
	df["evaluation"] = df["evaluation"].clip(-clip_val, clip_val)
	return df

def load_features(path):
	data = load_json(path)
	if isinstance(data, dict):
		rows = [{"position": k, "features": v} for k, v in data.items()]
		return pd.DataFrame(rows)
	elif isinstance(data, list):
		return pd.DataFrame(data)
	else:
		raise ValueError("Unsupported features format")

def load_static(path):
	data = load_json(path)
	if isinstance(data, dict):
		return {k: int(v) for k, v in data.items()}
	elif isinstance(data, list):
		return {rec["position"]: int(rec.get("static_eval")) for rec in data if "position" in rec and "static_eval" in rec}
	else:
		raise ValueError("Unsupported static format")

def expand_features(df):
	df = df.copy()
	df["features_arr"] = df["features"].apply(lambda x: np.asarray(x, dtype=float))
	lengths = df["features_arr"].apply(len)
	if lengths.nunique() != 1:
		common = lengths.mode()[0]
		df = df[lengths == common].reset_index(drop=True)
	X = np.stack(df["features_arr"].values)
	return df.reset_index(drop=True), X

def metrics(y_true, y_pred):
	return {
		"mse": float(mean_squared_error(y_true, y_pred)),
		"mae": float(mean_absolute_error(y_true, y_pred)),
		"r2": float(r2_score(y_true, y_pred))
	}

if __name__ == "__main__":
	p = argparse.ArgumentParser()
	p.add_argument("--evals", default="logs/evaluations.json")
	p.add_argument("--features", default="logs/position_features.json")
	p.add_argument("--static", default="logs/position_static.json")
	p.add_argument("--test-size", type=float, default=0.1)
	p.add_argument("--random-state", type=int, default=42)
	p.add_argument("--clip", type=int, default=6000, help="Clip evaluations to [-clip, clip] before processing")
	# MLP training hyperparams
	p.add_argument("--mlp-epochs", type=int, default=50)
	p.add_argument("--mlp-lr", type=float, default=1e-3)
	p.add_argument("--mlp-batch", type=int, default=256)
	args = p.parse_args()

	evals_df = load_evals(args.evals, clip_val=args.clip)
	feats_df = load_features(args.features)
	static_map = load_static(args.static)

	merged = pd.merge(evals_df, feats_df, on="position", how="inner")
	if merged.empty:
		raise ValueError("No matching positions between evals and features")

	merged, X_full = expand_features(merged)
	y_full = merged["evaluation"].values
	positions = merged["position"].values

	# split ORIGINAL rows to avoid leakage between original and its symmetric counterpart
	idx = np.arange(len(merged))
	idx_train, idx_test = train_test_split(idx, test_size=args.test_size, random_state=args.random_state)

	# symmetric augmentation approach:
	L = X_full.shape[1]
	if L % 2 != 0:
		raise ValueError("Feature vector length must be even for symmetric mode")
	N = L // 2

	def augment_indices(indices):
		X_aug = []
		y_aug = []
		static_aug = []
		pos_aug = []
		for i in indices:
			f = X_full[i]
			y = float(y_full[i])
			pos = positions[i]
			s = static_map.get(pos, None)
			# original
			X_aug.append(f.copy())
			y_aug.append(y)
			static_aug.append(None if s is None else float(s))
			pos_aug.append(pos)
			# swapped counterpart
			f_swapped = np.concatenate([f[N:], f[:N]])
			X_aug.append(f_swapped)
			y_aug.append(-y)
			static_aug.append(None if s is None else -float(s))
			pos_aug.append(pos + "_sym")  # mark as different to avoid accidental joins
		return np.vstack(X_aug), np.array(y_aug), np.array(static_aug, dtype=object), pos_aug

	X_train_aug, y_train_aug, static_train_aug, pos_train_aug = augment_indices(idx_train)
	X_test_aug, y_test_aug, static_test_aug, pos_test_aug = augment_indices(idx_test)

	# fit linear model on FULL (2N) vectors but force intercept=0 to satisfy equivalence
	model = LinearRegression(fit_intercept=False)
	model.fit(X_train_aug, y_train_aug)
	y_pred_test = model.predict(X_test_aug)

	reg_metrics_full = metrics(y_test_aug, y_pred_test)

	# prepare static predictor metrics: only on test augmented samples that have static
	mask_static = np.array([s is not None for s in static_test_aug])
	n_test_with_static = int(mask_static.sum())
	if n_test_with_static == 0:
		static_metrics = None
	else:
		y_test_sub = y_test_aug[mask_static].astype(float)
		static_preds = np.array([s for s in static_test_aug[mask_static]], dtype=float)
		static_metrics = metrics(y_test_sub, static_preds)

	print(f"n_original_samples: {len(merged)}, n_features_full: {L}, N_half: {N}")
	print(f"train_size (augmented): {len(X_train_aug)}, test_size (augmented): {len(X_test_aug)}, test_with_static: {n_test_with_static}")
	print("\nLinear Regression (symmetric augmentation, intercept=0) on test set:")
	print(f"  MSE: {reg_metrics_full['mse']:.4f}")
	print(f"  MAE: {reg_metrics_full['mae']:.4f}")
	print(f"  R2:  {reg_metrics_full['r2']:.4f}")

	if static_metrics:
		print("\nStatic_eval predictor (on same augmented subset):")
		print(f"  MSE: {static_metrics['mse']:.4f}")
		print(f"  MAE: {static_metrics['mae']:.4f}")
		print(f"  R2:  {static_metrics['r2']:.4f}")
	else:
		print("\nNo static_eval available in test set for comparison.")

	coeffs = model.coef_.astype(float)
	w1 = coeffs[:N]
	w2 = coeffs[N:]
	symmetry_err = np.max(np.abs(w2 + w1))
	print("\nLinear model parameters (full-length coeffs):")
	print("Intercept: 0.0 (enforced)")
	print("Coefficients (length 2N):", list(map(float, coeffs)))
	print(f"Max |w2 + w1| (symmetry check): {symmetry_err:.6g}")
	print("Reduced weights (first-half, equivalent to model on (first-half - second-half)):")
	print(list(map(float, w1)))

	# -------------------- MLP -------------------------

	print("-- MLP --")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	torch.manual_seed(args.random_state)

	class HalfMLP(nn.Module):
		def __init__(self, n_in, out_dim=4):
			super().__init__()
			self.net = nn.Sequential(
				nn.Linear(n_in, 8),
				nn.ReLU(),
				nn.Linear(8, 6),
				nn.ReLU(),
				nn.Linear(6, out_dim),
				nn.ReLU()
			)
		def forward(self, x):
			return self.net(x)

	class FinalMLP(nn.Module):
		def __init__(self, in_dim=8):
			super().__init__()
			self.net = nn.Sequential(
				nn.Linear(in_dim, 6),
				nn.ReLU(),
				nn.Linear(6, 4),
				nn.ReLU(),
				nn.Linear(4, 1)
			)
		def forward(self, x):
			return self.net(x).squeeze(-1) * args.clip

	shared = HalfMLP(N, out_dim=4).to(device)
	final = FinalMLP(in_dim=8).to(device)
	optimizer = torch.optim.Adam(list(shared.parameters()) + list(final.parameters()), lr=args.mlp_lr)
	loss_fn = nn.MSELoss()

	# prepare dataloaders
	Xtr = torch.from_numpy(X_train_aug.astype(np.float32))
	Ytr = torch.from_numpy(y_train_aug.astype(np.float32))
	Xte = torch.from_numpy(X_test_aug.astype(np.float32))
	Yte = torch.from_numpy(y_test_aug.astype(np.float32))

	train_ds = TensorDataset(Xtr, Ytr)
	train_loader = DataLoader(train_ds, batch_size=args.mlp_batch, shuffle=True)

	best_val_loss = float('inf')
	best_state = None
	best_epoch = -1

	shared.train(); final.train()
	for epoch in range(args.mlp_epochs):
		running_loss = 0.0
		total = 0
		for xb, yb in train_loader:
			xb, yb = xb.to(device), yb.to(device)
			h1 = shared(xb[:, :N]); h2 = shared(xb[:, N:])
			conc = torch.cat([h1, h2], dim=1)
			pred = final(conc)
			loss = loss_fn(pred, yb)
			optimizer.zero_grad(); loss.backward(); optimizer.step()
			bs = xb.size(0)
			running_loss += loss.item() * bs
			total += bs
		train_loss = running_loss / total

		shared.eval(); final.eval()
		with torch.no_grad():
			Xte_dev = Xte.to(device); Yte_dev = Yte.to(device)
			h1 = shared(Xte_dev[:, :N]); h2 = shared(Xte_dev[:, N:])
			conc = torch.cat([h1, h2], dim=1)
			val_pred = final(conc)
			val_loss = loss_fn(val_pred, Yte_dev).item()

			if val_loss < best_val_loss:
				best_val_loss = val_loss
				best_epoch = epoch + 1
				best_state = {
					'shared': {k: v.cpu().clone() for k, v in shared.state_dict().items()},
					'final': {k: v.cpu().clone() for k, v in final.state_dict().items()}
				}

		print(f"Epoch {epoch+1}/{args.mlp_epochs} - train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")

	if best_state is not None:
		shared.load_state_dict({k: v.to(device) for k, v in best_state['shared'].items()})
		final.load_state_dict({k: v.to(device) for k, v in best_state['final'].items()})

	shared.eval(); final.eval()
	with torch.no_grad():
		Xte_dev = Xte.to(device)
		h1 = shared(Xte_dev[:, :N])
		h2 = shared(Xte_dev[:, N:])
		conc = torch.cat([h1, h2], dim=1)
		mlp_preds = final(conc).cpu().numpy()

	mlp_metrics = metrics(y_test_aug, mlp_preds)

	print(f"\nBest epoch (by val loss): {best_epoch}, val_loss: {best_val_loss:.4f}")
	print("\nMLP on test set (using best-epoch weights):")
	print(f"  MSE: {mlp_metrics['mse']:.4f}")
	print(f"  MAE: {mlp_metrics['mae']:.4f}")
	print(f"  R2:  {mlp_metrics['r2']:.4f}")

	print()
	params = list(chain(shared.named_parameters(), final.named_parameters()))

	next_layer = 0
	last_bias_layer = None
	for name, p in params:
		arr = p.detach().cpu().numpy()
		if arr.ndim == 2:
			r, c = arr.shape
			ident = f"W{next_layer}{next_layer+1}"
			rows = []
			for row in arr:
				rows.append("[" + ", ".join(f"{float(x):.8e}f32" for x in row) + "]")
			body = "[\n  " + ",\n  ".join(rows) + "\n]"
			print(f"\tconst {ident}: [[f32; {c}]; {r}] = {body};\n")
			last_bias_layer = next_layer + 1
			next_layer += 1
		elif arr.ndim == 1:
			n = arr.shape[0]
			layer_idx = last_bias_layer if last_bias_layer is not None else next_layer
			ident = f"B{layer_idx}"
			body = "[" + ", ".join(f"{float(x):.8e}f32" for x in arr) + "]"
			print(f"\tconst {ident}: [f32; {n}] = {body};\n")
