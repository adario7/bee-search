import argparse
import json
import os
import gc
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from itertools import chain
import matplotlib.pyplot as plt

def load_json(path):
	if not os.path.exists(path):
		raise FileNotFoundError(path)
	with open(path, "r") as f:
		return json.load(f)

def load_data(evals_path, features_path, static_path, clip_val):
	print(f"Loading evals from {evals_path}...")
	raw_evals = load_json(evals_path)
	print(f"Loading features from {features_path}...")
	raw_feats = load_json(features_path)

	static_map = {}
	if os.path.exists(static_path):
		print(f"Loading static evals from {static_path}...")
		raw_static = load_json(static_path)
		if isinstance(raw_static, dict):
			static_map = {k: float(v) for k, v in raw_static.items()}
		elif isinstance(raw_static, list):
			static_map = {rec["position"]: float(rec["static_eval"]) for rec in raw_static if "position" in rec and "static_eval" in rec}
		del raw_static

	feats_map = {}
	if isinstance(raw_feats, dict):
		for k, v in raw_feats.items():
			if v is not None and len(v) > 0:
				feats_map[k] = np.asarray(v, dtype=np.int16)
	elif isinstance(raw_feats, list):
		for rec in raw_feats:
			pos = rec.get("position")
			vec = rec.get("features")
			if pos and vec is not None and len(vec) > 0:
				feats_map[pos] = np.asarray(vec, dtype=np.int16)
	del raw_feats

	matched_feats = []
	matched_y = []
	matched_w = []
	matched_static = []

	for rec in raw_evals:
		pos = rec.get("position")
		if not pos or pos not in feats_map:
			continue
		vec = feats_map[pos]
		if vec.ndim != 1 or vec.shape[0] == 0:
			continue

		ev = rec.get("evaluation", 0)
		try:
			ev = float(ev)
		except (ValueError, TypeError):
			ev = 0.0
		ev = float(np.clip(ev, -clip_val, clip_val))

		win = rec.get("winner", 0.0)
		try:
			win = float(win)
		except (ValueError, TypeError):
			win = 0.0
		win = float(np.clip(win, 0.0, 1.0))

		matched_feats.append(vec)
		matched_y.append(ev)
		matched_w.append(win)
		matched_static.append(static_map.get(pos, None))

	del raw_evals, feats_map, static_map
	gc.collect()

	if not matched_feats:
		raise ValueError("No matching positions between evals and features")

	# Find most common feature vector length to discard corrupted or partial rows
	lengths = [f.shape[0] for f in matched_feats]
	common_len = max(set(lengths), key=lengths.count)

	valid_indices = [i for i, f in enumerate(matched_feats) if f.shape[0] == common_len]
	if len(valid_indices) < len(matched_feats):
		print(f"Warning: filtered {len(matched_feats) - len(valid_indices)} positions with mismatched feature vector lengths (expected length {common_len})")

	X_full = np.vstack([matched_feats[i] for i in valid_indices])
	y_full = np.array([matched_y[i] for i in valid_indices], dtype=np.float32)
	winner_full = np.array([matched_w[i] for i in valid_indices], dtype=np.float32)
	matched_static = [matched_static[i] for i in valid_indices]

	return X_full, y_full, winner_full, matched_static

def metrics(y_true, y_pred):
	mse = float(mean_squared_error(y_true, y_pred))
	return {
		"rmse": float(np.sqrt(mse)),
		"mae": float(mean_absolute_error(y_true, y_pred)),
		"r2": float(r2_score(y_true, y_pred))
	}

def adj_idx(c1, c2):
	if c1 > c2:
		c1, c2 = c2, c1
	if c1 == c2 and (c1 % 8 == 0 or c1 % 8 >= 5):
		return None
	idx = 0
	i = 0
	while i < 16:
		j = i
		while j < 16:
			if not (i == j and (i % 8 == 0 or i % 8 >= 5)):
				if i == c1 and j == c2:
					return idx
				idx += 1
			j += 1
		i += 1
	return None

_adj_perm = None
def get_adj_perm():
	global _adj_perm
	if _adj_perm is None:
		perm = [0] * 128
		for i in range(16):
			for j in range(i, 16):
				idx1 = adj_idx(i, j)
				if idx1 is not None:
					inv_i = (i + 8) % 16
					inv_j = (j + 8) % 16
					idx2 = adj_idx(inv_i, inv_j)
					perm[idx1] = idx2
		_adj_perm = np.array(perm, dtype=int)
	return _adj_perm

if __name__ == "__main__":
	p = argparse.ArgumentParser()
	p.add_argument("--evals", default="logs/evaluations.json")
	p.add_argument("--features", default="logs/position_features.json")
	p.add_argument("--static", default="logs/position_static.json")
	p.add_argument("--test-size", type=float, default=0.1)
	p.add_argument("--random-state", type=int, default=42)
	p.add_argument("--clip", type=int, default=6000, help="Clip evaluations to [-clip, clip] before processing")
	# MLP training hyperparams
	p.add_argument("--mlp-epochs", type=int, default=150)
	p.add_argument("--mlp-lr", type=float, default=5e-4)
	p.add_argument("--mlp-batch", type=int, default=2048)
	p.add_argument("--mlp-lambda", type=float, default=2e-3, help="Fixed weight for winner loss term")
	p.add_argument("--mlp-l2", type=float, default=2e-5, help="L2 regularization coefficient")
	p.add_argument("--skip-lr", action="store_true", help="Skip Linear Regression and do only MLP")
	p.add_argument("--no-adj", action="store_true", help="Zero out all 128 adjacency matrix features for ablation testing")
	p.add_argument("--load-ckpt", default=None, help="Path to checkpoint .pth to load")
	args = p.parse_args()

	X_full, y_full, winner_full, static_list = load_data(args.evals, args.features, args.static, clip_val=args.clip)
	n_original = len(y_full)
	print(f"Loaded {n_original} matched positions")

	# symmetric augmentation approach:
	L = X_full.shape[1]
	ADJ = 128
	N = (L - ADJ) // 2

	if args.no_adj:
		print("Flag --no-adj enabled: Zeroing out all 128 adjacency matrix features.")
		X_full[:, 2*N:] = 0

	# split ORIGINAL rows to avoid leakage between original and its symmetric counterpart
	idx = np.arange(n_original)
	idx_train, idx_test = train_test_split(idx, test_size=args.test_size, random_state=args.random_state)

	def augment_indices(indices):
		n_sub = len(indices)
		adj_perm = get_adj_perm()
		X_aug = np.empty((n_sub * 2, L), dtype=np.float32)
		y_aug = np.empty(n_sub * 2, dtype=np.float32)
		w_aug = np.empty(n_sub * 2, dtype=np.float32)
		static_aug = []

		for idx_out, i in enumerate(indices):
			f = X_full[i]
			y = y_full[i]
			w = winner_full[i]
			s = static_list[i]

			# original
			row_orig = idx_out * 2
			X_aug[row_orig] = f
			y_aug[row_orig] = y
			w_aug[row_orig] = w
			static_aug.append(s)

			# swapped counterpart
			row_swap = idx_out * 2 + 1
			f_swapped = np.concatenate([f[N:2*N], f[:N], f[2*N:][adj_perm]])
			X_aug[row_swap] = f_swapped
			y_aug[row_swap] = -y
			w_aug[row_swap] = 1.0 - w
			static_aug.append(-s if s is not None else None)

		return X_aug, y_aug, w_aug, static_aug

	X_train_aug, y_train_aug, winner_train_aug, static_train_aug = augment_indices(idx_train)
	X_test_aug, y_test_aug, winner_test_aug, static_test_aug = augment_indices(idx_test)

	# Free initial dataset arrays to minimize RAM usage
	zero_ratio = float(np.mean(X_full == 0))
	del X_full, y_full, winner_full, static_list
	gc.collect()

	# prepare static predictor metrics: only on test augmented samples that have static
	mask_static = np.array([s is not None for s in static_test_aug])
	n_test_with_static = int(mask_static.sum())
	if n_test_with_static == 0:
		static_metrics = None
	else:
		y_test_sub = y_test_aug[mask_static].astype(float)
		static_preds = np.array([s for s in static_test_aug if s is not None], dtype=float)
		static_metrics = metrics(y_test_sub, static_preds)

	if static_metrics:
		print("\nStatic_eval predictor (on same augmented subset):")
		print(f"  RMSE: {static_metrics['rmse']:.4f}")
		print(f"  MAE:  {static_metrics['mae']:.4f}")
		print(f"  R2:   {static_metrics['r2']:.4f}")
	else:
		print("\nNo static_eval available in test set for comparison.")

	if not args.skip_lr:
		# fit linear model on FULL vectors but force intercept=0 to satisfy equivalence
		model = LinearRegression(fit_intercept=False)
		model.fit(X_train_aug, y_train_aug)
		y_pred_test = model.predict(X_test_aug)

		reg_metrics_full = metrics(y_test_aug, y_pred_test)

		print(f"n_original_samples: {n_original}, n_features_full: {L}, N_half: {N}, ADJ_len: {ADJ}")
		print(f"train_size (augmented): {len(X_train_aug)}, test_size (augmented): {len(X_test_aug)}, test_with_static: {n_test_with_static}")
		print("\nLinear Regression (symmetric augmentation, intercept=0) on test set:")
		print(f"  RMSE: {reg_metrics_full['rmse']:.4f}")
		print(f"  MAE:  {reg_metrics_full['mae']:.4f}")
		print(f"  R2:   {reg_metrics_full['r2']:.4f}")

		coeffs = model.coef_.astype(float)
		w1 = coeffs[:N]
		w2 = coeffs[N:2*N]
		symmetry_err = np.max(np.abs(w2 + w1))
		print("\nLinear model parameters (full-length coeffs):")
		print("Intercept: 0.0 (enforced)")
		print("Coefficients:", list(map(float, coeffs)))
		print(f"Max |w2 + w1| (symmetry check): {symmetry_err:.6g}")
		print("Reduced weights (first-half, equivalent to model on (first-half - second-half)):")
		print(list(map(float, w1)))
	else:
		print("\nSkipping Linear Regression.")

	# -------------------- MLP -------------------------

	print("-- MLP --")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	torch.manual_seed(args.random_state)

	class SingleMLP(nn.Module):
		def __init__(self, in_dim):
			super().__init__()
			# output two values: [eval_raw, winner_raw]
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
			eval_out = torch.nn.functional.softsign(out[:, 0]) * 1.5 * args.clip
			winner_out = torch.sigmoid(out[:, 1])
			return eval_out, winner_out

	model = SingleMLP(in_dim=L).to(device)

	def count_model_stats(model, zero_ratio):
		params = sum(p.numel() for p in model.parameters())
		max_flops = 0
		expected_flops = 0
		is_first = True
		for layer in model.net:
			if isinstance(layer, nn.Linear):
				in_f = layer.in_features
				out_f = layer.out_features
				bias_flops = out_f if layer.bias is not None else 0
				layer_max = 2 * in_f * out_f + bias_flops
				if is_first:
					nonzero_in = in_f * (1.0 - zero_ratio)
					layer_exp = 2 * nonzero_in * out_f + bias_flops
					is_first = False
				else:
					layer_exp = layer_max
				max_flops += layer_max
				expected_flops += layer_exp
		return params, max_flops, expected_flops

	if args.load_ckpt:
		if os.path.exists(args.load_ckpt):
			ckpt = torch.load(args.load_ckpt, map_location=device)
			if 'model' in ckpt:
				model.load_state_dict(ckpt['model'])
			elif isinstance(ckpt, dict):
				model.load_state_dict(ckpt)
			print(f"Loaded checkpoint from {args.load_ckpt}")
		else:
			print(f"Warning: checkpoint not found at {args.load_ckpt}")

	all_params = list(model.parameters())
	params, max_flops, expected_flops = count_model_stats(model, zero_ratio)
	print("\nModel Summary:")
	print(f"  Total parameters: {params:,}")
	print(f"  Input feature dimension: {L}")
	print(f"  Average zero features: {zero_ratio*100:.2f}% ({L * zero_ratio:.1f} / {L} entries)")
	print(f"  Max FLOPs (100% non-zero inputs): {max_flops:,}")
	print(f"  Expected FLOPs (skipping zero inputs in layer 1): {expected_flops:,.1f}\n")

	optimizer = torch.optim.Adam(all_params, lr=args.mlp_lr)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.mlp_epochs, eta_min=args.mlp_lr*1e-2)
	loss_fn_eval = nn.MSELoss()
	loss_fn_winner = nn.BCELoss()
	lambda_w = args.mlp_lambda
	l2_coef = args.mlp_l2

	# prepare dataloaders
	Xtr = torch.from_numpy(X_train_aug.astype(np.float32))
	Ytr = torch.from_numpy(y_train_aug.astype(np.float32))
	Wtr = torch.from_numpy(winner_train_aug.astype(np.float32))
	Xte = torch.from_numpy(X_test_aug.astype(np.float32))
	Yte = torch.from_numpy(y_test_aug.astype(np.float32))
	Wte = torch.from_numpy(winner_test_aug.astype(np.float32))

	train_ds = TensorDataset(Xtr, Ytr, Wtr)
	train_loader = DataLoader(train_ds, batch_size=args.mlp_batch, shuffle=True)

	best_val_loss = float('inf')
	best_state = None
	best_epoch = -1
	history = {'train_loss': [], 'test_loss': []}

	for epoch in range(args.mlp_epochs):
		running_eval_loss = 0.0
		running_eval_rmse = 0.0
		running_winner_loss = 0.0
		running_winner_mae = 0.0
		running_reg_loss = 0.0
		running_total_loss = 0.0
		total = 0
		model.train()
		for xb, yb, wb in train_loader:
			xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
			pred_eval, pred_winner = model(xb)
			loss_eval = loss_fn_eval(pred_eval/args.clip, yb/args.clip)
			loss_winner = lambda_w * loss_fn_winner(pred_winner, wb)
			l2_reg = l2_coef * sum(p.pow(2).sum() for p in all_params)
			loss = loss_eval + loss_winner + l2_reg
			optimizer.zero_grad(); loss.backward(); optimizer.step()
			bs = xb.size(0)
			running_eval_loss += loss_eval.item() * bs
			running_eval_rmse += ((pred_eval - yb)**2).sum().item()
			running_winner_loss += loss_winner.item() * bs
			running_winner_mae += (pred_winner - wb).abs().sum().item()
			running_reg_loss += l2_reg.item() * bs
			running_total_loss += loss.item() * bs
			total += bs
		train_eval_loss = running_eval_loss / total
		train_eval_rmse = (running_eval_rmse / total) ** 0.5
		train_winner_loss = running_winner_loss / total
		train_winner_mae = running_winner_mae / total
		train_reg_loss = running_reg_loss / total
		train_total_loss = running_total_loss / total

		model.eval()
		with torch.no_grad():
			Xte_dev = Xte.to(device); Yte_dev = Yte.to(device); Wte_dev = Wte.to(device)
			val_eval_pred, val_winner_pred = model(Xte_dev)
			val_eval_loss = loss_fn_eval(val_eval_pred/args.clip, Yte_dev/args.clip).item()
			val_eval_rmse = ((val_eval_pred - Yte_dev)**2).mean().item() ** 0.5
			val_winner_loss = lambda_w * loss_fn_winner(val_winner_pred, Wte_dev).item()
			val_winner_mae = (val_winner_pred - Wte_dev).abs().mean().item()
			val_l2_reg = l2_coef * sum(p.pow(2).sum() for p in all_params).item()
			val_reg_loss = val_l2_reg
			val_total_loss = val_eval_loss + val_winner_loss + val_reg_loss

			if val_eval_loss < best_val_loss:
				best_val_loss = val_eval_loss
				best_epoch = epoch + 1
				best_state = {'model': {k: v.cpu().clone() for k, v in model.state_dict().items()}}
		
		history['train_loss'].append(train_total_loss)
		history['test_loss'].append(val_total_loss)

		cur_lr = optimizer.param_groups[0]['lr']
		scheduler.step()

		model.train()
		print(
			f"Epoch {epoch+1}/{args.mlp_epochs} lr={cur_lr:.2e} | "
			f"TRAIN eval={train_eval_rmse:.1f} win={train_winner_mae:.4f} loss={100*train_total_loss:.4f}={100*train_eval_loss:.4f}+{100*train_winner_loss:.4f}+{100*train_reg_loss:.4f} | "
			f"TEST  eval={val_eval_rmse:.1f} win={val_winner_mae:.4f} loss={100*val_total_loss:.4f}={100*val_eval_loss:.4f}+{100*val_winner_loss:.4f}+{100*val_reg_loss:.4f}"
		)

	if best_state is not None:
		model.load_state_dict({k: v.to(device) for k, v in best_state['model'].items()})
		
		# Save best model
		save_path = "logs/best_mlp_eval.pth"
		torch.save(best_state, save_path)
		print(f"Best model saved to {save_path}")

	# plot loss history
	plt.figure(figsize=(10, 6))
	plt.plot(history['train_loss'], label='Train Total Loss')
	plt.plot(history['test_loss'], label='Test Total Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.title('Training and Test Total Loss over Time')
	plt.legend()
	plt.grid(True)
	plt.savefig("logs/training_loss.png")
	print("\nLoss plot saved to logs/training_loss.png")
	with open("logs/training_history.json", "w") as f:
		json.dump(history, f)

	model.eval()
	with torch.no_grad():
		Xte_dev = Xte.to(device)
		eval_preds, winner_preds = model(Xte_dev)
		mlp_preds = eval_preds.cpu().numpy()

	mlp_metrics = metrics(y_test_aug, mlp_preds)

	print(f"\nBest epoch (by val loss): {best_epoch}, val_total_loss: {best_val_loss:.6f}")
	print("\nMLP on test set (using best-epoch weights) - evaluation output:")
	print(f"  RMSE: {mlp_metrics['rmse']:.4f}")
	print(f"  MAE:  {mlp_metrics['mae']:.4f}")
	print(f"  R2:   {mlp_metrics['r2']:.4f}")

	print()
	def write_module_weights(f, named_params, prefix, start_layer):
		"""Write weights for one module, returns next_layer index."""
		params_list = list(named_params)
		weight_names = [name for name, _ in params_list if name.endswith(".weight")]
		last_w_name = weight_names[-1] if weight_names else ""
		last_b_name = last_w_name.replace(".weight", ".bias")

		next_layer = start_layer
		last_bias_layer = None
		for name, p in params_list:
			arr = p.detach().cpu().numpy()
			is_final_last_w = (name == last_w_name)
			is_final_last_b = (name == last_b_name)
			if arr.ndim == 2:
				# if this is the final layer producing 2 outputs, keep only the first row (eval head)
				if is_final_last_w and arr.shape[0] == 2:
					arr_print = arr[0:1, :]
				else:
					arr_print = arr
				# PyTorch weights are (out_features, in_features). Transpose to (in_features, out_features) for Rust.
				arr_print = arr_print.T
				r, c = arr_print.shape
				ident = f"{prefix}W{next_layer}{next_layer+1}"
				rows = []
				for row in arr_print:
					rows.append("[" + ", ".join(f"{float(x):.8e}f32" for x in row) + "]")
				body = "[\n  " + ",\n  ".join(rows) + "\n]"
				f.write(f"\tconst {ident}: [[f32; {c}]; {r}] = {body};\n\n")
				last_bias_layer = next_layer + 1
				next_layer += 1
			elif arr.ndim == 1:
				# if this is the final bias for the 2-output layer, keep only the first element
				if is_final_last_b and arr.shape[0] == 2:
					arr_print = arr[:1]
				else:
					arr_print = arr
				n = arr_print.shape[0]
				layer_idx = last_bias_layer if last_bias_layer is not None else next_layer
				ident = f"{prefix}B{layer_idx}"
				body = "[" + ", ".join(f"{float(x):.8e}f32" for x in arr_print) + "]"
				f.write(f"\tconst {ident}: [f32; {n}] = {body};\n\n")
		return next_layer

	rust_out_path = "logs/eval_mlp.rs"
	with open(rust_out_path, "w") as f:
		f.write("\t// --- single mlp (all features) ---\n")
		write_module_weights(f, model.named_parameters(), "M_", 0)

	print(f"Rust weights written to {rust_out_path}")

