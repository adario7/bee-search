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
import matplotlib.pyplot as plt

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
	# new winner field 0..1
	if "winner" not in df.columns:
		raise KeyError("No 'winner' in evals")
	df["winner"] = pd.to_numeric(df["winner"], errors="coerce").fillna(0.0).astype(float)
	df["winner"] = df["winner"].clip(0.0, 1.0)
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
	mse = float(mean_squared_error(y_true, y_pred))
	return {
		"rmse": float(np.sqrt(mse)),
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
	p.add_argument("--mlp-epochs", type=int, default=100)
	p.add_argument("--mlp-lr", type=float, default=1e-3)
	p.add_argument("--mlp-batch", type=int, default=256)
	p.add_argument("--mlp-lambda", type=float, default=2e-3, help="Fixed weight for winner loss term")
	p.add_argument("--mlp-l2", type=float, default=2e-5, help="L2 regularization coefficient")
	p.add_argument("--mlp-l2-sym", type=float, default=1e-5, help="L2 symmetry regularization pushing half_a and half_b toward same weights")
	p.add_argument("--skip-lr", action="store_true", help="Skip Linear Regression and do only MLP")
	p.add_argument("--load-ckpt", default=None, help="Path to checkpoint .pth to load")
	args = p.parse_args()

	evals_df = load_evals(args.evals, clip_val=args.clip)
	print(f"Loaded {len(evals_df)} evals")
	feats_df = load_features(args.features)
	print(f"Loaded {len(feats_df)} features")
	static_map = load_static(args.static)
	print(f"Loaded {len(static_map)} static evals")

	merged = pd.merge(evals_df, feats_df, on="position", how="inner")
	if merged.empty:
		raise ValueError("No matching positions between evals and features")

	merged, X_full = expand_features(merged)
	y_full = merged["evaluation"].values
	winner_full = merged["winner"].values
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
		w_aug = []
		static_aug = []
		pos_aug = []
		for i in indices:
			f = X_full[i]
			y = float(y_full[i])
			w = float(winner_full[i])
			pos = positions[i]
			s = static_map.get(pos, None)
			# original
			X_aug.append(f.copy())
			y_aug.append(y)
			w_aug.append(w)
			static_aug.append(None if s is None else float(s))
			pos_aug.append(pos)
			# swapped counterpart
			f_swapped = np.concatenate([f[N:], f[:N]])
			X_aug.append(f_swapped)
			y_aug.append(-y)
			w_aug.append(1.0 - w)
			static_aug.append(None if s is None else -float(s))
			pos_aug.append(pos + "_sym")  # mark as different to avoid accidental joins
		return np.vstack(X_aug), np.array(y_aug), np.array(w_aug, dtype=float), np.array(static_aug, dtype=object), pos_aug

	X_train_aug, y_train_aug, winner_train_aug, static_train_aug, pos_train_aug = augment_indices(idx_train)
	X_test_aug, y_test_aug, winner_test_aug, static_test_aug, pos_test_aug = augment_indices(idx_test)

	# prepare static predictor metrics: only on test augmented samples that have static
	mask_static = np.array([s is not None for s in static_test_aug])
	n_test_with_static = int(mask_static.sum())
	if n_test_with_static == 0:
		static_metrics = None
	else:
		y_test_sub = y_test_aug[mask_static].astype(float)
		static_preds = np.array([s for s in static_test_aug[mask_static]], dtype=float)
		static_metrics = metrics(y_test_sub, static_preds)

	if static_metrics:
		print("\nStatic_eval predictor (on same augmented subset):")
		print(f"  RMSE: {static_metrics['rmse']:.4f}")
		print(f"  MAE:  {static_metrics['mae']:.4f}")
		print(f"  R2:   {static_metrics['r2']:.4f}")
	else:
		print("\nNo static_eval available in test set for comparison.")

	if not args.skip_lr:
		# fit linear model on FULL (2N) vectors but force intercept=0 to satisfy equivalence
		model = LinearRegression(fit_intercept=False)
		model.fit(X_train_aug, y_train_aug)
		y_pred_test = model.predict(X_test_aug)

		reg_metrics_full = metrics(y_test_aug, y_pred_test)


		print(f"n_original_samples: {len(merged)}, n_features_full: {L}, N_half: {N}")
		print(f"train_size (augmented): {len(X_train_aug)}, test_size (augmented): {len(X_test_aug)}, test_with_static: {n_test_with_static}")
		print("\nLinear Regression (symmetric augmentation, intercept=0) on test set:")
		print(f"  RMSE: {reg_metrics_full['rmse']:.4f}")
		print(f"  MAE:  {reg_metrics_full['mae']:.4f}")
		print(f"  R2:   {reg_metrics_full['r2']:.4f}")

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
	else:
		print("\nSkipping Linear Regression.")

	# -------------------- MLP -------------------------

	print("-- MLP --")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	torch.manual_seed(args.random_state)

	class HalfMLP(nn.Module):
		def __init__(self, n_in, out_dim):
			super().__init__()
			self.net = nn.Sequential(
				nn.Linear(n_in, 10),
				nn.ReLU(),
				nn.Linear(10, 8),
				nn.ReLU(),
				nn.Linear(8, out_dim),
				nn.ReLU()
			)
		def forward(self, x):
			return self.net(x)

	class FinalMLP(nn.Module):
		def __init__(self, in_dim):
			super().__init__()
			# output two values: [eval_raw, winner_raw]
			self.net = nn.Sequential(
				nn.Linear(in_dim, 8),
				nn.ReLU(),
				nn.Linear(8, 6),
				nn.ReLU(),
				nn.Linear(6, 2)
			)
		def forward(self, x):
			out = self.net(x)
			#eval_out = torch.nn.functional.softsign(out[:, 0]) * args.clip
			#eval_out = out[:, 0] * args.clip
			eval_out = torch.nn.functional.softsign(out[:, 0]) * 1.5 * args.clip
			winner_out = torch.sigmoid(out[:, 1])
			return eval_out, winner_out

	mid_half_dim = 6
	half_a = HalfMLP(N, out_dim=mid_half_dim).to(device)
	half_b = HalfMLP(N, out_dim=mid_half_dim).to(device)
	final = FinalMLP(in_dim=mid_half_dim*2).to(device)

	if args.load_ckpt:
		if os.path.exists(args.load_ckpt):
			ckpt = torch.load(args.load_ckpt, map_location=device)
			half_a.load_state_dict(ckpt['half_a'])
			half_b.load_state_dict(ckpt['half_b'])
			final.load_state_dict(ckpt['final'])
			print(f"Loaded checkpoint from {args.load_ckpt}")
		else:
			print(f"Warning: checkpoint not found at {args.load_ckpt}")

	all_params = list(half_a.parameters()) + list(half_b.parameters()) + list(final.parameters())
	print("Total parameters:", sum(p.numel() for p in all_params))
	optimizer = torch.optim.Adam(all_params, lr=args.mlp_lr)
	loss_fn_eval = nn.MSELoss()
	loss_fn_winner = nn.BCELoss()
	lambda_w = args.mlp_lambda
	l2_coef = args.mlp_l2
	l2_sym_coef = args.mlp_l2_sym

	def symmetry_reg():
		"""L2 penalty on the difference between half_a and half_b parameters."""
		loss = 0.0
		for pa, pb in zip(half_a.parameters(), half_b.parameters()):
			loss = loss + (pa - pb).pow(2).sum()
		return l2_sym_coef * loss

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
		running_sym_loss = 0.0
		running_total_loss = 0.0
		total = 0
		half_a.train(); half_b.train(); final.train()
		for xb, yb, wb in train_loader:
			xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
			h1 = half_a(xb[:, :N]); h2 = half_b(xb[:, N:])
			conc = torch.cat([h1, h2], dim=1)
			pred_eval, pred_winner = final(conc)
			loss_eval = loss_fn_eval(pred_eval/args.clip, yb/args.clip)
			loss_winner = lambda_w * loss_fn_winner(pred_winner, wb)
			l2_reg = l2_coef * sum(p.pow(2).sum() for p in all_params)
			sym_reg = symmetry_reg()
			loss = loss_eval + loss_winner + l2_reg + sym_reg
			optimizer.zero_grad(); loss.backward(); optimizer.step()
			bs = xb.size(0)
			running_eval_loss += loss_eval.item() * bs
			running_eval_rmse += ((pred_eval - yb)**2).sum().item()
			running_winner_loss += loss_winner.item() * bs
			running_winner_mae += (pred_winner - wb).abs().sum().item()
			running_reg_loss += l2_reg.item() * bs
			running_sym_loss += sym_reg.item() * bs
			running_total_loss += loss.item() * bs
			total += bs
		train_eval_loss = running_eval_loss / total
		train_eval_rmse = (running_eval_rmse / total) ** 0.5
		train_winner_loss = running_winner_loss / total
		train_winner_mae = running_winner_mae / total
		train_reg_loss = running_reg_loss / total
		train_sym_loss = running_sym_loss / total
		train_total_loss = running_total_loss / total

		half_a.eval(); half_b.eval(); final.eval()
		with torch.no_grad():
			Xte_dev = Xte.to(device); Yte_dev = Yte.to(device); Wte_dev = Wte.to(device)
			h1 = half_a(Xte_dev[:, :N]); h2 = half_b(Xte_dev[:, N:])
			conc = torch.cat([h1, h2], dim=1)
			val_eval_pred, val_winner_pred = final(conc)
			val_eval_loss = loss_fn_eval(val_eval_pred/args.clip, Yte_dev/args.clip).item()
			val_eval_rmse = ((val_eval_pred - Yte_dev)**2).mean().item() ** 0.5
			val_winner_loss = lambda_w * loss_fn_winner(val_winner_pred, Wte_dev).item()
			val_winner_mae = (val_winner_pred - Wte_dev).abs().mean().item()
			val_l2_reg = l2_coef * sum(p.pow(2).sum() for p in all_params).item()
			val_sym_reg = symmetry_reg().item()
			val_reg_loss = val_l2_reg
			val_total_loss = val_eval_loss + val_winner_loss + val_reg_loss + val_sym_reg

			if val_eval_loss < best_val_loss:
				best_val_loss = val_eval_loss
				best_epoch = epoch + 1
				best_state = {
					'half_a': {k: v.cpu().clone() for k, v in half_a.state_dict().items()},
					'half_b': {k: v.cpu().clone() for k, v in half_b.state_dict().items()},
					'final': {k: v.cpu().clone() for k, v in final.state_dict().items()}
				}
		
		history['train_loss'].append(train_total_loss)
		history['test_loss'].append(val_total_loss)

		half_a.train(); half_b.train(); final.train()
		print(
			f"Epoch {epoch+1}/{args.mlp_epochs} | "
			f"TRAIN eval={train_eval_rmse:.1f} win={train_winner_mae:.4f} loss={100*train_total_loss:.4f}={100*train_eval_loss:.4f}+{100*train_winner_loss:.4f}+{100*train_reg_loss:.4f}+{100*train_sym_loss:.4f} | "
			f"TEST  eval={val_eval_rmse:.1f} win={val_winner_mae:.4f} loss={100*val_total_loss:.4f}={100*val_eval_loss:.4f}+{100*val_winner_loss:.4f}+{100*val_reg_loss:.4f}+{100*val_sym_reg:.4f}"
		)

	if best_state is not None:
		half_a.load_state_dict({k: v.to(device) for k, v in best_state['half_a'].items()})
		half_b.load_state_dict({k: v.to(device) for k, v in best_state['half_b'].items()})
		final.load_state_dict({k: v.to(device) for k, v in best_state['final'].items()})
		
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

	half_a.eval(); half_b.eval(); final.eval()
	with torch.no_grad():
		Xte_dev = Xte.to(device)
		h1 = half_a(Xte_dev[:, :N])
		h2 = half_b(Xte_dev[:, N:])
		conc = torch.cat([h1, h2], dim=1)
		eval_preds, winner_preds = final(conc)
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
		next_layer = start_layer
		last_bias_layer = None
		for name, p in named_params:
			arr = p.detach().cpu().numpy()
			is_final_last_w = name.endswith("net.4.weight")
			is_final_last_b = name.endswith("net.4.bias")
			if arr.ndim == 2:
				# if this is the final layer producing 2 outputs, keep only the first row (eval head)
				if is_final_last_w and arr.shape[0] == 2:
					arr_print = arr[0:1, :]
				else:
					arr_print = arr
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
		f.write("\t// --- half_a (current player) ---\n")
		next_l = write_module_weights(f, half_a.named_parameters(), "A_", 0)
		f.write("\t// --- half_b (opponent) ---\n")
		next_l = write_module_weights(f, half_b.named_parameters(), "B_", 0)
		f.write("\t// --- head ---\n")
		write_module_weights(f, final.named_parameters(), "H_", next_l)

	print(f"Rust weights written to {rust_out_path}")
