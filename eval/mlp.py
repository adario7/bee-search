import argparse
import json
import os
import gc
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

TOTAL_FN = 436
PIECE_FN = 11
PIECES_PER_SIDE = 14
PIECE_FEATURES_LEN = PIECES_PER_SIDE * 2 * PIECE_FN # 308
ADJ_SIZE = 128

SAMPLE_DTYPE = np.dtype([
	('features', np.int16, (TOTAL_FN,)),
	('eval', np.float32),
	('winner', np.float32),
	('static_eval', np.float32),
	('turn_num', np.uint16),
	('_pad', np.uint8, 2),
])

def load_json(path):
	if not os.path.exists(path):
		raise FileNotFoundError(path)
	with open(path, "r") as f:
		return json.load(f)

def load_data_from_json(evals_path, features_path, static_path, clip_val):
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

		parts = pos.split(";")
		is_black_turn = len(parts) >= 3 and parts[2].split("[")[0].strip().lower() == "black"

		ev = rec.get("evaluation", 0)
		try:
			ev = float(ev)
		except (ValueError, TypeError):
			ev = 0.0
		# Convert search evaluation to White perspective
		ev_white = -ev if is_black_turn else ev
		ev_white = float(np.clip(ev_white, -clip_val, clip_val))

		win = rec.get("winner", 0.0)
		try:
			win = float(win)
		except (ValueError, TypeError):
			win = 0.0
		win_white = (1.0 - win) if is_black_turn else win
		win_white = float(np.clip(win_white, 0.0, 1.0))

		st = static_map.get(pos, None)
		if st is not None and is_black_turn:
			st = -st

		matched_feats.append(vec)
		matched_y.append(ev_white)
		matched_w.append(win_white)
		matched_static.append(st)

	del raw_evals, feats_map, static_map
	gc.collect()

	if not matched_feats:
		raise ValueError("No matching positions between evals and features")

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
		perm = [0] * ADJ_SIZE
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

def sort_piece_group(cand, start_slot, count):
	chunks = [cand[(start_slot + i) * PIECE_FN : (start_slot + i + 1) * PIECE_FN].copy() for i in range(count)]
	chunks.sort(key=lambda x: tuple(x))
	for i in range(count):
		cand[(start_slot + i) * PIECE_FN : (start_slot + i + 1) * PIECE_FN] = chunks[i]

def canonicalize_features(raw):
	"""
	Fast canonicalization: prunes candidate D6 transforms using WQ->BQ vector (raw[3..5]),
	sorts duplicate piece groups, and selects the lexicographically minimal candidate.
	"""
	transforms = [
		lambda q, r, s: (q, r, s),
		lambda q, r, s: (-s, -q, -r),
		lambda q, r, s: (r, s, q),
		lambda q, r, s: (-q, -r, -s),
		lambda q, r, s: (s, q, r),
		lambda q, r, s: (-r, -s, -q),
		lambda q, r, s: (-q, -s, -r),
		lambda q, r, s: (r, q, s),
		lambda q, r, s: (-s, -r, -q),
		lambda q, r, s: (q, s, r),
		lambda q, r, s: (-r, -q, -s),
		lambda q, r, s: (s, r, q),
	]

	vq = (raw[3], raw[4], raw[5])
	if vq != (0, 0, 0):
		min_v = (999999, 999999, 999999)
		candidates = []
		for k in range(12):
			vk = transforms[k](*vq)
			if vk < min_v:
				min_v = vk
				candidates = [k]
			elif vk == min_v:
				candidates.append(k)
	else:
		candidates = list(range(12))

	best = None
	for k in candidates:
		t = transforms[k]
		cand = raw.copy()
		for p in range(PIECES_PER_SIDE * 2):
			base = p * PIECE_FN
			qw, rw, sw = t(cand[base + 0], cand[base + 1], cand[base + 2])
			qb, rb, sb = t(cand[base + 3], cand[base + 4], cand[base + 5])
			cand[base + 0] = qw
			cand[base + 1] = rw
			cand[base + 2] = sw
			cand[base + 3] = qb
			cand[base + 4] = rb
			cand[base + 5] = sb

		# White piece groups
		sort_piece_group(cand, 1, 3)
		sort_piece_group(cand, 4, 2)
		sort_piece_group(cand, 6, 3)
		sort_piece_group(cand, 9, 2)

		# Black piece groups
		sort_piece_group(cand, 15, 3)
		sort_piece_group(cand, 18, 2)
		sort_piece_group(cand, 20, 3)
		sort_piece_group(cand, 23, 2)

		if len(candidates) == 1:
			return cand

		if best is None or tuple(cand) < tuple(best):
			best = cand

	return best

def swap_features_color(f, adj_perm=None):
	"""
	Swaps White and Black perspective:
	- White piece blocks (0..154) <-> Black piece blocks (154..308)
	- Inside each piece: dq/dr/ds wrt WQ <-> dq/dr/ds wrt BQ, n_white <-> n_black
	- Adjacency histogram (308..436) permuted via adj_perm
	- Canonicalized across D6 symmetries and piece ordering
	"""
	if adj_perm is None:
		adj_perm = get_adj_perm()

	f_new = np.empty_like(f)
	half_pieces = PIECES_PER_SIDE * PIECE_FN # 154

	# Swap White and Black pieces, with internal coordinate / neighbor swap
	for slot in range(PIECES_PER_SIDE):
		w_base = slot * PIECE_FN
		b_base = half_pieces + w_base

		# New White piece (from old Black piece)
		f_new[w_base + 0:w_base + 3] = f[b_base + 3:b_base + 6] # wrt WQ is old wrt BQ
		f_new[w_base + 3:w_base + 6] = f[b_base + 0:b_base + 3] # wrt BQ is old wrt WQ
		f_new[w_base + 6] = f[b_base + 7]                       # n_white is old n_black
		f_new[w_base + 7] = f[b_base + 6]                       # n_black is old n_white
		f_new[w_base + 8:w_base + 11] = f[b_base + 8:b_base + 11]

		# New Black piece (from old White piece)
		f_new[b_base + 0:b_base + 3] = f[w_base + 3:w_base + 6]
		f_new[b_base + 3:b_base + 6] = f[w_base + 0:w_base + 3]
		f_new[b_base + 6] = f[w_base + 7]
		f_new[b_base + 7] = f[w_base + 6]
		f_new[b_base + 8:b_base + 11] = f[w_base + 8:w_base + 11]

	# Adjacency histogram permutation
	f_new[PIECE_FEATURES_LEN:PIECE_FEATURES_LEN + ADJ_SIZE] = f[PIECE_FEATURES_LEN:PIECE_FEATURES_LEN + ADJ_SIZE][adj_perm]
	return canonicalize_features(f_new)

class LazyBinaryFeatureDataset(Dataset):
	"""
	Memory-Mapped Dataset reading directly from .bin file.
	"""
	def __init__(self, raw_mmap, indices, augment=False):
		self.raw_mmap = raw_mmap
		self.indices = np.asarray(indices, dtype=np.int64)
		self.augment = augment
		self.multiplier = 2 if augment else 1
		self.adj_perm = get_adj_perm() if augment else None

	def __len__(self):
		return len(self.indices) * self.multiplier

	def __getitem__(self, idx):
		if self.augment:
			orig_idx = self.indices[idx // 2]
			is_swapped = (idx % 2 == 1)
		else:
			orig_idx = self.indices[idx]
			is_swapped = False

		record = self.raw_mmap[orig_idx]
		feat = record['features'].astype(np.float32)
		y = np.float32(record['eval'])
		w = np.float32(record['winner'])
		s = np.float32(record['static_eval'])

		if is_swapped:
			feat = swap_features_color(feat, self.adj_perm)
			y = np.float32(-y)
			w = np.float32(1.0 - w)
			s = np.float32(-s)

		return feat, y, w, s


if __name__ == "__main__":
	p = argparse.ArgumentParser()
	p.add_argument("--bin", default="logs/features_mlp.bin", help="Path to binary features file (from extract-features)")
	p.add_argument("--evals", default="logs/evaluations.json")
	p.add_argument("--features", default="logs/position_features.json")
	p.add_argument("--static", default="logs/position_static.json")
	p.add_argument("--limit", type=int, default=None, help="Limit number of dataset samples to load")
	p.add_argument("--test-size", type=float, default=0.1)
	p.add_argument("--random-state", type=int, default=42)
	p.add_argument("--clip", type=int, default=6000, help="Clip evaluations to [-clip, clip] before processing")
	# Training hyperparams
	p.add_argument("--epochs", type=int, default=150)
	p.add_argument("--lr", type=float, default=5e-4)
	p.add_argument("--batch", type=int, default=2048)
	p.add_argument("--lambda-w", type=float, default=2e-3, help="Fixed weight for winner loss term")
	p.add_argument("--l2", type=float, default=3e-5, help="L2 regularization coefficient")
	p.add_argument("--skip-lr", action="store_true", help="Skip Linear Regression and do only MLP")
	p.add_argument("--no-adj", action="store_true", help="Zero out all 128 adjacency matrix features for ablation testing")
	p.add_argument("--load-ckpt", default=None, help="Path to checkpoint .pth to load")
	args = p.parse_args()

	using_bin = os.path.exists(args.bin)
	if using_bin:
		print(f"Loading binary dataset from {args.bin}...")
		mmap_data = np.memmap(args.bin, dtype=SAMPLE_DTYPE, mode='r')
		n_samples = len(mmap_data)
		if args.limit is not None and args.limit < n_samples:
			n_samples = args.limit
			mmap_data = mmap_data[:n_samples]
			print(f"Limiting to first {n_samples:,} samples (from --limit)")
		print(f"Loaded {n_samples:,} samples from {args.bin}")
		L = TOTAL_FN

		# Split by position pairs (2 samples per position: original and precomputed symmetric variant)
		n_pairs = n_samples // 2
		if n_pairs > 0:
			pair_idx = np.arange(n_pairs)
			pair_train, pair_test = train_test_split(pair_idx, test_size=args.test_size, random_state=args.random_state)
			idx_train = np.sort(np.concatenate([pair_train * 2, pair_train * 2 + 1]))
			idx_test = np.sort(np.concatenate([pair_test * 2, pair_test * 2 + 1]))
		else:
			idx = np.arange(n_samples)
			idx_train, idx_test = train_test_split(idx, test_size=args.test_size, random_state=args.random_state)

		print(f"Loading {len(idx_train):,} train & {len(idx_test):,} test tensors into RAM for zero-overhead GPU feeding...")
		X_train = torch.from_numpy(mmap_data['features'][idx_train].astype(np.float32))
		y_train = torch.from_numpy(mmap_data['eval'][idx_train].astype(np.float32))
		w_train = torch.from_numpy(mmap_data['winner'][idx_train].astype(np.float32))
		s_train = torch.from_numpy(mmap_data['static_eval'][idx_train].astype(np.float32))

		X_test = torch.from_numpy(mmap_data['features'][idx_test].astype(np.float32))
		y_test = torch.from_numpy(mmap_data['eval'][idx_test].astype(np.float32))
		w_test = torch.from_numpy(mmap_data['winner'][idx_test].astype(np.float32))
		s_test = torch.from_numpy(mmap_data['static_eval'][idx_test].astype(np.float32))

		class FastTensorDataset(Dataset):
			def __init__(self, x, y, w, s):
				self.x = x
				self.y = y
				self.w = w
				self.s = s
			def __len__(self):
				return len(self.x)
			def __getitem__(self, i):
				return self.x[i], self.y[i], self.w[i], self.s[i]

		train_ds = FastTensorDataset(X_train, y_train, w_train, s_train)
		test_ds = FastTensorDataset(X_test, y_test, w_test, s_test)

		train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, pin_memory=torch.cuda.is_available())
		test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False)

		# Estimate zero ratio from a sample
		sample_feats = mmap_data['features'][:min(1000, n_samples)]
		zero_ratio = float(np.mean(sample_feats == 0))
	else:
		X_full, y_full, winner_full, static_list = load_data_from_json(args.evals, args.features, args.static, clip_val=args.clip)
		n_original = len(y_full)
		if args.limit is not None and args.limit < n_original:
			n_original = args.limit
			X_full = X_full[:n_original]
			y_full = y_full[:n_original]
			winner_full = winner_full[:n_original]
			static_list = static_list[:n_original]
			print(f"Limiting to first {n_original:,} samples (from --limit)")
		print(f"Loaded {n_original} matched positions from JSON")
		L = X_full.shape[1]

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

				row_orig = idx_out * 2
				X_aug[row_orig] = f
				y_aug[row_orig] = y
				w_aug[row_orig] = w
				static_aug.append(s)

				row_swap = idx_out * 2 + 1
				f_swapped = swap_features_color(f, adj_perm)
				X_aug[row_swap] = f_swapped
				y_aug[row_swap] = -y
				w_aug[row_swap] = 1.0 - w
				static_aug.append(-s if s is not None else None)

			return X_aug, y_aug, w_aug, static_aug

		X_train_aug, y_train_aug, winner_train_aug, static_train_aug = augment_indices(idx_train)
		X_test_aug, y_test_aug, winner_test_aug, static_test_aug = augment_indices(idx_test)

		zero_ratio = float(np.mean(X_full == 0))
		del X_full, y_full, winner_full, static_list
		gc.collect()

		class SimpleTensorDataset(Dataset):
			def __init__(self, x, y, w, s):
				self.x = torch.from_numpy(x)
				self.y = torch.from_numpy(y)
				self.w = torch.from_numpy(w)
				self.s = s
			def __len__(self):
				return len(self.x)
			def __getitem__(self, i):
				s_val = self.s[i] if self.s[i] is not None else 0.0
				return self.x[i], self.y[i], self.w[i], s_val

		train_loader = DataLoader(SimpleTensorDataset(X_train_aug, y_train_aug, winner_train_aug, static_train_aug), batch_size=args.batch, shuffle=True)
		test_loader = DataLoader(SimpleTensorDataset(X_test_aug, y_test_aug, winner_test_aug, static_test_aug), batch_size=args.batch, shuffle=False)

	# -------------------- MLP -------------------------

	print("\n-- MLP --")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	torch.manual_seed(args.random_state)

	class SingleMLP(nn.Module):
		def __init__(self, in_dim):
			super().__init__()
			self.net = nn.Sequential(
				nn.Linear(in_dim, 64),
				nn.ReLU(),
				nn.Linear(64, 32),
				nn.ReLU(),
				nn.Linear(32, 16),
				nn.ReLU(),
				nn.Linear(16, 8),
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
	print("Model Summary:")
	print(f"  Total parameters: {params:,}")
	print(f"  Input feature dimension: {L}")
	print(f"  Average zero features: {zero_ratio*100:.2f}% ({L * zero_ratio:.1f} / {L} entries)")
	print(f"  Max FLOPs (100% non-zero inputs): {max_flops:,}")
	print(f"  Expected FLOPs (skipping zero inputs in layer 1): {expected_flops:,.1f}\n")

	optimizer = torch.optim.Adam(all_params, lr=args.lr)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr*2e-2)
	loss_fn_eval = nn.MSELoss()
	loss_fn_winner = nn.BCELoss()
	lambda_w = args.lambda_w
	l2_coef = args.l2

	best_val_loss = float('inf')
	best_state = None
	best_epoch = -1
	history = {'train_loss': [], 'test_loss': []}

	for epoch in range(args.epochs):
		running_eval_loss = torch.tensor(0.0, device=device)
		running_eval_sq_err = torch.tensor(0.0, device=device)
		running_winner_loss = torch.tensor(0.0, device=device)
		running_winner_abs_err = torch.tensor(0.0, device=device)
		running_total_loss = torch.tensor(0.0, device=device)
		total = 0
		model.train()
		for xb, yb, wb, _ in train_loader:
			xb, yb, wb = xb.to(device).float(), yb.to(device).float(), wb.to(device).float()
			pred_eval, pred_winner = model(xb)
			loss_eval = loss_fn_eval(pred_eval/args.clip, yb/args.clip)
			loss_winner = lambda_w * loss_fn_winner(pred_winner, wb)
			l2_reg = l2_coef * sum(p.pow(2).sum() for p in all_params)
			loss = loss_eval + loss_winner + l2_reg
			optimizer.zero_grad(); loss.backward(); optimizer.step()
			bs = xb.size(0)
			running_eval_loss += loss_eval.detach() * bs
			running_eval_sq_err += ((pred_eval.detach() - yb)**2).sum()
			running_winner_loss += loss_winner.detach() * bs
			running_winner_abs_err += (pred_winner.detach() - wb).abs().sum()
			running_total_loss += loss.detach() * bs
			total += bs
		train_eval_loss = (running_eval_loss / total).item()
		train_eval_rmse = ((running_eval_sq_err / total) ** 0.5).item()
		train_winner_loss = (running_winner_loss / total).item()
		train_winner_mae = (running_winner_abs_err / total).item()
		train_reg_loss = (l2_coef * sum(p.pow(2).sum() for p in all_params)).item()
		train_total_loss = (running_total_loss / total).item()

		model.eval()
		val_eval_loss_t = torch.tensor(0.0, device=device)
		val_eval_sq_err = torch.tensor(0.0, device=device)
		val_winner_loss_t = torch.tensor(0.0, device=device)
		val_winner_abs_err = torch.tensor(0.0, device=device)
		val_total = 0
		with torch.no_grad():
			for xb, yb, wb, _ in test_loader:
				xb, yb, wb = xb.to(device).float(), yb.to(device).float(), wb.to(device).float()
				val_eval_pred, val_winner_pred = model(xb)
				bs = xb.size(0)
				val_eval_loss_t += loss_fn_eval(val_eval_pred/args.clip, yb/args.clip) * bs
				val_eval_sq_err += ((val_eval_pred - yb)**2).sum()
				val_winner_loss_t += lambda_w * loss_fn_winner(val_winner_pred, wb) * bs
				val_winner_abs_err += (val_winner_pred - wb).abs().sum()
				val_total += bs

		val_eval_loss = (val_eval_loss_t / val_total).item()
		val_eval_rmse = ((val_eval_sq_err / val_total) ** 0.5).item()
		val_winner_loss = (val_winner_loss_t / val_total).item()
		val_winner_mae = (val_winner_abs_err / val_total).item()
		val_reg_loss = (l2_coef * sum(p.pow(2).sum() for p in all_params)).item()
		val_total_loss = val_eval_loss + val_winner_loss + val_reg_loss

		if val_eval_loss < best_val_loss:
			best_val_loss = val_eval_loss
			best_epoch = epoch + 1
			best_state = {'model': {k: v.cpu().clone() for k, v in model.state_dict().items()}}

		history['train_loss'].append(train_total_loss)
		history['test_loss'].append(val_total_loss)

		cur_lr = optimizer.param_groups[0]['lr']
		scheduler.step()

		print(
			f"Epoch {epoch+1:3d}/{args.epochs} lr={cur_lr:.2e} | "
			f"TRAIN eval={train_eval_rmse:.1f} win={train_winner_mae:.4f} loss={100*train_total_loss:.4f}={100*train_eval_loss:.4f}+{100*train_winner_loss:.4f}+{100*train_reg_loss:.4f} | "
			f"TEST  eval={val_eval_rmse:.1f} win={val_winner_mae:.4f} loss={100*val_total_loss:.4f}={100*val_eval_loss:.4f}+{100*val_winner_loss:.4f}+{100*val_reg_loss:.4f}"
		)

	if best_state is not None:
		model.load_state_dict({k: v.to(device) for k, v in best_state['model'].items()})
		
		# Save best model
		save_path = "logs/best_mlp_eval.pth"
		torch.save(best_state, save_path)
		print(f"\nBest model (epoch {best_epoch}) saved to {save_path}")

	# plot loss history
	os.makedirs("logs", exist_ok=True)
	plt.figure(figsize=(10, 6))
	plt.plot(history['train_loss'], label='Train Total Loss')
	plt.plot(history['test_loss'], label='Test Total Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.title('Training and Test Total Loss over Time')
	plt.legend()
	plt.grid(True)
	plt.savefig("logs/training_loss.png")
	print("Loss plot saved to logs/training_loss.png")
	with open("logs/training_history.json", "w") as f:
		json.dump(history, f)

	model.eval()
	all_preds, all_y = [], []
	with torch.no_grad():
		for xb, yb, _, _ in test_loader:
			xb = xb.to(device).float()
			eval_preds, _ = model(xb)
			all_preds.append(eval_preds.cpu().numpy())
			all_y.append(yb.numpy())

	mlp_preds = np.concatenate(all_preds)
	y_test_true = np.concatenate(all_y)
	mlp_metrics = metrics(y_test_true, mlp_preds)

	print(f"\nBest epoch (by val loss): {best_epoch}, val_total_loss: {best_val_loss:.6f}")
	print("MLP on test set (using best-epoch weights) - evaluation output:")
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
				if is_final_last_w and arr.shape[0] == 2:
					arr_print = arr[0:1, :]
				else:
					arr_print = arr
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


