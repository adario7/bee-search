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

import zstandard as zstd

TOTAL_FN = 2152
OLD_TOTAL_FN = 232
FN = 52
FN2 = 104
ADJ_SIZE = 128
HEXES_PER_QUEEN = 60
TOTAL_PCTS = 16
PCT_COUNT = 8
PCT_SQUARE_PER_QUEEN = HEXES_PER_QUEEN * TOTAL_PCTS # 960

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

def swap_features_color(f, adj_perm=None):
	"""
	Swaps White and Black perspective:
	- Tactical features:
	  - STM block (0..52) <-> Opponent block (52..104)
	  - Inside each block: liberties ally <-> enemy, near_pct ally <-> enemy
	  - Adjacency histogram (104..232): permuted via adj_perm
	- Pct-Square features:
	  - White Queen block (232..1192) <-> Black Queen block (1192..2152)
	  - Inside each hex (16 pcts): White pieces (0..7) <-> Black pieces (8..15)
	"""
	if adj_perm is None:
		adj_perm = get_adj_perm()

	f_new = np.empty_like(f)
	# 1. Tactical features
	for block in (0, 1):
		src = block * FN
		dst = (1 - block) * FN
		f_new[dst + 0] = f[src + 2]
		f_new[dst + 1] = f[src + 3]
		f_new[dst + 2] = f[src + 0]
		f_new[dst + 3] = f[src + 1]
		for i in range(PCT_COUNT):
			f_new[dst + 4 + i*6 + 0] = f[src + 4 + i*6 + 0] # movable
			f_new[dst + 4 + i*6 + 1] = f[src + 4 + i*6 + 1] # fixed
			f_new[dst + 4 + i*6 + 2] = f[src + 4 + i*6 + 2] # buried
			f_new[dst + 4 + i*6 + 3] = f[src + 4 + i*6 + 3] # move_n
			f_new[dst + 4 + i*6 + 4] = f[src + 4 + i*6 + 5] # near_enemy is old near_ally
			f_new[dst + 4 + i*6 + 5] = f[src + 4 + i*6 + 4] # near_ally is old near_enemy

	# 2. Adjacency
	f_new[FN2:FN2 + ADJ_SIZE] = f[FN2:FN2 + ADJ_SIZE][adj_perm]

	# 3. Pct-square: swap White Queen and Black Queen blocks
	for q_src in (0, 1):
		q_dst = 1 - q_src
		src_q = OLD_TOTAL_FN + q_src * PCT_SQUARE_PER_QUEEN
		dst_q = OLD_TOTAL_FN + q_dst * PCT_SQUARE_PER_QUEEN
		for h in range(HEXES_PER_QUEEN):
			src_h = src_q + h * TOTAL_PCTS
			dst_h = dst_q + h * TOTAL_PCTS
			f_new[dst_h : dst_h + PCT_COUNT] = f[src_h + PCT_COUNT : src_h + TOTAL_PCTS]
			f_new[dst_h + PCT_COUNT : dst_h + TOTAL_PCTS] = f[src_h : src_h + PCT_COUNT]

	return f_new

CSR_HEADER_DTYPE = np.dtype([
	('magic', 'S4'),
	('version', np.uint32),
	('num_samples', np.uint32),
	('feature_dim', np.uint32),
	('total_nnz', np.uint64),
])

def load_sparse_csr_dataset(bin_path, limit=None):
	print(f"Loading Sparse CSR dataset from {bin_path}...")
	if bin_path.endswith(".zst") or bin_path.endswith(".zstd"):
		dctx = zstd.ZstdDecompressor()
		with open(bin_path, "rb") as f:
			with dctx.stream_reader(f) as reader:
				decompressed = bytearray(reader.read())
	else:
		with open(bin_path, "rb") as f:
			decompressed = bytearray(f.read())

	header = np.frombuffer(decompressed[:24], dtype=CSR_HEADER_DTYPE)[0]
	magic = header['magic'].decode('ascii', errors='ignore')
	if magic != 'BCSR':
		raise ValueError(f"Invalid magic header in {bin_path}: {magic}, expected 'BCSR'")

	num_samples = int(header['num_samples'])
	feature_dim = int(header['feature_dim'])
	total_nnz = int(header['total_nnz'])

	pos = 24
	offsets = np.frombuffer(decompressed, dtype=np.uint64, count=num_samples + 1, offset=pos)
	pos += (num_samples + 1) * 8

	evals = np.frombuffer(decompressed, dtype=np.float32, count=num_samples, offset=pos)
	pos += num_samples * 4

	winners = np.frombuffer(decompressed, dtype=np.float32, count=num_samples, offset=pos)
	pos += num_samples * 4

	static_evals = np.frombuffer(decompressed, dtype=np.float32, count=num_samples, offset=pos)
	pos += num_samples * 4

	turn_nums = np.frombuffer(decompressed, dtype=np.uint16, count=num_samples, offset=pos)
	pos += num_samples * 2

	indices = np.frombuffer(decompressed, dtype=np.uint16, count=total_nnz, offset=pos)
	pos += total_nnz * 2

	values = np.frombuffer(decompressed, dtype=np.int16, count=total_nnz, offset=pos)

	if limit is not None and limit < num_samples:
		num_samples = limit
		offsets = offsets[:num_samples + 1]
		evals = evals[:num_samples]
		winners = winners[:num_samples]
		static_evals = static_evals[:num_samples]
		turn_nums = turn_nums[:num_samples]
		indices = indices[:offsets[num_samples]]
		values = values[:offsets[num_samples]]
		print(f"Limiting to first {limit:,} samples (from --limit)")

	ram_mb = len(decompressed) / (1024 * 1024)
	avg_nnz = float(offsets[num_samples]) / max(1, num_samples)
	print(f"Loaded {num_samples:,} samples (dim={feature_dim}, avg_nnz={avg_nnz:.1f}/{feature_dim}, RAM={ram_mb:.2f} MB)")
	return {
		'num_samples': num_samples,
		'feature_dim': feature_dim,
		'offsets': offsets,
		'evals': evals,
		'winners': winners,
		'static_evals': static_evals,
		'turn_nums': turn_nums,
		'indices': indices,
		'values': values,
	}

class GPUSparseBatchLoader:
	"""
	Direct GPU-unpacked batch loader for Sparse CSR datasets.
	Transfers 1D sparse indices & values directly to CUDA VRAM and unpacks dense batches
	with instantaneous GPU tensor indexing, eliminating CPU worker overhead.
	"""
	def __init__(self, csr_data, sample_indices, batch_size=2048, shuffle=True, device='cpu'):
		self.offsets = csr_data['offsets']
		self.indices = csr_data['indices']
		self.values = csr_data['values']
		self.evals = csr_data['evals']
		self.winners = csr_data['winners']
		self.static_evals = csr_data['static_evals']
		self.sample_indices = np.asarray(sample_indices, dtype=np.int64)
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.device = torch.device(device)
		self.n = len(self.sample_indices)
		self.total_fn = int(csr_data['feature_dim'])

	def __len__(self):
		return (self.n + self.batch_size - 1) // self.batch_size

	def __iter__(self):
		if self.shuffle:
			curr_indices = self.sample_indices[np.random.permutation(self.n)]
		else:
			curr_indices = self.sample_indices

		for start_idx in range(0, self.n, self.batch_size):
			end_idx = min(start_idx + self.batch_size, self.n)
			batch_s_ids = curr_indices[start_idx:end_idx]
			bs = len(batch_s_ids)

			starts = self.offsets[batch_s_ids]
			ends = self.offsets[batch_s_ids + 1]
			lengths = (ends - starts).astype(np.int64)

			row_idx = np.repeat(np.arange(bs, dtype=np.int64), lengths)
			col_idx = np.concatenate([self.indices[s:e] for s, e in zip(starts, ends)]).astype(np.int64)
			val_slice = np.concatenate([self.values[s:e] for s, e in zip(starts, ends)]).astype(np.float32)

			d_rows = torch.from_numpy(row_idx).to(self.device, non_blocking=True)
			d_cols = torch.from_numpy(col_idx).to(self.device, non_blocking=True)
			d_vals = torch.from_numpy(val_slice).to(self.device, non_blocking=True)

			batch_x = torch.zeros((bs, self.total_fn), device=self.device, dtype=torch.float32)
			batch_x[d_rows, d_cols] = d_vals

			batch_y = torch.from_numpy(self.evals[batch_s_ids].astype(np.float32)).to(self.device, non_blocking=True)
			batch_w = torch.from_numpy(self.winners[batch_s_ids].astype(np.float32)).to(self.device, non_blocking=True)
			batch_s = torch.from_numpy(self.static_evals[batch_s_ids].astype(np.float32)).to(self.device, non_blocking=True)

			yield batch_x, batch_y, batch_w, batch_s


if __name__ == "__main__":
	p = argparse.ArgumentParser()
	p.add_argument("--bin", default="logs/features_mlp.bin.zst", help="Path to binary features file (from extract-features)")
	p.add_argument("--evals", default="logs/evaluations.json")
	p.add_argument("--features", default="logs/position_features.json")
	p.add_argument("--static", default="logs/position_static.json")
	p.add_argument("--limit", type=int, default=None, help="Limit number of dataset samples to load")
	p.add_argument("--test-size", type=float, default=0.1)
	p.add_argument("--random-state", type=int, default=42)
	p.add_argument("--clip", type=int, default=6000, help="Clip evaluations to [-clip, clip] before processing")
	# Training hyperparams
	p.add_argument("--epochs", type=int, default=150)
	p.add_argument("--lr0", type=float, default=2e-4)
	p.add_argument("--lr1", type=float, default=1e-5)
	p.add_argument("--batch", type=int, default=2048)
	p.add_argument("--dropout", type=float, default=0.1, help="Dropout rate after hidden ReLU layers")
	p.add_argument("--input-dropout", type=float, default=0.0, help="Dropout rate directly on input features (default: 0.0)")
	p.add_argument("--lambda-w", type=float, default=2e-3, help="Fixed weight for winner loss term")
	p.add_argument("--l2", type=float, default=5e-5, help="L2 regularization coefficient")
	p.add_argument("--skip-lr", action="store_true", help="Skip Linear Regression and do only MLP")
	p.add_argument("--no-adj", action="store_true", help="Zero out all 128 adjacency matrix features for ablation testing")
	p.add_argument("--load-ckpt", default=None, help="Path to checkpoint .pth to load")
	args = p.parse_args()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using compute device: {device}")

	bin_file = args.bin
	if not os.path.exists(bin_file) and os.path.exists(bin_file + ".zst"):
		bin_file = bin_file + ".zst"
	elif not os.path.exists(bin_file) and bin_file.endswith(".bin") and os.path.exists("logs/features_mlp.bin.zst"):
		bin_file = "logs/features_mlp.bin.zst"

	using_bin = os.path.exists(bin_file)
	if using_bin:
		csr_data = load_sparse_csr_dataset(bin_file, args.limit)
		n_samples = csr_data['num_samples']
		L = csr_data['feature_dim']

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

		train_loader = GPUSparseBatchLoader(csr_data, idx_train, batch_size=args.batch, shuffle=True, device=device)
		test_loader = GPUSparseBatchLoader(csr_data, idx_test, batch_size=args.batch, shuffle=False, device=device)

		# Compute exact zero ratio across the dataset
		total_entries = n_samples * L
		total_nnz = int(csr_data['offsets'][n_samples])
		zero_ratio = float(total_entries - total_nnz) / float(total_entries)
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
		def __init__(self, in_dim, dropout=0.15, input_dropout=0.0):
			super().__init__()
			self.input_drop = nn.Dropout(input_dropout) if input_dropout > 0.0 else nn.Identity()
			self.fc1 = nn.Linear(in_dim, 64)
			self.drop1 = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
			self.fc2 = nn.Linear(64, 32)
			self.drop2 = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
			self.fc3 = nn.Linear(32, 16)
			self.drop3 = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
			self.fc4 = nn.Linear(16, 8)
			self.drop4 = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
			self.fc5 = nn.Linear(8, 2)
			self.linear_layers = [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]

		def forward(self, x):
			x = self.input_drop(x)
			x = self.drop1(torch.relu(self.fc1(x)))
			x = self.drop2(torch.relu(self.fc2(x)))
			x = self.drop3(torch.relu(self.fc3(x)))
			x = self.drop4(torch.relu(self.fc4(x)))
			out = self.fc5(x)
			eval_out = torch.nn.functional.softsign(out[:, 0]) * 1.5 * args.clip
			winner_out = torch.sigmoid(out[:, 1])
			return eval_out, winner_out

	model = SingleMLP(in_dim=L, dropout=args.dropout, input_dropout=args.input_dropout).to(device)

	def count_model_stats(model, zero_ratio):
		params = sum(p.numel() for p in model.parameters())
		max_flops = 0
		expected_flops = 0
		is_first = True
		for layer in model.linear_layers:
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
	print(f"  Dropout rates: hidden={args.dropout}, input={args.input_dropout}")
	print(f"  Average zero features: {zero_ratio*100:.2f}% ({L * zero_ratio:.1f} / {L} entries)")
	print(f"  Max FLOPs (100% non-zero inputs): {max_flops:,}")
	print(f"  Expected FLOPs (skipping zero inputs in layer 1): {expected_flops:,.1f}\n")

	optimizer = torch.optim.Adam(all_params, lr=args.lr0)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr1)
	loss_fn_eval = nn.MSELoss()
	loss_fn_winner = nn.BCELoss()
	lambda_w = args.lambda_w
	l2_coef = args.l2

	best_val_loss = float('inf')
	best_state = None
	best_epoch = -1
	history = {'train_loss': [], 'test_loss': []}

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

	def export_rust_weights_and_stats(rust_out_path, m, epoch_num, total_epochs, val_loss, val_rmse, val_mae_w, zero_r):
		dataset_src = bin_file if using_bin else args.evals
		n_tr = len(idx_train)
		n_te = len(idx_test)
		with open(rust_out_path, "w") as f:
			f.write("/*\n")
			f.write("Training & Evaluation Summary\n")
			f.write("-----------------------------\n")
			f.write(f"Dataset:            {dataset_src} ({n_tr + n_te:,} samples: {n_tr:,} train / {n_te:,} test)\n")
			f.write(f"Epochs:             {total_epochs} (Best Epoch: {epoch_num})\n")
			f.write(f"Best Val Loss:      {val_loss:.6f}\n\n")
			f.write("Validation Metrics (Test Set):\n")
			f.write(f"  RMSE:             {val_rmse:.4f} cp\n")
			f.write(f"  Winner MAE:       {val_mae_w:.4f}\n\n")
			f.write("Model Summary:\n")
			f.write(f"  Parameters:       {params:,}\n")
			f.write(f"  Input Dimension:  {L}\n")
			f.write(f"  Zero Features:    {zero_r*100:.2f}% ({L * zero_r:.1f} / {L} entries)\n")
			f.write(f"  Max FLOPs:        {max_flops:,}\n")
			f.write(f"  Expected FLOPs:   {expected_flops:,.1f} (skipping zero inputs in layer 1)\n")
			f.write("*/\n\n")
			f.write("\t// --- single mlp (all features) ---\n")
			write_module_weights(f, m.named_parameters(), "M_", 0)

	try:
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

			# If new best model found, save checkpoint and Rust code immediately
			if val_eval_loss < best_val_loss:
				best_val_loss = val_eval_loss
				best_epoch = epoch + 1
				best_state = {'model': {k: v.cpu().clone() for k, v in model.state_dict().items()}}
				torch.save(best_state, "logs/best_mlp_eval.pth")
				export_rust_weights_and_stats("logs/eval_mlp.rs", model, best_epoch, args.epochs, best_val_loss, val_eval_rmse, val_winner_mae, zero_ratio)

			history['train_loss'].append(train_total_loss)
			history['test_loss'].append(val_total_loss)

			cur_lr = optimizer.param_groups[0]['lr']
			scheduler.step()

			print(
				f"Epoch {epoch+1:3d}/{args.epochs} lr={cur_lr:.2e} | "
				f"TRAIN eval={train_eval_rmse:.1f} win={train_winner_mae:.4f} loss={100*train_total_loss:.4f}={100*train_eval_loss:.4f}+{100*train_winner_loss:.4f}+{100*train_reg_loss:.4f} | "
				f"TEST  eval={val_eval_rmse:.1f} win={val_winner_mae:.4f} loss={100*val_total_loss:.4f}={100*val_eval_loss:.4f}+{100*val_winner_loss:.4f}+{100*val_reg_loss:.4f}"
			)
	except KeyboardInterrupt:
		print(f"\nTraining interrupted by user at epoch {epoch+1}. Loading best model checkpoint (epoch {best_epoch})...")

	if best_state is not None:
		model.load_state_dict({k: v.to(device) for k, v in best_state['model'].items()})
		save_path = "logs/best_mlp_eval.pth"
		torch.save(best_state, save_path)
		print(f"\nBest model (epoch {best_epoch}) confirmed at {save_path}")

	# plot loss history
	os.makedirs("logs", exist_ok=True)
	if len(history['train_loss']) > 0:
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
			all_y.append(yb.cpu().numpy())

	mlp_preds = np.concatenate(all_preds)
	y_test_true = np.concatenate(all_y)
	mlp_metrics = metrics(y_test_true, mlp_preds)

	errors = np.abs(mlp_preds - y_test_true)
	p50_err = float(np.percentile(errors, 50))
	p90_err = float(np.percentile(errors, 90))
	p99_err = float(np.percentile(errors, 99))
	valid_sign = (y_test_true != 0)
	sign_acc = float(np.mean(np.sign(mlp_preds[valid_sign]) == np.sign(y_test_true[valid_sign])) * 100.0) if np.any(valid_sign) else 0.0

	print(f"\nBest epoch (by val loss): {best_epoch}/{args.epochs}, val_total_loss: {best_val_loss:.6f}")
	print("MLP on test set (using best-epoch weights) - evaluation output:")
	print(f"  RMSE:            {mlp_metrics['rmse']:.4f} cp")
	print(f"  MAE:             {mlp_metrics['mae']:.4f} cp")
	print(f"  R² Score:        {mlp_metrics['r2']:.4f} ({mlp_metrics['r2']*100:.2f}% variance explained)")
	print(f"  Sign Accuracy:   {sign_acc:.2f}%")
	print(f"  Median Error:    {p50_err:.2f} cp")
	print(f"  P90 Error:       {p90_err:.2f} cp")
	print(f"  P99 Error:       {p99_err:.2f} cp")

	rust_out_path = "logs/eval_mlp.rs"
	export_rust_weights_and_stats(rust_out_path, model, best_epoch, args.epochs, best_val_loss, mlp_metrics['rmse'], mlp_metrics['mae'], zero_ratio)
	print(f"Rust weights and stats written to {rust_out_path}")




