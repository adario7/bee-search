import argparse
import json
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
    p.add_argument("--symmetric", dest="symmetric", action="store_true")
    p.add_argument("--no-symmetric", dest="symmetric", action="store_false")
    p.set_defaults(symmetric=True)
    p.add_argument("--clip", type=int, default=6000, help="Clip evaluations to [-clip, clip] before processing")
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

    if not args.symmetric:
        X_train = X_full[idx_train]
        X_test = X_full[idx_test]
        y_train = y_full[idx_train]
        y_test = y_full[idx_test]

        # static predictions available for subset
        pos_test = positions[idx_test]
        static_test = np.array([static_map.get(p, None) for p in pos_test], dtype=object)

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)

        reg_metrics_full = metrics(y_test, y_pred_test)

        # test subset that has static_eval
        mask_static = np.array([p in static_map for p in pos_test])
        n_test_with_static = int(mask_static.sum())
        if n_test_with_static == 0:
            static_metrics = None
        else:
            y_test_sub = y_test[mask_static]
            static_preds = np.array([static_map[p] for p in pos_test[mask_static]], dtype=float)
            static_metrics = metrics(y_test_sub, static_preds)

        print(f"n_samples: {len(merged)}, n_features: {X_full.shape[1]}")
        print(f"train_size: {len(X_train)}, test_size: {len(X_test)}, test_with_static: {n_test_with_static}")
        print("\nLinear Regression (on test set):")
        print(f"  MSE: {reg_metrics_full['mse']:.4f}")
        print(f"  MAE: {reg_metrics_full['mae']:.4f}")
        print(f"  R2:  {reg_metrics_full['r2']:.4f}")

        if static_metrics:
            print("\nStatic_eval predictor (on same subset):")
            print(f"  MSE: {static_metrics['mse']:.4f}")
            print(f"  MAE: {static_metrics['mae']:.4f}")
            print(f"  R2:  {static_metrics['r2']:.4f}")
        else:
            print("\nNo static_eval available in test set for comparison.")

        print("\nLinear model parameters:")
        print("Intercept:", float(model.intercept_))
        print("Coefficients:", list(map(float, model.coef_)))
    else:
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
