import argparse, json, os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

CLIP = 6000

def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r") as f:
        return json.load(f)

def load_evals(path):
    data = load_json(path)
    df = pd.DataFrame(data)
    if "evaluation" not in df.columns:
        raise KeyError("No 'evaluation' in evals")
    df["evaluation"] = pd.to_numeric(df["evaluation"], errors="coerce").fillna(0).astype(int)
    df["evaluation"] = df["evaluation"].clip(-CLIP, CLIP)
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
    args = p.parse_args()

    evals_df = load_evals(args.evals)
    feats_df = load_features(args.features)
    static_map = load_static(args.static)

    merged = pd.merge(evals_df, feats_df, on="position", how="inner")
    if merged.empty:
        raise ValueError("No matching positions between evals and features")

    merged, X = expand_features(merged)
    y = merged["evaluation"].values
    positions = merged["position"].values

    X_train, X_test, y_train, y_test, pos_train, pos_test = train_test_split(
        X, y, positions, test_size=args.test_size, random_state=args.random_state
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)

    reg_metrics_full = metrics(y_test, y_pred_test)

    # Prepare subset of test set that has static_eval
    mask_static = np.array([p in static_map for p in pos_test])
    n_test_with_static = int(mask_static.sum())

    if n_test_with_static == 0:
        raise ValueError("No test positions have static_eval available for comparison")

    pos_test_sub = pos_test[mask_static]
    y_test_sub = y_test[mask_static]
    y_pred_sub = y_pred_test[mask_static]
    static_preds = np.array([static_map[p] for p in pos_test_sub], dtype=float)

    reg_metrics_sub = metrics(y_test_sub, y_pred_sub)
    static_metrics = metrics(y_test_sub, static_preds)

    # Output
    print(f"n_samples: {len(merged)}, n_features: {X.shape[1]}")
    print(f"train_size: {len(X_train)}, test_size: {len(X_test)}, test_with_static: {n_test_with_static}")
    print("\nLinear Regression (on full test set):")
    print(f"  MSE: {reg_metrics_full['mse']:.4f}")
    print(f"  MAE: {reg_metrics_full['mae']:.4f}")
    print(f"  R2:  {reg_metrics_full['r2']:.4f}")

    print("\nLinear Regression (on test subset with static_eval):")
    print(f"  MSE: {reg_metrics_sub['mse']:.4f}")
    print(f"  MAE: {reg_metrics_sub['mae']:.4f}")
    print(f"  R2:  {reg_metrics_sub['r2']:.4f}")

    print("\nStatic_eval predictor (on same subset):")
    print(f"  MSE: {static_metrics['mse']:.4f}")
    print(f"  MAE: {static_metrics['mae']:.4f}")
    print(f"  R2:  {static_metrics['r2']:.4f}")

    print("\nLinear model parameters:")
    print("Intercept:", float(model.intercept_))
    print("Coefficients:", list(map(float, model.coef_)))
