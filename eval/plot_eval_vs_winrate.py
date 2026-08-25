import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def sigmoid_model(x, a):
    # P(win) = 1 / (1 + exp(-a * x))
    return 1.0 / (1.0 + np.exp(-a * x))

def lichess_model(x, c):
    # P(win) = 1 / (1 + 10^(-x / c))
    return 1.0 / (1.0 + 10.0 ** (-x / c))

def main():
    evals_path = "/home/alessandro/local/logs/evaluations.json"
    print(f"Loading evaluations from {evals_path}...")

    with open(evals_path, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data):,} total records. Extracting evaluation and winner values...")

    evals = []
    winners = []

    for r in data:
        ev = r.get("evaluation")
        w = r.get("winner")
        if ev is not None and w is not None:
            try:
                ev_f = float(ev)
                w_f = float(w)
                # Keep valid range
                if -6000 <= ev_f <= 6000 and 0.0 <= w_f <= 1.0:
                    evals.append(ev_f)
                    winners.append(w_f)
            except (ValueError, TypeError):
                continue

    evals = np.array(evals, dtype=np.float64)
    winners = np.array(winners, dtype=np.float64)
    n_samples = len(evals)
    print(f"Valid paired samples: {n_samples:,}")

    # -------------------------------------------------------------
    # Binning
    # -------------------------------------------------------------
    bin_width = 100.0  # 100 centipawns per bin
    min_eval = -4000.0
    max_eval = 4000.0
    bins = np.arange(min_eval, max_eval + bin_width, bin_width)
    bin_indices = np.digitize(evals, bins) - 1

    bin_centers = []
    bin_win_rates = []
    bin_counts = []
    bin_stds = []

    for i in range(len(bins) - 1):
        mask = (bin_indices == i)
        count = np.sum(mask)
        if count >= 30:  # Only bins with sufficient sample size
            b_eval = evals[mask]
            b_win = winners[mask]
            mean_eval = np.mean(b_eval)
            mean_win = np.mean(b_win)
            # Standard error of the mean
            sem = np.std(b_win) / np.sqrt(count)
            bin_centers.append(mean_eval)
            bin_win_rates.append(mean_win)
            bin_counts.append(count)
            bin_stds.append(sem)

    bin_centers = np.array(bin_centers)
    bin_win_rates = np.array(bin_win_rates)
    bin_counts = np.array(bin_counts)
    bin_stds = np.array(bin_stds)

    # -------------------------------------------------------------
    # Curve Fitting
    # -------------------------------------------------------------
    # Fit logistic sigmoid: P(win) = 1 / (1 + exp(-a * x))
    popt_sig, _ = curve_fit(sigmoid_model, bin_centers, bin_win_rates, p0=[0.001], sigma=bin_stds, absolute_sigma=True)
    a_fit = popt_sig[0]

    # Fit base-10 logistic: P(win) = 1 / (1 + 10^(-x / C))
    popt_c, _ = curve_fit(lichess_model, bin_centers, bin_win_rates, p0=[1200.0], sigma=bin_stds, absolute_sigma=True)
    c_fit = popt_c[0]

    print(f"\nFitted Sigmoid parameter: a = {a_fit:.6e}")
    print(f"Fitted Base-10 Scaling Constant: C = {c_fit:.1f} cp")
    print(f"Formula: P(win) = 1 / (1 + 10^(-eval / {c_fit:.1f}))")

    # -------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------
    os.makedirs("logs", exist_ok=True)
    out_path = "logs/eval_vs_winrate.png"

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    # Upper subplot: Win Rate vs Centipawns
    x_curve = np.linspace(-4000, 4000, 500)
    y_curve = lichess_model(x_curve, c_fit)

    # Empirical bin points with 95% CI error bars
    ax1.errorbar(
        bin_centers,
        bin_win_rates * 100.0,
        yerr=bin_stds * 1.96 * 100.0,
        fmt="o",
        color="#1f77b4",
        ecolor="#6baed6",
        elinewidth=1.5,
        capsize=3,
        capthick=1.5,
        markersize=5,
        alpha=0.85,
        label=f"Empirical Binned Data (100 cp bins, N={n_samples:,})",
        zorder=3,
    )

    # Fitted sigmoid curve
    ax1.plot(
        x_curve,
        y_curve * 100.0,
        color="#d62728",
        linewidth=2.5,
        label=rf"Fitted Logistic Model: $P(\mathrm{{win}}) = \frac{{1}}{{1 + 10^{{-\mathrm{{eval}} / {c_fit:.0f}}}}}$",
        zorder=4,
    )

    # Standard 1200 cp reference curve (common in chess / lichess)
    y_ref = lichess_model(x_curve, 1200.0)
    ax1.plot(
        x_curve,
        y_ref * 100.0,
        color="#7f7f7f",
        linestyle="--",
        linewidth=1.5,
        label=r"Reference Curve ($C = 1200\mathrm{\ cp}$)",
        zorder=2,
    )

    # Reference grid & markers
    ax1.axhline(50.0, color="gray", linestyle=":", alpha=0.6)
    ax1.axvline(0.0, color="gray", linestyle=":", alpha=0.6)
    ax1.set_ylabel("Empirical Win Rate (%)", fontsize=12, fontweight="bold")
    ax1.set_title("True Evaluation (Centipawns) vs True Win Rate in Hive (evaluations.json)", fontsize=14, fontweight="bold", pad=12)
    ax1.set_ylim(-2, 102)
    ax1.legend(fontsize=11, loc="lower right", framealpha=0.95)
    ax1.grid(True, alpha=0.3)

    # Lower subplot: Sample Count Histogram (log scale)
    bar_width = bin_width * 0.85
    ax2.bar(
        bin_centers,
        bin_counts,
        width=bar_width,
        color="#3182bd",
        edgecolor="#08519c",
        alpha=0.75,
        zorder=3,
    )
    ax2.set_yscale("log")
    ax2.set_ylabel("Positions\n(Log Scale)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Search Evaluation (Centipawns)", fontsize=12, fontweight="bold")
    ax2.set_xlim(-4100, 4100)
    ax2.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Plot successfully saved to {out_path}")

if __name__ == "__main__":
    main()
