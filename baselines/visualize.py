import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def plot_gamma_heatmap(gamma_mean, title, out_path):
    """Plot a KT×KS heatmap of mean Gamma."""
    G = np.array(gamma_mean)
    fig, ax = plt.subplots(figsize=(max(3, G.shape[1]), max(3, G.shape[0])))
    im = ax.imshow(G, aspect="auto", cmap="Blues", vmin=0)
    ax.set_xlabel("Student layer j")
    ax.set_ylabel("Teacher layer k")
    ax.set_xticks(range(G.shape[1]))
    ax.set_yticks(range(G.shape[0]))
    ax.set_xticklabels([f"S{j}" for j in range(G.shape[1])])
    ax.set_yticklabels([f"T{k}" for k in range(G.shape[0])])
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            ax.text(j, i, f"{G[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_loss_curves(metrics_json_path, out_path):
    """Plot train loss components and val accuracy from a metrics.json file."""
    with open(metrics_json_path) as f:
        metrics = json.load(f)

    epochs = [m["epoch"] for m in metrics]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, [m["train_loss_total"] for m in metrics], label="total")
    ax1.plot(epochs, [m["train_loss_task"] for m in metrics], label="task")
    ax1.plot(epochs, [m["train_loss_kd"] for m in metrics], label="kd")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.set_title("Train loss")

    ax2.plot(epochs, [m["val_acc"] for m in metrics], color="green")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val accuracy")
    ax2.set_title("Validation accuracy")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_results_table(results_dir, out_path=None):
    """Scan eval_*.json files and print a markdown table of best val acc ± std per run type."""
    # Collect from metrics.json files
    run_accs = {}
    for metrics_path in glob.glob(os.path.join(results_dir, "*", "metrics.json")):
        run_name = os.path.basename(os.path.dirname(metrics_path))
        # Strip seed suffix to group by configuration
        parts = run_name.rsplit("_seed", 1)
        config = parts[0] if len(parts) == 2 else run_name
        with open(metrics_path) as f:
            metrics = json.load(f)
        best_val = max(m["val_acc"] for m in metrics)
        run_accs.setdefault(config, []).append(best_val)

    rows = []
    for config, accs in sorted(run_accs.items()):
        mean = np.mean(accs)
        std = np.std(accs)
        rows.append((config, mean, std, len(accs)))

    header = "| Configuration | Val Acc (mean) | Val Acc (std) | Seeds |"
    sep = "|---------------|----------------|---------------|-------|"
    lines = [header, sep]
    for config, mean, std, n in rows:
        lines.append(f"| {config} | {mean:.4f} | {std:.4f} | {n} |")

    table = "\n".join(lines)
    print(table)

    if out_path:
        with open(out_path, "w") as f:
            f.write(table + "\n")
    return table


def plot_tau_sweep(results_dir, teacher, student, direction, out_path):
    """Plot val accuracy vs tau for B4 sweep."""
    taus, accs = [], []
    for tau in [0.01, 0.1, 1.0, 10.0, 100.0]:
        pattern = os.path.join(
            results_dir,
            f"b4_{teacher}_{student}_tau{tau}_dir{direction}_seed*",
            "metrics.json",
        )
        seed_accs = []
        for path in glob.glob(pattern):
            with open(path) as f:
                metrics = json.load(f)
            seed_accs.append(max(m["val_acc"] for m in metrics))
        if seed_accs:
            taus.append(tau)
            accs.append(np.mean(seed_accs))

    if not taus:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(taus, accs, marker="o")
    ax.set_xlabel("Temperature τ")
    ax.set_ylabel("Best val accuracy")
    ax.set_title(f"B4 τ sweep — teacher={teacher}, student={student}, dir={direction}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    # Results table
    make_results_table(
        args.results_dir, out_path=os.path.join(args.out_dir, "results_table.md")
    )

    # Loss curves for each run
    for metrics_path in glob.glob(os.path.join(args.results_dir, "*", "metrics.json")):
        run_name = os.path.basename(os.path.dirname(metrics_path))
        plot_loss_curves(
            metrics_path, os.path.join(args.out_dir, f"loss_{run_name}.png")
        )

    # Gamma heatmaps from eval JSON files
    for eval_path in glob.glob(os.path.join(args.results_dir, "eval_*.json")):
        with open(eval_path) as f:
            ev = json.load(f)
        if ev.get("gamma_mean") is not None:
            run_name = os.path.splitext(os.path.basename(eval_path))[0]
            title = (
                f"{ev['baseline']} T={ev['teacher']} S={ev['student']} "
                f"τ={ev['tau']} dir={ev['direction']}"
            )
            plot_gamma_heatmap(
                ev["gamma_mean"],
                title,
                os.path.join(args.out_dir, f"gamma_{run_name}.png"),
            )

    # Tau sweep plots
    for teacher in ("rnn", "cnn"):
        student = "cnn" if teacher == "rnn" else "rnn"
        for direction in ("row", "col"):
            plot_tau_sweep(
                args.results_dir,
                teacher,
                student,
                direction,
                os.path.join(
                    args.out_dir, f"tau_sweep_{teacher}_{student}_{direction}.png"
                ),
            )

    print(f"Figures written to {args.out_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results")
    p.add_argument("--out_dir", default="figures")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
