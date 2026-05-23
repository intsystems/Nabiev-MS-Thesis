import argparse
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from data import get_cifar10
from models import SmallCNN, SmallRNN
from projectors import build_projectors

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_model(arch, device):
    if arch == 'cnn':
        return SmallCNN().to(device)
    return SmallRNN().to(device)

def best_val_accs(run_dirs):
    """Return list of best val acc per run directory (one per seed)."""
    accs = []
    for d in run_dirs:
        path = os.path.join(d, 'metrics.json')
        if not os.path.exists(path):
            continue
        m = json.load(open(path))
        accs.append(max(ep['val_acc'] for ep in m))
    return accs


def find_runs(results_dir, prefix, seeds=(0, 42)):
    return [os.path.join(results_dir, f"{prefix}_seed{s}")
            for s in seeds
            if os.path.isdir(os.path.join(results_dir, f"{prefix}_seed{s}"))]

def build_bar_chart(results_dir, out_path):
    seeds = (0, 42)

    # Each direction entry: display label, ordered list of (bar_label, run_prefix),
    # and a prefix for the teacher upper-bound dashed line.
    directions = [
        {
            'title': 'RNN → CNN  (teacher=RNN, student=CNN)',
            'bars': [
                ('B0\n(no KD)',    'b0_cnn_cnn'),   # CNN trained standalone
                ('B1\n(resp. KD)', 'b1_rnn_cnn'),
                ('B2\n(hard)',     'b2_rnn_cnn'),
                ('B3\n(uniform)',  'b3_rnn_cnn_tau1.0_dirrow'),
                ('Proposed',       'proposed_rnn_cnn_tau1.0_dirrow'),
            ],
            'teacher_prefix': 'b0_rnn_rnn',
        },
        {
            'title': 'CNN → RNN  (teacher=CNN, student=RNN)',
            'bars': [
                ('B0\n(no KD)',    'b0_rnn_rnn'),   # RNN trained standalone
                ('B1\n(resp. KD)', 'b1_cnn_rnn'),
                ('B2\n(hard)',     'b2_cnn_rnn'),
                ('B3\n(uniform)',  'b3_cnn_rnn_tau1.0_dirrow'),
                ('Proposed',       'proposed_cnn_rnn_tau1.0_dirrow'),
            ],
            'teacher_prefix': 'b0_cnn_cnn',
        },
    ]

    palette = ['#aec7e8', '#ffbb78', '#98df8a', '#c5b0d5', '#d62728']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, dcfg in zip(axes, directions):
        labels, means, stds = [], [], []
        for bar_label, prefix in dcfg['bars']:
            accs = best_val_accs(find_runs(results_dir, prefix, seeds))
            labels.append(bar_label)
            means.append(float(np.mean(accs)) if accs else 0.0)
            stds.append(float(np.std(accs)) if accs else 0.0)

        x = np.arange(len(labels))
        bars = ax.bar(x, means, yerr=stds, capsize=5,
                      color=palette, edgecolor='black', linewidth=0.7, width=0.6)

        # Teacher accuracy as dashed line (upper bound)
        t_accs = best_val_accs(find_runs(results_dir, dcfg['teacher_prefix'], seeds))
        if t_accs:
            t_mean = float(np.mean(t_accs))
            ax.axhline(t_mean, color='black', linestyle='--', linewidth=1.2,
                       label=f'Teacher  {t_mean:.3f}')
            ax.legend(fontsize=9, loc='lower right')

        # Value annotations above bars
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    m + s + 0.008,
                    f'{m:.3f}', ha='center', va='bottom', fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel('Val accuracy', fontsize=11)
        ax.set_title(dcfg['title'], fontsize=11)
        ax.set_ylim(0, min(1.0, max(means) + 0.12))
        ax.yaxis.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)

    fig.suptitle('Baseline comparison — CIFAR-10 val accuracy (mean ± std over 2 seeds)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved → {out_path}")

def resolve_ckpt_path(path, baselines_dir):
    """Resolve a checkpoint path that may be relative to the baselines dir."""
    if os.path.isabs(path) and os.path.exists(path):
        return path
    candidate = os.path.join(baselines_dir, path)
    if os.path.exists(candidate):
        return candidate
    return path  # return as-is; caller will handle missing file


def reconstruct_teacher_proj(seed, student_arch, teacher_arch, d, device):
    """Replay the exact random-init sequence from train.py to get teacher_proj.

    In train_one_run the order is:
      set_seed(seed)
      ... (data loader construction, no torch weight-init random consumed) ...
      student = make_model(student_arch, device)   <- consumes torch rng
      teacher = make_model(teacher_arch, device)   <- consumes torch rng
      teacher_proj, student_proj = build_projectors(...)  <- next torch rng state
    """
    set_seed(seed)
    make_model(student_arch, device)   # advance rng to match training
    make_model(teacher_arch, device)   # advance rng to match training
    teacher_proj, _ = build_projectors(teacher_arch, student_arch, d)
    return teacher_proj.to(device).eval()


def load_proposed_run(run_dir, baselines_dir, device):
    """Load student + projectors for the proposed method.
    Returns (student, teacher, teacher_proj, student_proj, KT, KS, tau, direction).
    """
    ckpt = torch.load(os.path.join(run_dir, 'best.pt'),
                      map_location=device, weights_only=True)
    a = ckpt['args']
    seed = int(a['seed'])
    d = int(a.get('d', 32))
    student_arch = a['student']
    teacher_arch = a['teacher']
    tau = float(ckpt.get('tau', a.get('tau', 1.0)))
    direction = a.get('direction', 'row')

    # Teacher projector — reconstruct from seed
    teacher_proj = reconstruct_teacher_proj(seed, student_arch, teacher_arch, d, device)

    # Student projector — load trained weights
    _, student_proj = build_projectors(teacher_arch, student_arch, d)
    student_proj = student_proj.to(device)
    student_proj.load_state_dict(ckpt['student_proj_state'])
    student_proj.eval()

    # Student model
    student = make_model(student_arch, device)
    student.load_state_dict(ckpt['model_state'])
    student.eval()

    # Teacher model
    teacher = make_model(teacher_arch, device)
    t_path = a.get('teacher_ckpt', '')
    t_path_resolved = resolve_ckpt_path(t_path, baselines_dir)
    if os.path.exists(t_path_resolved):
        t_ckpt = torch.load(t_path_resolved, map_location=device, weights_only=True)
        teacher.load_state_dict(t_ckpt['model_state'])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    KT = 3 if teacher_arch == 'cnn' else 4
    KS = 3 if student_arch == 'cnn' else 4
    return student, teacher, teacher_proj, student_proj, KT, KS, tau, direction


@torch.no_grad()
def compute_gamma(student, teacher, t_proj, s_proj, x, KT, KS, tau, direction, device):
    """Return mean Γ (KT × KS) over batch."""
    x = x.to(device)
    _, t_zs = teacher(x)
    _, s_zs = student(x)
    F_T = [t_proj[k](t_zs[k]) for k in range(KT)]
    F_S = [s_proj[j](s_zs[j]) for j in range(KS)]

    B = x.size(0)
    C = torch.zeros(B, KT, KS, device=device)
    for k in range(KT):
        for j in range(KS):
            diff = F_T[k] - F_S[j]
            C[:, k, j] = (diff ** 2).sum(dim=(1, 2))

    if direction == 'row':
        Gamma = F.softmax(-C / tau, dim=2)
    else:
        Gamma = F.softmax(-C / tau, dim=1)

    return Gamma.mean(dim=0).cpu().numpy()   # (KT, KS)


def b2_fixed_matrix(KT, KS):
    """Binary (KT × KS) matrix encoding the hard B2 correspondence."""
    M = np.zeros((KT, KS))
    for k in range(KT):
        j_star = int(k * KS / KT)
        M[k, j_star] = 1.0
    return M


def plot_heatmap_cell(ax, gamma, title, row_labels, col_labels, cmap, vmax):
    im = ax.imshow(gamma, aspect='auto', cmap=cmap, vmin=0, vmax=vmax)
    KT, KS = gamma.shape
    threshold = vmax * 0.55
    for ki in range(KT):
        for ji in range(KS):
            color = 'white' if gamma[ki, ji] > threshold else 'black'
            ax.text(ji, ki, f'{gamma[ki, ji]:.2f}',
                    ha='center', va='center', fontsize=8, color=color)
    ax.set_xticks(range(KS))
    ax.set_yticks(range(KT))
    ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xlabel('Student layer', fontsize=8)
    ax.set_title(title, fontsize=10, pad=6)
    return im


def build_heatmap_figure(results_dir, baselines_dir, out_path, val_batch, device):
    x_batch = val_batch.to(device)   # (16, 3, 32, 32)

    directions = [
        {
            'row_label': 'RNN → CNN',
            'teacher': 'rnn', 'student': 'cnn',
            'KT': 4, 'KS': 3,
            'proposed_prefix': 'proposed_rnn_cnn_tau1.0_dirrow',
        },
        {
            'row_label': 'CNN → RNN',
            'teacher': 'cnn', 'student': 'rnn',
            'KT': 3, 'KS': 4,
            'proposed_prefix': 'proposed_cnn_rnn_tau1.0_dirrow',
        },
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    col_titles = ['B2  (hard fixed)', 'B3  (uniform)', 'Proposed  (learned)']

    for row_idx, dcfg in enumerate(directions):
        KT, KS = dcfg['KT'], dcfg['KS']
        t_labels = [f'T{k}' for k in range(KT)]
        s_labels = [f'S{j}' for j in range(KS)]

        # --- Column 0: B2 fixed correspondence ---
        gamma_b2 = b2_fixed_matrix(KT, KS)
        im = plot_heatmap_cell(axes[row_idx][0], gamma_b2,
                               col_titles[0] if row_idx == 0 else '',
                               t_labels, s_labels, 'Greys', vmax=1.0)
        plt.colorbar(im, ax=axes[row_idx][0], shrink=0.85)

        # --- Column 1: B3 uniform ---
        gamma_b3 = np.ones((KT, KS)) / (KT * KS)
        im = plot_heatmap_cell(axes[row_idx][1], gamma_b3,
                               col_titles[1] if row_idx == 0 else '',
                               t_labels, s_labels, 'Blues', vmax=gamma_b3.max() * 1.5)
        plt.colorbar(im, ax=axes[row_idx][1], shrink=0.85)

        # --- Column 2: Proposed learned Γ ---
        run_dir = os.path.join(results_dir, f"{dcfg['proposed_prefix']}_seed0")
        if os.path.isdir(run_dir):
            (student, teacher, t_proj, s_proj,
             KT_, KS_, tau, direction) = load_proposed_run(run_dir, baselines_dir, device)
            gamma_prop = compute_gamma(student, teacher, t_proj, s_proj,
                                       x_batch, KT_, KS_, tau, direction, device)
        else:
            gamma_prop = np.zeros((KT, KS))
        im = plot_heatmap_cell(axes[row_idx][2], gamma_prop,
                               col_titles[2] if row_idx == 0 else '',
                               t_labels, s_labels, 'Blues', vmax=max(gamma_prop.max(), 0.01))
        plt.colorbar(im, ax=axes[row_idx][2], shrink=0.85)

        # Row label on the left
        axes[row_idx][0].set_ylabel(f'{dcfg["row_label"]}\n\nTeacher layer', fontsize=9)

    fig.suptitle('Alignment matrix Γ  (mean over 16 CIFAR-10 val images, seed=0)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved → {out_path}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--results_dir', default='results')
    p.add_argument('--out_dir', default='figures')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    baselines_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(baselines_dir, args.results_dir) \
        if not os.path.isabs(args.results_dir) else args.results_dir
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Building comparison bar chart...")
    build_bar_chart(
        results_dir,
        os.path.join(args.out_dir, 'comparison_bar_chart.png'),
    )

    print("Loading val batch for heatmap...")
    _, val_loader, _ = get_cifar10()
    x_batch = next(iter(val_loader))[0][:16]   # fixed first 16 images

    print("Building Γ heatmap...")
    build_heatmap_figure(
        results_dir, baselines_dir,
        os.path.join(args.out_dir, 'gamma_heatmap.png'),
        x_batch, device,
    )

    print("Done.")
