import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
    TextColumn, TimeElapsedColumn, TimeRemainingColumn,
)

from data import get_synthetic, get_cifar10
from models import SmallCNN, SmallRNN
from projectors import build_projectors
from alignment import SoftAlignment, distillation_loss, gamma_entropy

console = Console()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_model(arch, device):
    if arch == 'cnn':
        return SmallCNN().to(device)
    return SmallRNN().to(device)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total


def log_proj_norms(F_list, prefix):
    return {f"{prefix}_proj_norm_layer{i}": F_list[i].norm(dim=(1, 2)).mean().item()
            for i in range(len(F_list))}

class HardAlignmentProjectors(nn.Module):
    """For each teacher layer k, a Linear(d*d, d*d) aligner to its fixed student pair."""

    def __init__(self, KT, d):
        super().__init__()
        self.projectors = nn.ModuleList([nn.Linear(d * d, d * d) for _ in range(KT)])
        self.d = d

    def forward(self, F_T_list, F_S_list, KS):
        KT = len(F_T_list)
        loss = torch.tensor(0.0, device=F_T_list[0].device)
        for k in range(KT):
            j_star = int(k * KS / KT)
            ft_flat = F_T_list[k].view(F_T_list[k].size(0), -1)
            ft_proj = self.projectors[k](ft_flat)
            fs_flat = F_S_list[j_star].view(F_S_list[j_star].size(0), -1)
            loss = loss + F.mse_loss(ft_proj, fs_flat.detach())
        return loss

def train_one_run(args):
    tau = args.tau
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Output directory
    direction_tag = f"_dir{args.direction}" if args.baseline in ('proposed', 'b3', 'b5') else ""
    tau_tag = f"_tau{tau}" if args.baseline in ('proposed', 'b3', 'b5') else ""
    out_dir = os.path.join(
        args.out_dir,
        f"{args.baseline}_{args.teacher}_{args.student}{tau_tag}{direction_tag}_seed{args.seed}"
    )
    os.makedirs(out_dir, exist_ok=True)

    # Data
    if args.dataset == 'synthetic':
        train_loader, val_loader, _ = get_synthetic(seed=args.seed)
        epochs = args.epochs if args.epochs != 50 else 20
    else:
        train_loader, val_loader, _ = get_cifar10()
        epochs = args.epochs

    # Student model
    student = make_model(args.student, device)
    KS = 3 if args.student == 'cnn' else 4

    # Teacher model
    teacher = None
    if args.baseline != 'b0' or args.teacher_ckpt:
        teacher = make_model(args.teacher, device)
        if args.teacher_ckpt:
            ckpt = torch.load(args.teacher_ckpt, map_location=device, weights_only=True)
            teacher.load_state_dict(ckpt['model_state'])
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
    else:
        # b0 without teacher_ckpt: train the teacher architecture itself
        student = make_model(args.teacher, device)

    KT = 3 if args.teacher == 'cnn' else 4

    # Projectors
    d = args.d
    teacher_proj, student_proj = build_projectors(args.teacher, args.student, d)
    teacher_proj = teacher_proj.to(device)
    student_proj = student_proj.to(device)

    alignment = SoftAlignment(tau=tau, direction=args.direction).to(device)

    hard_proj = None
    if args.baseline == 'b2':
        hard_proj = HardAlignmentProjectors(KT, d).to(device)

    param_groups = list(student.parameters()) + list(student_proj.parameters())
    if args.baseline == 'b2' and hard_proj is not None:
        param_groups += list(hard_proj.parameters())

    optimizer = torch.optim.Adam(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    all_metrics = []

    run_label = (f"[bold cyan]{args.baseline}[/bold cyan] "
                 f"T=[green]{args.teacher}[/green] S=[yellow]{args.student}[/yellow] "
                 f"τ={tau} dir={args.direction} seed={args.seed}")

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(bar_width=30),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("[dim]{task.fields[stats]}[/dim]"),
        console=console,
        refresh_per_second=10,
    ) as progress:
        epoch_task = progress.add_task(run_label, total=epochs, stats="")

        for epoch in range(epochs):
            student.train()
            if teacher is not None:
                teacher.eval()

            total_loss_sum = 0.0
            task_loss_sum = 0.0
            kd_loss_sum = 0.0
            gamma_ent_sum = 0.0
            mean_cost_sum = 0.0
            n_batches = 0

            batch_task = progress.add_task(
                f"  [dim]batches[/dim]", total=len(train_loader), stats=""
            )

            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()

                student_logits, student_zs = student(x)
                loss_task = F.cross_entropy(student_logits, y)

                loss_kd = torch.tensor(0.0, device=device)
                ent = 0.0
                mean_c = 0.0

                if args.baseline == 'b0':
                    pass

                elif args.baseline == 'b1':
                    with torch.no_grad():
                        teacher_logits, _ = teacher(x)
                    T_kd = 4.0
                    loss_kd = F.kl_div(
                        F.log_softmax(student_logits / T_kd, dim=1),
                        F.softmax(teacher_logits / T_kd, dim=1),
                        reduction='batchmean'
                    ) * (T_kd ** 2)

                elif args.baseline in ('b2', 'b3', 'proposed'):
                    with torch.no_grad():
                        _, teacher_zs = teacher(x)
                        F_T = [teacher_proj[k](teacher_zs[k]) for k in range(KT)]
                    F_S = [student_proj[j](student_zs[j]) for j in range(KS)]

                    if args.baseline == 'b2':
                        loss_kd = hard_proj(F_T, F_S, KS)
                    else:
                        if args.baseline == 'b3':
                            B = x.size(0)
                            _, C = alignment(F_T, F_S)
                            Gamma = torch.ones(B, KT, KS, device=device) / (KT * KS)
                        else:
                            Gamma, C = alignment(F_T, F_S)
                        loss_kd = distillation_loss(Gamma, C)
                        ent = gamma_entropy(Gamma, args.direction)
                        mean_c = C.mean().item()

                loss = loss_task + args.lambda_kd * loss_kd
                loss.backward()
                optimizer.step()

                total_loss_sum += loss.item()
                task_loss_sum += loss_task.item()
                kd_loss_sum += loss_kd.item() if isinstance(loss_kd, torch.Tensor) else loss_kd
                gamma_ent_sum += ent
                mean_cost_sum += mean_c
                n_batches += 1

                progress.update(batch_task, advance=1,
                                stats=f"loss={loss.item():.4f} task={loss_task.item():.4f}")

            progress.remove_task(batch_task)
            scheduler.step()
            val_acc = evaluate(student, val_loader, device)

            # Rank-collapse guard
            proj_norms = {}
            if args.baseline not in ('b0', 'b1'):
                with torch.no_grad():
                    sample_x = next(iter(val_loader))[0][:8].to(device)
                    _, s_zs = student(sample_x)
                    F_S_sample = [student_proj[j](s_zs[j]) for j in range(KS)]
                    proj_norms = log_proj_norms(F_S_sample, 'student')
                    if any(v < 1e-3 for v in proj_norms.values()):
                        console.print(f"[bold yellow][WARN][/bold yellow] epoch {epoch}: "
                                      f"rank collapse in student projector")

            epoch_metrics = {
                'epoch': epoch,
                'train_loss_total': total_loss_sum / n_batches,
                'train_loss_task': task_loss_sum / n_batches,
                'train_loss_kd': kd_loss_sum / n_batches,
                'val_acc': val_acc,
                'gamma_entropy': gamma_ent_sum / n_batches,
                'mean_cost': mean_cost_sum / n_batches,
                **proj_norms,
            }
            all_metrics.append(epoch_metrics)

            progress.update(
                epoch_task, advance=1,
                stats=(f"loss={epoch_metrics['train_loss_total']:.4f}  "
                       f"val={val_acc:.4f}  best={max(best_val_acc, val_acc):.4f}")
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                ckpt = {
                    'epoch': epoch,
                    'model_state': student.state_dict(),
                    'student_proj_state': student_proj.state_dict(),
                    'val_acc': val_acc,
                    'args': vars(args),
                    'tau': tau,
                }
                if hard_proj is not None:
                    ckpt['hard_proj_state'] = hard_proj.state_dict()
                torch.save(ckpt, os.path.join(out_dir, 'best.pt'))

    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(all_metrics, f, indent=2)

    console.print(f"[bold green]Done.[/bold green] Best val acc: {best_val_acc:.4f} → {out_dir}")
    return best_val_acc, out_dir

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline', choices=['b0', 'b1', 'b2', 'b3', 'proposed'], required=True)
    p.add_argument('--teacher', choices=['rnn', 'cnn'], required=True)
    p.add_argument('--student', choices=['rnn', 'cnn'], required=True)
    p.add_argument('--tau', type=float, default=1.0)
    p.add_argument('--direction', choices=['row', 'col'], default='row')
    p.add_argument('--lambda_kd', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--dataset', choices=['synthetic', 'cifar10'], default='cifar10')
    p.add_argument('--teacher_ckpt', type=str, default=None)
    p.add_argument('--out_dir', type=str, default='results')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--d', type=int, default=32)
    return p.parse_args()


if __name__ == '__main__':
    train_one_run(parse_args())
