import argparse
import json
import os

import torch

from data import get_synthetic, get_cifar10
from models import SmallCNN, SmallRNN
from projectors import build_projectors
from alignment import SoftAlignment


def make_model(arch, device):
    if arch == 'cnn':
        return SmallCNN().to(device)
    return SmallRNN().to(device)


def evaluate_checkpoint(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(args.ckpt, map_location=device)
    saved_args = ckpt.get('args', {})
    tau = ckpt.get('tau', saved_args.get('tau', 1.0))
    direction = saved_args.get('direction', 'row')
    d = saved_args.get('d', 32)
    baseline = saved_args.get('baseline', 'proposed')

    student = make_model(args.student, device)
    student.load_state_dict(ckpt['model_state'])
    student.eval()

    _, student_proj = build_projectors(args.teacher, args.student, d)
    student_proj = student_proj.to(device)
    if 'student_proj_state' in ckpt:
        student_proj.load_state_dict(ckpt['student_proj_state'])
    student_proj.eval()

    # Teacher for Gamma visualization
    teacher = make_model(args.teacher, device)
    if args.teacher_ckpt:
        t_ckpt = torch.load(args.teacher_ckpt, map_location=device)
        teacher.load_state_dict(t_ckpt['model_state'])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    teacher_proj, _ = build_projectors(args.teacher, args.student, d)
    teacher_proj = teacher_proj.to(device)
    teacher_proj.eval()

    alignment = SoftAlignment(tau=tau, direction=direction).to(device)

    if args.dataset == 'synthetic':
        _, _, test_loader = get_synthetic()
        val_loader = _  # reuse val
        _, val_loader, _ = get_synthetic()
    else:
        _, val_loader, test_loader = get_cifar10()

    # Test accuracy
    eval_loader = test_loader if test_loader is not None else val_loader
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in eval_loader:
            x, y = x.to(device), y.to(device)
            logits, _ = student(x)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    test_acc = correct / total

    # Gamma heatmap for a fixed batch of 16
    KT = 3 if args.teacher == 'cnn' else 4
    KS = 3 if args.student == 'cnn' else 4
    gamma_mean = None
    if baseline not in ('b0', 'b1'):
        sample_x = next(iter(eval_loader))[0][:16].to(device)
        with torch.no_grad():
            _, t_zs = teacher(sample_x)
            _, s_zs = student(sample_x)
            F_T = [teacher_proj[k](t_zs[k]) for k in range(KT)]
            F_S = [student_proj[j](s_zs[j]) for j in range(KS)]
            if baseline == 'b3':
                B = sample_x.size(0)
                _, C = alignment(F_T, F_S)
                Gamma = torch.ones(B, KT, KS, device=device) / (KT * KS)
            else:
                Gamma, _ = alignment(F_T, F_S)
            gamma_mean = Gamma.mean(dim=0).cpu().tolist()   # (KT, KS)

    result = {
        'test_acc': test_acc,
        'gamma_mean': gamma_mean,
        'baseline': baseline,
        'teacher': args.teacher,
        'student': args.student,
        'tau': tau,
        'direction': direction,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    run_name = os.path.basename(os.path.dirname(args.ckpt))
    out_path = os.path.join(args.out_dir, f"eval_{run_name}.json")
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Results saved to {out_path}")
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--teacher', choices=['rnn', 'cnn'], required=True)
    p.add_argument('--student', choices=['rnn', 'cnn'], required=True)
    p.add_argument('--teacher_ckpt', type=str, default=None)
    p.add_argument('--dataset', choices=['synthetic', 'cifar10'], default='cifar10')
    p.add_argument('--out_dir', type=str, default='results')
    return p.parse_args()


if __name__ == '__main__':
    evaluate_checkpoint(parse_args())
