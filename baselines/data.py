import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
import torchvision
import torchvision.transforms as transforms

BATCH_SIZE = 128


def get_synthetic(seed=0):
    rng = np.random.RandomState(seed)
    N_train, N_val = 5000, 1000
    N = N_train + N_val
    X = torch.tensor(rng.randn(N, 3, 32, 32), dtype=torch.float32)
    y = torch.tensor(rng.randint(0, 10, (N,)), dtype=torch.long)
    train_ds = TensorDataset(X[:N_train], y[:N_train])
    val_ds = TensorDataset(X[N_train:], y[N_train:])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, val_loader, None  # no test split for synthetic


def get_cifar10(data_root="./data"):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    full_train = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True,
                                               transform=train_transform)
    full_train_eval = torchvision.datasets.CIFAR10(root=data_root, train=True, download=False,
                                                    transform=eval_transform)
    test_ds = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True,
                                            transform=eval_transform)

    # 45k train / 5k val
    indices = list(range(len(full_train)))
    train_idx, val_idx = indices[:45000], indices[45000:]

    train_ds = Subset(full_train, train_idx)
    val_ds = Subset(full_train_eval, val_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
                               pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
                             pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
                              pin_memory=True)
    return train_loader, val_loader, test_loader
