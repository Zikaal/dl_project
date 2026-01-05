from dataclasses import dataclass
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from src.optim_scratch_torch import OptimCfg, SGD_Scratch, Adam_Scratch
from src.section4_data_torch import DataCfg, make_loaders
from src.section4_models_torch import SmallResNet, PlainNet, pretrained_resnet18


@dataclass
class TrainCfg:
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cpu"
    optimizer: str = "sgd_momentum"  # "sgd" | "sgd_momentum" | "adam"


def _acc(logits, y):
    return (logits.argmax(1) == y).float().mean().item()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += _acc(logits, y) * bs
        n += bs
    return total_loss / n, total_acc / n


def grad_flow_stats(model):
    """
    early = первые ~20% trainable параметров
    late  = последние ~20% trainable параметров
    Работает для любых моделей (PlainNet/ResNet/Pretrained).
    """
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().norm().item())

    if len(grads) == 0:
        return {"early": 0.0, "late": 0.0}

    k = max(1, len(grads) // 5)  # 20%
    early = sum(grads[:k])
    late  = sum(grads[-k:])
    return {"early": early, "late": late}



def make_optimizer_from_scratch(model, cfg: TrainCfg):
    ocfg = OptimCfg(lr=cfg.lr, momentum=(0.9 if cfg.optimizer == "sgd_momentum" else 0.0),
                    weight_decay=cfg.weight_decay)

    if cfg.optimizer in ["sgd", "sgd_momentum"]:
        return SGD_Scratch(model.parameters(), ocfg)
    elif cfg.optimizer == "adam":
        return Adam_Scratch(model.parameters(), ocfg)
    else:
        raise ValueError("optimizer must be: sgd | sgd_momentum | adam")


def train(model, train_loader, val_loader, cfg: TrainCfg):
    model.to(cfg.device)
    loss_fn = nn.CrossEntropyLoss()
    opt = make_optimizer_from_scratch(model, cfg)

    hist = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "grad_early": [], "grad_late": []
    }

    for ep in range(1, cfg.epochs + 1):
        model.train()
        total_loss, total_acc, n = 0.0, 0.0, 0

        for x, y in train_loader:
            x, y = x.to(cfg.device), y.to(cfg.device)

            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()

            g = grad_flow_stats(model)
            hist["grad_early"].append(g["early"])
            hist["grad_late"].append(g["late"])

            opt.step()

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_acc += _acc(logits, y) * bs
            n += bs

        train_loss = total_loss / n
        train_acc = total_acc / n
        val_loss, val_acc = evaluate(model, val_loader, cfg.device)

        hist["train_loss"].append(train_loss)
        hist["train_acc"].append(train_acc)
        hist["val_loss"].append(val_loss)
        hist["val_acc"].append(val_acc)

        print(f"[{model.__class__.__name__}] Epoch {ep}/{cfg.epochs} "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

    return hist


def plot_hist(hist, title):
    plt.figure()
    plt.plot(hist["train_loss"], label="train")
    plt.plot(hist["val_loss"], label="val")
    plt.title(f"{title} Loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend()
    plt.show()

    plt.figure()
    plt.plot(hist["train_acc"], label="train")
    plt.plot(hist["val_acc"], label="val")
    plt.title(f"{title} Accuracy")
    plt.xlabel("epoch"); plt.ylabel("acc"); plt.legend()
    plt.show()

    plt.figure()
    plt.plot(hist["grad_early"], label="early grad norm")
    plt.plot(hist["grad_late"], label="late grad norm")
    plt.title(f"{title} Gradient Flow (per step)")
    plt.xlabel("step"); plt.ylabel("grad norm"); plt.legend()
    plt.show()


# -------- Experiments --------

def exp_resnet_vs_plain(data_cfg: DataCfg, train_cfg: TrainCfg, augment=True):
    train_loader, val_loader = make_loaders(data_cfg, augment=augment)

    plain = PlainNet(num_classes=2)
    h_plain = train(plain, train_loader, val_loader, train_cfg)
    plot_hist(h_plain, "PlainNet (no skip)")

    res = SmallResNet(num_classes=2)
    h_res = train(res, train_loader, val_loader, train_cfg)
    plot_hist(h_res, "SmallResNet (skip connections)")

    return h_plain, h_res


def exp_transfer_learning(data_cfg: DataCfg, train_cfg: TrainCfg, augment=True):
    train_loader, val_loader = make_loaders(data_cfg, augment=augment)

    # 1) from scratch
    scratch = SmallResNet(num_classes=2)
    h_scratch = train(scratch, train_loader, val_loader, train_cfg)
    plot_hist(h_scratch, "From scratch (SmallResNet)")

    # 2) pretrained frozen
    frozen = pretrained_resnet18(num_classes=2, freeze_backbone=True)
    h_frozen = train(frozen, train_loader, val_loader, train_cfg)
    plot_hist(h_frozen, "Pretrained ResNet18 (frozen backbone)")

    # 3) fine-tune last layer block + head (simple)
    ft = pretrained_resnet18(num_classes=2, freeze_backbone=True)
    # unfreeze layer4 + fc
    for name, p in ft.named_parameters():
        if name.startswith("layer4") or name.startswith("fc"):
            p.requires_grad = True
    h_ft = train(ft, train_loader, val_loader, train_cfg)
    plot_hist(h_ft, "Pretrained ResNet18 (fine-tune layer4+fc)")

    return h_scratch, h_frozen, h_ft


def exp_augmentation_effect(data_cfg: DataCfg, train_cfg: TrainCfg):
    # no aug
    tr0, va0 = make_loaders(data_cfg, augment=False)
    m0 = SmallResNet(num_classes=2)
    h0 = train(m0, tr0, va0, train_cfg)

    # with aug
    tr1, va1 = make_loaders(data_cfg, augment=True)
    m1 = SmallResNet(num_classes=2)
    h1 = train(m1, tr1, va1, train_cfg)

    plt.figure()
    plt.plot(h0["val_acc"], label="no aug")
    plt.plot(h1["val_acc"], label="with aug")
    plt.title("Augmentation impact on val accuracy")
    plt.xlabel("epoch"); plt.ylabel("val_acc"); plt.legend()
    plt.show()

    return h0, h1
