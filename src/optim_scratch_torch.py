from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Dict
import torch


@dataclass
class OptimCfg:
    lr: float = 1e-3
    momentum: float = 0.9
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0


class SGD_Scratch:
    """
    SGD / SGD+Momentum implemented from scratch.
    No torch.optim used.
    """
    def __init__(self, params: Iterable[torch.nn.Parameter], cfg: OptimCfg):
        self.params = list(params)
        self.cfg = cfg
        self.v: Dict[int, torch.Tensor] = {}  # momentum buffer

    @torch.no_grad()
    def step(self):
        lr = self.cfg.lr
        mu = self.cfg.momentum
        wd = self.cfg.weight_decay

        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad

            if wd != 0.0:
                g = g + wd * p  # L2 weight decay

            if mu == 0.0:
                p.add_(-lr * g)
            else:
                pid = id(p)
                if pid not in self.v:
                    self.v[pid] = torch.zeros_like(p)
                self.v[pid].mul_(mu).add_(g)
                p.add_(-lr * self.v[pid])

    @torch.no_grad()
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()


class Adam_Scratch:
    """
    Adam implemented from scratch (optional for your experiments).
    """
    def __init__(self, params: Iterable[torch.nn.Parameter], cfg: OptimCfg):
        self.params = list(params)
        self.cfg = cfg
        self.m: Dict[int, torch.Tensor] = {}
        self.v: Dict[int, torch.Tensor] = {}
        self.t = 0

    @torch.no_grad()
    def step(self):
        self.t += 1
        lr = self.cfg.lr
        b1 = self.cfg.beta1
        b2 = self.cfg.beta2
        eps = self.cfg.eps
        wd = self.cfg.weight_decay

        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad

            if wd != 0.0:
                g = g + wd * p

            pid = id(p)
            if pid not in self.m:
                self.m[pid] = torch.zeros_like(p)
                self.v[pid] = torch.zeros_like(p)

            self.m[pid].mul_(b1).add_((1 - b1) * g)
            self.v[pid].mul_(b2).add_((1 - b2) * (g * g))

            m_hat = self.m[pid] / (1 - (b1 ** self.t))
            v_hat = self.v[pid] / (1 - (b2 ** self.t))

            p.add_(-lr * m_hat / (torch.sqrt(v_hat) + eps))

    @torch.no_grad()
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
