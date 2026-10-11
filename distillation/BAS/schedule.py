"""Algo 4 counters with an explicit, documented acquisition event.
Source: https://arxiv.org/pdf/2208.07497v3
Ref.: "Patience Routine", p. 5 upper right
"""
from dataclasses import dataclass
import math


@dataclass
class Patience:
    lr: float = .001
    minimum: float = .000001
    maximum: float = .01
    decrease: float = .5
    increase: float = 2.
    patience_lr: int = 5
    patience_acquire: int = 15
    best: float = math.inf
    rho1: int = 0
    rho2: int = 0

    def __post_init__(self):
        if not (0 < self.minimum <= self.lr <= self.maximum and 0 < self.decrease < 1
                and self.increase > 1 and min(self.patience_lr, self.patience_acquire) >= 0):
            raise ValueError('Invalid patience schedule')

    def step(self, value, *, can_acquire=True):
        if not math.isfinite(value):
            raise FloatingPointError('Nonfinite panel loss')
        if value >= self.best:
            self.rho1 += 1
            self.rho2 += 1
        else:
            self.best, self.rho1 = value, 0
            # Algorithm 4 deliberately does not reset rho2 on an improvement. 
        if self.rho1 > self.patience_lr:
            self.lr = min(self.maximum, max(self.minimum, self.lr * self.decrease))
            self.rho1 = 0
        acquire = self.rho2 > self.patience_acquire and can_acquire
        
        if acquire:
            self.lr = min(self.maximum, max(self.minimum, self.lr * self.increase))
            self.rho2 = 0
            
        return acquire
