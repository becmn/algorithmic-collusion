"""
Funções comuns aos scripts de análise (revisão da Fase 1, 16/10/2026).

Regras únicas para o artigo:
  - média: IC 95% t (graus de liberdade n − 1);
  - fração: IC 95% de Wilson com z = quantil 0,975 da normal (1,959964...);
  - Δ de preços sorteados ao acaso na grade: calculado da tabela de lucros (não digitado).
"""

import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
sys.path.insert(0, str(ROOT / "qlearning"))

Z95 = float(stats.norm.ppf(0.975))


def t_ci(x):
    """Média e IC 95% t de uma amostra."""
    x = np.asarray(x, dtype=float)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))
    return float(x.mean()), float(x.mean() - h), float(x.mean() + h)


def t_ci_summary(mean, sd, n):
    """IC 95% t a partir de média, DP e n (para resumos sem os dados por sessão)."""
    h = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
    return float(mean), float(mean - h), float(mean + h)


def wilson(k, n, z=Z95):
    """Fração k/n e IC 95% de Wilson; com k = 0 ou k = n o limite é exato."""
    p = k / n
    den = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    lo = 0.0 if k == 0 else c - h
    hi = 1.0 if k == n else c + h
    return p, lo, hi


def delta_random_prices():
    """Δ esperado com as duas firmas sorteando preços uniformemente na grade de 15 preços."""
    from base_case import setup
    b = setup()
    pbar = 0.5 * (b["PI"][:, :, 0] + b["PI"][:, :, 1]).mean()
    return float((pbar - b["piN"]) / (b["piM"] - b["piN"]))
