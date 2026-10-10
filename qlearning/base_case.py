"""
Caso base de Calvano et al. (2020): parâmetros da Ficha de parâmetros (Pesquisa 1)
e as tabelas que a sessão usa (grade de preços, lucros e Q inicial).
"""

import numpy as np
from scipy.optimize import fsolve

from session import calvano_grid, demand, initial_q, profit_table

# Ficha de parâmetros
PARAMS = {
    "n": 2, "a": 2.0, "a0": 0.0, "mu": 0.25, "c": 1.0,
    "delta": 0.95, "alpha": 0.15, "beta": 4e-6,
    "m": 15, "xi": 0.1, "k": 1,
    "t_stable": 100_000, "t_max": 1_000_000_000,
}


def setup(p=PARAMS):
    """Nash e monopólio contínuos, grade, tabela de lucros e Q inicial."""
    a, a0, mu, c = p["a"], p["a0"], p["mu"], p["c"]

    def foc_nash(x):
        q = demand(x, a, a0, mu)
        return 1 - (x - c) * (1 - q) / mu

    def foc_mon(x):
        q = demand(x, a, a0, mu)
        return 1 - (x - c) * (1 - q) / mu + (x[::-1] - c) * q[::-1] / mu

    pN = fsolve(foc_nash, np.array([1.5, 1.5]))[0]
    pM = fsolve(foc_mon, np.array([1.9, 1.9]))[0]
    piN = (pN - c) * demand(np.array([pN, pN]), a, a0, mu)[0]
    piM = (pM - c) * demand(np.array([pM, pM]), a, a0, mu)[0]
    prices = calvano_grid(pN, pM, p["m"], p["xi"])
    PI = profit_table(prices, a, a0, mu, c)
    Q0 = initial_q(PI, p["delta"])
    return {"pN": pN, "pM": pM, "piN": piN, "piM": piM,
            "prices": prices, "PI": PI, "Q0": Q0}
