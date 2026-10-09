"""
Fase 1.2: mede uma sessão de Q-learning (caso base de Calvano et al., Ficha de parâmetros).
Roda 20 sessões (sementes 1 a 20) e mostra segundos por sessão (sem a 1ª chamada,
que inclui a compilação), períodos até convergir e Δ.
Referência (Pesquisa 9): ~0,11 s por sessão, ~1,87 milhão de períodos, Δ ≈ 0,85.
"""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import fsolve

sys.path.insert(0, str(Path(__file__).resolve().parent))
from session import calvano_grid, demand, initial_q, profit_table, run_session

# ---- Ficha de parâmetros (Calvano et al., 2020; Pesquisa 1)
A, A0, MU, C = 2.0, 0.0, 0.25, 1.0
DELTA, ALPHA, BETA = 0.95, 0.15, 4e-6
M, XI = 15, 0.1
T_STABLE, T_MAX = 100_000, 1_000_000_000

# ---- Nash e monopólio conjunto (contínuos)
def foc_nash(p):
    q = demand(p, A, A0, MU)
    return 1 - (p - C) * (1 - q) / MU

def foc_mon(p):
    q = demand(p, A, A0, MU)
    return 1 - (p - C) * (1 - q) / MU + (p[::-1] - C) * q[::-1] / MU

pN = fsolve(foc_nash, np.array([1.5, 1.5]))[0]
pM = fsolve(foc_mon, np.array([1.9, 1.9]))[0]
piN = (pN - C) * demand(np.array([pN, pN]), A, A0, MU)[0]
piM = (pM - C) * demand(np.array([pM, pM]), A, A0, MU)[0]

prices = calvano_grid(pN, pM, M, XI)
PI = profit_table(prices, A, A0, MU, C)
Q0 = initial_q(PI, DELTA)

print(f"pN = {pN:.6f}  piN = {piN:.6f}  pM = {pM:.6f}  piM = {piM:.6f}")
print(f"grade: {prices[0]:.4f} a {prices[-1]:.4f}, passo {prices[1] - prices[0]:.4f}")

# ---- conferências contra a Ficha
checks = [
    ("pN", pN, 1.472927, 1e-5), ("piN", piN, 0.222927, 1e-5),
    ("pM", pM, 1.924981, 1e-5), ("piM", piM, 0.337490, 1e-5),
    ("grade min", prices[0], 1.4277, 1e-4), ("grade max", prices[-1], 1.9702, 1e-4),
]
for name, got, want, tol in checks:
    print(f"  {'OK ' if abs(got - want) <= tol else 'ERRO'} {name}: {got:.6f} (Ficha {want})")

def one(seed):
    return run_session(seed, PI, Q0, ALPHA, DELTA, BETA, T_STABLE, T_MAX, piN, piM)

# ---- 1ª chamada: compila (não entra na medição)
t0 = time.perf_counter()
one(0)
print(f"\n1ª chamada (com compilação): {time.perf_counter() - t0:.2f} s")

# ---- reprodutibilidade
r1, r2 = one(1), one(1)
same = (r1[1] == r2[1]) and (r1[2] == r2[2]) and np.array_equal(r1[5], r2[5])
print(f"Semente 1 rodada 2 vezes, resultados idênticos: {same}")

# ---- 20 sessões
print("\nsemente  convergiu  períodos    Δ       ciclo  segundos")
secs, periods, deltas, conv = [], [], [], []
for seed in range(1, 21):
    t0 = time.perf_counter()
    ok, t, d, n, pidx, Q, g = one(seed)
    dt = time.perf_counter() - t0
    secs.append(dt); periods.append(t); deltas.append(d); conv.append(ok)
    print(f"{seed:>7}  {str(ok):>9}  {t:>9,}  {d:6.3f}  {n:>5}  {dt:8.3f}")

secs, periods, deltas = map(np.array, (secs, periods, deltas))
print("\nResumo (20 sessões)")
print(f"  segundos por sessão: média {secs.mean():.3f} (mín {secs.min():.3f}; máx {secs.max():.3f})")
print(f"  períodos até parar:  média {periods.mean():,.0f}")
print(f"  Δ: média {deltas.mean():.3f} (DP {deltas.std(ddof=1):.3f})")
print(f"  sessões que convergiram: {sum(conv)} de 20")
print("Referência (Pesquisa 9): ~0,11 s; ~1,87 milhão de períodos; Δ ≈ 0,85")
