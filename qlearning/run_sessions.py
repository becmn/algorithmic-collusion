"""
Fase 1.3: 1.000 sessões do caso base (sementes 1 a 1.000), em paralelo com prange.
Roda duas vezes, confere que os resultados são idênticos e salva em results/:
  - qlearning_base_sessions.csv: uma linha por sessão
  - qlearning_base_meta.json: parâmetros, versões, tempos e resumo
Uso: python qlearning/run_sessions.py            (1.000 sessões)
     python qlearning/run_sessions.py 20         (teste rápido, não salva)
"""

import csv
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numba
import numpy as np
import scipy

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from base_case import PARAMS, setup
from batch import run_many

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
SAVE = N == 1000
OUT = HERE.parent / "results"

b = setup()
P = PARAMS
args = (b["PI"], b["Q0"], P["alpha"], P["delta"], P["beta"],
        P["t_stable"], P["t_max"], b["piN"], b["piM"])
seeds = np.arange(1, N + 1, dtype=np.int64)

print(f"Núcleos usados pelo numba: {numba.get_num_threads()}")

# 1ª chamada: compila (não entra na medição)
t0 = time.perf_counter()
run_many(np.array([0], dtype=np.int64), *args)
print(f"Compilação (1 sessão): {time.perf_counter() - t0:.1f} s")

times, runs = [], []
for r in (1, 2):
    t0 = time.perf_counter()
    runs.append(run_many(seeds, *args))
    times.append(time.perf_counter() - t0)
    print(f"Rodada {r}: {N} sessões em {times[-1]:.1f} s")

identical = all(np.array_equal(x, y) for x, y in zip(runs[0], runs[1]))
print(f"Rodadas 1 e 2 idênticas: {identical}")

converged, periods, deltas, cyc_len, cycles = runs[0]
prices = b["prices"]
m = P["m"]

rows = []
for k in range(N):
    cyc = cycles[k][cycles[k] >= 0]
    a1, a2 = cyc // m, cyc % m
    if not converged[k]:
        kind = "nao_convergiu"
    elif cyc_len[k] == 1:
        kind = "preco_constante"
    else:
        kind = "ciclo"
    p1 = prices[a1].mean() if len(cyc) else float("nan")
    p2 = prices[a2].mean() if len(cyc) else float("nan")
    rows.append({
        "seed": int(seeds[k]), "converged": bool(converged[k]),
        "periods": int(periods[k]), "delta": float(deltas[k]),
        "outcome": kind, "cycle_len": int(cyc_len[k]),
        "p1_mean": p1, "p2_mean": p2,
        "cycle_price_idx": ";".join(f"{i + 1}-{j + 1}" for i, j in zip(a1, a2)),
    })

conv_frac = converged.mean()
cyc_frac = np.mean([r["outcome"] == "ciclo" for r in rows])
summary = {
    "n_sessions": N,
    "converged": int(converged.sum()),
    "delta_mean": float(deltas.mean()),
    "delta_sd": float(deltas.std(ddof=1)),
    "delta_mc_error_95": float(1.96 * deltas.std(ddof=1) / np.sqrt(N)),
    "periods_mean": float(periods.mean()),
    "periods_min": int(periods.min()), "periods_max": int(periods.max()),
    "share_cycle": float(cyc_frac),
    "seconds_run1": times[0], "seconds_run2": times[1],
    "identical_runs": bool(identical),
}

print("\nResumo")
print(f"  convergiram: {summary['converged']} de {N}")
print(f"  Δ: média {summary['delta_mean']:.3f} (DP {summary['delta_sd']:.3f}; "
      f"erro de Monte Carlo ±{summary['delta_mc_error_95']:.3f})")
print(f"  períodos até convergir: média {summary['periods_mean']:,.0f} "
      f"(mín {summary['periods_min']:,}; máx {summary['periods_max']:,})")
print(f"  fração que termina em ciclo (tamanho > 1): {cyc_frac:.3f}")
print("Referências: Calvano Δ 0,849 (DP 0,112); Pesquisa 9: Δ 0,850 (DP 0,114), "
      "1,87 milhão de períodos, 1.000 sessões em 48 s com 2 núcleos; ~36% em ciclo")

if SAVE and identical:
    OUT.mkdir(exist_ok=True)
    with open(OUT / "qlearning_base_sessions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    meta = {
        "description": "Q-learning, caso base de Calvano et al. (2020), sementes 1 a 1000",
        "date": datetime.now().isoformat(timespec="seconds"),
        "params": P,
        "reference_points": {k: float(b[k]) for k in ("pN", "pM", "piN", "piM")},
        "price_grid": [float(x) for x in prices],
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "numba": numba.__version__, "scipy": scipy.__version__,
                     "platform": platform.platform()},
        "numba_threads": numba.get_num_threads(),
        "summary": summary,
        "notes": "cycle_price_idx: pares (índice firma 1 - índice firma 2) na grade, 1 a 15. "
                 "Δ no ciclo-limite (Q congelada, ε = 0).",
    }
    with open(OUT / "qlearning_base_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\nSalvo em {OUT}")
elif SAVE:
    print("\nNADA SALVO: as duas rodadas deram resultados diferentes.")
