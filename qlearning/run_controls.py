"""
Fase 1.5, parte 2: controles em que a colusão por punição não deveria aparecer.
Três condições, 1.000 sessões cada (sementes 1 a 1.000):
  1. sem_memoria_d095: k = 0, δ = 0,95, α = 0,15, β = 4e-6   (Protocolo, controle 1)
  2. memoria1_d0:      k = 1, δ = 0,    α = 0,15, β = 4e-6   (Protocolo, controle 2)
  3. tabela_A1:        k = 0, δ = 0,    α = 0,25, β = 1e-4   (validação: Calvano, Tabela A1)
Os controles 1 e 2 mudam só um fator em relação ao caso base. A condição 3 é a do apêndice
de Calvano, que tem alvos numéricos (Δ 0,185; fração em Nash 0,273; Q-loss 0,013).
Antes, confere que a sessão genérica com k = 1 e os parâmetros do caso base devolve o mesmo Δ
da Fase 1.3 e a mesma fração em Nash da parte 1 (sementes 1 a 100).
Salva em results/: qlearning_controls_sessions.csv e qlearning_controls_meta.json.
Uso: python qlearning/run_controls.py          (1.000 sessões por condição)
     python qlearning/run_controls.py 20       (teste rápido, não salva)
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
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from base_case import PARAMS, setup
from controls import run_many_mem
from session import initial_q

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
SAVE = N == 1000
OUT = HERE.parent / "results"
N_CHECK = 100
P = PARAMS
b = setup()
PI, prices, pN = b["PI"], b["prices"], b["pN"]

CONDITIONS = {
    "sem_memoria_d095": {"memory": 0, "delta": 0.95, "alpha": P["alpha"], "beta": P["beta"],
                         "fonte": "Protocolo, controle 1 (Pesquisa 5, P3)"},
    "memoria1_d0": {"memory": 1, "delta": 0.0, "alpha": P["alpha"], "beta": P["beta"],
                    "fonte": "Protocolo, controle 2 (Pesquisa 5, P3)"},
    "tabela_A1": {"memory": 0, "delta": 0.0, "alpha": 0.25, "beta": 1e-4,
                  "fonte": "Calvano, Apêndice, Tabela A1 (Pesquisa 1)"},
}
TARGET_A1 = {"delta": 0.185, "nash_share": 0.273, "qloss": 0.013}
# caso base lido de results/ (revisão de 16/10/2026: antes eram números digitados)
with open(OUT / "qlearning_base_meta.json", encoding="utf-8") as _f:
    _bd = json.load(_f)["summary"]["delta_mean"]
with open(OUT / "qlearning_bestresponse_meta.json", encoding="utf-8") as _f:
    _bn = json.load(_f)["summary"]["by_version"]["calvano"]["nash_share"]
BASE = {"delta": round(_bd, 3), "nash_share_projeto": round(_bn, 3)}


def q0_for(delta, memory):
    Q0 = initial_q(PI, delta)
    return Q0 if memory == 1 else Q0[:, :1, :].copy()


def run(cond, seeds):
    c = CONDITIONS[cond] if isinstance(cond, str) else cond
    return run_many_mem(seeds, PI, q0_for(c["delta"], c["memory"]), c["alpha"], c["delta"],
                        c["beta"], P["t_stable"], P["t_max"], b["piN"], b["piM"], c["memory"])


def wilson(x, n, z=1.96):
    p = x / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


print(f"Núcleos usados pelo numba: {numba.get_num_threads()}")
t0 = time.perf_counter()
run("tabela_A1", np.array([1], dtype=np.int64))
run("memoria1_d0", np.array([1], dtype=np.int64))
print(f"Compilação: {time.perf_counter() - t0:.1f} s")

# ---------- conferência: k = 1 com o caso base reproduz as Fases 1.3 e 1.5 (parte 1)
base_cond = {"memory": 1, "delta": P["delta"], "alpha": P["alpha"], "beta": P["beta"]}
chk_seeds = np.arange(1, N_CHECK + 1, dtype=np.int64)
t0 = time.perf_counter()
r = run(base_cond, chk_seeds)
print(f"Conferência com o caso base ({N_CHECK} sessões, {time.perf_counter() - t0:.1f} s):")
same_delta = same_nash = None
f13 = OUT / "qlearning_base_sessions.csv"
f15 = OUT / "qlearning_bestresponse_sessions.csv"
if f13.exists():
    with open(f13, encoding="utf-8") as f:
        base = {int(x["seed"]): float(x["delta"]) for x in csv.DictReader(f)}
    same_delta = all(abs(base[int(s)] - d) < 1e-12 for s, d in zip(chk_seeds, r[2]))
    print(f"  Δ igual ao da Fase 1.3: {same_delta}")
if f15.exists():
    with open(f15, encoding="utf-8") as f:
        brn = {int(x["seed"]): x["nash_on_path_calvano"] == "True" for x in csv.DictReader(f)}
    same_nash = all(brn[int(s)] == bool(nh) for s, nh in zip(chk_seeds, r[7]))
    print(f"  Nash no caminho igual ao da parte 1: {same_nash}")

# ---------- as três condições
seeds = np.arange(1, N + 1, dtype=np.int64)
rows, summary = [], {}
for cond, c in CONDITIONS.items():
    t0 = time.perf_counter()
    conv, per, dlt, clen, pidx, p1, p2, nash, qlp, qla = run(cond, seeds)
    secs = time.perf_counter() - t0
    ok = conv
    n_ok = int(ok.sum())
    d = dlt  # sessões sem convergência entram com a média dos últimos 100.000 períodos (Ficha, amb. 4)
    se = d.std(ddof=1) / np.sqrt(N)
    tcrit = stats.t.ppf(0.975, N - 1)
    price = np.where(pidx >= 0, prices[0] + pidx * (prices[1] - prices[0]), np.nan)
    n_nash = int(nash[ok].sum())
    summary[cond] = {
        "params": {k: c[k] for k in ("memory", "delta", "alpha", "beta")}, "fonte": c["fonte"],
        "n_sessions": N, "n_converged": n_ok, "seconds": secs,
        "periods_mean": float(per.mean()), "periods_min": int(per.min()), "periods_max": int(per.max()),
        "delta_mean": float(d.mean()), "delta_sd": float(d.std(ddof=1)),
        "delta_ci95_t": [float(d.mean() - tcrit * se), float(d.mean() + tcrit * se)],
        "delta_median": float(np.median(d)),
        "delta_iqr": [float(np.percentile(d, 25)), float(np.percentile(d, 75))],
        "share_delta_gt_05": float((d > 0.5).mean()), "share_delta_lt_0": float((d < 0).mean()),
        "share_cycle": float((clen[ok] > 1).mean()) if n_ok else None,
        "price_mean": float(np.nanmean(price)),
        "price_pct_over_nash": float(100 * (np.nanmean(price) / pN - 1)),
        "nash_share": n_nash / n_ok if n_ok else None,
        "nash_share_wilson95": wilson(n_nash, n_ok) if n_ok else None,
        "qloss_path_mean": float(np.nanmean(qlp)), "qloss_all_mean": float(np.nanmean(qla)),
    }
    for k in range(N):
        rows.append({"condition": cond, "seed": int(seeds[k]), "converged": bool(conv[k]),
                     "periods": int(per[k]), "delta": float(dlt[k]), "cycle_len": int(clen[k]),
                     "price_idx_mean": float(pidx[k]),
                     "price_idx_firm1": int(p1[k]), "price_idx_firm2": int(p2[k]),
                     "nash_on_path_calvano": bool(nash[k]),
                     "qloss_path": float(qlp[k]), "qloss_all": float(qla[k])})

# ---------- saída
print(f"\nResumo ({N} sessões por condição; caso base: Δ {BASE['delta']}, "
      f"Nash {BASE['nash_share_projeto']})")
for cond, s in summary.items():
    pr = s["params"]
    lo, hi = s["delta_ci95_t"]
    print(f"\n[{cond}] k = {pr['memory']}, δ = {pr['delta']}, α = {pr['alpha']}, β = {pr['beta']}"
          f"   ({s['seconds']:.1f} s)")
    print(f"  convergiram: {s['n_converged']} de {N}; períodos: média {s['periods_mean']:,.0f} "
          f"(mín. {s['periods_min']:,}; máx. {s['periods_max']:,})")
    print(f"  Δ: média {s['delta_mean']:.3f} [IC 95% {lo:.3f}; {hi:.3f}], DP {s['delta_sd']:.3f}, "
          f"mediana {s['delta_median']:.3f}, IQR {s['delta_iqr'][0]:.3f}–{s['delta_iqr'][1]:.3f}")
    print(f"  fração Δ > 0,5: {s['share_delta_gt_05']:.3f}; Δ < 0: {s['share_delta_lt_0']:.3f}; "
          f"em ciclo: {s['share_cycle']:.3f}")
    print(f"  preço médio: {s['price_mean']:.4f} ({s['price_pct_over_nash']:+.1f}% sobre Nash 1,4729)")
    if s["nash_share"] is not None:
        nlo, nhi = s["nash_share_wilson95"]
        print(f"  fração em Nash no caminho: {s['nash_share']:.3f} [IC 95% {nlo:.3f}; {nhi:.3f}]; "
              f"Q-loss no caminho {s['qloss_path_mean']:.4f}, em todos os estados "
              f"{s['qloss_all_mean']:.4f}")
    if cond == "tabela_A1":
        print(f"  alvos da Tabela A1: Δ {TARGET_A1['delta']}; Nash {TARGET_A1['nash_share']}; "
              f"Q-loss {TARGET_A1['qloss']}")

if SAVE:
    OUT.mkdir(exist_ok=True)
    with open(OUT / "qlearning_controls_sessions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    meta = {
        "description": "Controles do Q-learning (sem memória; δ = 0; Tabela A1), sementes 1 a 1000",
        "date": datetime.now().isoformat(timespec="seconds"),
        "fixed": {"n": 2, "a": P["a"], "a0": P["a0"], "mu": P["mu"], "c": P["c"], "m": P["m"],
                  "xi": P["xi"], "t_stable": P["t_stable"], "t_max": P["t_max"]},
        "method": {"q0": "eq. 8 com o δ de cada condição; k = 0 usa 1 estado",
                   "delta_index": "ciclo-limite; sem convergência, média dos últimos 100.000 períodos",
                   "best_response": "leitura de Calvano (desvio de um período), como na parte 1",
                   "ci": "IC 95% t para Δ; Wilson para frações"},
        "check_vs_base": {"n": N_CHECK, "same_delta_as_phase_1_3": same_delta,
                          "same_nash_as_part_1": same_nash},
        "targets_table_A1": TARGET_A1, "base_case": BASE,
        "summary": summary,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "numba": numba.__version__, "scipy": scipy.__version__,
                     "platform": platform.platform()},
        "numba_threads": numba.get_num_threads(),
    }
    with open(OUT / "qlearning_controls_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\nSalvo em {OUT}")
