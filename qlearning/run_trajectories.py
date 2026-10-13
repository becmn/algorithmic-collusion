"""
Fase 1.6, parte 2: refaz as 1.000 sessões do caso base (mesmas sementes da Fase 1.3)
registrando a trajetória em blocos de escala logarítmica e os caminhos de preço do
teste de desvio. Confere que períodos, Δ e a curva média R(τ) são iguais aos já salvos
e grava em results/:
  - qlearning_trajectory_blocks.csv    Δ e preço médio por bloco, entre as sessões
  - qlearning_trajectory_examples.csv  preços por bloco de 3 sessões de exemplo
  - qlearning_deviation_paths.csv      preços médios por τ nos dois ramos do desvio e R(τ),
                                       com IC t de 95%
  - qlearning_trajectories_meta.json   regras, conferências e resumo
Sessões de exemplo (regra fixada antes de olhar as trajetórias): as de Δ mais próximo dos
percentis 25, 50 e 75 da distribuição da Fase 1.3 (empate: menor semente).
Uso: python qlearning/run_trajectories.py          (1.000 sessões)
     python qlearning/run_trajectories.py 20       (teste rápido, não salva)
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
from deviation import T_DEV
from trajectories import log_block_edges, run_many_traced

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
SAVE = N == 1000
OUT = HERE.parent / "results"
T_END_BLOCKS = 10_000_000     # último limite dos blocos (acima do máximo de 3,18 milhões)
N_BLOCKS = 241                # pontos do espaçamento logarítmico antes de tirar repetidos

b = setup()
P = PARAMS
prices = b["prices"]
piN, piM = b["piN"], b["piM"]
edges = log_block_edges(T_END_BLOCKS, N_BLOCKS)
nb = len(edges) - 1
width = np.diff(edges)
args = (b["PI"], b["Q0"], P["alpha"], P["delta"], P["beta"], P["t_stable"], P["t_max"],
        piN, piM, prices, edges)
seeds = np.arange(1, N + 1, dtype=np.int64)

print(f"Núcleos usados pelo numba: {numba.get_num_threads()}")
print(f"Blocos: {nb} (de 1 período no início a {width[-1]:,} no fim)")
t0 = time.perf_counter()
run_many_traced(np.array([0], dtype=np.int64), *args)
print(f"Compilação (1 sessão): {time.perf_counter() - t0:.1f} s")

t0 = time.perf_counter()
conv, per, dlt, clen, SP1, SP2, SPI, CNT, paths, rcurve = run_many_traced(seeds, *args)
secs = time.perf_counter() - t0
print(f"{N} sessões em {secs:.1f} s")

# ---------------------------------------------------------------- conferências
checks = {}
checks["todas_convergiram"] = bool(conv.all())
checks["contagem_blocos_igual_periodos"] = bool((CNT.sum(axis=1) == per).all())

base_csv = OUT / "qlearning_base_sessions.csv"
with open(base_csv, encoding="utf-8") as f:
    base = {int(r["seed"]): r for r in csv.DictReader(f)}
checks["delta_igual_fase13"] = all(abs(float(base[int(s)]["delta"]) - d) < 1e-12
                                   for s, d in zip(seeds, dlt))
checks["periodos_iguais_fase13"] = all(int(base[int(s)]["periods"]) == int(t)
                                       for s, t in zip(seeds, per))

mean_rcurve = rcurve[conv].mean(axis=0)
if SAVE:
    dev_meta = json.load(open(OUT / "qlearning_deviation_meta.json", encoding="utf-8"))
    saved = np.array(dev_meta["mean_curve_rival_dev_over_ctrl_minus_1"])
    checks["curva_R_igual_fase14"] = bool(np.max(np.abs(saved - mean_rcurve)) < 1e-12)

# ---------------------------------------------------------------- trajetória média
full = CNT == width[None, :]                 # sessão ainda rodando no bloco inteiro
with np.errstate(invalid="ignore", divide="ignore"):
    d_block = (SPI / CNT - piN) / (piM - piN)
    p_block = 0.5 * (SP1 + SP2) / CNT
block_rows = []
for j in range(nb):
    m = full[:, j]
    k = int(m.sum())
    if k == 0:
        continue
    d = d_block[m, j]
    q = np.percentile(d, [10, 25, 50, 75, 90])
    block_rows.append({
        "block": j, "t_start": int(edges[j]), "t_end": int(edges[j + 1]),
        "t_mid_geom": float(np.sqrt(max(edges[j], 1) * edges[j + 1])),
        "n_sessions": k,
        "delta_mean": float(d.mean()), "delta_p10": float(q[0]), "delta_p25": float(q[1]),
        "delta_p50": float(q[2]), "delta_p75": float(q[3]), "delta_p90": float(q[4]),
        "price_mean": float(p_block[m, j].mean()),
        "epsilon_start": float(np.exp(-P["beta"] * edges[j])),
    })
t_all = int(edges[max(j for j in range(nb) if full[:, j].all()) + 1])
checks["blocos_com_todas_as_sessoes"] = t_all >= 1_000_000

# ---------------------------------------------------------------- sessões de exemplo
d_base = np.array([float(base[s]["delta"]) for s in sorted(base)])
s_base = np.array(sorted(base))
examples = {}
for name, q in (("p25", 25), ("p50", 50), ("p75", 75)):
    target = np.percentile(d_base, q)
    dist = np.abs(d_base - target)
    examples[name] = {"seed": int(s_base[np.flatnonzero(dist == dist.min())[0]]),
                      "delta_target": float(target)}
ex_rows = []
for name, e in examples.items():
    k = e["seed"] - 1
    if k >= N:
        continue
    e["delta"] = float(dlt[k])
    e["periods"] = int(per[k])
    e["cycle_len"] = int(clen[k])
    for j in range(nb):
        if CNT[k, j] == 0:
            continue
        ex_rows.append({
            "role": name, "seed": e["seed"], "block": j,
            "t_start": int(edges[j]), "t_end": int(min(edges[j + 1], per[k])),
            "n_periods": int(CNT[k, j]),
            "price_firm1": float(SP1[k, j] / CNT[k, j]),
            "price_firm2": float(SP2[k, j] / CNT[k, j]),
            "delta_block": float(d_block[k, j]),
        })

# ---------------------------------------------------------------- caminhos do desvio
ok = conv
series = {
    "ctrl_p1": paths[ok, 0, 0, :], "ctrl_p2": paths[ok, 0, 1, :],
    "dev_p1": paths[ok, 1, 0, :], "dev_p2": paths[ok, 1, 1, :],
    "R_tau": rcurve[ok, :],
}
n_ok = int(ok.sum())
tcrit = stats.t.ppf(0.975, n_ok - 1)
dev_rows = []
for tau in range(T_DEV + 1):
    row = {"tau": tau}
    for name, x in series.items():
        mu = float(x[:, tau].mean())
        sd = float(x[:, tau].std(ddof=1))
        h = tcrit * sd / np.sqrt(n_ok)
        row[f"{name}_mean"] = mu
        row[f"{name}_sd"] = sd
        row[f"{name}_ci_lo"] = mu - h
        row[f"{name}_ci_hi"] = mu + h
    row["n"] = n_ok
    dev_rows.append(row)

# ---------------------------------------------------------------- saída na tela
print("\nConferências:")
for k, v in checks.items():
    print(f"  {k}: {v}")
print(f"\nBlocos com as {N} sessões: até o período {t_all:,}")
for t_show in (300, 10_000, 100_000, 500_000, 1_000_000):
    r = min((r for r in block_rows if r["n_sessions"] == N),
            key=lambda r: abs(np.log(r["t_mid_geom"] / t_show)))
    print(f"  bloco [{r['t_start']:,}; {r['t_end']:,}): Δ médio {r['delta_mean']:.3f} "
          f"(P10 {r['delta_p10']:.3f}; P90 {r['delta_p90']:.3f}), ε no início {r['epsilon_start']:.3f}")
print("\nSessões de exemplo (Δ mais próximo dos percentis 25, 50 e 75):")
for name, e in examples.items():
    if "delta" in e:
        print(f"  {name}: semente {e['seed']}, Δ {e['delta']:.3f} (alvo {e['delta_target']:.3f}), "
              f"{e['periods']:,} períodos, ciclo de tamanho {e['cycle_len']}")
print(f"\nDesvio, médias entre {n_ok} sessões (τ = 0, 1, 2, 5, 10, 15):")
for tau in (0, 1, 2, 5, 10, 15):
    r = dev_rows[tau]
    print(f"  τ={tau:>2}: desvio p1 {r['dev_p1_mean']:.3f}, p2 {r['dev_p2_mean']:.3f}; "
          f"controle p1 {r['ctrl_p1_mean']:.3f}, p2 {r['ctrl_p2_mean']:.3f}; "
          f"R(τ) {r['R_tau_mean']:+.4f} [{r['R_tau_ci_lo']:+.4f}; {r['R_tau_ci_hi']:+.4f}]")

if SAVE:
    def write_csv(name, rows):
        with open(OUT / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    write_csv("qlearning_trajectory_blocks.csv", block_rows)
    write_csv("qlearning_trajectory_examples.csv", ex_rows)
    write_csv("qlearning_deviation_paths.csv", dev_rows)
    meta = {
        "description": "Fase 1.6, parte 2: trajetórias em blocos logarítmicos e caminhos de "
                       "preço do teste de desvio, Q-learning caso base, sementes 1 a 1000",
        "date": datetime.now().isoformat(timespec="seconds"),
        "blocks": {"n_blocks": int(nb), "t_end": T_END_BLOCKS,
                   "rule": "limites = 0 e floor(geomspace(1, t_end, 241)) sem repetidos; "
                           "média do bloco = média dos períodos do bloco",
                   "aggregation": "por bloco, só sessões que ainda rodam no bloco inteiro; "
                                  "a figura usa os blocos com as 1.000 sessões",
                   "last_period_with_all_sessions": t_all},
        "examples_rule": "sessões com Δ mais próximo dos percentis 25, 50 e 75 da Fase 1.3 "
                         "(empate: menor semente), fixada antes de ver as trajetórias",
        "examples": examples,
        "deviation_paths": "preço médio por τ nos ramos de controle e com desvio (firma 1 "
                           "desvia em τ = 1), média sobre os pontos do ciclo = 1 observação; "
                           "IC t de 95% entre sessões",
        "checks": checks,
        "seconds": secs,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "numba": numba.__version__, "scipy": scipy.__version__,
                     "platform": platform.platform()},
        "numba_threads": numba.get_num_threads(),
    }
    with open(OUT / "qlearning_trajectories_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\nSalvo em {OUT}")
