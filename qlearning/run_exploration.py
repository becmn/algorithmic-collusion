"""
Fase 1.5, parte 3a: sensibilidade à exploração (Protocolo: caso base e pelo menos um β menor,
dentro da grade de Calvano, β de 0 a ~2e-5; Pesquisa 1). 1.000 sessões por β (sementes 1 a 1.000),
demais parâmetros do caso base. Em cada sessão: Δ, períodos até convergir, teste de melhor
resposta (leitura de Calvano) e teste de desvio comum (R, G, V e classes, θ = 2%).
β menor = exploração mais longa. Pergunta (Pesquisa 5): o preço alto some com mais aprendizado?
  β = 1e-5  (menos exploração; ponto médio da grade de Calvano)
  β = 4e-6  (caso base; confere com as Fases 1.3, 1.4 e 1.5, parte 1)
  β = 2e-6  e  β = 1e-6  (mais exploração)
Salva em results/: qlearning_exploration_sessions.csv (uma linha por β e sessão) e
qlearning_exploration_meta.json (resumo por β). Cada β é rodado num comando separado.
Uso: python qlearning/run_exploration.py BETA [N]   (ex.: 1e-6; N padrão 1000; com menos não salva)
     python qlearning/run_exploration.py resumo     (tabela com os β já salvos)
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
from deviation import classify
from sensitivity import run_many_full

ARG = sys.argv[1] if len(sys.argv) > 1 else "resumo"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
SAVE = N == 1000
OUT = HERE.parent / "results"
CSV = OUT / "qlearning_exploration_sessions.csv"
META = OUT / "qlearning_exploration_meta.json"
ALLOWED = (1e-5, 4e-6, 2e-6, 1e-6)
THETA = 0.02
P = PARAMS
b = setup()
prices = b["prices"]
step = prices[1] - prices[0]


def wilson(x, n, z=1.96):
    p = x / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def load_meta():
    if META.exists():
        with open(META, encoding="utf-8") as f:
            return json.load(f)
    return {"description": "Sensibilidade à exploração, Q-learning, sementes 1 a 1000",
            "fixed": {k: P[k] for k in ("n", "a", "a0", "mu", "c", "delta", "alpha", "m", "xi",
                                        "k", "t_stable", "t_max")},
            "method": {"delta_index": "ciclo-limite (Q congelada, ε = 0)",
                       "best_response": "leitura de Calvano (desvio de um período), como na parte 1",
                       "deviation": "teste de desvio comum da Fase 1.4; classes com θ = 2%"},
            "by_beta": {}}


def print_table(meta):
    rows = sorted(meta["by_beta"].items(), key=lambda kv: -float(kv[0]))
    print("\n β        Δ médio [IC 95%]        períodos   em Nash   Q-loss   R       G < 0   A      B      C")
    for beta, s in rows:
        lo, hi = s["delta_ci95"]
        c = s["classes"]
        print(f" {float(beta):<8.0e} {s['delta_mean']:.3f} [{lo:.3f}; {hi:.3f}]   "
              f"{s['periods_mean'] / 1e6:5.2f} mi  {s['nash_share']:.3f}    "
              f"{s['qloss_path_mean']:.4f}   {s['R_mean']:+.3f}  {s['share_G_lt_0']:.3f}   "
              f"{c['A']:<5d}  {c['B']:<5d}  {c['C']:<5d}")
    print(" (alvos do caso base, Calvano: Δ 0,849; Nash 0,505; Q-loss 0,002; R −0,127; G < 0 0,936)")


if ARG == "resumo":
    print_table(load_meta())
    sys.exit()

beta = float(ARG)
if not any(abs(beta - a) < 1e-15 for a in ALLOWED):
    sys.exit(f"β fora da lista do protocolo: {ALLOWED}")
seeds = np.arange(1, N + 1, dtype=np.int64)
args = (b["PI"], b["Q0"], P["alpha"], P["delta"], beta, P["t_stable"], P["t_max"],
        b["piN"], b["piM"], prices, step)

print(f"Núcleos usados pelo numba: {numba.get_num_threads()}; β = {beta:g}; sessões: {N}")
t0 = time.perf_counter()
run_many_full(np.array([1], dtype=np.int64), *args)
print(f"Compilação (1 sessão): {time.perf_counter() - t0:.1f} s")
t0 = time.perf_counter()
conv, per, dlt, clen, pidx, nash, qlp, qla, R, G, V = run_many_full(seeds, *args)
secs = time.perf_counter() - t0
print(f"{N} sessões + testes em {secs:.1f} s")

# Conferência: com o β do caso base, tudo igual às fases anteriores
checks = {}
if abs(beta - P["beta"]) < 1e-15:
    def col(path, key, cast):
        with open(path, encoding="utf-8") as f:
            return {int(x["seed"]): cast(x[key]) for x in csv.DictReader(f)}
    pairs = [("qlearning_base_sessions.csv", "delta", float, dlt, "Δ (Fase 1.3)"),
             ("qlearning_bestresponse_sessions.csv", "nash_on_path_calvano",
              lambda v: v == "True", nash, "Nash no caminho (Fase 1.5, parte 1)"),
             ("qlearning_deviation_sessions.csv", "G", float, G, "G (Fase 1.4)")]
    for fname, key, cast, arr, label in pairs:
        f = OUT / fname
        if f.exists():
            ref = col(f, key, cast)
            if cast is float:
                same = all(abs(ref[int(s)] - x) < 1e-12 for s, x in zip(seeds, arr) if conv[int(s) - 1])
            else:
                same = all(ref[int(s)] == bool(x) for s, x in zip(seeds, arr))
            checks[label] = same
            print(f"  igual às fases anteriores, {label}: {same}")

ok = conv
n_ok = int(ok.sum())
cls = [classify(R[k], G[k], V[k], THETA) if ok[k] else "" for k in range(N)]
d = dlt
se = d.std(ddof=1) / np.sqrt(N)
n_nash = int(nash[ok].sum())
n_unp = int((G[ok] < 0).sum())
price = prices[0] + pidx[ok] * step
s = {
    "beta": beta, "n_sessions": N, "n_converged": n_ok, "seconds": secs,
    "eps_at_mean_convergence": float(np.exp(-beta * per.mean())),
    "periods_mean": float(per.mean()), "periods_min": int(per.min()), "periods_max": int(per.max()),
    "delta_mean": float(d.mean()), "delta_sd": float(d.std(ddof=1)),
    "delta_ci95": [float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)],
    "delta_median": float(np.median(d)), "share_delta_gt_05": float((d > 0.5).mean()),
    "share_cycle": float((clen[ok] > 1).mean()),
    "price_pct_over_nash": float(100 * (price.mean() / b["pN"] - 1)),
    "nash_share": n_nash / n_ok, "nash_share_wilson95": wilson(n_nash, n_ok),
    "qloss_path_mean": float(np.nanmean(qlp)), "qloss_all_mean": float(np.nanmean(qla)),
    "R_mean": float(np.nanmean(R)), "G_mean": float(np.nanmean(G)),
    "share_G_lt_0": n_unp / n_ok, "share_G_lt_0_wilson95": wilson(n_unp, n_ok),
    "V_mean_uncensored": float(V[ok][V[ok] >= 0].mean()),
    "classes": {c: cls.count(c) for c in ("A", "B", "C", "outro")},
    "checks_vs_previous_phases": checks,
}
lo, hi = s["delta_ci95"]
print(f"\nβ = {beta:g} ({n_ok} de {N} convergiram; períodos: média {s['periods_mean']:,.0f}, "
      f"mín. {s['periods_min']:,}, máx. {s['periods_max']:,})")
print(f"  Δ: média {s['delta_mean']:.3f} [IC 95% {lo:.3f}; {hi:.3f}], DP {s['delta_sd']:.3f}, "
      f"mediana {s['delta_median']:.3f}; preço {s['price_pct_over_nash']:+.1f}% sobre Nash")
nlo, nhi = s["nash_share_wilson95"]
print(f"  em Nash no caminho: {s['nash_share']:.3f} [{nlo:.3f}; {nhi:.3f}]; Q-loss no caminho "
      f"{s['qloss_path_mean']:.4f}, em todos os estados {s['qloss_all_mean']:.4f}")
glo, ghi = s["share_G_lt_0_wilson95"]
print(f"  desvio: R {s['R_mean']:+.3f}; G médio {s['G_mean']:+.4f}; G < 0 em {s['share_G_lt_0']:.3f} "
      f"[{glo:.3f}; {ghi:.3f}]; V {s['V_mean_uncensored']:.2f}; classes {s['classes']}")

if SAVE:
    OUT.mkdir(exist_ok=True)
    key = f"{beta:g}"
    rows = []
    if CSV.exists():
        with open(CSV, encoding="utf-8") as f:
            rows = [x for x in csv.DictReader(f) if x["beta"] != key]
    for k in range(N):
        rows.append({"beta": key, "seed": int(seeds[k]), "converged": bool(conv[k]),
                     "periods": int(per[k]), "delta": float(dlt[k]), "cycle_len": int(clen[k]),
                     "nash_on_path_calvano": bool(nash[k]), "qloss_path": float(qlp[k]),
                     "qloss_all": float(qla[k]), "R": float(R[k]), "G": float(G[k]),
                     "V": float(V[k]) if ok[k] and V[k] >= 0 else None,
                     "class_theta2": cls[k]})
    rows.sort(key=lambda x: (-float(x["beta"]), int(x["seed"])))
    with open(CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    meta = load_meta()
    meta["by_beta"][key] = {"date": datetime.now().isoformat(timespec="seconds"), **s}
    meta["versions"] = {"python": platform.python_version(), "numpy": np.__version__,
                        "numba": numba.__version__, "scipy": scipy.__version__,
                        "platform": platform.platform()}
    meta["numba_threads"] = numba.get_num_threads()
    with open(META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\nβ = {key} salvo em {OUT}")
    print_table(meta)
