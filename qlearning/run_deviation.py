"""
Fase 1.4: teste de desvio comum nas 1.000 sessões do caso base (sementes 1 a 1.000).
Refaz as sessões (mesmo código e sementes da Fase 1.3), confere que o Δ de cada sessão
é igual ao de results/qlearning_base_sessions.csv e salva em results/:
  - qlearning_deviation_sessions.csv: R, G, V e classes por sessão
  - qlearning_deviation_meta.json: protocolo, resumo, comparação com Calvano e curva média
Uso: python qlearning/run_deviation.py          (1.000 sessões)
     python qlearning/run_deviation.py 20       (teste rápido, não salva)
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
from deviation import T_DEV, classify, run_many_with_deviation

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
SAVE = N == 1000
OUT = HERE.parent / "results"
THETAS = (0.02, 0.05)          # θ principal e sensibilidade (Protocolo)

# Alvos da Ficha de parâmetros (Calvano et al., Apêndice, Tabela A5)
TARGET_IR, TARGET_IC, TARGET_DUR = -0.127, 0.936, 5.7

b = setup()
P = PARAMS
prices = b["prices"]
step = prices[1] - prices[0]
args = (b["PI"], b["Q0"], P["alpha"], P["delta"], P["beta"], P["t_stable"], P["t_max"],
        b["piN"], b["piM"], prices, step)
seeds = np.arange(1, N + 1, dtype=np.int64)

print(f"Núcleos usados pelo numba: {numba.get_num_threads()}")
t0 = time.perf_counter()
run_many_with_deviation(np.array([0], dtype=np.int64), *args)
print(f"Compilação (1 sessão): {time.perf_counter() - t0:.1f} s")

t0 = time.perf_counter()
conv, dlt, clen, R, G, Rc, V, Vp, cens, nodev, curves = run_many_with_deviation(seeds, *args)
secs = time.perf_counter() - t0
print(f"{N} sessões + teste de desvio em {secs:.1f} s")

# Conferência com a Fase 1.3
base_csv = OUT / "qlearning_base_sessions.csv"
same_delta = None
if base_csv.exists():
    with open(base_csv, encoding="utf-8") as f:
        base = {int(r["seed"]): float(r["delta"]) for r in csv.DictReader(f)}
    same_delta = all(abs(base[int(s)] - d) < 1e-12 for s, d in zip(seeds, dlt) if int(s) in base)
    print(f"Δ igual ao da Fase 1.3 em todas as sessões: {same_delta}")

ok = conv
n_ok = int(ok.sum())
rows = []
for k in range(N):
    row = {"seed": int(seeds[k]), "converged": bool(conv[k]), "delta": float(dlt[k]),
           "cycle_len": int(clen[k]),
           "R": float(R[k]), "G": float(G[k]), "R_calvano": float(Rc[k]),
           "V": float(V[k]) if ok[k] and V[k] >= 0 else None,
           "V_censored": bool(ok[k] and V[k] < 0),
           "V_phase": float(Vp[k]) if ok[k] and Vp[k] >= 0 else None,
           "V_phase_censored": bool(ok[k] and Vp[k] < 0),
           "share_starts_censored": float(cens[k]),
           "share_starts_no_deviation": float(nodev[k])}
    for th in THETAS:
        row[f"class_theta{int(th * 100)}"] = (classify(R[k], G[k], V[k], th) if ok[k] else "")
    rows.append(row)

Rk, Gk, Rck = R[ok], G[ok], Rc[ok]
Vk = V[ok]
v_ok = Vk[Vk >= 0]


def wilson(x, n, z=1.96):
    p = x / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


classes = {}
for th in THETAS:
    col = [r[f"class_theta{int(th * 100)}"] for r in rows if r["converged"]]
    classes[f"theta_{th}"] = {c: col.count(c) for c in ("A", "B", "C", "outro")}

n_unprof = int((Gk < 0).sum())
summary = {
    "n_sessions": N, "n_tested": n_ok,
    "R_mean": float(Rk.mean()), "R_sd": float(Rk.std(ddof=1)),
    "R_calvano_mean": float(Rck.mean()),
    "G_mean": float(Gk.mean()), "G_sd": float(Gk.std(ddof=1)),
    "share_unprofitable_G_lt_0": n_unprof / n_ok,
    "share_unprofitable_wilson95": wilson(n_unprof, n_ok),
    "V_mean_uncensored": float(v_ok.mean()) if len(v_ok) else None,
    "V_median_uncensored": float(np.median(v_ok)) if len(v_ok) else None,
    "n_V_censored": int((Vk < 0).sum()),
    "n_V_censored_constant": int(((Vk < 0) & (clen[ok] == 1)).sum()),
    "n_V_censored_cycle": int(((Vk < 0) & (clen[ok] > 1)).sum()),
    "n_V_phase_censored": int((Vp[ok] < 0).sum()),
    "n_sessions_cycle": int((clen[ok] > 1).sum()),
    "n_sessions_with_start_without_deviation": int((nodev[ok] > 0).sum()),
    "n_sessions_never_deviate": int((nodev[ok] == 1).sum()),
    "classes": classes,
    "seconds": secs,
    "same_delta_as_phase_1_3": same_delta,
}
mean_curve = np.nanmean(curves[ok], axis=0)

print("\nResumo (sessões testadas: %d)" % n_ok)
print(f"  R (reação do rival em τ = 2, contra o controle): média {summary['R_mean']:.3f} "
      f"(DP {summary['R_sd']:.3f})   alvo IR {TARGET_IR}")
print(f"  R no estilo de Calvano (contra o preço de τ = 0): média {summary['R_calvano_mean']:.3f}")
print(f"  G (ganho do desvio): média {summary['G_mean']:.4f} (DP {summary['G_sd']:.4f}); "
      f"Calvano: −3% a −4%")
lo, hi = summary["share_unprofitable_wilson95"]
print(f"  desvio não lucrativo (G < 0): {summary['share_unprofitable_G_lt_0']:.3f} "
      f"[IC 95% {lo:.3f}; {hi:.3f}]   alvo IC {TARGET_IC}")
print(f"  V (volta), sessões sem censura: média {summary['V_mean_uncensored']:.2f}, "
      f"mediana {summary['V_median_uncensored']:.1f}; censuradas: {summary['n_V_censored']}"
      f"   alvo duração ~{TARGET_DUR}")
print(f"  sessões com algum ponto sem desvio de fato (já joga a melhor resposta): "
      f"{summary['n_sessions_with_start_without_deviation']} "
      f"(nunca desviam: {summary['n_sessions_never_deviate']})")
print(f"  censuradas: {summary['n_V_censored_constant']} de preço constante, "
      f"{summary['n_V_censored_cycle']} de {summary['n_sessions_cycle']} em ciclo; "
      f"pela regra preço a preço seriam {summary['n_V_phase_censored']}")
for th in THETAS:
    c = classes[f"theta_{th}"]
    print(f"  classes θ = {int(th * 100)}%: A {c['A']}, B {c['B']}, C {c['C']}, outro {c['outro']}")
print("  curva média p2_dev/p2_ctrl − 1, τ = 0..15:")
print("   " + " ".join(f"{x:+.3f}" for x in mean_curve))

if SAVE:
    OUT.mkdir(exist_ok=True)
    with open(OUT / "qlearning_deviation_sessions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    meta = {
        "description": "Teste de desvio comum, Q-learning caso base, sementes 1 a 1000",
        "date": datetime.now().isoformat(timespec="seconds"),
        "protocol": {
            "start": "estratégias gulosas finais; Q congelada; ε = 0",
            "deviation": "firma 1, τ = 1, melhor resposta estática na grade ao preço da firma 2 em τ = 0",
            "horizon": T_DEV, "discount": P["delta"], "grid_step": float(step),
            "V_rule": "1º τ ≥ 2 com os dois preços a até 1 degrau do controle por 3 períodos "
                      "seguidos dentro de τ ≤ 15; senão censurado",
            "cycles": "desvio a partir de cada ponto do ciclo; média = 1 observação; "
                      "V da sessão censurado se algum ponto do ciclo for censurado",
            "V_rule_cycles": "em sessões em ciclo, volta = par de preços de novo no ciclo-limite, "
                             "em qualquer fase, por 3 períodos seguidos (decisão de 13/10/2026); "
                             "a regra preço a preço fica em V_phase",
            "thetas": list(THETAS),
        },
        "targets_calvano_table_A5": {"IR": TARGET_IR, "IC": TARGET_IC, "duration": TARGET_DUR},
        "summary": summary,
        "mean_curve_rival_dev_over_ctrl_minus_1": [float(x) for x in mean_curve],
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "numba": numba.__version__, "scipy": scipy.__version__,
                     "platform": platform.platform()},
        "numba_threads": numba.get_num_threads(),
    }
    with open(OUT / "qlearning_deviation_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\nSalvo em {OUT}")
