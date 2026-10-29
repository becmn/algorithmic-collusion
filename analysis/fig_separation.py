"""
Fase 1.8: figura e tabela do teste de separação (memória × falha de aprendizado), a partir de
results/qlearning_separation_sessions.csv (gravado por qlearning/run_separation.py; nada é
simulado aqui).

Gera:
  - paper/figures/fig_separation.png/.pdf     Δ com e sem memória sob a mesma exploração
  - paper/tables/table_separation.tex          a mesma comparação em LaTeX, com D e IC de Welch
  - results/qlearning_separation_table.csv     uma linha por nível de exploração
Regra de decisão (fixada antes de rodar, 16/10/2026): ver qlearning/run_separation.py.
Uso: python analysis/fig_separation.py
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from common import RES, ROOT, delta_random_prices, t_ci, wilson

FIG = ROOT / "paper" / "figures"
TAB = ROOT / "paper" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

DECAY = [("beta1e-05", "β = 10⁻⁵\n(shorter)"), ("beta4e-06", "β = 4×10⁻⁶\n(base case)"),
         ("beta2e-06", "β = 2×10⁻⁶"), ("beta1e-06", "β = 10⁻⁶\n(longest)")]
FLOOR = [("epsmin0.001", "ε ≥ 0.1%"), ("epsmin0.01", "ε ≥ 1%"), ("epsmin0.05", "ε ≥ 5%")]
DECISION_LEVELS = ("beta1e-06", "epsmin0.05")

with open(RES / "qlearning_separation_sessions.csv", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
by = {}
for r in rows:
    by.setdefault(r["condition"], []).append(r)

table = []
for lv, _ in DECAY + FLOOR:
    r0, r1 = by[f"k0_{lv}"], by[f"k1_{lv}"]
    d0 = np.array([float(r["delta"]) for r in r0])
    d1 = np.array([float(r["delta"]) for r in r1])
    tt = stats.ttest_ind(d1, d0, equal_var=False)
    ci = tt.confidence_interval(0.95)
    n1 = len(r1)
    m0, lo0, hi0 = t_ci(d0)
    m1, lo1, hi1 = t_ci(d1)
    _, nlo, nhi = wilson(sum(r["nash_on_path_1period"] == "True" for r in r1), n1)
    table.append({
        "level": lv, "n_k0": len(r0), "n_k1": n1,
        "delta_k0": m0, "delta_k0_lo": lo0, "delta_k0_hi": hi0,
        "delta_k1": m1, "delta_k1_lo": lo1, "delta_k1_hi": hi1,
        "D": m1 - m0, "D_lo": float(ci.low), "D_hi": float(ci.high), "p_welch": float(tt.pvalue),
        "nash1_k1": sum(r["nash_on_path_1period"] == "True" for r in r1) / n1,
        "nash1_k1_lo": nlo, "nash1_k1_hi": nhi,
        "nash_opt_k1": sum(r["nash_on_path_optimal"] == "True" for r in r1) / n1,
        "classA_k1": sum(r["class_theta2"] == "A" for r in r1) / n1,
        "converged_k1": sum(r["converged"] == "True" for r in r1) / n1,
    })

with open(RES / "qlearning_separation_table.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
    w.writeheader()
    w.writerows(table)

lows = [t["D_lo"] for t in table if t["level"] in DECISION_LEVELS]
highs = [t["D_hi"] for t in table if t["level"] in DECISION_LEVELS]
verdict = ("favorable" if all(x > 0 for x in lows) else
           "unfavorable" if all(x < 0 for x in highs) else "inconclusive")

# ---------------------------------------------------------------- figura
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
rand = delta_random_prices()
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True,
                         gridspec_kw={"width_ratios": [4, 3]})
tl = {t["level"]: t for t in table}
for ax, levels, title in ((axes[0], DECAY, "Decaying exploration ε = e^(−βt), until convergence"),
                          (axes[1], FLOOR, "Exploration floor, 4 million periods")):
    x = np.arange(len(levels))
    for key, lab, kw in (("k1", "One-period memory (k = 1): punishment possible",
                          dict(fmt="-o", color="black", mfc="black", ms=6)),
                         ("k0", "No memory (k = 0): punishment impossible",
                          dict(fmt="--s", color="0.45", mfc="white", ms=6))):
        m = np.array([tl[lv][f"delta_{key}"] for lv, _ in levels])
        lo = np.array([tl[lv][f"delta_{key}_lo"] for lv, _ in levels])
        hi = np.array([tl[lv][f"delta_{key}_hi"] for lv, _ in levels])
        ax.errorbar(x, m, yerr=[m - lo, hi - m], capsize=3, lw=1.4, label=lab, **kw)
        # rótulo direto só no último ponto de cada série
        ax.text(x[-1] + 0.12, m[-1], f"{m[-1]:.3f}", va="center", fontsize=8.5, color="0.2")
    ax.axhline(rand, color="0.5", lw=0.8, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in levels], fontsize=8.5)
    ax.set_xlim(-0.4, len(levels) - 0.4)
    ax.set_title(title, fontsize=9.5)
    ax.set_xlabel("More learning →", fontsize=9)
axes[0].text(-0.35, rand + 0.012, "random prices", fontsize=8, color="0.35")
axes[0].set_ylabel("Profit gain Δ (mean, 95% CI)")
axes[0].set_ylim(0.45, 1.02)
h, l = axes[0].get_legend_handles_labels()
fig.suptitle("Same exploration, with and without memory (1,000 sessions per point, δ = 0.95)",
             fontsize=10.5)
fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=2.0)
fig.legend(h, l, frameon=False, loc="lower center", ncol=2, fontsize=8.5)
saved = []
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"fig_separation.{ext}", dpi=200)
    saved.append(f"paper/figures/fig_separation.{ext}")
plt.close(fig)

# ---------------------------------------------------------------- LaTeX
labels = {"beta1e-05": r"$\beta = 10^{-5}$", "beta4e-06": r"$\beta = 4\times10^{-6}$ (base)",
          "beta2e-06": r"$\beta = 2\times10^{-6}$", "beta1e-06": r"$\beta = 10^{-6}$",
          "epsmin0.001": r"$\varepsilon \geq 0.1\%$", "epsmin0.01": r"$\varepsilon \geq 1\%$",
          "epsmin0.05": r"$\varepsilon \geq 5\%$"}
tex = [r"\begin{table}[t]", r"\centering\small",
       r"\caption{Separation test: profit gain $\Delta$ with and without memory under the same "
       r"exploration ($\delta = 0.95$, 1{,}000 sessions per cell). $D = \Delta_{k=1} - \Delta_{k=0}$ "
       r"with 95\% Welch interval. Nash: share of sessions in Nash equilibrium on path, "
       r"one-period deviations (any deviation in parentheses). Class A: punishment with return "
       r"($\theta = 2\%$). Decision rule fixed before running: punishment reading supported only "
       r"if $D > 0$ at both most-learning conditions ($\beta = 10^{-6}$ and $\varepsilon \geq 5\%$). "
       f"Result: {verdict}." + r"}",
       r"\label{tab:separation}", r"\begin{tabular}{lccccc}", r"\toprule",
       r"Exploration & $\Delta$, no memory & $\Delta$, memory & $D$ (95\% CI) & Nash, memory & Class A \\",
       r"\midrule"]
for t in table:
    tex.append(f"{labels[t['level']]} & {t['delta_k0']:.3f} & {t['delta_k1']:.3f} & "
               f"{t['D']:+.3f} ({t['D_lo']:+.3f} to {t['D_hi']:+.3f}) & "
               f"{t['nash1_k1']:.3f} ({t['nash_opt_k1']:.3f}) & {t['classA_k1']:.3f} \\\\")
    if t["level"] == "beta1e-06":
        tex.append(r"\midrule")
tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
(TAB / "table_separation.tex").write_text("\n".join(tex), encoding="utf-8")

print("Nível           Δ k0    Δ k1    D [IC 95% Welch]          Nash1 k1  Nash ótimo k1  classe A k1")
for t in table:
    print(f"  {t['level']:<13} {t['delta_k0']:.3f}   {t['delta_k1']:.3f}   {t['D']:+.3f} "
          f"[{t['D_lo']:+.3f}; {t['D_hi']:+.3f}]   {t['nash1_k1']:.3f}     {t['nash_opt_k1']:.3f}"
          f"          {t['classA_k1']:.3f}")
print(f"Decisão pela regra fixada antes: {verdict}")
print("Conferência arquivos_salvos:", all((ROOT / p).exists() for p in saved)
      and (TAB / "table_separation.tex").exists())
