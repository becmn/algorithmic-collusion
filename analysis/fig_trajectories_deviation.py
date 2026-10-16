"""
Fase 1.6, parte 2: figuras de trajetória e de resposta ao desvio, a partir de results/
(arquivos gravados por qlearning/run_trajectories.py; nada é simulado aqui).

Gera em paper/figures/ (PNG e PDF):
  - fig_learning_trajectory      Δ médio por bloco (eixo de períodos logarítmico), faixa
                                 P10–P90 entre as 1.000 sessões, e o Δ no ciclo-limite
  - fig_price_trajectories       preços das duas firmas em 3 sessões de exemplo
                                 (Δ mais próximo dos percentis 25, 50 e 75)
  - fig_deviation_response       preços médios nos dois ramos do teste de desvio (τ = 0 a 15)
                                 e a curva R(τ) do rival, com IC de 95%
Figuras em tons de cinza; as firmas se distinguem por estilo de linha e marcador, não por cor.
Uso: python analysis/fig_trajectories_deviation.py
Revisão de 16/10/2026: Δ de preços ao acaso calculado (analysis/common.py). A curva da direita
da resposta ao desvio é p₂ com desvio(τ) / p₂ controle(τ) − 1 (coluna R_tau_* do CSV); ela NÃO é
o R do Protocolo, que divide por p₂(0) (os dois só coincidem nas sessões de preço constante).
Por isso o rótulo diz "relative to control" e a linha de θ, que vale para o R do Protocolo, saiu.
"""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import delta_random_prices

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

base_meta = json.load(open(RES / "qlearning_base_meta.json", encoding="utf-8"))
DELTA_RANDOM = delta_random_prices()   # preços sorteados ao acaso na grade (0,4974; Fase 1.1)
LLM_HORIZON = 300              # períodos por execução de LLM (Protocolo)


def read(name):
    with open(RES / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
saved = []


def save(fig, name, tight=True):
    if tight:
        fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200)
        saved.append(f"paper/figures/{name}.{ext}")
    plt.close(fig)


# ================================================================ 1. trajetória de Δ
blocks = [r for r in read("qlearning_trajectory_blocks.csv") if int(r["n_sessions"]) == 1000]
t_mid = np.array([float(r["t_mid_geom"]) for r in blocks])
d_mean = np.array([float(r["delta_mean"]) for r in blocks])
d_p10 = np.array([float(r["delta_p10"]) for r in blocks])
d_p90 = np.array([float(r["delta_p90"]) for r in blocks])
t_last = int(blocks[-1]["t_end"])
base = read("qlearning_base_sessions.csv")
d_conv = np.array([float(r["delta"]) for r in base])
per_conv = np.array([int(r["periods"]) for r in base])

fig, ax = plt.subplots(figsize=(6.5, 4.0))
ax.fill_between(t_mid, d_p10, d_p90, color="0.85", lw=0, label="P10–P90 across sessions")
ax.plot(t_mid, d_mean, color="black", lw=1.6, label="Mean across 1,000 sessions")
# Δ final (ciclo-limite) desenhado à direita, na mediana dos períodos até convergir
x_end = float(np.median(per_conv))
q10, q90 = np.percentile(d_conv, [10, 90])
ax.errorbar([x_end], [d_conv.mean()], yerr=[[d_conv.mean() - q10], [q90 - d_conv.mean()]],
            fmt="s", color="black", ms=6, capsize=3, lw=1.2,
            label="At convergence (limit cycle): mean, P10–P90")
ax.axhline(0, color="0.3", lw=0.9, ls="--")
ax.text(1.2, 0.015, "Nash (Δ = 0)", fontsize=8.5, color="0.25", va="bottom")
ax.axhline(1, color="0.3", lw=0.9, ls="--")
ax.text(1.2, 0.985, "Monopoly (Δ = 1)", fontsize=8.5, color="0.25", va="top")
ax.axhline(DELTA_RANDOM, color="0.5", lw=0.8, ls=":")
ax.text(1.2, DELTA_RANDOM + 0.015, "random prices", fontsize=8.5, color="0.35", va="bottom")
ax.axvline(LLM_HORIZON, color="0.5", lw=0.8, ls="-.")
ax.text(LLM_HORIZON * 1.15, 0.08, f"period {LLM_HORIZON}\n(length of an LLM run)",
        fontsize=8.5, color="0.35", va="bottom")
ax.set_xscale("log")
ax.set_xlim(1, x_end * 2.2)
ax.set_ylim(-0.05, 1.05)
ax.set_xlabel("Period (log scale); each point averages one block of periods")
ax.set_ylabel("Profit gain Δ")
ax.set_title("Learning trajectory of Δ, Q-learning base case", fontsize=10)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=8.5)
save(fig, "fig_learning_trajectory")

# ================================================================ 2. preços de 3 sessões
tm = json.load(open(RES / "qlearning_trajectories_meta.json", encoding="utf-8"))
ex = read("qlearning_trajectory_examples.csv")
grid = base_meta["price_grid"]
pN = base_meta["reference_points"]["pN"]
pM = base_meta["reference_points"]["pM"]
fig, axes = plt.subplots(3, 1, figsize=(6.5, 7.2), sharex=True)
for ax, role in zip(axes, ("p25", "p50", "p75")):
    rows = [r for r in ex if r["role"] == role]
    e = tm["examples"][role]
    x = np.array([np.sqrt(max(int(r["t_start"]), 1) * int(r["t_end"])) for r in rows])
    ax.plot(x, [float(r["price_firm1"]) for r in rows], color="black", lw=1.3, label="Firm 1")
    ax.plot(x, [float(r["price_firm2"]) for r in rows], color="0.45", lw=1.3, ls="--",
            label="Firm 2")
    for y, lab in ((pN, "Nash"), (pM, "Monopoly")):
        ax.axhline(y, color="0.3", lw=0.8, ls=":")
        # rótulo fora da área do gráfico, à direita, para não ficar coberto pelas linhas de preço
        ax.text(1.01, y, lab, fontsize=8, color="0.25", va="center",
                transform=ax.get_yaxis_transform())
    ax.set_xscale("log")
    ax.set_ylim(min(grid) - 0.02, max(grid) + 0.02)
    ax.set_ylabel("Price")
    ax.set_title(f"Seed {e['seed']}: Δ = {e['delta']:.3f} (closest to the "
                 f"{role[1:]}th percentile), cycle length {e['cycle_len']}", fontsize=9.5)
axes[-1].set_xlabel("Period (log scale); each point averages one block of periods")
h, l = axes[0].get_legend_handles_labels()
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.legend(h, l, frameon=False, loc="lower center", ncol=2, fontsize=9)
save(fig, "fig_price_trajectories", tight=False)

# ================================================================ 3. resposta ao desvio
dv = read("qlearning_deviation_paths.csv")
tau = np.array([int(r["tau"]) for r in dv])
col = lambda k: np.array([float(r[k]) for r in dv])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.0, 3.8), gridspec_kw={"width_ratios": [1.15, 1]})
a1.fill_between(tau, col("dev_p1_ci_lo"), col("dev_p1_ci_hi"), color="0.8", lw=0)
a1.plot(tau, col("dev_p1_mean"), color="black", lw=1.5, marker="s", ms=4,
        label="Deviating firm (firm 1)")
a1.fill_between(tau, col("dev_p2_ci_lo"), col("dev_p2_ci_hi"), color="0.8", lw=0)
a1.plot(tau, col("dev_p2_mean"), color="0.35", lw=1.5, ls="--", marker="o", ms=4,
        mfc="white", label="Rival (firm 2)")
a1.plot(tau, 0.5 * (col("ctrl_p1_mean") + col("ctrl_p2_mean")), color="0.6", lw=1.0,
        ls="-.", label="Control branch (no deviation)")
for y, lab in ((pN, "Nash"), (pM, "Monopoly")):
    a1.axhline(y, color="0.3", lw=0.8, ls=":")
    a1.text(15, y, lab, fontsize=8, color="0.25", va="bottom", ha="right")
a1.set_xlabel("τ (deviation at τ = 1)")
a1.set_ylabel("Average price")
a1.set_xticks(range(0, 16, 3))
a1.set_title("Prices after a one-period forced deviation", fontsize=10)
a1.legend(frameon=False, loc="lower right", fontsize=8.5, bbox_to_anchor=(1.0, 0.08))

r_mean, r_lo, r_hi = col("R_tau_mean"), col("R_tau_ci_lo"), col("R_tau_ci_hi")
a2.fill_between(tau, 100 * r_lo, 100 * r_hi, color="0.8", lw=0, label="95% CI")
a2.plot(tau, 100 * r_mean, color="black", lw=1.5, marker="o", ms=4, label="Mean over sessions")
a2.axhline(0, color="0.3", lw=0.8)
a2.set_xlabel("τ (deviation at τ = 1)")
a2.set_ylabel("Rival price relative to control (%)")
a2.set_xticks(range(0, 16, 3))
a2.set_title("Rival's reaction: p₂ deviation / p₂ control − 1", fontsize=10)
a2.legend(frameon=False, loc="lower right", fontsize=8.5)
save(fig, "fig_deviation_response")

# ================================================================ conferências
checks = {
    "trajetoria_1000_sessoes": len(blocks) > 0 and all(int(r["n_sessions"]) == 1000 for r in blocks),
    "tres_exemplos": len({r["role"] for r in ex}) == 3,
    "R2_igual_fase14": abs(r_mean[2] - json.load(open(RES / "qlearning_deviation_meta.json",
                                                      encoding="utf-8"))
                           ["mean_curve_rival_dev_over_ctrl_minus_1"][2]) < 1e-12,
    "figuras_salvas": all((ROOT / p).exists() for p in saved),
}
print(f"Trajetória: {len(blocks)} blocos com as 1.000 sessões, até o período {t_last:,}")
print(f"Δ no período ~{LLM_HORIZON}: "
      f"{d_mean[np.argmin(np.abs(t_mid - LLM_HORIZON))]:.3f}; mínimo da média "
      f"{d_mean.min():.3f} perto do período {t_mid[np.argmin(d_mean)]:,.0f}; "
      f"no último bloco {d_mean[-1]:.3f}; no ciclo-limite {d_conv.mean():.3f}")
print(f"R(τ) em τ = 2: {100 * r_mean[2]:.2f}% [{100 * r_lo[2]:.2f}; {100 * r_hi[2]:.2f}]")
print("Figuras:", ", ".join(Path(p).name for p in saved))
for k, v in checks.items():
    print(f"Conferência {k}: {v}")
