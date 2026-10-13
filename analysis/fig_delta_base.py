"""
Fase 1.6, parte 1: gráficos e tabela do caso base a partir de results/ (sem simular nada).

Lê results/qlearning_base_sessions.csv (Fase 1.3, 1.000 sessões) e gera:
  - paper/figures/fig_delta_distribution.png e .pdf   (distribuição de Δ entre sessões)
  - paper/figures/fig_delta_vs_periods.png e .pdf     (Δ × períodos até convergir)
  - results/qlearning_delta_vs_periods_bands.csv      (tabela por faixa de Δ)
  - results/qlearning_figures_part1_meta.json         (números das figuras: correlações, frações, etc.)

Componente oficial Δ × períodos (decisão de 10/10/2026): descritivo e exploratório,
fora da família confirmatória, sem afirmar causa.

Uso: python analysis/fig_delta_base.py
"""

import csv
import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # salva arquivos sem abrir janela
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

DELTA_CALVANO = 0.849      # alvo, Tabela 1 de Calvano (Ficha de parâmetros)
DELTA_RANDOM = 0.4974      # preços sorteados ao acaso na grade (Fase 1.1)

# ---------- leitura ----------
rows = list(csv.DictReader(open(RES / "qlearning_base_sessions.csv", encoding="utf-8")))
delta = np.array([float(r["delta"]) for r in rows])
periods = np.array([int(r["periods"]) for r in rows])
cycle = np.array([int(r["cycle_len"]) > 1 for r in rows])
n = len(delta)


def wilson(k, n, z=1.959964):
    """IC de Wilson (95%) para uma fração k/n."""
    p = k / n
    den = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return [c - h, c + h]


# ---------- números da distribuição ----------
q1, med, q3 = np.percentile(delta, [25, 50, 75])
k_above = int((delta > 0.5).sum())
k_below = int((delta < 0).sum())
dist = {
    "n": n,
    "mean": float(delta.mean()),
    "sd": float(delta.std(ddof=1)),
    "median": float(med),
    "q1": float(q1),
    "q3": float(q3),
    "min": float(delta.min()),
    "max": float(delta.max()),
    "share_delta_gt_0_5": k_above / n,
    "share_delta_gt_0_5_wilson95": wilson(k_above, n),
    "share_delta_lt_0": k_below / n,
    "share_delta_lt_0_wilson95": wilson(k_below, n),
    "mean_constant_price": float(delta[~cycle].mean()),
    "mean_cycle": float(delta[cycle].mean()),
    "n_constant_price": int((~cycle).sum()),
    "n_cycle": int(cycle.sum()),
}

# ---------- figura 1: distribuição de Δ ----------
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
bins = np.arange(0.40, 1.0001, 0.02)
fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.hist([delta[~cycle], delta[cycle]], bins=bins, stacked=True,
        color=["0.35", "0.75"], edgecolor="white", linewidth=0.6,
        label=[f"Constant price (n = {dist['n_constant_price']})",
               f"Cycle (n = {dist['n_cycle']})"])
# a média desta replicação e o alvo de Calvano coincidem (0,849): uma linha só, com os dois no rótulo
ax.axvline(dist["mean"], color="black", lw=1.5, ls="-")
ax.text(dist["mean"], ax.get_ylim()[1] * 0.97,
        f"mean {dist['mean']:.3f}  \n(Calvano et al.: {DELTA_CALVANO})  ",
        va="top", ha="right", fontsize=9)
ax.axvline(DELTA_RANDOM, color="0.4", lw=1.0, ls=":")
ax.text(DELTA_RANDOM, ax.get_ylim()[1] * 0.97, "  random prices", va="top",
        ha="left", fontsize=9, color="0.3")
ax.set_xlabel("Profit gain Δ (0 = Nash, 1 = monopoly)")
ax.set_ylabel("Sessions")
ax.set_xlim(0.40, 1.0)
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.80))
ax.set_title(f"Distribution of Δ across {n} Q-learning sessions (base case)", fontsize=10)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"fig_delta_distribution.{ext}", dpi=200)
plt.close(fig)

# ---------- tabela Δ × períodos até convergir ----------
bands = [("Δ ≤ 0.6", delta <= 0.6),
         ("0.6 < Δ ≤ 0.7", (delta > 0.6) & (delta <= 0.7)),
         ("0.7 < Δ ≤ 0.8", (delta > 0.7) & (delta <= 0.8)),
         ("0.8 < Δ ≤ 0.9", (delta > 0.8) & (delta <= 0.9)),
         ("Δ > 0.9", delta > 0.9),
         ("|Δ − 0.849| ≤ 0.02", np.abs(delta - DELTA_CALVANO) <= 0.02),
         ("All sessions", np.ones(n, dtype=bool))]
table = []
for name, m in bands:
    p = periods[m]
    table.append({
        "band": name,
        "n": int(m.sum()),
        "periods_mean_millions": round(float(p.mean()) / 1e6, 3),
        "periods_median_millions": round(float(np.median(p)) / 1e6, 3),
        "periods_sd_millions": round(float(p.std(ddof=1)) / 1e6, 3),
        "share_cycle": round(float(cycle[m].mean()), 3),
    })
with open(RES / "qlearning_delta_vs_periods_bands.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
    w.writeheader()
    w.writerows(table)

pear = stats.pearsonr(delta, periods)
spear = stats.spearmanr(delta, periods)
corr = {
    "pearson_r": float(pear.statistic),
    "pearson_p": float(pear.pvalue),
    "spearman_rho": float(spear.statistic),
    "spearman_p": float(spear.pvalue),
}

# ---------- figura 2: Δ × períodos ----------
fig, ax = plt.subplots(figsize=(6.5, 4.4))
x = periods / 1e6
ax.scatter(x[~cycle], delta[~cycle], s=9, marker="o", facecolor="0.3",
           edgecolor="none", alpha=0.55, label="Constant price")
ax.scatter(x[cycle], delta[cycle], s=10, marker="^", facecolor="none",
           edgecolor="0.45", linewidth=0.6, alpha=0.7, label="Cycle")
# média de períodos por faixa (as 5 faixas exclusivas), desenhada no centro da faixa
centers = [0.55, 0.65, 0.75, 0.85, 0.95]
ax.plot([t["periods_mean_millions"] for t in table[:5]], centers, color="black",
        lw=1.5, marker="s", ms=5, label="Mean periods by Δ band")
ax.set_xlabel("Periods until convergence (millions)")
ax.set_ylabel("Profit gain Δ")
ax.set_ylim(0.40, 1.0)
# legenda abaixo do gráfico, para não cobrir pontos
leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=9,
                markerscale=1.8, frameon=False)
for h in leg.legend_handles:
    h.set_alpha(1)
ax.set_title(f"Δ vs. time to converge, {n} sessions "
             f"(Pearson r = {corr['pearson_r']:.2f}, Spearman ρ = {corr['spearman_rho']:.2f})",
             fontsize=10)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"fig_delta_vs_periods.{ext}", dpi=200)
plt.close(fig)

# ---------- metadados ----------
meta = {
    "description": "Fase 1.6, parte 1: distribuição de Δ e Δ × períodos até convergir "
                   "(caso base, 1.000 sessões da Fase 1.3). Descritivo e exploratório.",
    "date": datetime.now().isoformat(timespec="seconds"),
    "source": "results/qlearning_base_sessions.csv",
    "distribution": dist,
    "delta_vs_periods": {"correlation": corr, "bands": table},
    "figures": ["paper/figures/fig_delta_distribution.png", "paper/figures/fig_delta_distribution.pdf",
                "paper/figures/fig_delta_vs_periods.png", "paper/figures/fig_delta_vs_periods.pdf"],
    "versions": {"numpy": np.__version__, "matplotlib": matplotlib.__version__,
                 "scipy": __import__("scipy").__version__},
}
with open(RES / "qlearning_figures_part1_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

# ---------- saída na tela ----------
print(f"Sessões: {n}")
print(f"Δ: média {dist['mean']:.3f}, DP {dist['sd']:.3f}, mediana {dist['median']:.3f}, "
      f"IQR [{dist['q1']:.3f}; {dist['q3']:.3f}], mín {dist['min']:.3f}, máx {dist['max']:.3f}")
print(f"Fração Δ > 0,5: {dist['share_delta_gt_0_5']:.3f}  Wilson {np.round(dist['share_delta_gt_0_5_wilson95'], 3)}")
print(f"Fração Δ < 0:   {dist['share_delta_lt_0']:.3f}  Wilson {np.round(dist['share_delta_lt_0_wilson95'], 3)}")
print(f"Δ médio: preço constante {dist['mean_constant_price']:.3f} (n={dist['n_constant_price']}); "
      f"ciclo {dist['mean_cycle']:.3f} (n={dist['n_cycle']})")
print(f"Correlação Δ × períodos: Pearson {corr['pearson_r']:.3f}; Spearman {corr['spearman_rho']:.3f}")
print("Faixa | n | períodos médios (mi) | mediana (mi) | fração em ciclo")
for t in table:
    print(f"  {t['band']:<20} {t['n']:>4}  {t['periods_mean_millions']:.2f}  "
          f"{t['periods_median_millions']:.2f}  {t['share_cycle']:.2f}")

# ---------- conferências ----------
checks = {
    "n_1000": n == 1000,
    "media_igual_fase13": abs(dist["mean"] - 0.849348000076128) < 1e-9,
    "soma_faixas_1000": sum(t["n"] for t in table[:5]) == 1000,
    "figuras_salvas": all((ROOT / p).exists() for p in meta["figures"]),
}
for k, v in checks.items():
    print(f"Conferência {k}: {v}")
