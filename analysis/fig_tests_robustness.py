"""
Fase 1.6, parte 3: tabela e figuras dos testes obrigatórios da Fase 1.5, a partir de results/
(nada é simulado aqui).

Plano de figuras do artigo (decisão de 13/10/2026):
  texto principal (máx. 4): (1) Δ por sessão/execução e agente; (2) trajetória de Δ com o
  período 300; (3) resposta ao desvio; (4) teste de separação (Fase 1.8; revisão de 16/10/2026).
  Esta figura de robustez foi para o apêndice; os testes obrigatórios ficam no texto como tabela.

Gera:
  - results/qlearning_mandatory_tests_table.csv   uma linha por condição
  - paper/tables/table_mandatory_tests.tex         a mesma tabela em LaTeX (booktabs)
  - paper/figures/fig_robustness_tests.png/.pdf    figura 4 do texto principal
  - paper/figures/fig_memoryless_discount.png/.pdf apêndice: Δ sem memória por δ
  - results/qlearning_figures_part3_meta.json      números das figuras e conferências
Uso: python analysis/fig_tests_robustness.py
Revisão de 16/10/2026: a fração em Nash aparece nas duas leituras, rotuladas: desvios de 1 período
(leitura de Calvano) e qualquer desvio (melhor resposta ótima; caso base e β lidos de
results/qlearning_separation_sessions.csv, Fase 1.8). Nos três controles as duas coincidem por
construção (k = 0 tem um estado só; com δ = 0 a continuação não pesa). IC de Wilson, IC t e Δ de
preços ao acaso vêm de analysis/common.py; nenhuma conferência usa número digitado.
"""

import csv
import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from common import delta_random_prices, t_ci as _t_ci, t_ci_summary, wilson

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = ROOT / "paper" / "figures"
TAB = ROOT / "paper" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)


def read(name):
    with open(RES / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


t_ci = _t_ci
DELTA_RANDOM = delta_random_prices()


truthy = lambda v: str(v).strip().lower() == "true"

# ---------------------------------------------------------------- condições
expl = read("qlearning_exploration_sessions.csv")
ctrl = read("qlearning_controls_sessions.csv")
sep = read("qlearning_separation_sessions.csv")      # Nash com qualquer desvio (Fase 1.8)
conds = []   # (id, rótulo em inglês, linhas, tem teste de desvio, alvos de Calvano)
for beta, lab in ((4e-6, "Base case (β = 4×10⁻⁶)"), (1e-5, "Shorter exploration (β = 10⁻⁵)"),
                  (2e-6, "Longer exploration (β = 2×10⁻⁶)"), (1e-6, "Longer exploration (β = 10⁻⁶)")):
    rows = [r for r in expl if abs(float(r["beta"]) - beta) < 1e-12]
    tgt = {"delta": 0.849, "nash": 0.505, "unprofitable": 0.936} if beta == 4e-6 else {}
    conds.append((f"beta_{beta:g}", lab, rows, True, tgt))
for cid, lab, tgt in (("sem_memoria_d095", "No memory, δ = 0.95", {}),
                      ("memoria1_d0", "Memory 1, δ = 0 (myopic)", {}),
                      ("tabela_A1", "Calvano Table A1 control (k = 0, δ = 0, α = 0.25, β = 10⁻⁴)",
                       {"delta": 0.185, "nash": 0.273})):
    rows = [r for r in ctrl if r["condition"] == cid]
    conds.append((cid, lab, rows, False, tgt))

table = []
for cid, lab, rows, has_dev, tgt in conds:
    n = len(rows)
    d, dlo, dhi = t_ci([float(r["delta"]) for r in rows])
    kn = sum(truthy(r["nash_on_path_calvano"]) for r in rows)
    pn, pnlo, pnhi = wilson(kn, n)
    if cid.startswith("beta_"):
        srows = [r for r in sep if r["condition"] == f"k1_{cid.replace('_', '')}"]
        assert len(srows) == n and all(abs(float(a["delta"]) - float(b_["delta"])) < 1e-12
                                       for a, b_ in zip(sorted(srows, key=lambda r: int(r["seed"])),
                                                        sorted(rows, key=lambda r: int(r["seed"]))))
        ko = sum(truthy(r["nash_on_path_optimal"]) for r in srows)
    else:
        ko = kn          # controles: as duas leituras coincidem por construção
    po, polo, pohi = wilson(ko, n)
    row = {"condition": cid, "label": lab, "n": n,
           "converged": sum(truthy(r["converged"]) for r in rows),
           "periods_mean_millions": round(np.mean([float(r["periods"]) for r in rows]) / 1e6, 3),
           "delta_mean": d, "delta_ci_lo": dlo, "delta_ci_hi": dhi,
           "nash_share": pn, "nash_ci_lo": pnlo, "nash_ci_hi": pnhi,
           "nash_any_dev_share": po, "nash_any_dev_ci_lo": polo, "nash_any_dev_ci_hi": pohi,
           "unprofitable_share": None, "unprofitable_ci_lo": None, "unprofitable_ci_hi": None,
           "classA_share": None, "classA_ci_lo": None, "classA_ci_hi": None,
           "target_delta": tgt.get("delta"), "target_nash": tgt.get("nash"),
           "target_unprofitable": tgt.get("unprofitable")}
    if has_dev:
        ku = sum(float(r["G"]) < 0 for r in rows)
        ka = sum(r["class_theta2"] == "A" for r in rows)
        row["unprofitable_share"], row["unprofitable_ci_lo"], row["unprofitable_ci_hi"] = wilson(ku, n)
        row["classA_share"], row["classA_ci_lo"], row["classA_ci_hi"] = wilson(ka, n)
    table.append(row)

with open(RES / "qlearning_mandatory_tests_table.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
    w.writeheader()
    w.writerows(table)

# ---------------------------------------------------------------- LaTeX
def fmt(x, nd=3):
    return "--" if x is None else f"{x:.{nd}f}"


tex = [r"\begin{table}[t]", r"\centering\small",
       r"\caption{Mandatory tests for the Q-learning agents (1{,}000 sessions per row). "
       r"$\Delta$: mean with 95\% $t$ interval. Nash: share of sessions whose limit strategies are "
       r"a Nash equilibrium on path against one-period deviations (Calvano et al.'s test); in "
       r"parentheses, against any deviation. Unprofitable: share with "
       r"discounted deviation gain $G<0$. Class A: punishment with return ($\theta = 2\%$). "
       r"Calvano et al.\ targets in brackets. Deviation test not run for the controls (--).}",
       r"\label{tab:mandatory-tests}",
       r"\begin{tabular}{lcccc}", r"\toprule",
       r"Condition & $\Delta$ (95\% CI) & Nash & Unprofitable & Class A \\", r"\midrule"]
for r in table:
    lab = (r["label"].replace("β", r"$\beta$").replace("δ", r"$\delta$").replace("×", r"$\times$")
           .replace("10⁻⁶", r"$10^{-6}$").replace("10⁻⁵", r"$10^{-5}$"))
    d = f"{r['delta_mean']:.3f} ({r['delta_ci_lo']:.3f}--{r['delta_ci_hi']:.3f})"   # IC entre parênteses; alvo entre colchetes
    if r["target_delta"] is not None:
        d += f" [{r['target_delta']:.3f}]"
    nash = (fmt(r["nash_share"]) + (f" [{r['target_nash']:.3f}]" if r["target_nash"] is not None else "")
            + f" ({r['nash_any_dev_share']:.3f})")
    unp = fmt(r["unprofitable_share"]) + (f" [{r['target_unprofitable']:.3f}]"
                                         if r["target_unprofitable"] is not None else "")
    tex.append(f"{lab} & {d} & {nash} & {unp} & {fmt(r['classA_share'])} \\\\")
    if r["condition"] == "beta_1e-06":
        tex.append(r"\midrule")
tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
(TAB / "table_mandatory_tests.tex").write_text("\n".join(tex), encoding="utf-8")

# ---------------------------------------------------------------- figura 4 (texto principal)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
y = np.arange(len(table))[::-1]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True,
                             gridspec_kw={"width_ratios": [1.1, 1]})
for yi, r in zip(y, table):
    a1.errorbar(r["delta_mean"], yi, xerr=[[r["delta_mean"] - r["delta_ci_lo"]],
                                           [r["delta_ci_hi"] - r["delta_mean"]]],
                fmt="o", color="black", ms=5, capsize=2)
    a1.text(r["delta_ci_hi"] + 0.02, yi, f"{r['delta_mean']:.3f}", va="center", fontsize=8.5)
    if r["target_delta"] is not None:
        a1.plot(r["target_delta"], yi, marker="|", color="0.4", ms=12, mew=1.5)
a1.axvline(0, color="0.3", lw=0.8, ls="--")
a1.axvline(DELTA_RANDOM, color="0.5", lw=0.8, ls=":")
a1.text(DELTA_RANDOM, y.max() + 0.75, "random prices", fontsize=8, color="0.35", ha="center")
a1.text(0.01, y.max() + 0.75, "Nash", fontsize=8, color="0.3", ha="left")
a1.set_xlim(-0.05, 1.12)
a1.set_ylim(-0.6, y.max() + 1.1)
a1.set_yticks(y)
a1.set_yticklabels([r["label"].replace("control (", "control\n(") for r in table], fontsize=9)
a1.set_xlabel("Profit gain Δ (mean, 95% CI)")
a1.set_title("Prices above competitive levels", fontsize=10)
a1.axhline(y[3] - 0.5, color="0.8", lw=0.8)

for yi, r in zip(y, table):
    a2.errorbar(r["nash_share"], yi + 0.15, xerr=[[r["nash_share"] - r["nash_ci_lo"]],
                                                  [r["nash_ci_hi"] - r["nash_share"]]],
                fmt="o", color="black", ms=5, capsize=2,
                label="Nash on path, one-period deviations" if yi == y[0] else None)
    a2.plot(r["nash_any_dev_share"], yi + 0.15, marker="^", color="black", mfc="0.8", ms=6,
            ls="none", label="Nash on path, any deviation" if yi == y[0] else None)
    if r["classA_share"] is not None:
        a2.errorbar(r["classA_share"], yi - 0.15, xerr=[[r["classA_share"] - r["classA_ci_lo"]],
                                                        [r["classA_ci_hi"] - r["classA_share"]]],
                    fmt="s", color="0.45", mfc="white", ms=5, capsize=2,
                    label="Punishment with return (class A)" if yi == y[0] else None)
    if r["target_nash"] is not None:
        a2.plot(r["target_nash"], yi + 0.15, marker="|", color="0.4", ms=12, mew=1.5)
a2.set_xlim(-0.03, 1.03)
a2.set_xlabel("Share of sessions (95% Wilson CI)")
a2.set_title("Equilibrium and punishment", fontsize=10)
a2.axhline(y[3] - 0.5, color="0.8", lw=0.8)
h, l = a2.get_legend_handles_labels()
# o traço vertical "|" marca o alvo de Calvano et al.; explicado na legenda, não no eixo
h.append(plt.Line2D([], [], color="0.4", marker="|", ls="none", ms=12, mew=1.5))
l.append("Calvano et al. target")
fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=2.5)
fig.legend(h, l, frameon=False, loc="lower center", ncol=3, fontsize=8.5)
saved = []
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"fig_robustness_tests.{ext}", dpi=200)
    saved.append(f"paper/figures/fig_robustness_tests.{ext}")
plt.close(fig)

# ---------------------------------------------------------------- apêndice: sem memória por δ
diag = json.load(open(RES / "qlearning_memoryless_diagnosis.json", encoding="utf-8"))["blocks"]
deltas = sorted(float(k) for k in diag["delta"] if k != "date")


def get(dd, k):
    for key in (str(k), f"{k:.1f}", f"{k:g}"):
        if key in dd:
            return dd[key]
    raise KeyError(k)


pts = [get(diag["delta"], k) for k in deltas]
mem1 = {0.0: next(r for r in table if r["condition"] == "memoria1_d0"),
        0.95: next(r for r in table if r["condition"] == "beta_4e-06")}
fig, ax = plt.subplots(figsize=(6.5, 3.9))
# IC t a partir de média, DP e n do diagnóstico (o JSON guarda IC normal; aqui fica igual ao resto)
cis = [t_ci_summary(p["delta_mean"], p["delta_sd"], 1000) for p in pts]
ax.errorbar(deltas, [p["delta_mean"] for p in pts],
            yerr=[[c[0] - c[1] for c in cis], [c[2] - c[0] for c in cis]],
            fmt="-o", color="black", ms=5, capsize=2, lw=1.3, label="No memory (k = 0)")
ax.errorbar(list(mem1), [r["delta_mean"] for r in mem1.values()],
            yerr=[[r["delta_mean"] - r["delta_ci_lo"] for r in mem1.values()],
                  [r["delta_ci_hi"] - r["delta_mean"] for r in mem1.values()]],
            fmt="s", color="0.45", mfc="white", ms=6, capsize=2,
            label="One-period memory (k = 1): δ = 0 control and base case")
ax.axhline(0, color="0.3", lw=0.8, ls="--")
ax.axhline(DELTA_RANDOM, color="0.5", lw=0.8, ls=":")
ax.text(0.01, DELTA_RANDOM + 0.015, "random prices", fontsize=8, color="0.35")
ax.set_xlabel("Discount factor δ")
ax.set_ylabel("Profit gain Δ (mean, 95% CI)")
ax.set_ylim(-0.05, 1.05)
ax.set_title("Without memory, Δ rises with the discount factor (1,000 sessions per point)",
             fontsize=10)
ax.legend(frameon=False, loc="upper left", fontsize=8.5)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"fig_memoryless_discount.{ext}", dpi=200)
    saved.append(f"paper/figures/fig_memoryless_discount.{ext}")
plt.close(fig)

# ---------------------------------------------------------------- conferências
base_meta = json.load(open(RES / "qlearning_base_meta.json", encoding="utf-8"))["summary"]
dev_meta = json.load(open(RES / "qlearning_deviation_meta.json", encoding="utf-8"))["summary"]
ctrl_meta = json.load(open(RES / "qlearning_controls_meta.json", encoding="utf-8"))["summary"]
b = table[0]
checks = {
    "sete_condicoes_1000_sessoes": len(table) == 7 and all(r["n"] == 1000 for r in table),
    "todas_convergiram": all(r["converged"] == r["n"] for r in table),
    "delta_base_igual_fase13": abs(b["delta_mean"] - base_meta["delta_mean"]) < 1e-9,
    "nao_lucrativo_base_igual_fase14": abs(b["unprofitable_share"]
                                           - dev_meta["share_unprofitable_G_lt_0"]) < 1e-12,
    "classe_A_base_igual_fase14": round(b["classA_share"] * 1000) == dev_meta["classes"]["theta_0.02"]["A"],
    "nash_base_igual_fase15": abs(b["nash_share"] - json.load(open(RES / "qlearning_bestresponse_meta.json",
        encoding="utf-8"))["summary"]["by_version"]["calvano"]["nash_share"]) < 1e-12,
    "nash_qualquer_desvio_base_igual_fase15": abs(b["nash_any_dev_share"] - json.load(open(
        RES / "qlearning_bestresponse_meta.json", encoding="utf-8"))["summary"]["by_version"]["otima"]["nash_share"]) < 1e-12,
    "controles_iguais_fase15": all(abs(next(r for r in table if r["condition"] == c)["delta_mean"]
                                       - ctrl_meta[c]["delta_mean"]) < 1e-9
                                   for c in ("sem_memoria_d095", "memoria1_d0", "tabela_A1")),
    "arquivos_salvos": all((ROOT / p).exists() for p in saved)
                       and (TAB / "table_mandatory_tests.tex").exists(),
}
meta = {"description": "Fase 1.6, parte 3: tabela e figuras dos testes obrigatórios (Fase 1.5)",
        "date": datetime.now().isoformat(timespec="seconds"),
        "sources": ["results/qlearning_exploration_sessions.csv",
                    "results/qlearning_separation_sessions.csv",
                    "results/qlearning_controls_sessions.csv",
                    "results/qlearning_memoryless_diagnosis.json"],
        "article_plan": {"main": ["fig_delta_distribution (vira Δ por agente na Fase 4)",
                                  "fig_learning_trajectory", "fig_deviation_response",
                                  "fig_separation (revisão de 16/10/2026)"],
                         "main_tables": ["table_mandatory_tests", "table_separation"],
                         "appendix": ["fig_robustness_tests", "fig_delta_vs_periods",
                                      "fig_price_trajectories", "fig_memoryless_discount"]},
        "table": table, "figures": saved, "checks": checks}
with open(RES / "qlearning_figures_part3_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("Condição | Δ [IC 95%] | Nash | não lucrativo | classe A")
for r in table:
    print(f"  {r['label']:<42} {r['delta_mean']:.3f} [{r['delta_ci_lo']:.3f}; {r['delta_ci_hi']:.3f}]"
          f"  {r['nash_share']:.3f}  {fmt(r['unprofitable_share'])}  {fmt(r['classA_share'])}")
print("Δ sem memória por δ: " + "; ".join(f"{k:g}: {p['delta_mean']:.3f}" for k, p in zip(deltas, pts)))
for k, v in checks.items():
    print(f"Conferência {k}: {v}")
