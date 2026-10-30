"""
Fase 4 — Comparação Q-learning × LLM pelo Plano estatístico do Protocolo (Pesquisa 8),
fixado antes da 1ª execução completa.

Unidade: 1 observação por sessão (Q-learning, Δ no ciclo-limite) ou por execução (LLM, Δ nos
períodos 251–300). Execuções que pararam sem completar os 300 períodos não têm Δ: são
contadas e reportadas, não entram nas médias.

Família confirmatória: H1 P1 × Q-learning; H2 P2 × Q-learning; H3 P1 × P2.
  - Welch bicaudal; IC 95% t de cada grupo; Holm a 5% nos 3 p-valores;
  - IC das diferenças com nível de família por Bonferroni (1 − 0,05/3 = 98,3%);
  - conferência por bootstrap BCa (10.000 reamostragens, semente fixa): se t e BCa
    discordarem sobre excluir o zero, o resultado é "frágil";
  - decisão: "diferente" (IC ajustado exclui 0); "equivalente" (IC 90% de Welch dentro de
    ±0,10, TOST); "inconclusivo" nos demais casos. Nunca "igual" a partir de p > 0,05.
Descrição: n, média, DP, mediana, intervalo interquartil; frações Δ > 0,5 e Δ < 0 (Wilson).
Família secundária (desvio): fração da classe A de cada prompt × a do Q-learning, teste
binomial exato contra a proporção do Q-learning, Holm próprio. Placebo: o padrão do LLM só é
chamado de punição se o IC de Wilson da fração A ficar inteiro acima do IC do placebo.

Uso: python analysis/llm_compare.py   (lê results/llm_runs.csv e results/qlearning_*.csv;
     grava results/llm_comparison.json)
Código próprio do projeto.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RES, t_ci, wilson  # noqa: E402

ALPHA_FAMILY = 0.05
N_FAMILY = 3
EQ_MARGIN = 0.10          # margem de equivalência (Decisões de 08/10/2026)
N_BOOT = 10_000
BOOT_SEED = 20261008      # semente registrada do bootstrap
THETA_COL = "class_theta_0.02"


def describe(x) -> dict:
    x = np.asarray(x, float)
    n = len(x)
    mean, lo, hi = t_ci(x) if n > 1 else (float(x.mean()), float("nan"), float("nan"))
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    k_hi, k_lo = int((x > 0.5).sum()), int((x < 0).sum())
    return {"n": n, "mean": mean, "ci95": [lo, hi], "sd": float(x.std(ddof=1)) if n > 1 else float("nan"),
            "median": float(med), "iqr": [float(q1), float(q3)],
            "share_above_0_5": wilson(k_hi, n), "share_below_0": wilson(k_lo, n)}


def welch(x, y, level):
    """Diferença de médias (x − y), IC de Welch no nível pedido e p bicaudal."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    se = math.sqrt(vx + vy)
    df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
    d = float(x.mean() - y.mean())
    h = stats.t.ppf(1 - (1 - level) / 2, df) * se
    p = float(stats.ttest_ind(x, y, equal_var=False).pvalue)
    return d, (d - h, d + h), p, df


def bca(x, y, level):
    res = stats.bootstrap((np.asarray(x, float), np.asarray(y, float)),
                          lambda a, b, axis=-1: np.mean(a, axis=axis) - np.mean(b, axis=axis),
                          n_resamples=N_BOOT, confidence_level=level, method="BCa",
                          vectorized=True, random_state=np.random.default_rng(BOOT_SEED))
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def holm(pvals: dict, alpha=ALPHA_FAMILY) -> dict:
    """p ajustado de Holm (monótono) e decisão de rejeitar a 5%."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, run, out = len(items), 0.0, {}
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = {"p": p, "p_holm": run, "reject": run <= alpha}
    return out


def compare(groups: dict) -> dict:
    """groups: {'Q': Δs, 'P1': Δs, 'P2': Δs}. Devolve a família confirmatória completa."""
    level_fam = 1 - ALPHA_FAMILY / N_FAMILY
    pairs = {"H1": ("P1", "Q"), "H2": ("P2", "Q"), "H3": ("P1", "P2")}
    res, pv = {}, {}
    for h, (a, b) in pairs.items():
        d, ci_fam, p, df = welch(groups[a], groups[b], level_fam)
        _, ci90, _, _ = welch(groups[a], groups[b], 0.90)
        bc = bca(groups[a], groups[b], level_fam)
        excl_t = ci_fam[0] > 0 or ci_fam[1] < 0
        excl_b = bc[0] > 0 or bc[1] < 0
        if excl_t:
            decision = "diferente"
        elif -EQ_MARGIN < ci90[0] and ci90[1] < EQ_MARGIN:
            decision = "equivalente"
        else:
            decision = "inconclusivo"
        res[h] = {"comparison": f"{a} − {b}", "diff": d, "ci_family": list(ci_fam),
                  "ci_family_level": level_fam, "ci90": list(ci90), "df": df, "bca_family": list(bc),
                  "fragile": excl_t != excl_b, "decision": decision}
        pv[h] = p
    for h, v in holm(pv).items():
        res[h].update(v)
    return res


def secondary(llm: pd.DataFrame, q_share_A: float) -> dict:
    """Fração A de cada prompt × proporção do Q-learning (binomial exato) e regra do placebo."""
    out, pv = {}, {}
    for p in ("P1", "P2"):
        g = llm[(llm.prompt == p) & llm[THETA_COL].notna()]
        n, k = len(g), int((g[THETA_COL] == "A").sum())
        if n == 0:
            continue
        kp = int(g["placebo_theta_0.02"].astype(str).str.lower().eq("true").sum())
        fa, fp = wilson(k, n), wilson(kp, n)
        out[p] = {"n": n, "share_A": fa, "placebo": fp,
                  "punishment_pattern": fa[1] > fp[2],      # IC de A inteiro acima do IC do placebo
                  "q_share_A": q_share_A}
        pv[p] = float(stats.binomtest(k, n, q_share_A).pvalue)
    for p, v in holm(pv).items():
        out[p].update(v)
    return out


def main():
    llm = pd.read_csv(RES / "llm_runs.csv")
    q = pd.read_csv(RES / "qlearning_base_sessions.csv")
    qd = pd.read_csv(RES / "qlearning_deviation_sessions.csv")
    complete = llm[(llm.status == "done") & llm["delta"].notna()]
    groups = {"Q": q["delta"].to_numpy(), "P1": complete[complete.prompt == "P1"]["delta"].to_numpy(),
              "P2": complete[complete.prompt == "P2"]["delta"].to_numpy()}
    q_share_A = float((qd["class_theta2"] == "A").mean())
    out = {"n_runs": int(len(llm)), "n_complete": int(len(complete)),
           "incomplete": llm.loc[~llm.index.isin(complete.index), ["run_id", "status", "reason"]].to_dict("records"),
           "describe": {k: describe(v) for k, v in groups.items()},
           "describe_stabilized_only": {p: describe(complete[(complete.prompt == p) & complete.stabilized.astype(str).str.lower().eq("true")]["delta"])
                                        for p in ("P1", "P2")
                                        if complete[(complete.prompt == p) & complete.stabilized.astype(str).str.lower().eq("true")].shape[0] > 1},
           "confirmatory": compare(groups),
           "secondary_deviation": secondary(complete, q_share_A),
           "rules": {"family_alpha": ALPHA_FAMILY, "eq_margin": EQ_MARGIN, "n_boot": N_BOOT, "boot_seed": BOOT_SEED}}
    (RES / "llm_comparison.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    for h, r in out["confirmatory"].items():
        print(f"{h} {r['comparison']}: diferença {r['diff']:+.3f}, IC {100 * r['ci_family_level']:.1f}% "
              f"[{r['ci_family'][0]:+.3f}; {r['ci_family'][1]:+.3f}], p Holm {r['p_holm']:.4f} → {r['decision']}"
              f"{' (frágil)' if r['fragile'] else ''}")


if __name__ == "__main__":
    main()
