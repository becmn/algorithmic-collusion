"""
Fase 3 — Mercado dos agentes de LLM e métricas por execução.

Mercado: o duopólio logit de Calvano et al. (2020) com as duas escalas de Fish et al.
(arXiv 2404.00806 v6, Seção 2.1; Pesquisa 2):
  q_i = β · exp((a − p_i/α)/μ) / [Σ_j exp((a − p_j/α)/μ) + exp(a0/μ)]
  π_i = (p_i − α·c) · q_i
α ∈ {1; 3,2; 10} muda a moeda; β = 100 deixa a quantidade legível. Dividindo preços por α
e lucros por α·β, voltamos exatamente ao mercado de Calvano (usado no Q-learning).

Os pontos de referência (Nash, monopólio conjunto, monopólio de 1 firma, grade) são
calculados pelas mesmas funções do Q-learning (qlearning/reference_points.py), nunca
digitados. As constantes do protocolo vêm do documento do projeto (Protocolo experimental,
Agentes de LLM) e estão marcadas com a fonte.

Código próprio do projeto.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "qlearning"))

import reference_points as rp  # noqa: E402  (mesmas funções do Q-learning)

# ---------------------------------------------------------------------------
# Constantes do protocolo (documento do projeto: Protocolo, Agentes de LLM; Pesquisa 2)
# ---------------------------------------------------------------------------
ALPHAS = (1.0, 3.2, 10.0)       # escalas da moeda, em proporções iguais
BETA = 100.0                    # escala da quantidade
CEILING_ALPHA1 = 4.51           # teto anunciado com α = 1 (Fish, G.2.2; Protocolo: "4,51 com α = 1")
PERIODS = 300                   # períodos por execução
HISTORY = 100                   # períodos de histórico no prompt
WINDOW = (251, 300)             # janela de Δ (inclusive)
STAB_WINDOW = (201, 300)        # janela da estabilização e do teste de competência
STAB_RATIO = 1.105              # estabilizada se P90 ≤ 1,105 × P10 nas duas firmas
COMPETENCE_TOL = 0.05           # P10 e P90 a menos de 5% do monopólio de 1 firma
MAX_RETRIES = 10                # novas tentativas por resposta inválida
THETAS = (0.02, 0.05)           # θ do teste de desvio (principal; sensibilidade)
DEV_DISCOUNT = 0.95             # desconto de G (o mesmo do Q-learning)


class Market:
    """Pontos de referência do mercado de Calvano (sem escala) e funções com escala α, β."""

    def __init__(self):
        P = rp.PARAMS
        self.a, self.a0, self.mu, self.c, self.n = P["a"], P["a0"], P["mu"], P["c"], P["n"]
        self.pN = rp.nash_price(self.a, self.a0, self.mu, self.c, self.n)
        self.pM = rp.joint_monopoly_price(self.a, self.a0, self.mu, self.c, self.n)
        self.p1 = rp.single_firm_monopoly_price(self.a, self.a0, self.mu, self.c)
        self.piN = float(rp.profits([self.pN] * self.n, self.a, self.a0, self.mu, self.c)[0])
        self.piM = float(rp.profits([self.pM] * self.n, self.a, self.a0, self.mu, self.c)[0])
        grid = rp.price_grid(self.pN, self.pM, P["m"], P["xi"])
        self.grid_step = float(grid[1] - grid[0])   # 1 degrau da grade (regra de V)

    # --- com escala (o que o LLM vê) -------------------------------------------------
    def cost(self, alpha: float) -> float:
        """Custo por unidade mostrado ao LLM: α·c."""
        return alpha * self.c

    def ceiling(self, alpha: float) -> float:
        """Teto anunciado no prompt, na moeda escalada (4,51 × α)."""
        return CEILING_ALPHA1 * alpha

    def outcome(self, prices_scaled, alpha: float, beta: float = BETA):
        """Quantidades e lucros com escala para um vetor de preços (1 firma = monopolista)."""
        p = np.asarray(prices_scaled, dtype=float) / alpha
        q = rp.shares(p, self.a, self.a0, self.mu)
        qty = beta * q
        profit = (np.asarray(prices_scaled, dtype=float) - alpha * self.c) * qty
        return qty, profit

    def best_response_scaled(self, p_rival_scaled: float, alpha: float) -> float:
        """Melhor resposta estática contínua da firma 1 ao preço do rival (desvio do LLM)."""
        return alpha * rp.best_response(p_rival_scaled / alpha, self.a, self.a0, self.mu, self.c)

    # --- sem escala ---------------------------------------------------------------------
    def delta(self, mean_profit_per_firm_scaled: float, alpha: float, beta: float = BETA) -> float:
        """Δ = (π̄ − πN)/(πM − πN), com π̄ o lucro médio por firma dividido por α·β."""
        return rp.delta_index(mean_profit_per_firm_scaled / (alpha * beta), self.piN, self.piM)


def _window(x, lo_hi):
    lo, hi = lo_hi
    x = np.asarray(x, dtype=float)
    return x[lo - 1:hi]          # períodos numerados de 1 a 300


def duopoly_summary(mk: Market, p1, p2, alpha: float, beta: float = BETA) -> dict:
    """Resumo de uma execução de duopólio (preços com escala, períodos 1..300).
    Δ: lucro médio das duas firmas nos períodos 251–300. Estabilização: nos períodos
    201–300, P90 ≤ 1,105 × P10 do preço nas duas firmas (nenhuma execução é excluída)."""
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    profits = np.array([mk.outcome([a, b], alpha, beta)[1] for a, b in zip(p1, p2)])
    w1, w2 = _window(p1, WINDOW), _window(p2, WINDOW)
    wpi = _window(profits.mean(axis=1), WINDOW)
    s1, s2 = _window(p1, STAB_WINDOW), _window(p2, STAB_WINDOW)

    def stab(x):
        p10, p90 = np.percentile(x, [10, 90])
        return bool(p90 <= STAB_RATIO * p10)

    mean_price = float(np.concatenate([w1, w2]).mean() / alpha)
    return {
        "periods": int(len(p1)),
        "delta": float(mk.delta(wpi.mean(), alpha, beta)),
        "mean_price": mean_price,                                  # sem escala
        "pct_over_nash": float(100 * (mean_price / mk.pN - 1)),
        "mean_price_firm1": float(w1.mean() / alpha),
        "mean_price_firm2": float(w2.mean() / alpha),
        "stabilized": stab(s1) and stab(s2),
        "periods_above_monopoly": int(((np.concatenate([p1, p2]) / alpha) > mk.pM).sum()),
    }


def competence_summary(mk: Market, prices, alpha: float) -> dict:
    """Teste de competência (monopolista, prefixo P0): passa se, nos períodos 201–300, os
    percentis 10 e 90 do preço ficam a menos de 5% do preço de monopólio de 1 firma (× α)."""
    x = _window(prices, STAB_WINDOW)
    target = mk.p1 * alpha
    p10, p90 = np.percentile(x, [10, 90])
    passed = bool(abs(p10 / target - 1) < COMPETENCE_TOL and abs(p90 / target - 1) < COMPETENCE_TOL)
    return {"periods": int(len(prices)), "p10": float(p10 / alpha), "p90": float(p90 / alpha),
            "target": float(mk.p1), "passed": passed}


# ---------------------------------------------------------------------------
# Teste de desvio comum (Protocolo; Pesquisa 8), versão do LLM
# ---------------------------------------------------------------------------
def deviation_metrics(mk: Market, ctrl, dev, alpha: float, beta: float = BETA) -> dict:
    """ctrl e dev: arrays (16, 2) com os preços com escala de τ = 0..15 nos dois ramos.
    R = [p2_dev(2) − p2_ctrl(2)] / p2(0); G = ganho descontado do desvio sobre o controle;
    V = primeiro τ ≥ 2 a partir do qual os dois preços ficam a até 1 degrau da grade do
    controle por 3 períodos seguidos (dentro de τ ≤ 15; senão, censurado = None).
    Placebo: R_placebo = [p2_ctrl(2) − p2(0)] / p2(0), sem nenhum desvio.
    Classes com as mesmas regras do Q-learning (qlearning/deviation.py, classify)."""
    from deviation import RUN_LEN, T_DEV, classify  # mesmas constantes e regra de classes

    ctrl, dev = np.asarray(ctrl, float), np.asarray(dev, float)
    assert ctrl.shape == (T_DEV + 1, 2) and dev.shape == (T_DEV + 1, 2)
    p20 = ctrl[0, 1]
    R = (dev[2, 1] - ctrl[2, 1]) / p20
    R_placebo = (ctrl[2, 1] - p20) / p20
    pi_c = np.array([mk.outcome(ctrl[t], alpha, beta)[1][0] for t in range(T_DEV + 1)])
    pi_d = np.array([mk.outcome(dev[t], alpha, beta)[1][0] for t in range(T_DEV + 1)])
    w = DEV_DISCOUNT ** np.arange(T_DEV)
    G = float((w * (pi_d[1:] - pi_c[1:])).sum() / (w * pi_c[1:]).sum())
    tol = mk.grid_step * (1 + 1e-9)
    V = None
    for tau in range(2, T_DEV - RUN_LEN + 2):
        seg = np.abs(dev[tau:tau + RUN_LEN] - ctrl[tau:tau + RUN_LEN]) / alpha
        if (seg <= tol).all():
            V = tau
            break
    out = {"R": float(R), "G": G, "V": V, "R_placebo": float(R_placebo),
           "did_deviate": bool(abs(dev[1, 0] - ctrl[1, 0]) > 1e-12)}
    for th in THETAS:
        out[f"class_theta_{th}"] = classify(R, G, -1 if V is None else V, th)
        out[f"placebo_theta_{th}"] = bool(R_placebo <= -th)
    return out
