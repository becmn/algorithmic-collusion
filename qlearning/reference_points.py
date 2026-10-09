"""
Fase 1.1 — Pontos de referência do mercado de Calvano et al. (2020).

Calcula Nash, monopólio conjunto, monopólio de 1 firma e a grade de preços,
confere com a Ficha de parâmetros do documento do projeto e roda os testes
de sanidade de Δ. Código próprio do projeto (não derivado de Courthoud nem
de Schildknecht).

Como rodar (na raiz do repositório, com o venv ativo):
    python qlearning/reference_points.py
"""

import itertools

import numpy as np
from scipy.optimize import brentq, minimize_scalar

# ---------------------------------------------------------------------------
# Parâmetros do caso base (Ficha de parâmetros; Calvano et al., p. 3274)
# ---------------------------------------------------------------------------
PARAMS = dict(
    n=2,        # número de firmas
    c=1.0,      # custo marginal
    a=2.0,      # qualidade do produto (a − c = 1)
    a0=0.0,     # bem externo
    mu=0.25,    # diferenciação horizontal
    delta=0.95, # fator de desconto
    m=15,       # preços na grade
    xi=0.1,     # extensão da grade além de Nash e monopólio
)


# ---------------------------------------------------------------------------
# Demanda logit e lucro (eq. 5)
# ---------------------------------------------------------------------------
def shares(prices, a, a0, mu):
    """Fatia de mercado q_i de cada firma, dado o vetor de preços."""
    prices = np.asarray(prices, dtype=float)
    e = np.exp((a - prices) / mu)
    return e / (e.sum() + np.exp(a0 / mu))


def profits(prices, a, a0, mu, c):
    """Lucro por período π_i = (p_i − c) q_i de cada firma."""
    prices = np.asarray(prices, dtype=float)
    return (prices - c) * shares(prices, a, a0, mu)


# ---------------------------------------------------------------------------
# Pontos de referência
# ---------------------------------------------------------------------------
def best_response(p_rival, a, a0, mu, c):
    """Melhor resposta contínua da firma 1 a um preço fixo da firma 2."""
    res = minimize_scalar(
        lambda p: -profits([p, p_rival], a, a0, mu, c)[0],
        bounds=(c, c + 5), method="bounded", options={"xatol": 1e-12},
    )
    return res.x


def nash_price(a, a0, mu, c, n):
    """Nash simétrico: p − c = μ / (1 − q), com q a fatia de cada firma."""
    def foc(p):
        q = shares([p] * n, a, a0, mu)[0]
        return (p - c) - mu / (1 - q)
    return brentq(foc, c + 1e-9, c + 5)


def joint_monopoly_price(a, a0, mu, c, n):
    """Preço igual nas n firmas que maximiza o lucro somado."""
    res = minimize_scalar(
        lambda p: -profits([p] * n, a, a0, mu, c).sum(),
        bounds=(c, c + 5), method="bounded", options={"xatol": 1e-12},
    )
    return res.x


def single_firm_monopoly_price(a, a0, mu, c):
    """Preço ótimo de uma firma sozinha no mercado (com o bem externo)."""
    res = minimize_scalar(
        lambda p: -profits([p], a, a0, mu, c)[0],
        bounds=(c, c + 5), method="bounded", options={"xatol": 1e-12},
    )
    return res.x


def price_grid(pN, pM, m, xi):
    """m preços igualmente espaçados em [pN − ξ(pM − pN), pM + ξ(pM − pN)]."""
    span = pM - pN
    return np.linspace(pN - xi * span, pM + xi * span, m)


def delta_index(pi_bar, piN, piM):
    """Índice de ganho de lucro Δ (eq. 9), com πN e πM contínuos."""
    return (pi_bar - piN) / (piM - piN)


# ---------------------------------------------------------------------------
# Execução e conferência
# ---------------------------------------------------------------------------
def check(nome, calculado, esperado, tol):
    ok = abs(calculado - esperado) <= tol
    print(f"  {'OK  ' if ok else 'FALHA'} {nome:<38} calculado {calculado:>9.4f}"
          f"   Ficha {esperado:>8.4f}   (tol {tol})")
    return ok


def main():
    P = PARAMS
    a, a0, mu, c, n = P["a"], P["a0"], P["mu"], P["c"], P["n"]
    resultados = []

    # 1. Nash e monopólio
    pN = nash_price(a, a0, mu, c, n)
    pM = joint_monopoly_price(a, a0, mu, c, n)
    p1 = single_firm_monopoly_price(a, a0, mu, c)
    qN, piN = shares([pN] * n, a, a0, mu)[0], profits([pN] * n, a, a0, mu, c)[0]
    qM, piM = shares([pM] * n, a, a0, mu)[0], profits([pM] * n, a, a0, mu, c)[0]

    print("\n1. Pontos de referência (Ficha de parâmetros)")
    resultados += [
        check("Nash: preço", pN, 1.4729, 5e-5),
        check("Nash: fatia q", qN, 0.4714, 5e-5),
        check("Nash: lucro por firma (πN)", piN, 0.2229, 5e-5),
        check("Monopólio conjunto: preço", pM, 1.9250, 5e-5),
        check("Monopólio conjunto: fatia q", qM, 0.3649, 5e-5),
        check("Monopólio conjunto: lucro (πM)", piM, 0.3375, 5e-5),
        check("Monopólio de 1 firma: preço", p1, 1.802, 5e-4),
    ]

    # 2. Grade
    grid = price_grid(pN, pM, P["m"], P["xi"])
    print("\n2. Grade de preços")
    resultados += [
        check("Grade: 1º preço", grid[0], 1.4277, 5e-5),
        check("Grade: 15º preço", grid[-1], 1.9702, 5e-5),
        check("Grade: passo", grid[1] - grid[0], 0.0387, 5e-5),
        check("Grade: 10º preço", grid[9], 1.7764, 5e-5),
    ]
    ok_pos = grid[1] < pN < grid[2] and grid[12] < pM < grid[13]
    print(f"  {'OK  ' if ok_pos else 'FALHA'} Nash entre 2ª e 3ª posição; monopólio entre 13ª e 14ª")
    resultados.append(ok_pos)

    # 3. Verificações indiretas do artigo
    pD = best_response(pM, a, a0, mu, c)                  # desvio ao monopólio
    piD = profits([pD, pM], a, a0, mu, c)[0]
    delta_crit = (piD - piM) / (piD - piN)                # grim trigger
    print("\n3. Verificações indiretas do artigo")
    resultados += [
        check("Margem (p − c)/c em Nash (≈ 47%)", (pN - c) / c, 0.473, 5e-4),
        check("Margem (p − c)/c no monopólio (≈ 2×)", (pM - c) / c, 0.925, 5e-4),
        check("δ crítico do grim trigger (≈ 0,39)", delta_crit, 0.39, 5e-3),
    ]
    print(f"       (nota 21 = testes de sanidade das posições 2 e 3, abaixo)")

    # 4. Testes de sanidade de Δ
    def delta_fixo(i):
        pi = profits([grid[i]] * n, a, a0, mu, c).mean()
        return delta_index(pi, piN, piM)

    pares = itertools.product(grid, repeat=n)
    pi_acaso = np.mean([profits(list(par), a, a0, mu, c).mean() for par in pares])

    print("\n4. Testes de sanidade de Δ")
    resultados += [
        check("Δ, as duas firmas na 2ª posição", delta_fixo(1), -0.024, 5e-4),
        check("Δ, as duas firmas na 3ª posição", delta_fixo(2), 0.117, 5e-4),
        check("Δ, as duas firmas na 14ª posição", delta_fixo(13), 1.000, 5e-4),
        check("Δ, preços sorteados na grade", delta_index(pi_acaso, piN, piM), 0.497, 5e-4),
    ]

    total, ok = len(resultados), sum(resultados)
    print(f"\nResultado: {ok} de {total} conferências OK.")
    print(f"Valores completos: pN = {pN:.6f}, πN = {piN:.6f}, pM = {pM:.6f}, "
          f"πM = {piM:.6f}, p1 = {p1:.6f}, δ* = {delta_crit:.4f}")


if __name__ == "__main__":
    main()
