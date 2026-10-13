"""
Fase 1.6, parte 2: trajetórias de aprendizado e caminhos de preço do teste de desvio.

Duas funções compiladas com numba:
  - run_session_traced: a mesma sessão de session.run_session (mesmo código, mesma ordem
    de sorteios, logo o mesmo resultado para a mesma semente), que além disso soma, em
    blocos de períodos com tamanho crescente (escala logarítmica), o preço de cada firma
    e o lucro médio das duas. A média de cada bloco é uma média móvel cuja janela cresce
    com o tempo, para caberem milhões de períodos num eixo logarítmico.
  - run_many_traced: roda várias sessões em paralelo (prange) e, nas que convergem,
    guarda os caminhos de preço dos dois ramos do teste de desvio (Fase 1.4) para
    τ = 0 a 15, com média sobre os pontos do ciclo-limite (1 observação por sessão).
Código próprio do projeto; o laço da sessão repete o de session.py (adaptado de
Courthoud, MIT; aviso em THIRD_PARTY_LICENSES.md) só para registrar os blocos.
"""

import numpy as np
from numba import njit, prange

from deviation import T_DEV, deviation_paths
from session import _row_argmax_max, limit_cycle


def log_block_edges(t_end, n_blocks):
    """Limites dos blocos: 0, 1, 2, ... até t_end, com espaçamento logarítmico
    (blocos de largura ≥ 1, sem repetição)."""
    e = np.unique(np.floor(np.geomspace(1, t_end, n_blocks)).astype(np.int64))
    return np.concatenate(([0], e)).astype(np.int64)


@njit(cache=True)
def run_session_traced(seed, PI, Q0, alpha, delta, beta, t_stable, t_max, pi_nash, pi_mon,
                       prices, edges):
    """Igual a session.run_session, mais as somas por bloco.
    Devolve: convergiu, períodos, Δ, tamanho do ciclo, estratégia gulosa final,
    estado final, soma do preço da firma 1, soma do preço da firma 2, soma do lucro médio
    das duas firmas e número de períodos de cada bloco."""
    np.random.seed(seed)
    m = PI.shape[0]
    S = m * m
    Q = Q0.copy()
    nb = edges.shape[0] - 1
    sp1 = np.zeros(nb)
    sp2 = np.zeros(nb)
    spi = np.zeros(nb)
    cnt = np.zeros(nb, dtype=np.int64)

    greedy = np.empty((2, S), dtype=np.int64)
    maxq = np.empty((2, S))
    for i in range(2):
        for s in range(S):
            g, v = _row_argmax_max(Q[i, s])
            greedy[i, s] = g
            maxq[i, s] = v

    s = np.random.randint(0, S)
    stable = 0
    ring = np.zeros(t_stable)
    t = 0
    converged = False
    act = np.empty(2, dtype=np.int64)
    b = 0

    while t < t_max:
        eps = np.exp(-beta * t)
        for i in range(2):
            if np.random.random() < eps:
                act[i] = np.random.randint(0, m)
            else:
                act[i] = greedy[i, s]
        s1 = act[0] * m + act[1]

        changed = False
        for i in range(2):
            pi_i = PI[act[0], act[1], i]
            target = pi_i + delta * maxq[i, s1]
            Q[i, s, act[i]] = (1.0 - alpha) * Q[i, s, act[i]] + alpha * target
            g, v = _row_argmax_max(Q[i, s])
            if g != greedy[i, s]:
                changed = True
                greedy[i, s] = g
            maxq[i, s] = v

        pbar_t = 0.5 * (PI[act[0], act[1], 0] + PI[act[0], act[1], 1])
        ring[t % t_stable] = pbar_t

        # registro por bloco (não afeta a sessão)
        while b < nb - 1 and t >= edges[b + 1]:
            b += 1
        if t < edges[nb]:
            sp1[b] += prices[act[0]]
            sp2[b] += prices[act[1]]
            spi[b] += pbar_t
            cnt[b] += 1

        if changed:
            stable = 0
        else:
            stable += 1
        s = s1
        t += 1
        if stable >= t_stable:
            converged = True
            break

    if converged:
        cyc = limit_cycle(greedy, s, m)
        n = cyc.shape[0]
        pbar = 0.0
        for k in range(n):
            a1 = cyc[k] // m
            a2 = cyc[k] % m
            pbar += 0.5 * (PI[a1, a2, 0] + PI[a1, a2, 1])
        pbar /= n
    else:
        n = 0
        pbar = ring.mean()

    delta_idx = (pbar - pi_nash) / (pi_mon - pi_nash)
    return converged, t, delta_idx, n, greedy, s, sp1, sp2, spi, cnt


@njit(parallel=True, cache=True)
def run_many_traced(seeds, PI, Q0, alpha, delta, beta, t_stable, t_max, pi_nash, pi_mon,
                    prices, edges):
    """Várias sessões em paralelo. Por sessão: convergiu, períodos, Δ, tamanho do ciclo,
    somas e contagens por bloco e os caminhos do teste de desvio
    paths[k, ramo, firma, τ] (ramo 0 = controle, 1 = com desvio), em preço, com média
    sobre os pontos do ciclo-limite, e a curva R(τ) = p2_desvio(τ)/p2_controle(τ) − 1 da
    sessão (média sobre os pontos do ciclo, como em deviation.py)."""
    n = seeds.shape[0]
    m = PI.shape[0]
    nb = edges.shape[0] - 1
    conv = np.zeros(n, dtype=np.bool_)
    per = np.zeros(n, dtype=np.int64)
    dlt = np.zeros(n)
    clen = np.zeros(n, dtype=np.int64)
    SP1 = np.zeros((n, nb))
    SP2 = np.zeros((n, nb))
    SPI = np.zeros((n, nb))
    CNT = np.zeros((n, nb), dtype=np.int64)
    paths = np.full((n, 2, 2, T_DEV + 1), np.nan)
    rcurve = np.full((n, T_DEV + 1), np.nan)
    for k in prange(n):
        ok, t, d, nc, greedy, s_end, sp1, sp2, spi, cnt = run_session_traced(
            seeds[k], PI, Q0, alpha, delta, beta, t_stable, t_max, pi_nash, pi_mon,
            prices, edges)
        conv[k] = ok
        per[k] = t
        dlt[k] = d
        clen[k] = nc
        SP1[k] = sp1
        SP2[k] = sp2
        SPI[k] = spi
        CNT[k] = cnt
        if not ok:
            continue
        cyc = limit_cycle(greedy, s_end, m)
        L = cyc.shape[0]
        acc = np.zeros((2, 2, T_DEV + 1))
        racc = np.zeros(T_DEV + 1)
        for j in range(L):
            ctrl, dev = deviation_paths(greedy, PI, cyc[j])
            for tau in range(T_DEV + 1):
                racc[tau] += prices[dev[tau, 1]] / prices[ctrl[tau, 1]] - 1.0
                for i in range(2):
                    acc[0, i, tau] += prices[ctrl[tau, i]]
                    acc[1, i, tau] += prices[dev[tau, i]]
        for r in range(2):
            for i in range(2):
                for tau in range(T_DEV + 1):
                    paths[k, r, i, tau] = acc[r, i, tau] / L
        for tau in range(T_DEV + 1):
            rcurve[k, tau] = racc[tau] / L
    return conv, per, dlt, clen, SP1, SP2, SPI, CNT, paths, rcurve
