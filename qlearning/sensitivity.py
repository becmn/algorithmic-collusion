"""
Sensibilidade à exploração (Fase 1.5, parte 3a; Protocolo; Pesquisa 5, proposta P4).
Roda cada sessão uma vez (mesma função da Fase 1.3) e aplica nela os dois testes já
validados: o teste de desvio comum (deviation.py, Fase 1.4) e o teste de melhor resposta
na leitura de Calvano (best_response.py, Fase 1.5, parte 1).
Código próprio do projeto.
"""

import numpy as np
from numba import njit, prange

from best_response import best_response_one
from deviation import T_DEV, metrics_one_start
from session import limit_cycle, run_session


@njit(parallel=True, cache=True)
def run_many_full(seeds, PI, Q0, alpha, delta, beta, t_stable, t_max, pi_nash, pi_mon,
                  prices, step):
    """Por sessão: convergiu, períodos, Δ, tamanho do ciclo, preço médio (índice), Nash no
    caminho (leitura de Calvano), Q-loss no caminho e em todos os estados, e as médias sobre
    o ciclo de R, G e V do teste de desvio (V = −1 se algum ponto do ciclo for censurado)."""
    n = seeds.shape[0]
    m = PI.shape[0]
    conv = np.zeros(n, dtype=np.bool_)
    periods = np.zeros(n, dtype=np.int64)
    dlt = np.zeros(n)
    clen = np.zeros(n, dtype=np.int64)
    pidx = np.zeros(n)
    nash = np.zeros(n, dtype=np.bool_)
    qlp = np.full(n, np.nan)
    qla = np.full(n, np.nan)
    R = np.full(n, np.nan)
    G = np.full(n, np.nan)
    V = np.full(n, np.nan)
    for k in prange(n):
        ok, t, d, nc, pi_, Q, greedy, s_end = run_session(
            seeds[k], PI, Q0, alpha, delta, beta, t_stable, t_max, pi_nash, pi_mon)
        conv[k] = ok
        periods[k] = t
        dlt[k] = d
        clen[k] = nc
        pidx[k] = pi_
        if not ok:
            continue
        cyc = limit_cycle(greedy, s_end, m)
        L = cyc.shape[0]
        bp, ba, lp, la = best_response_one(greedy, PI, delta, cyc, False)
        nash[k] = bp[0] and bp[1]
        qlp[k] = lp
        qla[k] = la
        in_cycle = np.zeros(m * m, dtype=np.bool_)
        for j in range(L):
            in_cycle[cyc[j]] = True
        sR = 0.0
        sG = 0.0
        sV = 0.0
        cens = False
        for j in range(L):
            r, g, v, vp, didv, rc, cu = metrics_one_start(greedy, PI, prices, delta, step,
                                                          cyc[j], in_cycle, L > 1)
            sR += r
            sG += g
            if v < 0:
                cens = True
            else:
                sV += v
        R[k] = sR / L
        G[k] = sG / L
        V[k] = -1.0 if cens else sV / L
    return conv, periods, dlt, clen, pidx, nash, qlp, qla, R, G, V
