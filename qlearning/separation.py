"""
Teste de separação (Fase 1.8, decidido na revisão da Fase 1 em 16/10/2026).

Pergunta: os preços altos do caso base vêm da possibilidade de punir (memória) ou da mesma
falha de aprendizado que deixa o controle sem memória em preço alto (Δ 0,957; Fase 1.5)?
Desenho: pares de condições que diferem SÓ na memória (k = 0 × k = 1), com o mesmo δ = 0,95,
α = 0,15, Q inicial da eq. 8 e o mesmo esquema de exploração, cada vez mais longo:
  - exploração decrescente ε = e^(−βt), β = 1e-5, 4e-6 (caso base), 2e-6 e 1e-6, até convergir;
  - exploração com piso ε = max(e^(−βt), ε mínimo), ε mínimo = 0,1%, 1% e 5%, por 4 milhões de
    períodos fixos (a regra de parada não se aplica, porque a exploração nunca some).
Se a falha de aprendizado explica os preços altos, mais aprendizado derruba os dois braços;
se a punição sustenta parte deles, o braço com memória fica acima do sem memória.

Por sessão: convergiu, períodos, Δ do ciclo-limite da estratégia gulosa no momento em que a
sessão para (igual ao Δ usual quando converge), tamanho do ciclo, melhor resposta (Nash no
caminho na leitura de Calvano, de 1 período, e com a melhor resposta ótima, a qualquer desvio),
Q-loss no caminho e, com memória, o teste de desvio comum (R, G, V; Fase 1.4).
Código próprio do projeto; a sessão é run_session_mem de controls.py, que com k = 1 e
ε mínimo = 0 reproduz session.py (conferido no runner).
"""

import numpy as np
from numba import njit, prange

from best_response import best_response_one
from controls import best_response_mem, cycle_actions, run_session_mem
from deviation import metrics_one_start


@njit(parallel=True, cache=True)
def run_many_sep(seeds, PI, Q0, alpha, delta, beta, t_stable, t_max, pi_nash, pi_mon,
                 prices, step, memory, eps_min):
    n = seeds.shape[0]
    m = PI.shape[0]
    conv = np.zeros(n, dtype=np.bool_)
    periods = np.zeros(n, dtype=np.int64)
    dg = np.zeros(n)
    clen = np.zeros(n, dtype=np.int64)
    nash1 = np.zeros(n, dtype=np.bool_)        # Nash no caminho, desvios de 1 período
    nash_opt = np.zeros(n, dtype=np.bool_)     # Nash no caminho, qualquer desvio
    qlp = np.full(n, np.nan)
    R = np.full(n, np.nan)
    G = np.full(n, np.nan)
    V = np.full(n, np.nan)
    for k in prange(n):
        ok, t, d, nc, pi_, greedy, s_end, Q = run_session_mem(
            seeds[k], PI, Q0, alpha, delta, beta, t_stable, t_max, pi_nash, pi_mon,
            memory, eps_min)
        conv[k] = ok
        periods[k] = t
        cyc, acts = cycle_actions(greedy, s_end, m, memory)
        L = cyc.shape[0]
        clen[k] = L
        pbar = 0.0
        for j in range(L):
            pbar += 0.5 * (PI[acts[j, 0], acts[j, 1], 0] + PI[acts[j, 0], acts[j, 1], 1])
        dg[k] = (pbar / L - pi_nash) / (pi_mon - pi_nash)
        if memory == 0:
            # Sem memória o rival não reage (R = 0 por construção) e, com um estado só,
            # a melhor resposta de 1 período coincide com a ótima.
            nh, lp, la = best_response_mem(greedy, PI, delta, cyc, m, memory)
            nash1[k] = nh
            nash_opt[k] = nh
            qlp[k] = lp
            continue
        bp, ba, lp, la = best_response_one(greedy, PI, delta, cyc, False)
        nash1[k] = bp[0] and bp[1]
        qlp[k] = lp
        bp2, ba2, lp2, la2 = best_response_one(greedy, PI, delta, cyc, True)
        nash_opt[k] = bp2[0] and bp2[1]
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
    return conv, periods, dg, clen, nash1, nash_opt, qlp, R, G, V
