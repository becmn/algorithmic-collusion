"""
Teste de melhor resposta (Fase 1.5, parte 1), como em Calvano et al. (2020), Seção III.D e
Tabela 1 (Pesquisa 1; Pesquisa 5, proposta P2, aprovada no Protocolo).

Duas versões da Q "verdadeira" de cada agente, com o rival jogando para sempre a sua
estratégia final (gulosa, Q congelada, ε = 0); b = preço do rival no estado s e s' = (a, b):
  1. Leitura de Calvano (principal): o valor de continuação da eq. (3) é fixado pela
     estratégia-limite do PRÓPRIO agente, o que dá um sistema linear (o artigo fala em
     "sistema linear"):  Q(s, a) = π(a, b) + δ · Q(s', σ(s')).
     Testa desvios de UM período: desviar uma vez e depois voltar à própria estratégia.
     Foi a leitura que reproduziu a Tabela 1 no teste de 13/10/2026.
  2. Melhor resposta ótima (exploratória, mais exigente): Q*(s, a) = π(a, b) + δ · max Q*(s', ·),
     por iteração de valor; testa qualquer desvio, inclusive de vários períodos.
Nas duas, resolvido por iteração até a diferença ficar abaixo de 1e-12. Depois:
  - Nash no caminho: nos estados do ciclo-limite, os DOIS agentes jogam a melhor ação.
  - Equilíbrio perfeito em subjogos ("SPE"): o mesmo em todos os 225 estados.
  - Q-loss (leitura adotada pelo projeto; o artigo diz só "em %"): em cada estado,
    [max_a Q(s, a) − Q(s, σ(s))] / max_a Q(s, a); média sobre os estados (do caminho ou
    todos) e sobre os dois agentes.
Empates: a ação aprendida conta como melhor resposta se ficar a menos de 1e-9 (relativo)
do máximo. Código próprio do projeto (não derivado de Courthoud nem de Schildknecht).
"""

import numpy as np
from numba import njit, prange

from session import limit_cycle, run_session

TOL_VI = 1e-12   # parada da iteração de valor
TOL_BR = 1e-9    # tolerância relativa para considerar a ação aprendida uma melhor resposta


@njit(cache=True)
def true_q(greedy, PI, delta, i, optimal):
    """Q verdadeira do agente i (0 ou 1) contra a estratégia final do rival.
    optimal=False: continuação pela própria estratégia final (leitura de Calvano).
    optimal=True: continuação ótima (melhor resposta a qualquer desvio)."""
    m = PI.shape[0]
    S = m * m
    Q = np.zeros((S, m))
    V = np.zeros(S)
    for it in range(100_000):
        diff = 0.0
        for s in range(S):
            b = greedy[1 - i, s]                 # preço do rival neste estado
            for a in range(m):
                if i == 0:
                    r = PI[a, b, 0]
                    s1 = a * m + b
                else:
                    r = PI[b, a, 1]
                    s1 = b * m + a
                q = r + delta * V[s1]
                d = abs(q - Q[s, a])
                if d > diff:
                    diff = d
                Q[s, a] = q
        for s in range(S):
            if optimal:
                v = Q[s, 0]
                for a in range(1, m):
                    if Q[s, a] > v:
                        v = Q[s, a]
            else:
                v = Q[s, greedy[i, s]]
            V[s] = v
        if diff < TOL_VI:
            break
    return Q


@njit(cache=True)
def check_state(Q, s, a_learned):
    """(é melhor resposta?, perda relativa) no estado s."""
    m = Q.shape[1]
    best = Q[s, 0]
    for a in range(1, m):
        if Q[s, a] > best:
            best = Q[s, a]
    loss = (best - Q[s, a_learned]) / best
    return loss <= TOL_BR, loss


@njit(cache=True)
def best_response_one(greedy, PI, delta, cyc, optimal):
    """Para uma sessão convergida: Nash no caminho, SPE, cada agente em melhor resposta
    no caminho, Q-loss no caminho e em todos os estados."""
    m = PI.shape[0]
    S = m * m
    br_path = np.ones(2, dtype=np.bool_)
    br_all = np.ones(2, dtype=np.bool_)
    loss_path = 0.0
    loss_all = 0.0
    for i in range(2):
        Q = true_q(greedy, PI, delta, i, optimal)
        for k in range(cyc.shape[0]):
            ok, loss = check_state(Q, cyc[k], greedy[i, cyc[k]])
            if not ok:
                br_path[i] = False
            loss_path += loss
        for s in range(S):
            ok, loss = check_state(Q, s, greedy[i, s])
            if not ok:
                br_all[i] = False
            loss_all += loss
    loss_path /= 2.0 * cyc.shape[0]
    loss_all /= 2.0 * S
    return br_path, br_all, loss_path, loss_all


@njit(parallel=True, cache=True)
def run_many_best_response(seeds, PI, Q0, alpha, delta, beta, t_stable, t_max,
                           pi_nash, pi_mon):
    """Refaz as sessões (mesmas sementes) e aplica o teste de melhor resposta."""
    n = seeds.shape[0]
    m = PI.shape[0]
    conv = np.zeros(n, dtype=np.bool_)
    dlt = np.zeros(n)
    clen = np.zeros(n, dtype=np.int64)
    sym = np.zeros(n, dtype=np.bool_)          # preço constante e igual nas duas firmas
    # índice [k, v, i]: sessão k, versão v (0 = Calvano, 1 = ótima), agente i
    br_path = np.zeros((n, 2, 2), dtype=np.bool_)
    br_all = np.zeros((n, 2, 2), dtype=np.bool_)
    qloss_path = np.full((n, 2), np.nan)
    qloss_all = np.full((n, 2), np.nan)
    for k in prange(n):
        ok, t, d, nc, pidx, Q, greedy, s_end = run_session(
            seeds[k], PI, Q0, alpha, delta, beta, t_stable, t_max, pi_nash, pi_mon)
        conv[k] = ok
        dlt[k] = d
        clen[k] = nc
        if ok:
            cyc = limit_cycle(greedy, s_end, m)
            sym[k] = nc == 1 and (cyc[0] // m) == (cyc[0] % m)
            for v in range(2):
                bp, ba, lp, la = best_response_one(greedy, PI, delta, cyc, v == 1)
                for i in range(2):
                    br_path[k, v, i] = bp[i]
                    br_all[k, v, i] = ba[i]
                qloss_path[k, v] = lp
                qloss_all[k, v] = la
    return conv, dlt, clen, sym, br_path, br_all, qloss_path, qloss_all
