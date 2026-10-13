"""
Controles da Fase 1.5, parte 2 (Protocolo, testes obrigatórios; Pesquisa 5, proposta P3):
mercados em que a colusão por punição não deveria aparecer.
  - sem memória (k = 0): o agente não vê os preços do período anterior (1 estado só),
    então não consegue condicionar o preço ao que o rival fez e não pode punir;
  - δ = 0: o agente só se importa com o lucro do próprio período (míope), então
    punir o rival amanhã não vale nada para ele.

Sessão genérica com memória k = 0 ou k = 1. Com k = 1, é a mesma conta de session.py
(mesma ordem de sorteios), e o runner confere que ela devolve Δ idêntico ao da Fase 1.3.
Estado: k = 1 → s = a1 * m + a2 (225 estados); k = 0 → s = 0 (1 estado).
Também calcula o teste de melhor resposta na leitura de Calvano (ver best_response.py),
para comparar o controle da Tabela A1 com a fração em Nash 0,273 e o Q-loss 0,013.
Fase 1.5, parte 3: a sessão ganhou eps_min (piso de exploração, 0 = regra da Ficha) e passou
a devolver a Q final, e run_many_mem_diag serve ao diagnóstico do controle sem memória;
com eps_min = 0, os resultados da parte 2 não mudam.
Código próprio do projeto; a sessão segue session.py (adaptado de Courthoud, MIT).
"""

import numpy as np
from numba import njit, prange

TOL_VI = 1e-12
TOL_BR = 1e-9


@njit(cache=True)
def _argmax_max(row):
    best = 0
    v = row[0]
    for j in range(1, row.shape[0]):
        if row[j] > v:
            v = row[j]
            best = j
    return best, v


@njit(cache=True)
def _state(a1, a2, m, memory):
    if memory == 1:
        return a1 * m + a2
    return 0


@njit(cache=True)
def cycle_actions(greedy, s_start, m, memory):
    """Joga as estratégias gulosas a partir de s_start até o estado se repetir.
    Devolve os estados do ciclo e os pares de preços (índices) jogados a partir deles."""
    S = greedy.shape[1]
    first = np.full(S, -1, dtype=np.int64)
    states = np.empty(S + 1, dtype=np.int64)
    s = s_start
    k = 0
    while first[s] < 0:
        first[s] = k
        states[k] = s
        k += 1
        s = _state(greedy[0, s], greedy[1, s], m, memory)
    cyc = states[first[s]:k].copy()
    acts = np.empty((cyc.shape[0], 2), dtype=np.int64)
    for j in range(cyc.shape[0]):
        acts[j, 0] = greedy[0, cyc[j]]
        acts[j, 1] = greedy[1, cyc[j]]
    return cyc, acts


@njit(cache=True)
def run_session_mem(seed, PI, Q0, alpha, delta, beta, t_stable, t_max, pi_nash, pi_mon,
                    memory, eps_min):
    """Uma sessão até convergir (estratégia gulosa igual por t_stable períodos) ou t_max.
    ε = max(e^(−βt), eps_min); eps_min = 0 é a regra da Ficha (usado só no diagnóstico).
    Devolve: convergiu, períodos, Δ, tamanho do ciclo, preço médio (índice), estratégia
    gulosa final, estado final e Q final."""
    np.random.seed(seed)
    m = PI.shape[0]
    S = Q0.shape[1]
    Q = Q0.copy()
    greedy = np.empty((2, S), dtype=np.int64)
    maxq = np.empty((2, S))
    for i in range(2):
        for s in range(S):
            g, v = _argmax_max(Q[i, s])
            greedy[i, s] = g
            maxq[i, s] = v

    s = np.random.randint(0, S)
    stable = 0
    ring = np.zeros(t_stable)
    t = 0
    converged = False
    act = np.empty(2, dtype=np.int64)
    while t < t_max:
        eps = max(np.exp(-beta * t), eps_min)
        for i in range(2):
            if np.random.random() < eps:
                act[i] = np.random.randint(0, m)
            else:
                act[i] = greedy[i, s]
        s1 = _state(act[0], act[1], m, memory)
        changed = False
        for i in range(2):
            target = PI[act[0], act[1], i] + delta * maxq[i, s1]
            Q[i, s, act[i]] = (1.0 - alpha) * Q[i, s, act[i]] + alpha * target
            g, v = _argmax_max(Q[i, s])
            if g != greedy[i, s]:
                changed = True
                greedy[i, s] = g
            maxq[i, s] = v
        ring[t % t_stable] = 0.5 * (PI[act[0], act[1], 0] + PI[act[0], act[1], 1])
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
        cyc, acts = cycle_actions(greedy, s, m, memory)
        n = cyc.shape[0]
        pbar = 0.0
        pidx = 0.0
        for j in range(n):
            pbar += 0.5 * (PI[acts[j, 0], acts[j, 1], 0] + PI[acts[j, 0], acts[j, 1], 1])
            pidx += 0.5 * (acts[j, 0] + acts[j, 1])
        pbar /= n
        pidx /= n
    else:
        n = 0
        pbar = ring.mean()
        pidx = -1.0
    return converged, t, (pbar - pi_nash) / (pi_mon - pi_nash), n, pidx, greedy, s, Q


@njit(cache=True)
def true_q_calvano(greedy, PI, delta, i, m, memory):
    """Q verdadeira do agente i, leitura de Calvano: rival fixo na estratégia final e
    continuação pela própria estratégia final (desvio de um período)."""
    S = greedy.shape[1]
    Q = np.zeros((S, m))
    V = np.zeros(S)
    for it in range(100_000):
        for s in range(S):
            b = greedy[1 - i, s]
            for a in range(m):
                if i == 0:
                    r = PI[a, b, 0]
                    s1 = _state(a, b, m, memory)
                else:
                    r = PI[b, a, 1]
                    s1 = _state(b, a, m, memory)
                Q[s, a] = r + delta * V[s1]
        diff = 0.0
        for s in range(S):
            v = Q[s, greedy[i, s]]
            d = abs(v - V[s])
            if d > diff:
                diff = d
            V[s] = v
        if diff < TOL_VI:
            break
    return Q


@njit(cache=True)
def best_response_mem(greedy, PI, delta, cyc, m, memory):
    """(Nash no caminho, Q-loss no caminho, Q-loss em todos os estados)."""
    S = greedy.shape[1]
    nash = True
    lp = 0.0
    la = 0.0
    for i in range(2):
        Q = true_q_calvano(greedy, PI, delta, i, m, memory)
        for s in range(S):
            best = Q[s, 0]
            for a in range(1, m):
                if Q[s, a] > best:
                    best = Q[s, a]
            loss = (best - Q[s, greedy[i, s]]) / best
            la += loss
            for j in range(cyc.shape[0]):
                if cyc[j] == s:
                    lp += loss
                    if loss > TOL_BR:
                        nash = False
    return nash, lp / (2.0 * cyc.shape[0]), la / (2.0 * S)


@njit(parallel=True, cache=True)
def run_many_mem(seeds, PI, Q0, alpha, delta, beta, t_stable, t_max, pi_nash, pi_mon,
                 memory):
    """Várias sessões em paralelo, com o teste de melhor resposta (leitura de Calvano)."""
    n = seeds.shape[0]
    m = PI.shape[0]
    conv = np.zeros(n, dtype=np.bool_)
    periods = np.zeros(n, dtype=np.int64)
    dlt = np.zeros(n)
    clen = np.zeros(n, dtype=np.int64)
    pidx = np.zeros(n)
    p1 = np.full(n, -1, dtype=np.int64)          # preço (índice) de cada firma, se constante
    p2 = np.full(n, -1, dtype=np.int64)
    nash = np.zeros(n, dtype=np.bool_)
    qlp = np.full(n, np.nan)
    qla = np.full(n, np.nan)
    for k in prange(n):
        ok, t, d, nc, pi_, greedy, s_end, Q = run_session_mem(
            seeds[k], PI, Q0, alpha, delta, beta, t_stable, t_max, pi_nash, pi_mon, memory, 0.0)
        conv[k] = ok
        periods[k] = t
        dlt[k] = d
        clen[k] = nc
        pidx[k] = pi_
        if ok:
            cyc, acts = cycle_actions(greedy, s_end, m, memory)
            if nc == 1:
                p1[k] = acts[0, 0]
                p2[k] = acts[0, 1]
            nh, lp, la = best_response_mem(greedy, PI, delta, cyc, m, memory)
            nash[k] = nh
            qlp[k] = lp
            qla[k] = la
    return conv, periods, dlt, clen, pidx, p1, p2, nash, qlp, qla


@njit(parallel=True, cache=True)
def run_many_mem_diag(seeds, PI, Q0, alpha, delta, beta, t_stable, t_max, pi_nash, pi_mon,
                      memory, eps_min):
    """Diagnóstico (Fase 1.5, parte 3). Por sessão: convergiu, períodos, Δ do ciclo-limite
    da estratégia gulosa no momento em que a sessão parou (igual ao Δ usual se convergiu),
    preços finais e, com memória 0 e preço constante, as notas que explicam por que o
    agente não corta o preço:
      q_play[i]  = Q aprendida do preço jogado;  q_fix[i] = π(jogado)/(1 − δ);
      q_br[i]    = Q aprendida da melhor resposta estática ao preço do rival;
      v_br[i]    = valor verdadeiro de cortar 1 vez para ela: π(MR) + δ · Q aprendida do jogado."""
    n = seeds.shape[0]
    m = PI.shape[0]
    conv = np.zeros(n, dtype=np.bool_)
    periods = np.zeros(n, dtype=np.int64)
    dg = np.zeros(n)
    clen = np.zeros(n, dtype=np.int64)
    p1 = np.full(n, -1, dtype=np.int64)
    p2 = np.full(n, -1, dtype=np.int64)
    q_play = np.full((n, 2), np.nan)
    q_fix = np.full((n, 2), np.nan)
    q_br = np.full((n, 2), np.nan)
    v_br = np.full((n, 2), np.nan)
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
        if L == 1:
            p1[k] = acts[0, 0]
            p2[k] = acts[0, 1]
            if memory == 0:
                for i in range(2):
                    a = acts[0, i]
                    b = acts[0, 1 - i]
                    br = 0
                    best = -1.0
                    for x in range(m):
                        r = PI[x, b, 0] if i == 0 else PI[b, x, 1]
                        if r > best:
                            best = r
                            br = x
                    r_play = PI[a, b, 0] if i == 0 else PI[b, a, 1]
                    q_play[k, i] = Q[i, 0, a]
                    q_fix[k, i] = r_play / (1.0 - delta)
                    q_br[k, i] = Q[i, 0, br]
                    v_br[k, i] = best + delta * Q[i, 0, a]
    return conv, periods, dg, clen, p1, p2, q_play, q_fix, q_br, v_br
