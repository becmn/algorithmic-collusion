"""
Uma sessão de Q-learning no duopólio de Calvano et al. (2020), compilada com numba.

Origem: adaptado de Matteo Courthoud, Algorithmic-Collusion-Replication
(https://github.com/matteocourthoud/Algorithmic-Collusion-Replication),
licença MIT, © 2021 Matteo Courthoud (aviso completo em THIRD_PARTY_LICENSES.md).
Partes aproveitadas: demanda logit, Q inicial pela média contra rival aleatório
descontada, ε = exp(-β t) por agente, regra de atualização e desempate pelo menor preço.

Mudanças em relação ao original (Fase 1.2 do projeto):
  1. Grade com ξ = 0,1 (Calvano), em vez de 13 pontos de Nash a monopólio + 1 passo.
  2. Convergência contada 1 vez por período (o original contava 1 vez por agente).
  3. Teto de períodos (10^9) passado como argumento próprio (o original lia a chave errada).
  4. Laço compilado com numba; estratégia gulosa guardada e só a linha da Q que mudou
     é recalculada; semente passada como argumento e aplicada dentro da função compilada.
  5. Estado inicial sorteado pela semente (o original começava sempre em (0, 0)).
  6. Δ no ciclo-limite (Q congelada, ε = 0); sessão sem convergência usa a média
     dos últimos t_stable períodos.
Todos os parâmetros entram como argumentos (nada global), por causa do cache do numba.
"""

import numpy as np
from numba import njit


# ---------------------------------------------------------------- funções em Python
def demand(p, a, a0, mu):
    """Fatias de mercado logit para o vetor de preços p."""
    e = np.exp((a - p) / mu)
    return e / (e.sum() + np.exp(a0 / mu))


def calvano_grid(p_nash, p_mon, m, xi):
    """m preços igualmente espaçados em [pN − ξ(pM − pN), pM + ξ(pM − pN)]."""
    lo = p_nash - xi * (p_mon - p_nash)
    hi = p_mon + xi * (p_mon - p_nash)
    return np.linspace(lo, hi, m)


def profit_table(prices, a, a0, mu, c):
    """PI[a1, a2, i] = lucro da firma i quando as firmas jogam os índices a1 e a2."""
    m = len(prices)
    PI = np.zeros((m, m, 2))
    for i1 in range(m):
        for i2 in range(m):
            p = np.array([prices[i1], prices[i2]])
            PI[i1, i2, :] = (p - c) * demand(p, a, a0, mu)
    return PI


def initial_q(PI, delta):
    """Q0[i, s, a]: lucro médio contra rival que sorteia preços, dividido por (1 − δ).
    Igual em todos os estados (Calvano, eq. 8)."""
    m = PI.shape[0]
    Q0 = np.zeros((2, m * m, m))
    q1 = PI[:, :, 0].mean(axis=1) / (1 - delta)   # firma 1: média sobre a ação da firma 2
    q2 = PI[:, :, 1].mean(axis=0) / (1 - delta)   # firma 2: média sobre a ação da firma 1
    Q0[0, :, :] = q1
    Q0[1, :, :] = q2
    return Q0


# ---------------------------------------------------------------- funções compiladas
@njit(cache=True)
def _row_argmax_max(row):
    """Índice do maior valor (o primeiro, isto é, o menor preço, em caso de empate) e o valor."""
    best = 0
    v = row[0]
    for j in range(1, row.shape[0]):
        if row[j] > v:
            v = row[j]
            best = j
    return best, v


@njit(cache=True)
def limit_cycle(greedy, s_start, m):
    """Joga as estratégias gulosas (Q congelada, ε = 0) a partir de s_start até um
    estado se repetir. Devolve os estados do ciclo, em ordem."""
    S = m * m
    first_visit = np.full(S, -1, dtype=np.int64)
    path = np.empty(S + 1, dtype=np.int64)
    s = s_start
    k = 0
    while first_visit[s] < 0:
        first_visit[s] = k
        path[k] = s
        k += 1
        a1 = greedy[0, s]
        a2 = greedy[1, s]
        s = a1 * m + a2
    return path[first_visit[s]:k].copy()


@njit(cache=True)
def run_session(seed, PI, Q0, alpha, delta, beta, t_stable, t_max, pi_nash, pi_mon):
    """Roda uma sessão até convergir ou chegar a t_max.

    Estado s = a1 * m + a2 (índices de preço do período anterior; 225 estados).
    Devolve: convergiu (bool), períodos até parar, Δ, tamanho do ciclo,
             preço médio (índice) no ciclo, Q final, estratégia gulosa final,
             estado em que a sessão parou.
    """
    np.random.seed(seed)
    m = PI.shape[0]
    S = m * m
    Q = Q0.copy()

    # estratégia gulosa e valor máximo de cada linha da Q
    greedy = np.empty((2, S), dtype=np.int64)
    maxq = np.empty((2, S))
    for i in range(2):
        for s in range(S):
            g, v = _row_argmax_max(Q[i, s])
            greedy[i, s] = g
            maxq[i, s] = v

    s = np.random.randint(0, S)          # estado inicial sorteado
    stable = 0
    ring = np.zeros(t_stable)            # lucro médio das 2 firmas nos últimos t_stable períodos
    t = 0
    converged = False
    act = np.empty(2, dtype=np.int64)

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

        ring[t % t_stable] = 0.5 * (PI[act[0], act[1], 0] + PI[act[0], act[1], 1])

        if changed:
            stable = 0
        else:
            stable += 1             # 1 vez por período (correção 2)
        s = s1
        t += 1
        if stable >= t_stable:
            converged = True
            break

    if converged:
        cyc = limit_cycle(greedy, s, m)
        n = cyc.shape[0]
        pbar = 0.0
        pidx = 0.0
        for k in range(n):
            a1 = cyc[k] // m
            a2 = cyc[k] % m
            pbar += 0.5 * (PI[a1, a2, 0] + PI[a1, a2, 1])
            pidx += 0.5 * (a1 + a2)
        pbar /= n
        pidx /= n
    else:
        n = 0
        pbar = ring.mean()
        pidx = -1.0

    delta_idx = (pbar - pi_nash) / (pi_mon - pi_nash)
    return converged, t, delta_idx, n, pidx, Q, greedy, s
