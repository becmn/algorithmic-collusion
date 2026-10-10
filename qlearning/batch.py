"""
Várias sessões de Q-learning em paralelo (Fase 1.3), com prange do numba.
Cada sessão chama run_session (session.py) com a sua semente; a semente é aplicada
dentro da função compilada, então o resultado não depende de qual núcleo rodou a sessão.
"""

import numpy as np
from numba import njit, prange

from session import limit_cycle, run_session

MAX_CYCLE = 225  # um ciclo nunca passa do número de estados (15 x 15)


@njit(parallel=True, cache=True)
def run_many(seeds, PI, Q0, alpha, delta, beta, t_stable, t_max, pi_nash, pi_mon):
    """Roda uma sessão por semente. Devolve, por sessão: convergiu, períodos até parar,
    Δ, tamanho do ciclo e os estados do ciclo (-1 depois do fim do ciclo)."""
    n_sess = seeds.shape[0]
    m = PI.shape[0]
    converged = np.zeros(n_sess, dtype=np.bool_)
    periods = np.zeros(n_sess, dtype=np.int64)
    deltas = np.zeros(n_sess)
    cyc_len = np.zeros(n_sess, dtype=np.int64)
    cycles = np.full((n_sess, MAX_CYCLE), -1, dtype=np.int64)
    for k in prange(n_sess):
        ok, t, d, n, pidx, Q, greedy, s_end = run_session(
            seeds[k], PI, Q0, alpha, delta, beta, t_stable, t_max, pi_nash, pi_mon)
        converged[k] = ok
        periods[k] = t
        deltas[k] = d
        cyc_len[k] = n
        if ok:
            cyc = limit_cycle(greedy, s_end, m)
            for j in range(cyc.shape[0]):
                cycles[k, j] = cyc[j]
    return converged, periods, deltas, cyc_len, cycles
