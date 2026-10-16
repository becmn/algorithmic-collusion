"""
Teste de desvio comum (Fase 1.4), protocolo da Pesquisa 8 registrado no documento do projeto.

Ponto de partida: estratégias gulosas finais de uma sessão convergida (Q congelada, ε = 0).
Para cada ponto do ciclo-limite (τ = 0):
  - ramo de controle: os dois seguem a estratégia gulosa por 15 períodos;
  - ramo com desvio: em τ = 1 a firma 1 joga a melhor resposta estática, dentro da grade,
    ao preço da firma 2 em τ = 0; a firma 2 segue a estratégia; de τ = 2 a 15 os dois
    seguem a estratégia aprendida.
Métricas por ponto de partida (firma 1 desvia, firma 2 é o rival):
  R = [p2_dev(2) − p2_ctrl(2)] / p2(0)
  G = Σ 0,95^(τ−1) [π1_dev(τ) − π1_ctrl(τ)] / Σ 0,95^(τ−1) π1_ctrl(τ), τ = 1..15
  V = primeiro τ ≥ 2 a partir do qual os dois preços ficam a até 1 degrau da grade do
      controle por 3 períodos seguidos, dentro dos 15 períodos (senão, censurado).
      Em sessões que terminam em ciclo (tamanho > 1), V = primeiro τ ≥ 2 a partir do qual
      o par de preços está de novo no ciclo-limite, em qualquer fase, por 3 períodos
      seguidos (decisão de 13/10/2026: a regra preço a preço contava como "sem volta" o
      ciclo que volta fora de fase). A regra preço a preço também é guardada (V_phase).
Em ciclos, a média sobre os pontos do ciclo conta como 1 observação (Calvano, nota 29).
Código próprio do projeto (não derivado de Courthoud nem de Schildknecht).
"""

import numpy as np
from numba import njit, prange

from session import limit_cycle, run_session

T_DEV = 15      # períodos depois do desvio (τ = 1 a 15)
RUN_LEN = 3     # períodos seguidos perto do controle para contar a volta (V)


@njit(cache=True)
def static_best_response(PI, a2):
    """Índice da grade que maximiza o lucro da firma 1 contra a firma 2 no índice a2
    (empate: menor preço)."""
    m = PI.shape[0]
    best = 0
    v = PI[0, a2, 0]
    for a in range(1, m):
        if PI[a, a2, 0] > v:
            v = PI[a, a2, 0]
            best = a
    return best


@njit(cache=True)
def deviation_paths(greedy, PI, s0):
    """Trajetórias dos dois ramos a partir do estado s0 (preços de τ = 0).
    Devolve índices de preço ctrl[τ, i] e dev[τ, i] para τ = 0..T_DEV."""
    m = PI.shape[0]
    ctrl = np.empty((T_DEV + 1, 2), dtype=np.int64)
    dev = np.empty((T_DEV + 1, 2), dtype=np.int64)
    ctrl[0, 0] = s0 // m
    ctrl[0, 1] = s0 % m
    dev[0, 0] = ctrl[0, 0]
    dev[0, 1] = ctrl[0, 1]
    sc = s0
    sd = s0
    for tau in range(1, T_DEV + 1):
        c1 = greedy[0, sc]
        c2 = greedy[1, sc]
        if tau == 1:
            d1 = static_best_response(PI, dev[0, 1])
        else:
            d1 = greedy[0, sd]
        d2 = greedy[1, sd]
        ctrl[tau, 0] = c1
        ctrl[tau, 1] = c2
        dev[tau, 0] = d1
        dev[tau, 1] = d2
        sc = c1 * m + c2
        sd = d1 * m + d2
    return ctrl, dev


@njit(cache=True)
def metrics_one_start(greedy, PI, prices, delta, step, s0, in_cycle, is_cycle):
    """R, G, V (V = −1 se censurado), V preço a preço (V_phase), se houve desvio de fato,
    R no estilo de Calvano [p2_dev(2) − p2(0)] / p2(0) e a curva p2_dev(τ)/p2_ctrl(τ) − 1."""
    m = PI.shape[0]
    ctrl, dev = deviation_paths(greedy, PI, s0)
    p20 = prices[ctrl[0, 1]]
    R = (prices[dev[2, 1]] - prices[ctrl[2, 1]]) / p20
    R_calv = (prices[dev[2, 1]] - p20) / p20
    did_deviate = dev[1, 0] != ctrl[1, 0]

    num = 0.0
    den = 0.0
    w = 1.0
    for tau in range(1, T_DEV + 1):
        num += w * (PI[dev[tau, 0], dev[tau, 1], 0] - PI[ctrl[tau, 0], ctrl[tau, 1], 0])
        den += w * PI[ctrl[tau, 0], ctrl[tau, 1], 0]
        w *= delta
    G = num / den

    V_phase = -1
    tol = step * (1.0 + 1e-9)
    for tau in range(2, T_DEV - RUN_LEN + 2):
        ok = True
        for k in range(RUN_LEN):
            for i in range(2):
                if abs(prices[dev[tau + k, i]] - prices[ctrl[tau + k, i]]) > tol:
                    ok = False
        if ok:
            V_phase = tau
            break

    V_cyc = -1
    for tau in range(2, T_DEV - RUN_LEN + 2):
        ok = True
        for k in range(RUN_LEN):
            if not in_cycle[dev[tau + k, 0] * m + dev[tau + k, 1]]:
                ok = False
        if ok:
            V_cyc = tau
            break
    V = V_cyc if is_cycle else V_phase

    curve = np.empty(T_DEV + 1)
    for tau in range(T_DEV + 1):
        curve[tau] = prices[dev[tau, 1]] / prices[ctrl[tau, 1]] - 1.0
    return R, G, V, V_phase, did_deviate, R_calv, curve


@njit(parallel=True, cache=True)
def run_many_with_deviation(seeds, PI, Q0, alpha, delta, beta, t_stable, t_max,
                            pi_nash, pi_mon, prices, step):
    """Roda cada sessão (mesma função da Fase 1.3) e aplica o teste de desvio a partir
    de cada ponto do ciclo-limite. Por sessão devolve: convergiu, Δ, tamanho do ciclo,
    médias de R, G e R_calvano sobre o ciclo, V médio (−1 se algum ponto censurado),
    V preço a preço (V_phase), fração de pontos censurados, fração de pontos sem desvio de fato e a curva média."""
    n = seeds.shape[0]
    m = PI.shape[0]
    conv = np.zeros(n, dtype=np.bool_)
    dlt = np.zeros(n)
    clen = np.zeros(n, dtype=np.int64)
    R = np.full(n, np.nan)
    G = np.full(n, np.nan)
    Rc = np.full(n, np.nan)
    V = np.full(n, np.nan)
    Vp = np.full(n, np.nan)
    cens = np.full(n, np.nan)
    nodev = np.full(n, np.nan)
    curves = np.full((n, T_DEV + 1), np.nan)
    for k in prange(n):
        ok, t, d, nc, pidx, Q, greedy, s_end = run_session(
            seeds[k], PI, Q0, alpha, delta, beta, t_stable, t_max, pi_nash, pi_mon)
        conv[k] = ok
        dlt[k] = d
        clen[k] = nc
        if not ok:
            continue
        cyc = limit_cycle(greedy, s_end, m)
        L = cyc.shape[0]
        in_cycle = np.zeros(m * m, dtype=np.bool_)
        for j in range(L):
            in_cycle[cyc[j]] = True
        sVp = 0.0
        ncp = 0
        sR = 0.0
        sG = 0.0
        sRc = 0.0
        sV = 0.0
        nc_ = 0
        nd = 0
        cv = np.zeros(T_DEV + 1)
        for j in range(L):
            r, g, v, vp, didv, rc, cu = metrics_one_start(greedy, PI, prices, delta, step,
                                                          cyc[j], in_cycle, L > 1)
            if vp < 0:
                ncp += 1
            else:
                sVp += vp
            sR += r
            sG += g
            sRc += rc
            if v < 0:
                nc_ += 1
            else:
                sV += v
            if not didv:
                nd += 1
            cv += cu
        R[k] = sR / L
        G[k] = sG / L
        Rc[k] = sRc / L
        V[k] = sV / L if nc_ == 0 else -1.0
        Vp[k] = sVp / L if ncp == 0 else -1.0
        cens[k] = nc_ / L
        nodev[k] = nd / L
        for tau in range(T_DEV + 1):
            curves[k, tau] = cv[tau] / L
    return conv, dlt, clen, R, G, Rc, V, Vp, cens, nodev, curves


def classify(R, G, V, theta):
    """Classes do protocolo: A punição com volta (R ≤ −θ, G < 0, V ≤ 15);
    B punição sem volta (R ≤ −θ, V censurado); C sem punição (R > −θ).
    Casos com R ≤ −θ, volta e G ≥ 0 não cabem em A, B ou C: ficam como 'outro'."""
    if R > -theta:
        return "C"
    if V < 0:
        return "B"
    if G < 0:
        return "A"
    return "outro"
