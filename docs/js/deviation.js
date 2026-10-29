/*
 * Teste de desvio do simulador (Fase 2.2): o mesmo protocolo de qlearning/deviation.py
 * (Fase 1.4), em JavaScript, para o botão "Forçar desvio" funcionar sobre sessões ao vivo.
 *
 * A partir de um estado s0 do ciclo-limite (τ = 0), com Q congelada e ε = 0:
 *   - ramo de controle: as duas firmas seguem a estratégia aprendida por 15 períodos;
 *   - ramo com desvio: em τ = 1 a firma 1 joga a melhor resposta estática, dentro da grade,
 *     ao preço da firma 2 em τ = 0; depois as duas seguem a estratégia aprendida.
 * Métricas: R (reação do rival em τ = 2), G (ganho descontado do desvio), V (volta),
 * lucro descontado da firma 1 com e sem desvio. Regras de V iguais às do Python:
 * preço constante → os dois preços a até 1 degrau do controle por 3 períodos seguidos;
 * ciclo → o par de preços de novo no ciclo-limite, em qualquer fase, por 3 períodos.
 * Validado contra o JSON do Python em test_deviation.js. Código próprio do projeto.
 */
(function (root) {
  "use strict";
  const T_DEV = 15, RUN_LEN = 3;

  // Melhor resposta estática da firma 1 ao índice a2 da firma 2 (empate: menor preço).
  function staticBestResponse(mk, a2) {
    const m = mk.m;
    let best = 0, v = mk.PI[(0 * m + a2) * 2];
    for (let a = 1; a < m; a++) {
      const x = mk.PI[(a * m + a2) * 2];
      if (x > v) { v = x; best = a; }
    }
    return best;
  }

  // greedy: Int32Array/Array de 2·S (firma 1 em [0, S), firma 2 em [S, 2S)).
  function paths(mk, greedy, s0) {
    const m = mk.m, S = m * m;
    const ctrl = [[Math.floor(s0 / m), s0 % m]];
    const dev = [[Math.floor(s0 / m), s0 % m]];
    let sc = s0, sd = s0;
    for (let tau = 1; tau <= T_DEV; tau++) {
      const c1 = greedy[sc], c2 = greedy[S + sc];
      const d1 = tau === 1 ? staticBestResponse(mk, dev[0][1]) : greedy[sd];
      const d2 = greedy[S + sd];
      ctrl.push([c1, c2]); dev.push([d1, d2]);
      sc = c1 * m + c2; sd = d1 * m + d2;
    }
    return { ctrl: ctrl, dev: dev };
  }

  // cycle: estados do ciclo-limite (de QEngine.limitCycle); start: posição no ciclo.
  function run(mk, greedy, cycle, start, delta) {
    const m = mk.m, P = mk.prices, PI = mk.PI;
    const s0 = cycle[start || 0];
    const pt = paths(mk, greedy, s0);
    const ctrl = pt.ctrl, dev = pt.dev;
    const pr = (st) => [P[st[0]], P[st[1]]];
    const pi1 = (st) => PI[(st[0] * m + st[1]) * 2];
    const p20 = P[ctrl[0][1]];
    const R = (P[dev[2][1]] - P[ctrl[2][1]]) / p20;

    let num = 0, den = 0, w = 1, profDev = 0, profCtrl = 0;
    for (let tau = 1; tau <= T_DEV; tau++) {
      num += w * (pi1(dev[tau]) - pi1(ctrl[tau]));
      den += w * pi1(ctrl[tau]);
      profDev += w * pi1(dev[tau]);
      profCtrl += w * pi1(ctrl[tau]);
      w *= delta;
    }
    const G = num / den;

    const isCycle = cycle.length > 1;
    const inCycle = new Set(cycle);
    const step = P[1] - P[0];
    const tol = step * (1 + 1e-9);
    let V = null;
    for (let tau = 2; tau <= T_DEV - RUN_LEN + 1 && V === null; tau++) {
      let ok = true;
      for (let k = 0; k < RUN_LEN && ok; k++) {
        const d = dev[tau + k], c = ctrl[tau + k];
        if (isCycle) ok = inCycle.has(d[0] * m + d[1]);
        else ok = Math.abs(P[d[0]] - P[c[0]]) <= tol && Math.abs(P[d[1]] - P[c[1]]) <= tol;
      }
      if (ok) V = tau;
    }
    return {
      tau: ctrl.map((_, k) => k),
      control: { idx: ctrl, price: ctrl.map(pr) },
      deviation: { idx: dev, price: dev.map(pr) },
      did_deviate: dev[1][0] !== ctrl[1][0],
      R: R, G: G, V: V,
      discounted_profit_firm1_control: profCtrl,
      discounted_profit_firm1_deviation: profDev,
    };
  }

  const api = { T_DEV: T_DEV, staticBestResponse: staticBestResponse, paths: paths, run: run };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.QDeviation = api;
})(typeof self !== "undefined" ? self : this);
