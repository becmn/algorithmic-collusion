/*
 * Motor de Q-learning do simulador (Fase 2.1): o mesmo algoritmo de qlearning/session.py,
 * em JavaScript puro, para rodar num Web Worker (qlearning.worker.js) e no Node (test_engine.js).
 *
 * Caso base de Calvano, Calzolari, Denicolò e Pastorello (2020), AER 110(10), com os
 * valores da Ficha de parâmetros do projeto (lidos de docs/data/qlearning_params.json).
 * Código próprio do projeto; a lógica segue session.py (que adapta Courthoud, MIT).
 *
 * Diferença em relação ao Python: o gerador aleatório. O numba usa o gerador do numpy;
 * aqui o gerador é o sfc32 (semente expandida pelo splitmix32), porque Math.random()
 * não aceita semente. A mesma semente NÃO reproduz a mesma sessão do Python: a validação
 * compara distribuições (Pesquisa 9). A mesma semente reproduz sempre a mesma sessão em JS.
 *
 * Convenções (iguais às do Python):
 *   estado s = a1 * m + a2 (índices de preço do período anterior; 225 estados)
 *   Q[(i * S + s) * m + a]   nota do agente i no estado s para o preço de índice a
 *   PI[(a1 * m + a2) * 2 + i] lucro da firma i quando as firmas jogam a1 e a2
 *   desempate do argmax pelo menor preço (primeiro índice)
 */
(function (root) {
  "use strict";

  // ------------------------------------------------------------ gerador com semente
  // splitmix32: espalha a semente (um inteiro) nos 4 números de estado do sfc32.
  function splitmix32(seed) {
    let x = seed >>> 0;
    return function () {
      x = (x + 0x9e3779b9) >>> 0;
      let z = x;
      z = Math.imul(z ^ (z >>> 16), 0x85ebca6b);
      z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35);
      return (z ^ (z >>> 16)) >>> 0;
    };
  }

  // sfc32: devolve números em [0, 1) com 32 bits de resolução.
  function makeRng(seed) {
    const sm = splitmix32(seed);
    let a = sm(), b = sm(), c = sm(), d = sm();
    function next() {
      const t = (((a + b) >>> 0) + d) >>> 0;
      d = (d + 1) >>> 0;
      a = b ^ (b >>> 9);
      b = (c + (c << 3)) >>> 0;
      c = (c << 21) | (c >>> 11);
      c = (c + t) >>> 0;
      return t / 4294967296;
    }
    for (let k = 0; k < 12; k++) next();   // descarta as primeiras saídas
    return next;
  }

  // ------------------------------------------------------------ mercado
  // Fatias logit: q_i = exp((a − p_i)/μ) / [Σ_j exp((a − p_j)/μ) + exp(a0/μ)]
  function shares(p1, p2, a, a0, mu) {
    const e1 = Math.exp((a - p1) / mu);
    const e2 = Math.exp((a - p2) / mu);
    const den = e1 + e2 + Math.exp(a0 / mu);
    return [e1 / den, e2 / den];
  }

  // Raiz de f em [lo, hi] por bisseção (f(lo) e f(hi) com sinais opostos).
  function bisect(f, lo, hi) {
    let flo = f(lo);
    for (let k = 0; k < 200 && hi - lo > 1e-15; k++) {
      const mid = 0.5 * (lo + hi), fm = f(mid);
      if (fm === 0) return mid;
      if ((fm > 0) === (flo > 0)) { lo = mid; flo = fm; } else { hi = mid; }
    }
    return 0.5 * (lo + hi);
  }

  // Preços simétricos de Nash e de monopólio conjunto: as mesmas condições de primeira
  // ordem de qlearning/base_case.py, com p1 = p2 = p (q = fatia de cada firma):
  //   Nash:      1 − (p − c)(1 − q)/μ = 0
  //   monopólio: 1 − (p − c)(1 − 2q)/μ = 0
  function referencePoints(a, a0, mu, c) {
    const q = function (p) { return shares(p, p, a, a0, mu)[0]; };
    const pN = bisect(function (p) { return 1 - (p - c) * (1 - q(p)) / mu; }, c, c + 10);
    const pM = bisect(function (p) { return 1 - (p - c) * (1 - 2 * q(p)) / mu; }, c, c + 10);
    return { pN: pN, pM: pM, piN: (pN - c) * q(pN), piM: (pM - c) * q(pM) };
  }

  // Monta pontos de referência, grade, tabela de lucros e Q inicial a partir do JSON
  // de parâmetros (qlearning_params.json).
  function setupMarket(P) {
    const a = P.market.a.value, a0 = P.market.a0.value, mu = P.market.mu.value;
    const c = P.market.c.value;
    const L = P.learning;
    const m = L.m.value, xi = L.xi.value, delta = L.delta.value;
    const ref = referencePoints(a, a0, mu, c);
    const pN = ref.pN, pM = ref.pM;

    // m preços igualmente espaçados em [pN − ξ(pM − pN), pM + ξ(pM − pN)] (como np.linspace)
    const lo = pN - xi * (pM - pN), hi = pM + xi * (pM - pN);
    const prices = new Float64Array(m);
    const stepP = (hi - lo) / (m - 1);
    for (let k = 0; k < m; k++) prices[k] = lo + k * stepP;
    prices[m - 1] = hi;

    const PI = new Float64Array(m * m * 2);
    for (let i1 = 0; i1 < m; i1++) {
      for (let i2 = 0; i2 < m; i2++) {
        const q = shares(prices[i1], prices[i2], a, a0, mu);
        PI[(i1 * m + i2) * 2] = (prices[i1] - c) * q[0];
        PI[(i1 * m + i2) * 2 + 1] = (prices[i2] - c) * q[1];
      }
    }

    // Q inicial (eq. 8): lucro médio contra rival que sorteia preços, ÷ (1 − δ)
    const q0 = [new Float64Array(m), new Float64Array(m)];
    for (let k = 0; k < m; k++) {
      let s1 = 0, s2 = 0;
      for (let j = 0; j < m; j++) {
        s1 += PI[(k * m + j) * 2];         // firma 1 joga k, firma 2 joga j
        s2 += PI[(j * m + k) * 2 + 1];     // firma 2 joga k, firma 1 joga j
      }
      q0[0][k] = s1 / m / (1 - delta);
      q0[1][k] = s2 / m / (1 - delta);
    }

    return {
      m: m, S: m * m, prices: prices, PI: PI, q0: q0,
      pN: pN, pM: pM, piN: ref.piN, piM: ref.piM,
      alpha: L.alpha.value, beta: L.beta.value, delta: delta,
      tStable: L.convergence.t_stable, tMax: L.convergence.t_max,
    };
  }

  function deltaIndex(mk, pbar) {
    return (pbar - mk.piN) / (mk.piM - mk.piN);
  }

  // ------------------------------------------------------------ sessão
  // opts (opcional): alpha, beta, delta, tStable, tMax; o padrão é o do artigo.
  function createSession(mk, seed, opts) {
    opts = opts || {};
    const m = mk.m, S = mk.S;
    const delta = opts.delta !== undefined ? opts.delta : mk.delta;
    const sess = {
      mk: mk, seed: seed,
      alpha: opts.alpha !== undefined ? opts.alpha : mk.alpha,
      beta: opts.beta !== undefined ? opts.beta : mk.beta,
      delta: delta,
      tStable: opts.tStable !== undefined ? opts.tStable : mk.tStable,
      tMax: opts.tMax !== undefined ? opts.tMax : mk.tMax,
      rng: makeRng(seed),
      Q: new Float64Array(2 * S * m),
      greedy: new Int32Array(2 * S),
      maxq: new Float64Array(2 * S),
      s: 0, t: 0, stable: 0, converged: false, done: false,
      ring: null,
    };
    // Q inicial igual em todos os estados; com outro δ, recalcula a média ÷ (1 − δ)
    const scale = (1 - mk.delta) / (1 - delta);
    for (let i = 0; i < 2; i++) {
      for (let s = 0; s < S; s++) {
        const base = (i * S + s) * m;
        let best = 0, v = -Infinity;
        for (let a = 0; a < m; a++) {
          const q = mk.q0[i][a] * scale;
          sess.Q[base + a] = q;
          if (q > v) { v = q; best = a; }
        }
        sess.greedy[i * S + s] = best;
        sess.maxq[i * S + s] = v;
      }
    }
    sess.ring = new Float64Array(sess.tStable);
    sess.s = Math.floor(sess.rng() * S);   // estado inicial sorteado
    return sess;
  }

  // Roda até blockSize períodos (para antes se convergir ou chegar a tMax) e devolve
  // o resumo do bloco no mesmo formato de learning_blocks_linear (Fase 1.7).
  function runBlock(sess, blockSize) {
    const mk = sess.mk, m = mk.m, S = mk.S, PI = mk.PI, prices = mk.prices;
    const Q = sess.Q, greedy = sess.greedy, maxq = sess.maxq, ring = sess.ring;
    const rng = sess.rng, alpha = sess.alpha, beta = sess.beta, delta = sess.delta;
    const tStable = sess.tStable, tMax = sess.tMax;
    let s = sess.s, t = sess.t, stable = sess.stable, converged = false;
    const tStart = t, tEnd = Math.min(t + blockSize, tMax);
    let sp1 = 0, sp2 = 0, spi = 0;

    while (t < tEnd) {
      const eps = Math.exp(-beta * t);
      const a1 = rng() < eps ? Math.floor(rng() * m) : greedy[s];
      const a2 = rng() < eps ? Math.floor(rng() * m) : greedy[S + s];
      const s1 = a1 * m + a2;
      const pi1 = PI[s1 * 2], pi2 = PI[s1 * 2 + 1];

      let changed = false;
      for (let i = 0; i < 2; i++) {
        const ai = i === 0 ? a1 : a2;
        const row = (i * S + s) * m;
        const target = (i === 0 ? pi1 : pi2) + delta * maxq[i * S + s1];
        Q[row + ai] = (1 - alpha) * Q[row + ai] + alpha * target;
        // argmax da linha que mudou (primeiro índice = menor preço em caso de empate)
        let best = 0, v = Q[row];
        for (let j = 1; j < m; j++) {
          if (Q[row + j] > v) { v = Q[row + j]; best = j; }
        }
        if (best !== greedy[i * S + s]) { changed = true; greedy[i * S + s] = best; }
        maxq[i * S + s] = v;
      }

      const pbar = 0.5 * (pi1 + pi2);
      ring[t % tStable] = pbar;
      sp1 += prices[a1]; sp2 += prices[a2]; spi += pbar;

      stable = changed ? 0 : stable + 1;   // 1 vez por período
      s = s1;
      t += 1;
      if (stable >= tStable) { converged = true; break; }
    }

    sess.s = s; sess.t = t; sess.stable = stable;
    if (converged) { sess.converged = true; sess.done = true; }
    if (t >= tMax) sess.done = true;
    const n = t - tStart;
    return {
      t_start: tStart, n_periods: n,
      price_firm1: n ? sp1 / n : NaN, price_firm2: n ? sp2 / n : NaN,
      delta: n ? deltaIndex(mk, spi / n) : NaN,
      epsilon_start: Math.exp(-sess.beta * tStart),
    };
  }

  // Joga as estratégias gulosas (Q congelada, ε = 0) a partir de sStart até um estado
  // se repetir; devolve os estados do ciclo, em ordem (igual a limit_cycle do Python).
  // greedy: Int32Array(2 * S), firma 1 em [0, S) e firma 2 em [S, 2S).
  function limitCycle(greedy, sStart, m) {
    const S = m * m;
    const firstVisit = new Int32Array(S).fill(-1);
    const path = [];
    let s = sStart;
    while (firstVisit[s] < 0) {
      firstVisit[s] = path.length;
      path.push(s);
      s = greedy[s] * m + greedy[S + s];
    }
    return path.slice(firstVisit[s]);
  }

  // Resultado final: Δ no ciclo-limite (convergiu) ou média dos últimos tStable períodos.
  function finishSession(sess) {
    const mk = sess.mk, m = mk.m, PI = mk.PI;
    const out = { seed: sess.seed, converged: sess.converged, periods: sess.t };
    if (sess.converged) {
      const cyc = limitCycle(sess.greedy, sess.s, m);
      let pbar = 0;
      out.cycle_idx = cyc.map(function (st) { return [Math.floor(st / m), st % m]; });
      for (let k = 0; k < cyc.length; k++) pbar += 0.5 * (PI[cyc[k] * 2] + PI[cyc[k] * 2 + 1]);
      pbar /= cyc.length;
      out.cycle_len = cyc.length;
      out.cycle_prices = out.cycle_idx.map(function (p) { return [mk.prices[p[0]], mk.prices[p[1]]]; });
      out.delta = deltaIndex(mk, pbar);
    } else {
      let sum = 0;
      for (let k = 0; k < sess.ring.length; k++) sum += sess.ring[k];
      out.cycle_len = 0;
      out.delta = deltaIndex(mk, sum / sess.ring.length);
    }
    out.final_state_idx = [Math.floor(sess.s / m), sess.s % m];
    out.greedy = sess.greedy.slice();   // estratégia final (usada no teste de desvio)
    return out;
  }

  // Sessão completa, pelo mesmo caminho do Worker (blocos de blockSize períodos).
  function runSession(mk, seed, opts, blockSize) {
    const sess = createSession(mk, seed, opts);
    const bs = blockSize || 10000;
    while (!sess.done) runBlock(sess, bs);
    return finishSession(sess);
  }

  const api = {
    makeRng: makeRng, setupMarket: setupMarket, deltaIndex: deltaIndex,
    createSession: createSession, runBlock: runBlock, limitCycle: limitCycle,
    finishSession: finishSession, runSession: runSession,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;   // Node
  else root.QEngine = api;                                                      // Worker / página
})(typeof self !== "undefined" ? self : this);
