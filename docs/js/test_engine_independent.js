/*
 * Teste independente do motor (revisão da Fase 1, 16/10/2026).
 * test_engine.js compara o motor com o Python por DISTRIBUIÇÃO (os geradores são diferentes).
 * Este arquivo cobre o que ele não cobre: reimplementa o laço de aprendizado do zero, na mesma
 * estrutura de qlearning/session.py, com o mesmo gerador, e exige igualdade EXATA com engine.js
 * em períodos, Δ, ciclo, estado final, estratégia gulosa e matriz Q (sementes 1, 2 e 48, com
 * dois tamanhos de bloco). Também confere o gerador contra o sfc32 canônico.
 * Uso (na raiz do repositório): node docs/js/test_engine_independent.js
 * Código próprio do projeto.
 */
"use strict";
const path = require("path");
const fs = require("fs");
const ROOT = path.join(__dirname, "..", "..");
const E = require(path.join(ROOT, "docs", "js", "engine.js"));
const P = JSON.parse(fs.readFileSync(path.join(ROOT, "docs", "data", "qlearning_params.json"), "utf8"));
const checks = [];
const check = (label, ok) => { checks.push([label, ok]); };

// --- gerador escrito à parte: sfc32 canônico + o mesmo misturador de semente do motor
function seedMixer(seed) {
  let x = seed >>> 0;
  return () => {
    x = (x + 0x9e3779b9) >>> 0;
    let z = x;
    z = Math.imul(z ^ (z >>> 16), 0x85ebca6b);
    z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35);
    return (z ^ (z >>> 16)) >>> 0;
  };
}
function sfc32(a, b, c, d) {
  return function () {
    a |= 0; b |= 0; c |= 0; d |= 0;
    const t = (((a + b) | 0) + d) | 0;
    d = (d + 1) | 0;
    a = b ^ (b >>> 9);
    b = (c + (c << 3)) | 0;
    c = (c << 21) | (c >>> 11);
    c = (c + t) | 0;
    return (t >>> 0) / 4294967296;
  };
}
function rng(seed) {
  const s = seedMixer(seed);
  const r = sfc32(s(), s(), s(), s());
  for (let k = 0; k < 12; k++) r();            // descarte inicial, como no motor
  return r;
}
for (const seed of [0, 1, 48, 2 ** 32 - 1]) {
  const a = E.makeRng(seed), b = rng(seed);
  let ok = true;
  for (let k = 0; k < 1e6 && ok; k++) if (a() !== b()) ok = false;
  check(`gerador igual ao sfc32 canônico (semente ${seed}, 10⁶ saídas)`, ok);
}

// --- mercado: grade do motor (já conferida contra o JSON); lucros recalculados aqui
const m = 15, c = 1, a = 2, a0 = 0, mu = 0.25;
const delta = 0.95, alpha = 0.15, beta = 4e-6, tStable = 100000, tMax = 1e9;
const mk = E.setupMarket(P);
const prices = Array.from(mk.prices);
function demand(p) {
  const e = p.map((x) => Math.exp((a - x) / mu));
  const d = e[0] + e[1] + Math.exp(a0 / mu);
  return e.map((x) => x / d);
}
const PI = [];
for (let i = 0; i < m; i++) {
  PI.push([]);
  for (let j = 0; j < m; j++) {
    const q = demand([prices[i], prices[j]]);
    PI[i].push([(prices[i] - c) * q[0], (prices[j] - c) * q[1]]);
  }
}
let dPI = 0;
for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) for (let k = 0; k < 2; k++)
  dPI = Math.max(dPI, Math.abs(PI[i][j][k] - mk.PI[(i * m + j) * 2 + k]));
check("tabela de lucros recalculada igual à do motor (erro < 1e-15)", dPI < 1e-15);

// --- laço de aprendizado, reescrito na estrutura de session.py
function argmax(row) {
  let b = 0, v = row[0];
  for (let j = 1; j < row.length; j++) if (row[j] > v) { v = row[j]; b = j; }   // empate: menor preço
  return [b, v];
}
function session(seed) {
  const R = rng(seed), S = m * m;
  const Q = [[], []];
  for (let i = 0; i < 2; i++) {                // Q inicial da eq. 8: lucro médio contra rival uniforme
    const q0 = [];
    for (let k = 0; k < m; k++) {
      let s = 0;
      for (let j = 0; j < m; j++) s += i === 0 ? PI[k][j][0] : PI[j][k][1];
      q0.push(s / m / (1 - delta));
    }
    for (let s = 0; s < S; s++) Q[i].push(Float64Array.from(q0));
  }
  const greedy = [[], []], maxq = [[], []];
  for (let i = 0; i < 2; i++) for (let s = 0; s < S; s++) {
    const [g, v] = argmax(Q[i][s]); greedy[i].push(g); maxq[i].push(v);
  }
  let s = Math.floor(R() * S), stable = 0, t = 0, conv = false;
  const act = [0, 0];
  while (t < tMax) {
    const eps = Math.exp(-beta * t);
    for (let i = 0; i < 2; i++) act[i] = R() < eps ? Math.floor(R() * m) : greedy[i][s];
    const s1 = act[0] * m + act[1];
    let changed = false;
    for (let i = 0; i < 2; i++) {
      const target = PI[act[0]][act[1]][i] + delta * maxq[i][s1];
      Q[i][s][act[i]] = (1 - alpha) * Q[i][s][act[i]] + alpha * target;
      const [g, v] = argmax(Q[i][s]);
      if (g !== greedy[i][s]) { changed = true; greedy[i][s] = g; }
      maxq[i][s] = v;
    }
    stable = changed ? 0 : stable + 1;
    s = s1; t++;
    if (stable >= tStable) { conv = true; break; }
  }
  const first = new Array(S).fill(-1), pathS = [];
  let x = s;
  while (first[x] < 0) { first[x] = pathS.length; pathS.push(x); x = greedy[0][x] * m + greedy[1][x]; }
  const cyc = pathS.slice(first[x]);
  let pb = 0;
  for (const st of cyc) { const i1 = Math.floor(st / m), i2 = st % m; pb += 0.5 * (PI[i1][i2][0] + PI[i1][i2][1]); }
  return { conv, t, d: (pb / cyc.length - mk.piN) / (mk.piM - mk.piN), cyc, s, greedy, Q };
}

for (const seed of [1, 2, 48]) {
  const mine = session(seed);
  for (const bs of [10000, 7777]) {
    const sess = E.createSession(mk, seed);
    while (!sess.done) E.runBlock(sess, bs);
    const js = E.finishSession(sess);
    let gEq = true, dQ = 0;
    for (let i = 0; i < 2; i++) for (let st = 0; st < m * m; st++) {
      if (js.greedy[i * m * m + st] !== mine.greedy[i][st]) gEq = false;
      for (let k = 0; k < m; k++) dQ = Math.max(dQ, Math.abs(sess.Q[(i * m * m + st) * m + k] - mine.Q[i][st][k]));
    }
    check(`semente ${seed}, bloco ${bs}: convergência, períodos, Δ, ciclo e estado final iguais`,
      js.converged === mine.conv && js.periods === mine.t && js.delta === mine.d &&
      js.cycle_len === mine.cyc.length && sess.s === mine.s);
    check(`semente ${seed}, bloco ${bs}: estratégia gulosa e matriz Q idênticas`, gEq && dQ === 0);
  }
}

console.log("Conferências:");
for (const [label, ok] of checks) console.log(`  ${ok ? "True " : "False"}  ${label}`);
const nOk = checks.filter((x) => x[1]).length;
console.log(`\n${nOk} de ${checks.length} conferências True`);
process.exitCode = nOk === checks.length ? 0 : 1;
