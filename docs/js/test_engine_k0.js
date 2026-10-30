/*
 * Revisão de 30/10/2026: valida o motor SEM memória (engine.js, opts.memory = 0) contra o
 * Python (results/qlearning_separation_sessions.csv, condição k0_beta4e-06: 1.000 sessões sem
 * memória com os parâmetros do artigo, Fase 1.8).
 * Uso (na raiz do repositório): node docs/js/test_engine_k0.js
 *
 * Conferências, fixadas antes de rodar (as mesmas do motor com memória, test_engine.js):
 *   1. Laço independente: uma reimplementação do zero do laço sem memória, com o mesmo
 *      gerador, dá a mesma sessão que engine.js (períodos, Δ e preços finais) nas sementes 1 a 5.
 *   2. Distribuição de Δ: 100 sessões em JS (sementes 1 a 100) contra as 1.000 do Python
 *      (geradores diferentes: compara distribuições). 2a. todas convergem; 2b. |Δ médio JS −
 *      Δ médio Python| ≤ 2 erros-padrão da diferença; 2c. KS com p > 0,05 (Δ do JS arredondado
 *      a 6 casas nos dois lados: Δ sem memória tem poucos valores distintos e diferenças na 10ª casa
 *      fariam valores iguais parecerem diferentes).
 *   3. A estratégia final copiada para os 225 estados faz a rival não reagir ao corte.
 * Código próprio do projeto.
 */
"use strict";
const fs = require("fs");
const path = require("path");
const E = require("./engine.js");
const D = require("./deviation.js");

const ROOT = path.join(__dirname, "..", "..");
const P = JSON.parse(fs.readFileSync(path.join(ROOT, "docs", "data", "qlearning_params.json"), "utf8"));
const N_JS = 100;
const checks = [];
const check = (name, ok, detail) => { checks.push([name, !!ok]); if (!ok && detail !== undefined) console.log("  ->", detail); };
const mean = (x) => x.reduce((a, b) => a + b, 0) / x.length;
const sd = (x) => { const mu = mean(x); return Math.sqrt(x.reduce((a, b) => a + (b - mu) ** 2, 0) / (x.length - 1)); };

function ks2(x, y) {
  const a = [...x].sort((u, v) => u - v), b = [...y].sort((u, v) => u - v);
  let i = 0, j = 0, Dm = 0;
  while (i < a.length && j < b.length) {
    const v = Math.min(a[i], b[j]);
    while (i < a.length && a[i] <= v) i++;
    while (j < b.length && b[j] <= v) j++;
    Dm = Math.max(Dm, Math.abs(i / a.length - j / b.length));
  }
  const ne = (a.length * b.length) / (a.length + b.length);
  const lam = (Math.sqrt(ne) + 0.12 + 0.11 / Math.sqrt(ne)) * Dm;
  let p = 0;
  for (let k = 1; k <= 100; k++) p += 2 * (k % 2 ? 1 : -1) * Math.exp(-2 * k * k * lam * lam);
  return { D: Dm, p: Math.min(1, Math.max(0, p)) };
}

const mk = E.setupMarket(P);
const m = mk.m;

// ---------------------------------------------------------------- 1. laço independente
// Escrito do zero a partir de controls.run_session_mem (memory = 0, eps_min = 0), com o
// mesmo gerador e a mesma ordem de sorteios do engine.js (o estado inicial é sorteado e
// descartado, como no engine).
function loopK0(seed) {
  const rng = E.makeRng(seed);
  const Q = [Float64Array.from(mk.q0[0]), Float64Array.from(mk.q0[1])];
  const argmax = (q) => { let b = 0; for (let j = 1; j < m; j++) if (q[j] > q[b]) b = j; return b; };
  const g = [argmax(Q[0]), argmax(Q[1])];
  rng();                                   // estado inicial (um só estado: descartado)
  let t = 0, stable = 0;
  while (t < mk.tMax) {
    const eps = Math.exp(-mk.beta * t);
    const a = [0, 0];
    for (let i = 0; i < 2; i++) a[i] = rng() < eps ? Math.floor(rng() * m) : g[i];
    let changed = false;
    for (let i = 0; i < 2; i++) {
      const pi = mk.PI[(a[0] * m + a[1]) * 2 + i];
      const mx = Q[i][argmax(Q[i])];
      Q[i][a[i]] = (1 - mk.alpha) * Q[i][a[i]] + mk.alpha * (pi + mk.delta * mx);
      const nb = argmax(Q[i]);
      if (nb !== g[i]) { changed = true; g[i] = nb; }
    }
    stable = changed ? 0 : stable + 1;
    t += 1;
    if (stable >= mk.tStable) break;
  }
  return { periods: t, g: g };
}
let same = true;
for (let seed = 1; seed <= 5; seed++) {
  const a = E.runSession(mk, seed, { memory: 0 }), b = loopK0(seed);
  same = same && a.periods === b.periods && a.cycle_idx[0][0] === b.g[0] && a.cycle_idx[0][1] === b.g[1];
}
check("laço independente sem memória = engine.js (sementes 1 a 5)", same);

// ---------------------------------------------------------------- 2. distribuição × Python
const csv = fs.readFileSync(path.join(ROOT, "results", "qlearning_separation_sessions.csv"), "utf8").trim().split(/\r?\n/);
const head = csv[0].split(",");
const ci = head.indexOf("condition"), di = head.indexOf("delta");
const dPy = csv.slice(1).map((l) => l.split(",")).filter((r) => r[ci] === "k0_beta4e-06").map((r) => Math.round(Number(r[di]) * 1e6) / 1e6);   // 6 casas nos dois lados: Δ tem valores repetidos
check("Python: 1.000 sessões sem memória lidas de results/", dPy.length === 1000, dPy.length);
const t0 = Date.now();
const js = [];
for (let seed = 1; seed <= N_JS; seed++) js.push(E.runSession(mk, seed, { memory: 0 }));
const secs = (Date.now() - t0) / 1000;
const dJS = js.map((r) => Math.round(r.delta * 1e6) / 1e6);
check(`todas as ${N_JS} sessões em JS convergem`, js.every((r) => r.converged));
const se = Math.sqrt(sd(dJS) ** 2 / dJS.length + sd(dPy) ** 2 / dPy.length);
const z = (mean(dJS) - mean(dPy)) / se;
check("Δ médio JS × Python a até 2 erros-padrão", Math.abs(z) <= 2, z);
const ks = ks2(dJS, dPy);
check("KS de Δ com p > 0,05", ks.p > 0.05, ks);
console.log(`   Δ médio: JS ${mean(dJS).toFixed(4)} × Python ${mean(dPy).toFixed(4)} (${z.toFixed(2)} erros-padrão); KS D = ${ks.D.toFixed(3)}, p = ${ks.p.toFixed(3)}; ${(secs / N_JS).toFixed(2)} s por sessão`);

// ---------------------------------------------------------------- 3. corte sem memória
const r = js[0];
const cyc = [r.cycle_idx[0][0] * m + r.cycle_idx[0][1]];
const dv = D.run(mk, r.greedy, cyc, 0, mk.delta);
check("sem memória, a rival não reage ao corte (R = 0)",
  dv.R === 0 && dv.deviation.idx.slice(1).every((st, k) => st[1] === dv.control.idx[k + 1][1]));

console.log("Conferências:");
for (const [name, ok] of checks) console.log(`  ${ok ? "True " : "False"}  ${name}`);
const nOk = checks.filter((c) => c[1]).length;
console.log(`\n${nOk} de ${checks.length} conferências True`);
process.exitCode = nOk === checks.length ? 0 : 1;
