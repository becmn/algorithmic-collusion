/*
 * Fase 2.1: valida o motor em JavaScript (engine.js) contra o Python (Fases 1.3 e 1.7).
 * Uso (na raiz do repositório): node docs/js/test_engine.js
 *
 * Conferências, fixadas antes de rodar:
 *   1. Tabelas: Nash, monopólio, grade, lucros e Q inicial calculados em JS iguais aos do
 *      JSON (erro < 1e-6; o JSON tem 6 casas decimais).
 *   2. Reprodução: Δ e ciclo das 3 sessões gravadas no Python, recalculados em JS a partir
 *      da estratégia final, iguais aos do JSON (erro < 1e-5).
 *   3. Determinismo: a semente 1 rodada duas vezes dá a mesma sessão.
 *   4. Distribuição de Δ (o critério do roteiro): 100 sessões em JS (sementes 1 a 100)
 *      contra as 1.000 do Python. Os geradores aleatórios são diferentes, então a
 *      comparação é de distribuições, não sessão a sessão (Pesquisa 9).
 *      4a. todas convergem;
 *      4b. |Δ médio JS − Δ médio Python| ≤ 2 erros-padrão da diferença;
 *      4c. teste de Kolmogorov-Smirnov (KS) das duas distribuições de Δ com p > 0,05.
 *      Antes do KS, o Δ do JS é arredondado a 6 casas, como no JSON: Δ tem muitos valores
 *      repetidos (sessões que terminam no mesmo ciclo) e, sem o arredondamento, valores
 *      iguais pareceriam diferentes.
 *   Informativo (sem critério): períodos até convergir, fração em ciclo e tempo por sessão.
 * Código próprio do projeto.
 */
"use strict";
const fs = require("fs");
const path = require("path");
const E = require("./engine.js");

const DATA = path.join(__dirname, "..", "data");
const P = JSON.parse(fs.readFileSync(path.join(DATA, "qlearning_params.json"), "utf8"));
const REF = JSON.parse(fs.readFileSync(path.join(DATA, "qlearning_reference.json"), "utf8"));
const SES = JSON.parse(fs.readFileSync(path.join(DATA, "qlearning_sessions.json"), "utf8"));
const N_JS = 100;

const checks = [];
function check(name, ok) { checks.push([name, ok]); }
const mean = (x) => x.reduce((a, b) => a + b, 0) / x.length;
const sd = (x) => { const mu = mean(x); return Math.sqrt(x.reduce((a, b) => a + (b - mu) ** 2, 0) / (x.length - 1)); };
const f3 = (x) => x.toFixed(3), f4 = (x) => x.toFixed(4);

// Kolmogorov-Smirnov para duas amostras: estatística D e p-valor assintótico
function ks2(x, y) {
  const a = [...x].sort((u, v) => u - v), b = [...y].sort((u, v) => u - v);
  let i = 0, j = 0, D = 0;
  while (i < a.length && j < b.length) {
    const v = Math.min(a[i], b[j]);
    while (i < a.length && a[i] <= v) i++;
    while (j < b.length && b[j] <= v) j++;
    D = Math.max(D, Math.abs(i / a.length - j / b.length));
  }
  const ne = (a.length * b.length) / (a.length + b.length);
  const lam = (Math.sqrt(ne) + 0.12 + 0.11 / Math.sqrt(ne)) * D;
  let p = 0;
  for (let k = 1; k <= 100; k++) p += 2 * (k % 2 ? 1 : -1) * Math.exp(-2 * k * k * lam * lam);
  return { D: D, p: Math.min(1, Math.max(0, p)) };
}

// ---------------------------------------------------------------- 1. tabelas
const mk = E.setupMarket(P);
const m = mk.m;
const RP = P.reference_points;
const errRef = Math.max(Math.abs(mk.pN - RP.price_nash), Math.abs(mk.pM - RP.price_monopoly),
  Math.abs(mk.piN - RP.profit_nash), Math.abs(mk.piM - RP.profit_monopoly));
console.log(`1. Nash ${mk.pN.toFixed(6)} (lucro ${mk.piN.toFixed(6)}); monopólio ${mk.pM.toFixed(6)} (lucro ${mk.piM.toFixed(6)}); erro máximo ${errRef.toExponential(1)}`);
check("Nash e monopólio iguais ao JSON", errRef < 1e-6);
let errGrid = 0, errPI = 0, errQ0 = 0;
P.grid.prices.forEach((v, k) => { errGrid = Math.max(errGrid, Math.abs(mk.prices[k] - v)); });
for (let i1 = 0; i1 < m; i1++) for (let i2 = 0; i2 < m; i2++) for (let i = 0; i < 2; i++)
  errPI = Math.max(errPI, Math.abs(mk.PI[(i1 * m + i2) * 2 + i] - P.profit_table.values[i1][i2][i]));
for (let i = 0; i < 2; i++) for (let k = 0; k < m; k++)
  errQ0 = Math.max(errQ0, Math.abs(mk.q0[i][k] - P.q0.values[i][k]));
console.log(`   Tabelas: erro máximo grade ${errGrid.toExponential(1)}, lucros ${errPI.toExponential(1)}, Q inicial ${errQ0.toExponential(1)}`);
check("grade igual ao JSON", errGrid < 1e-6);
check("tabela de lucros igual ao JSON", errPI < 1e-6);
check("Q inicial igual ao JSON", errQ0 < 1e-6);

// ---------------------------------------------------------------- 2. reprodução das sessões gravadas
for (const s of SES.sessions) {
  const g = new Int32Array(2 * m * m);
  s.greedy_strategy.firm1.forEach((v, k) => { g[k] = v; });
  s.greedy_strategy.firm2.forEach((v, k) => { g[m * m + k] = v; });
  const sStart = s.final_state_idx[0] * m + s.final_state_idx[1];
  const cyc = E.limitCycle(g, sStart, m);
  let pbar = 0;
  for (const st of cyc) pbar += 0.5 * (mk.PI[st * 2] + mk.PI[st * 2 + 1]);
  const d = E.deltaIndex(mk, pbar / cyc.length);
  const sameCycle = JSON.stringify(cyc.map((st) => [Math.floor(st / m), st % m])) === JSON.stringify(s.cycle_idx);
  console.log(`2. Semente ${s.seed} (Python): Δ JS ${d.toFixed(6)} × JSON ${s.delta.toFixed(6)}; ciclo de ${cyc.length}, igual: ${sameCycle}`);
  check(`Δ da sessão gravada ${s.seed} igual`, Math.abs(d - s.delta) < 1e-5);
  check(`ciclo da sessão gravada ${s.seed} igual`, sameCycle);
}

// ---------------------------------------------------------------- 3. determinismo
const r1 = E.runSession(mk, 1), r1b = E.runSession(mk, 1);
console.log(`3. Semente 1 duas vezes: períodos ${r1.periods} e ${r1b.periods}; Δ ${r1.delta.toFixed(6)} e ${r1b.delta.toFixed(6)}`);
check("semente 1 reproduz a mesma sessão", r1.periods === r1b.periods && r1.delta === r1b.delta);

// ---------------------------------------------------------------- 4. 100 sessões
const t0 = process.hrtime.bigint();
const js = [];
for (let seed = 1; seed <= N_JS; seed++) js.push(E.runSession(mk, seed));
const secs = Number(process.hrtime.bigint() - t0) / 1e9;

const dJS = js.map((r) => r.delta), dPy = REF.all_sessions.delta;
const dJS6 = dJS.map((x) => Math.round(x * 1e6) / 1e6);   // mesma precisão do JSON, para o KS
const pJS = js.map((r) => r.periods), pPy = REF.all_sessions.periods;
const cycJS = js.filter((r) => r.cycle_len > 1).length / N_JS;
const cycPy = REF.all_sessions.cycle_len.filter((c) => c > 1).length / dPy.length;
const nConv = js.filter((r) => r.converged).length;
const se = Math.sqrt(sd(dJS) ** 2 / dJS.length + sd(dPy) ** 2 / dPy.length);
const diff = mean(dJS) - mean(dPy);
const ksD = ks2(dJS6, dPy), ksP = ks2(pJS, pPy);
const seP = Math.sqrt(sd(pJS) ** 2 / pJS.length + sd(pPy) ** 2 / pPy.length);

console.log(`4. ${N_JS} sessões em JS (sementes 1 a ${N_JS}) × ${dPy.length} no Python:`);
console.log(`   convergiram: ${nConv} de ${N_JS}`);
console.log(`   Δ médio: JS ${f3(mean(dJS))} (DP ${f3(sd(dJS))}) × Python ${f3(mean(dPy))} (DP ${f3(sd(dPy))}); diferença ${f4(diff)} = ${(diff / se).toFixed(2)} erros-padrão`);
console.log(`   KS de Δ: D = ${f3(ksD.D)}, p = ${f3(ksD.p)}`);
console.log(`   [info] períodos médios: JS ${Math.round(mean(pJS)).toLocaleString("pt-BR")} × Python ${Math.round(mean(pPy)).toLocaleString("pt-BR")} (${((mean(pJS) - mean(pPy)) / seP).toFixed(2)} erros-padrão; KS p = ${f3(ksP.p)})`);
console.log(`   [info] fração em ciclo: JS ${f3(cycJS)} × Python ${f3(cycPy)}`);
console.log(`   [info] tempo: ${secs.toFixed(1)} s no total, ${(secs / N_JS).toFixed(3)} s por sessão (Node ${process.version})`);
check(`as ${N_JS} sessões convergem`, nConv === N_JS);
check("Δ médio a até 2 erros-padrão do Python", Math.abs(diff) <= 2 * se);
check("KS de Δ com p > 0,05", ksD.p > 0.05);

// ---------------------------------------------------------------- resumo
console.log("\nConferências:");
for (const [name, ok] of checks) console.log(`  ${ok ? "True " : "False"}  ${name}`);
const allOk = checks.every((c) => c[1]);
console.log(`\n${checks.filter((c) => c[1]).length} de ${checks.length} conferências True`);
process.exit(allOk ? 0 : 1);
