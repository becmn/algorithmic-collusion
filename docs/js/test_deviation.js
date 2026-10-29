/*
 * Fase 2.2: valida o teste de desvio em JavaScript (deviation.js) contra o Python.
 * Para as 3 sessões gravadas (qlearning_sessions.json), a partir da estratégia final
 * gravada e de CADA ponto do ciclo, exige igualdade com o teste de desvio do Python:
 * trajetórias de preço dos dois ramos (índices), R, G, V e lucro descontado da firma 1
 * (erro < 1e-5; o JSON tem 6 casas decimais).
 * Uso (na raiz do repositório): node docs/js/test_deviation.js
 * Código próprio do projeto.
 */
"use strict";
const fs = require("fs");
const path = require("path");
const E = require("./engine.js");
const D = require("./deviation.js");

const DATA = path.join(__dirname, "..", "data");
const P = JSON.parse(fs.readFileSync(path.join(DATA, "qlearning_params.json"), "utf8"));
const SES = JSON.parse(fs.readFileSync(path.join(DATA, "qlearning_sessions.json"), "utf8"));
const mk = E.setupMarket(P);
const m = mk.m, S = m * m;
const close = (a, b) => (a === null && b === null) || (a !== null && b !== null && Math.abs(a - b) < 1e-5);
const checks = [];

for (const s of SES.sessions) {
  const g = new Int32Array(2 * S);
  s.greedy_strategy.firm1.forEach((v, k) => { g[k] = v; });
  s.greedy_strategy.firm2.forEach((v, k) => { g[S + k] = v; });
  const cyc = E.limitCycle(g, s.final_state_idx[0] * m + s.final_state_idx[1], m);
  for (const st of s.deviation_test.starts) {
    const js = D.run(mk, g, cyc, st.cycle_position, P.learning.delta.value);
    const samePaths =
      JSON.stringify(js.control.idx.map((x) => x[0])) === JSON.stringify(st.control.idx_firm1) &&
      JSON.stringify(js.control.idx.map((x) => x[1])) === JSON.stringify(st.control.idx_firm2) &&
      JSON.stringify(js.deviation.idx.map((x) => x[0])) === JSON.stringify(st.deviation.idx_firm1) &&
      JSON.stringify(js.deviation.idx.map((x) => x[1])) === JSON.stringify(st.deviation.idx_firm2);
    const sameMetrics = close(js.R, st.R) && close(js.G, st.G) && js.V === st.V &&
      close(js.discounted_profit_firm1_control, st.discounted_profit_firm1_control) &&
      close(js.discounted_profit_firm1_deviation, st.discounted_profit_firm1_deviation) &&
      js.did_deviate === st.did_deviate;
    console.log(`Semente ${s.seed}, ponto ${st.cycle_position} do ciclo: R ${js.R.toFixed(6)} × ${st.R}; ` +
      `G ${js.G.toFixed(6)} × ${st.G}; V ${js.V} × ${st.V}`);
    checks.push([`semente ${s.seed}, ponto ${st.cycle_position}: trajetórias iguais`, samePaths]);
    checks.push([`semente ${s.seed}, ponto ${st.cycle_position}: R, G, V e lucros iguais`, sameMetrics]);
  }
}

console.log("\nConferências:");
for (const [name, ok] of checks) console.log(`  ${ok ? "True " : "False"}  ${name}`);
const nOk = checks.filter((c) => c[1]).length;
console.log(`\n${nOk} de ${checks.length} conferências True`);
process.exitCode = nOk === checks.length ? 0 : 1;
