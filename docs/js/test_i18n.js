/*
 * Fase 2.2: confere os textos do simulador (i18n.js) e o HTML.
 *   1. Português e inglês têm exatamente as mesmas chaves, nenhuma vazia.
 *   2. Cada texto tem os mesmos {nomes} de valores nas duas línguas.
 *   3. Toda chave usada no index.html (data-i18n, data-i18n-html, data-i18n-attr) e no
 *      app.js (t("...")) existe no dicionário.
 *   4. O texto em inglês não tem palavras com acentos do português (sobra de tradução),
 *      fora os nomes próprios da lista abaixo.
 *   5. Nenhum texto chama o resultado do Q-learning de colusão: "colus"/"collu" só aparece
 *      nas frases que dizem que o resultado NÃO prova colusão ou que falam dos casos reais.
 * Uso (na raiz do repositório): node docs/js/test_i18n.js
 * Código próprio do projeto.
 */
"use strict";
const fs = require("fs");
const path = require("path");
const I = require("./i18n.js");

const checks = [];
const check = (name, ok, detail) => { checks.push([name, ok]); if (!ok && detail) console.log("  ->", detail); };
const pt = I.STR.pt, en = I.STR.en;
const kp = Object.keys(pt).sort(), ke = Object.keys(en).sort();

const onlyPt = kp.filter((k) => !(k in en)), onlyEn = ke.filter((k) => !(k in pt));
check("mesmas chaves em português e inglês", !onlyPt.length && !onlyEn.length, { onlyPt, onlyEn });
const empty = kp.filter((k) => !String(pt[k]).trim() || !String(en[k] || "").trim());
check("nenhum texto vazio", !empty.length, empty);

const vars = (s) => (String(s).match(/\{(\w+)\}/g) || []).sort().join(",");
const badVars = kp.filter((k) => k in en && vars(pt[k]) !== vars(en[k]));
check("mesmos {valores} nas duas línguas", !badVars.length, badVars);

const html = fs.readFileSync(path.join(__dirname, "..", "index.html"), "utf8");
const app = fs.readFileSync(path.join(__dirname, "app.js"), "utf8");
const used = new Set();
for (const m of html.matchAll(/data-i18n(?:-html)?="([\w]+)"/g)) used.add(m[1]);
for (const m of html.matchAll(/data-i18n-attr="([^"]+)"/g)) m[1].split(";").forEach((p) => used.add(p.split(":")[1].trim()));
for (const m of app.matchAll(/\bt\("([\w]+)"/g)) used.add(m[1]);
for (const m of app.matchAll(/"((?:status|btn|score|verdict|ann|delta|source|outcome|p)_[\w]+)"/g)) used.add(m[1]);
const missing = [...used].filter((k) => !(k in pt));
check(`as ${used.size} chaves usadas no HTML e no app.js existem`, !missing.length, missing);
// condições do teste de separação (chaves montadas a partir dos níveis do JSON)
const ref = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "data", "qlearning_reference.json"), "utf8"));
const condMissing = ref.separation_test.rows.map((r) => "cond_" + r.level.replace(/[-.]/g, "_")).filter((k) => !(k in pt));
check("há texto para todas as condições do teste de separação", !condMissing.length, condMissing);

// horizonte do teste do desvio: o do JSON (results/) tem de ser o de deviation.js
check("horizonte do desvio igual no JSON e em deviation.js", ref.deviation.protocol.horizon === require("./deviation.js").T_DEV);
// valores globais ({nsess}, {horizon}…) que a página preenche com setGlobals a partir dos JSON
const GLOBALS = ["nsess", "block", "stable", "tmax", "m", "horizon", "nrec"];
const appSrc = app;
check("todo valor global usado nos textos é definido no app.js", GLOBALS.every((g) => new RegExp("\\b" + g + ": \\(\\)").test(appSrc)));

const ALLOW = ["Denicolò", "Português"];   // nomes próprios e o botão que leva ao português
const ptChars = /[ãõçáéíóúâêôà]/i;
const leftovers = ke.filter((k) => {
  const plain = String(en[k]).replace(/<a [^>]*>/g, "").replace(/https?:\S+/g, "");
  return plain.split(/\s+/).some((w) => ptChars.test(w) && !ALLOW.some((a) => w.includes(a)));
});
check("inglês sem sobras de português", !leftovers.length, leftovers.map((k) => `${k}: ${en[k]}`));

const okCollusion = new Set(["dev_caveat", "why_not", "cite_original", "sep_rule_note"]);
const collusion = kp.filter((k) => /colus|collu/i.test(pt[k] + " " + en[k]) && !okCollusion.has(k));
check("nenhum texto chama o resultado do Q-learning de colusão", !collusion.length, collusion);

console.log("Conferências:");
for (const [name, ok] of checks) console.log(`  ${ok ? "True " : "False"}  ${name}`);
const nOk = checks.filter((c) => c[1]).length;
console.log(`\n${nOk} de ${checks.length} conferências True (${kp.length} textos em cada língua)`);
process.exitCode = nOk === checks.length ? 0 : 1;
