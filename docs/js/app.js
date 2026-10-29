/*
 * Página do simulador (Fase 2.2): carrega os JSON da Fase 1.7, desenha os gráficos (SVG),
 * roda sessões ao vivo no Web Worker e aplica o teste do desvio (deviation.js).
 * Estados: (a) sessão gravada no Python, estática; (b) ao vivo, aprendendo; (c) pausado;
 * (d) convergido, com "Forçar desvio" liberado; com movimento reduzido, sem animação.
 * Textos em i18n.js. Código próprio do projeto.
 */
(function () {
  "use strict";
  const I = window.I18N, E = window.QEngine, D = window.QDeviation;
  const t = I.t, num = I.num;
  const $ = (id) => document.getElementById(id);
  const reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const BLOCKS_PER_SECOND = 40;   // ritmo da animação (exibição; não muda o cálculo)
  const TAU_MS = 380;             // ritmo da animação do desvio

  const S = {
    params: null, ref: null, sessions: null, mk: null,
    mode: "recorded",   // "recorded" | "live"
    seed: null, blocks: [], shown: 0, final: null,
    running: false, paused: false, worker: null, runId: 0,
    dev: null, devShown: 16, devTimer: null, statusKey: null, statusVars: null,
  };

  // ------------------------------------------------------------------ dados
  function loadJSON(name) {
    return fetch("data/" + name).then((r) => {
      if (!r.ok) throw new Error(name + ": " + r.status);
      return r.json();
    });
  }

  function recordedSession() {
    const s = S.sessions.sessions.find((x) => x.opens_page) || S.sessions.sessions[0];
    const L = s.learning_blocks_linear, m = S.mk.m, Ssz = m * m;
    const blocks = L.t_start.map((t0, k) => ({
      t_start: t0, n_periods: L.n_periods[k], price_firm1: L.price_firm1[k],
      price_firm2: L.price_firm2[k], delta: L.delta[k], epsilon_start: L.epsilon_start[k],
    }));
    const greedy = new Int32Array(2 * Ssz);
    s.greedy_strategy.firm1.forEach((v, k) => { greedy[k] = v; });
    s.greedy_strategy.firm2.forEach((v, k) => { greedy[Ssz + k] = v; });
    return {
      seed: s.seed, blocks: blocks,
      final: { converged: s.converged, periods: s.periods, delta: s.delta, cycle_len: s.cycle_len,
               greedy: greedy, cycle: s.cycle_idx.map((p) => p[0] * m + p[1]) },
    };
  }

  // ------------------------------------------------------------------ SVG
  const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  function frame(el, hRatio, hMin, hMax) {
    const w = Math.max(260, Math.round(el.clientWidth || 600));
    const h = Math.round(Math.max(hMin, Math.min(hMax, w * hRatio)));
    return { w: w, h: h, fs: w < 420 ? 11.5 : 13 };
  }
  const svgOpen = (w, h) => `<svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" aria-hidden="true" focusable="false">`;
  function scale(d0, d1, r0, r1) { return (v) => r0 + (v - d0) / (d1 - d0) * (r1 - r0); }

  function priceAxis(o, y, x0, x1, fs) {
    let s = "";
    for (let p = 1.5; p <= 1.95; p += 0.1) {
      const yy = y(p).toFixed(1);
      s += `<line x1="${x0}" x2="${x1}" y1="${yy}" y2="${yy}" stroke="#ece9e2"/>`;
      s += `<text x="${x0 - 6}" y="${yy}" dy="0.32em" text-anchor="end" font-size="${fs}" fill="#55555b">${esc(num(p, 1))}</text>`;
    }
    return s;
  }

  function refLines(y, x0, x1, fs) {
    const rp = S.params.reference_points;
    const yN = y(rp.price_nash), yM = y(rp.price_monopoly);
    return `<rect x="${x0}" y="${yM}" width="${x1 - x0}" height="${yN - yM}" fill="rgba(31,95,168,0.07)"/>` +
      `<line x1="${x0}" x2="${x1}" y1="${yM}" y2="${yM}" stroke="#3f3f44" stroke-width="1.5" stroke-dasharray="7 3 2 3"/>` +
      `<line x1="${x0}" x2="${x1}" y1="${yN}" y2="${yN}" stroke="#3f3f44" stroke-width="1.5" stroke-dasharray="2 3"/>` +
      `<text x="${x1 - 4}" y="${yM - 6}" text-anchor="end" font-size="${fs}" fill="#3f3f44" paint-order="stroke" stroke="#fff" stroke-width="3">${esc(t("line_monopoly"))}</text>` +
      `<text x="${x1 - 4}" y="${yN + fs + 3}" text-anchor="end" font-size="${fs}" fill="#3f3f44" paint-order="stroke" stroke="#fff" stroke-width="3">${esc(t("line_nash"))}</text>`;
  }

  function yDomain() {
    const g = S.params.grid.prices;
    return [g[0] - 0.02, g[g.length - 1] + 0.02];
  }

  // Gráfico do aprendizado: preço médio por bloco de 10.000 períodos.
  function drawLearning() {
    const el = $("learn-chart");
    const f = frame(el, 0.52, 230, 360);
    const m = { l: 44, r: 10, t: 12, b: 40 };
    const blocks = S.blocks.slice(0, Math.floor(S.shown));
    const lastEnd = S.blocks.length ? S.blocks[S.blocks.length - 1].t_start + S.blocks[S.blocks.length - 1].n_periods : 0;
    const xMaxRaw = Math.max(S.final ? S.final.periods : lastEnd, 1.5e6);
    const xMax = Math.ceil(xMaxRaw / 5e5) * 5e5;
    const [y0, y1] = yDomain();
    const x = scale(0, xMax, m.l, f.w - m.r), y = scale(y0, y1, f.h - m.b, m.t);
    let s = svgOpen(f.w, f.h) + priceAxis(null, y, m.l, f.w - m.r, f.fs);
    for (let v = 0; v <= xMax + 1; v += 5e5) {
      const xx = x(v).toFixed(1);
      s += `<line x1="${xx}" x2="${xx}" y1="${f.h - m.b}" y2="${f.h - m.b + 4}" stroke="#55555b"/>`;
      s += `<text x="${xx}" y="${f.h - m.b + 16}" text-anchor="middle" font-size="${f.fs}" fill="#55555b">${esc(num(v / 1e6, 1))}</text>`;
    }
    s += `<text x="${(m.l + f.w - m.r) / 2}" y="${f.h - 4}" text-anchor="middle" font-size="${f.fs}" fill="#55555b">${esc(t("axis_periods_millions"))}</text>`;
    s += refLines(y, m.l, f.w - m.r, f.fs);
    const pts = (key) => blocks.map((b) => `${x(b.t_start + b.n_periods / 2).toFixed(1)},${y(b[key]).toFixed(1)}`).join(" ");
    if (blocks.length) {
      s += `<polyline points="${pts("price_firm2")}" fill="none" stroke="#a8480a" stroke-width="2" stroke-dasharray="6 4" stroke-linejoin="round"/>`;
      s += `<polyline points="${pts("price_firm1")}" fill="none" stroke="#1f5fa8" stroke-width="2" stroke-linejoin="round"/>`;
    }
    s += `<line x1="${m.l}" x2="${m.l}" y1="${m.t}" y2="${f.h - m.b}" stroke="#55555b"/>`;
    s += `<line x1="${m.l}" x2="${f.w - m.r}" y1="${f.h - m.b}" y2="${f.h - m.b}" stroke="#55555b"/>`;
    el.innerHTML = s + "</svg>";
  }

  // Gráfico do desvio: degraus de τ = 0 a 15, ramo com corte × ramo sem corte.
  function drawDeviation() {
    const el = $("dev-chart");
    const f = frame(el, 0.5, 240, 340);
    const m = { l: 44, r: 12, t: 3 * f.fs + 20, b: 40 };
    const [y0, y1] = yDomain();
    const x = scale(0, 16, m.l, f.w - m.r), y = scale(y0, y1, f.h - m.b, m.t);
    let s = svgOpen(f.w, f.h) + priceAxis(null, y, m.l, f.w - m.r, f.fs);
    for (let k = 0; k <= 15; k += (f.w < 420 ? 3 : 1)) {
      s += `<text x="${x(k + 0.5)}" y="${f.h - m.b + 16}" text-anchor="middle" font-size="${f.fs}" fill="#55555b">${k}</text>`;
    }
    s += `<text x="${(m.l + f.w - m.r) / 2}" y="${f.h - 4}" text-anchor="middle" font-size="${f.fs}" fill="#55555b">${esc(t("axis_tau"))}</text>`;
    s += refLines(y, m.l, f.w - m.r, f.fs);
    const d = S.dev;
    if (d) {
      const n = Math.min(16, S.devShown);
      const step = (prices, i, upto) => {
        let p = "";
        for (let k = 0; k < upto; k++) {
          const yy = y(prices[k][i]).toFixed(1);
          p += (k === 0 ? "M" : "L") + x(k).toFixed(1) + "," + yy + "L" + x(k + 1).toFixed(1) + "," + yy;
        }
        return p;
      };
      s += `<path d="${step(d.control.price, 0, 16)}" fill="none" stroke="#7d7d84" stroke-width="2" stroke-dasharray="1.5 3.5" stroke-linecap="round"/>`;
      s += `<path d="${step(d.control.price, 1, 16)}" fill="none" stroke="#7d7d84" stroke-width="2" stroke-dasharray="1.5 3.5" stroke-linecap="round"/>`;
      s += `<path d="${step(d.deviation.price, 1, n)}" fill="none" stroke="#a8480a" stroke-width="2.5" stroke-dasharray="6 4"/>`;
      s += `<path d="${step(d.deviation.price, 0, n)}" fill="none" stroke="#1f5fa8" stroke-width="2.5"/>`;
      // anotações em sequência
      const ann = (k, row, key) => {
        const xx = x(k).toFixed(1), yy = m.t - 6 - row * (f.fs + 4);
        return `<line x1="${xx}" x2="${xx}" y1="${yy + 4}" y2="${f.h - m.b}" stroke="#3f3f44" stroke-width="1" stroke-dasharray="3 3"/>` +
          `<text x="${xx}" y="${yy}" font-size="${f.fs}" fill="#1c1c1e" text-anchor="${k > 11 ? "end" : "start"}" dx="${k > 11 ? -3 : 3}">${esc(t(key))}</text>`;
      };
      if (n > 1 && d.did_deviate) s += ann(1, 2, "ann_cut");
      if (n > 2 && d.did_deviate) s += ann(2, 1, d.R <= -0.02 ? "ann_react" : "ann_noreact");
      if (d.V !== null && n > d.V && d.did_deviate) s += ann(d.V, 0, "ann_back");
    }
    s += `<line x1="${m.l}" x2="${m.l}" y1="${m.t}" y2="${f.h - m.b}" stroke="#55555b"/>`;
    s += `<line x1="${m.l}" x2="${f.w - m.r}" y1="${f.h - m.b}" y2="${f.h - m.b}" stroke="#55555b"/>`;
    el.innerHTML = s + "</svg>";
  }

  // Gráfico do teste de separação: Δ com e sem memória por condição, com IC 95%.
  function drawSeparation() {
    const el = $("sep-chart");
    const rows = S.ref.separation_test.rows;
    const fs = (el.clientWidth || 600) < 420 ? 11.5 : 13;
    const rowH = fs * 2 + 26, top = 6, bottom = 38;
    const w = Math.max(260, Math.round(el.clientWidth || 600)), h = top + rows.length * rowH + bottom;
    const m = { l: 10, r: 14 };
    const x = scale(0.5, 1.0, m.l + 8, w - m.r);
    let s = svgOpen(w, h);
    for (let v = 0.5; v <= 1.0001; v += 0.1) {
      const xx = x(v).toFixed(1);
      s += `<line x1="${xx}" x2="${xx}" y1="${top}" y2="${h - bottom}" stroke="#ece9e2"/>`;
      s += `<text x="${xx}" y="${h - bottom + 16}" text-anchor="middle" font-size="${fs}" fill="#55555b">${esc(num(v, 1))}</text>`;
    }
    s += `<text x="${(m.l + w - m.r) / 2}" y="${h - 4}" text-anchor="middle" font-size="${fs}" fill="#55555b">Δ</text>`;
    rows.forEach((r, i) => {
      const y0 = top + i * rowH;
      s += `<text x="${m.l}" y="${y0 + fs}" font-size="${fs}" fill="#1c1c1e">${esc(t(condKey(r.level)))}</text>`;
      const yW = y0 + fs + 10, yO = yW + 11;
      s += `<line x1="${x(r.delta_k1_lo)}" x2="${x(r.delta_k1_hi)}" y1="${yW}" y2="${yW}" stroke="#1f5fa8" stroke-width="2"/>`;
      s += `<circle cx="${x(r.delta_k1)}" cy="${yW}" r="5" fill="#1f5fa8"/>`;
      s += `<line x1="${x(r.delta_k0_lo)}" x2="${x(r.delta_k0_hi)}" y1="${yO}" y2="${yO}" stroke="#a8480a" stroke-width="2"/>`;
      s += `<rect x="${x(r.delta_k0) - 4.5}" y="${yO - 4.5}" width="9" height="9" fill="#fff" stroke="#a8480a" stroke-width="2.5"/>`;
    });
    el.innerHTML = s + "</svg>";
  }
  const condKey = (level) => "cond_" + level.replace(/[-.]/g, "_");

  // ------------------------------------------------------------------ textos e tabelas
  function table(head, rows, numericFrom) {
    let h = "<table><thead><tr>" + head.map((c, i) => `<th scope="col"${i >= numericFrom ? ' class="num"' : ""}>${esc(c)}</th>`).join("") + "</tr></thead><tbody>";
    rows.forEach((r) => { h += "<tr>" + r.map((c, i) => `<td${i >= numericFrom ? ' class="num"' : ""}>${esc(c)}</td>`).join("") + "</tr>"; });
    return h + "</tbody></table>";
  }

  function renderLearnTable() {
    const b = S.blocks.slice(0, Math.floor(S.shown));
    const every = Math.max(1, Math.ceil(b.length / 40));
    const rows = b.filter((_, k) => k % every === 0 || k === b.length - 1).map((x) =>
      [num((x.t_start + x.n_periods) / 1e6, 2), num(x.price_firm1, 3), num(x.price_firm2, 3), num(x.delta, 3)]);
    $("learn-table").innerHTML = table([t("th_periods"), t("th_price_a"), t("th_price_b"), t("th_delta_block")], rows, 1);
  }

  function renderSide() {
    const shownBlocks = Math.floor(S.shown);
    const done = S.final && shownBlocks >= S.blocks.length;
    let delta = null;
    if (done) delta = S.final.delta;
    else if (shownBlocks > 0) delta = S.blocks[shownBlocks - 1].delta;
    $("delta-value").textContent = delta === null ? "–" : num(delta, 2);
    $("delta-when").textContent = delta === null ? "" : t(done ? "delta_final" : "delta_during");
    $("meter-fill").style.left = delta === null ? "0%" : (Math.max(0, Math.min(1, delta)) * 100).toFixed(1) + "%";
    $("source-line").textContent = t(S.mode === "recorded" ? "source_recorded" : "source_live", { seed: S.seed });
    let outcome = "";
    if (done && S.final.converged) {
      outcome = S.final.cycle_len > 1 ? t("outcome_cycle", { n: S.final.cycle_len }) : t("outcome_constant");
    } else if (!done && shownBlocks > 0) {
      const b = S.blocks[shownBlocks - 1];
      outcome = t(S.paused ? "status_paused" : "status_learning",
                  { t: num((b.t_start + b.n_periods) / 1e3, 0), eps: num(b.epsilon_start * 100, 1) });
    }
    $("outcome").textContent = outcome;
  }

  // Mensagem da região viva: só muda em mudanças de estado (não a cada bloco).
  function setStatus(key, vars) {
    S.statusKey = key; S.statusVars = vars || null;
    $("status").textContent = key ? t(key, vars) : "";
  }
  function convergedStatus() {
    const f = S.final;
    const vars = { periods: num(f.periods / 1e6, 2), delta: num(f.delta, 2) };
    if (!f.converged) return ["status_not_converged", vars];
    return [f.delta >= 0.9 ? "status_converged_high" : f.delta <= 0.1 ? "status_converged_low" : "status_converged_mid", vars];
  }

  function renderButtons() {
    const run = $("btn-run");
    const learning = S.mode === "live" && !(S.final && S.shown >= S.blocks.length);
    run.textContent = t(learning ? (S.paused ? "btn_resume" : "btn_pause") : "btn_run");
    const canDeviate = !!(S.final && S.final.converged && S.shown >= S.blocks.length);
    $("btn-deviate").disabled = !canDeviate;
    $("btn-deviate").textContent = t("btn_deviate");
    $("btn-new").textContent = t("btn_new");
    $("deviate-hint").textContent = canDeviate ? "" : t("btn_deviate_hint");
  }

  function renderScore() {
    const d = S.dev;
    if (!d) { ["score-react", "score-back", "score-profit"].forEach((id) => { $(id).textContent = "–"; }); $("verdict").textContent = ""; return; }
    const finished = S.devShown >= 16;
    $("score-react").textContent = S.devShown > 2 ? t("score_react_value", { x: num(d.R * 100, 1) }) : "–";
    $("score-back").textContent = finished ? (d.V === null ? t("score_back_none") : String(d.V)) : "–";
    $("score-profit").textContent = finished ? t("score_profit_value", {
      d: num(d.discounted_profit_firm1_deviation, 2), c: num(d.discounted_profit_firm1_control, 2) }) : "–";
    let v = "";
    if (finished) {
      if (!d.did_deviate) v = t("verdict_none");
      else if (d.G < 0) v = t("verdict_unprofitable", { g: num(-d.G * 100, 1) });
      else v = t("verdict_profitable", { g: num(d.G * 100, 1) });
    }
    $("verdict").textContent = v;
  }

  function renderStatic() {
    const R = S.ref, P = S.params, base = R.base;
    const bt = R.mandatory_tests.find((x) => x.condition === "beta_4e-06");
    $("aggregate").textContent = t("aggregate", { n: num(base.n_sessions, 0), mean: num(base.delta_mean, 2),
      lo: num(bt.delta_ci_lo, 2), hi: num(bt.delta_ci_hi, 2), share: num(base.share_delta_above_0_5 * 100, 1) });
    const cls = R.deviation.classes["theta_0.02"];
    $("dev-aggregate").textContent = t("dev_aggregate", { n: num(base.n_sessions, 0),
      a: num(cls.A / base.n_sessions * 100, 0), u: num(R.deviation.share_unprofitable * 100, 0) });
    $("dev-caveat").innerHTML = esc(t("dev_caveat", { ref: "\u0000" })).replace("\u0000",
      `<a href="${I.LINKS.denboer}" rel="noopener" target="_blank">${esc(t("ref_denboer"))}</a>`);
    $("sep-text").textContent = t("sep_text", { k: R.separation_test.rows.length });
    $("sep-table").innerHTML = table([t("th_condition"), t("th_with"), t("th_without")],
      R.separation_test.rows.map((r) => [t(condKey(r.level)),
        `${num(r.delta_k1, 3)} [${num(r.delta_k1_lo, 3)}; ${num(r.delta_k1_hi, 3)}]`,
        `${num(r.delta_k0, 3)} [${num(r.delta_k0_lo, 3)}; ${num(r.delta_k0_hi, 3)}]`]), 1);

    // tabela de parâmetros (valores e páginas lidos do JSON)
    const mk = P.market, L = P.learning, en = I.getLang() === "en";
    const src = (x) => (x || "").replace(/;\s*Pesquisa \d+/g, "").replace(/ e /g, en ? " and " : " e ");
    const sup = (e) => String(e).replace(/-/g, "⁻").replace(/\d/g, (c) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[c]);
    const sci = (v) => { const e = Math.floor(Math.log10(v)), c = v / Math.pow(10, e); return `${num(c, Math.abs(c - Math.round(c)) < 1e-9 ? 0 : 1)} × 10${sup(e)}`; };
    const rows = [
      ["p_n", "n", num(mk.n.value, 0), src(mk.n.source)], ["p_c", "c", num(mk.c.value, 0), src(mk.c.source)],
      ["p_a", "a", num(mk.a.value, 0), src(mk.a.source)], ["p_a0", "a₀", num(mk.a0.value, 0), src(mk.a0.source)],
      ["p_mu", "μ", num(mk.mu.value, 2), src(mk.mu.source)], ["p_delta", "δ", num(L.delta.value, 2), src(L.delta.source)],
      ["p_alpha", "α", num(L.alpha.value, 2), src(L.alpha.source)], ["p_beta", "β", sci(L.beta.value), src(L.beta.source)],
      ["p_k", "k", num(L.k.value, 0), src(L.k.source)], ["p_m", "m", num(L.m.value, 0), src(L.m.source)],
      ["p_xi", "ξ", num(L.xi.value, 1), src(L.xi.source)],
      ["p_conv", "", t("p_conv_value", { stable: num(L.convergence.t_stable, 0) }), src(L.convergence.source)],
      ["p_sessions", "", t("p_sessions_value", { n: num(L.sessions.value, 0) }), src(L.sessions.source)],
    ];
    $("params-table").innerHTML = `<caption>${esc(t("method_params_caption"))}</caption>` +
      table([t("th_param"), t("th_symbol"), t("th_value"), t("th_source")], rows.map((r) => [t(r[0]), r[1], r[2], r[3]]), 9)
        .replace(/^<table>|<\/table>$/g, "");
    const rp = P.reference_points;
    $("method-measure").textContent = t("method_measure", { piN: num(rp.profit_nash, 4), piM: num(rp.profit_monopoly, 4),
      pN: num(rp.price_nash, 4), pM: num(rp.price_monopoly, 4) });
    $("lim-3").textContent = t("lim_3", { rand: num(R.learning_trajectory.delta_random_prices.value, 2) });
    $("cite-text").textContent = t("cite_text", { repo: I.LINKS.repo });
    $("updated").textContent = t("updated", { date: I.date(R.meta.generated) });
  }

  function renderAll() {
    I.apply();
    const other = I.getLang() === "pt" ? "en" : "pt";
    $("lang-btn-text").setAttribute("lang", other === "en" ? "en" : "pt-BR");
    renderStatic(); renderSide(); renderButtons(); renderScore(); renderLearnTable();
    if (S.statusKey) $("status").textContent = t(S.statusKey, S.statusVars);
    $("dev-cycle-note").hidden = !(S.final && S.final.cycle_len > 1);
    drawLearning(); drawDeviation(); drawSeparation();
  }

  // ------------------------------------------------------------------ animação do aprendizado
  let rafId = null, lastTs = null;
  function tick(ts) {
    rafId = null;
    if (!S.paused) {
      const dt = lastTs === null ? 0 : (ts - lastTs) / 1000;
      const target = reduceMotion ? (S.final ? S.blocks.length : 0) : S.shown + dt * BLOCKS_PER_SECOND;
      S.shown = Math.min(S.blocks.length, target);
    }
    lastTs = ts;
    drawLearning(); renderSide();
    const done = S.final && S.shown >= S.blocks.length;
    if (done) { finishLive(); return; }
    if (!S.paused) rafId = requestAnimationFrame(tick);
  }
  function startAnim() { if (rafId === null) { lastTs = null; rafId = requestAnimationFrame(tick); } }

  function finishLive() {
    S.running = false;
    const [k, v] = convergedStatus();
    setStatus(k, v);
    renderButtons(); renderLearnTable();
    $("dev-cycle-note").hidden = !(S.final.cycle_len > 1);
  }

  // ------------------------------------------------------------------ Worker
  function ensureWorker() {
    if (S.worker) return S.worker;
    const w = new Worker("js/qlearning.worker.js");
    w.onmessage = (ev) => {
      const msg = ev.data;
      if (msg.type === "blocks" && msg.runId === S.runId) {
        Array.prototype.push.apply(S.blocks, msg.blocks);
        startAnim();
      } else if (msg.type === "done" && msg.runId === S.runId) {
        const r = msg.result, m = S.mk.m;
        S.final = { converged: r.converged, periods: r.periods, delta: r.delta, cycle_len: r.cycle_len,
                    greedy: Int32Array.from(r.greedy),
                    cycle: r.cycle_idx ? r.cycle_idx.map((p) => p[0] * m + p[1]) : [] };
        startAnim();
      } else if (msg.type === "error") {
        workerFailed(msg.message);
      }
    };
    w.onerror = (e) => { e.preventDefault(); workerFailed(e.message || "Worker"); };
    w.postMessage({ type: "init", params: S.params });
    S.worker = w;
    return w;
  }
  // Worker conta as sessões por runId próprio; a página manda start/stop e acompanha a última.
  let workerRun = 0;

  function workerFailed(message) {
    S.running = false;
    showRecorded();
    setStatus("status_worker_error", { msg: message });
  }

  function startLive(seed) {
    let w;
    try { w = ensureWorker(); } catch (e) { workerFailed(e.message); return; }
    S.mode = "live"; S.seed = seed; S.blocks = []; S.shown = 0; S.final = null;
    S.paused = false; S.running = true;
    workerRun += 1; S.runId = workerRun;
    w.postMessage({ type: "start", seed: seed });
    S.dev = null; S.devShown = 16; clearTimeout(S.devTimer);
    setStatus("status_computing");
    renderSide(); renderButtons(); renderScore(); drawLearning(); drawDeviation();
    $("dev-cycle-note").hidden = true;
  }

  function showRecorded() {
    const r = recordedSession();
    S.mode = "recorded"; S.seed = r.seed; S.blocks = r.blocks; S.shown = r.blocks.length; S.final = r.final;
    S.paused = false; S.running = false;
    const [k, v] = convergedStatus();
    setStatus(k, v);
    applyDeviation(false);
    renderAll();
  }

  // ------------------------------------------------------------------ desvio
  function applyDeviation(animate) {
    const f = S.final;
    S.dev = D.run(S.mk, f.greedy, f.cycle, 0, S.params.learning.delta.value);
    clearTimeout(S.devTimer);
    S.devShown = animate && !reduceMotion ? 1 : 16;
    const stepAnim = () => {
      drawDeviation(); renderScore();
      if (S.devShown < 16) { S.devShown += 1; S.devTimer = setTimeout(stepAnim, TAU_MS); }
    };
    stepAnim();
  }

  // ------------------------------------------------------------------ botões
  function randomSeed() {
    const a = new Uint32Array(1);
    (window.crypto || window.msCrypto).getRandomValues(a);
    return 1 + (a[0] % 999999);
  }

  function onRun() {
    const learning = S.mode === "live" && !(S.final && S.shown >= S.blocks.length);
    if (!learning) { startLive(S.seed); return; }
    S.paused = !S.paused;
    if (S.paused) {
      const b = S.blocks[Math.max(0, Math.floor(S.shown) - 1)];
      setStatus("status_paused", { t: num(b ? (b.t_start + b.n_periods) / 1e3 : 0, 0) });
    } else {
      setStatus("status_computing"); startAnim();
    }
    renderSide(); renderButtons();
  }

  function onDeviate() {
    if (!S.final || !S.final.converged) return;
    applyDeviation(true);
    setStatus("status_deviation_done");
    const sec = $("deviation");
    const r = sec.getBoundingClientRect();
    if (r.top > window.innerHeight * 0.6) sec.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
  }

  function onLang() {
    const next = I.getLang() === "pt" ? "en" : "pt";
    I.setLang(next);
    try { localStorage.setItem("lang", next); } catch (e) { /* sem armazenamento */ }
    try {
      const u = new URL(location.href); u.searchParams.set("lang", next);
      history.replaceState(null, "", u.toString());
    } catch (e) { /* sem history */ }
    renderAll();
  }

  // ------------------------------------------------------------------ início
  function init() {
    I.setLang(I.initialLang());
    I.apply();
    Promise.all([loadJSON("qlearning_params.json"), loadJSON("qlearning_reference.json"), loadJSON("qlearning_sessions.json")])
      .then(([p, r, s]) => {
        S.params = p; S.ref = r; S.sessions = s; S.mk = E.setupMarket(p);
        $("btn-run").addEventListener("click", onRun);
        $("btn-deviate").addEventListener("click", onDeviate);
        $("btn-new").addEventListener("click", () => startLive(randomSeed()));
        $("lang-btn").addEventListener("click", onLang);
        let rs = null;
        window.addEventListener("resize", () => { clearTimeout(rs); rs = setTimeout(() => { drawLearning(); drawDeviation(); drawSeparation(); }, 120); });
        showRecorded();
        document.body.dataset.ready = "1";
      })
      .catch((e) => { $("status").textContent = "Erro ao carregar os dados / Error loading data: " + e.message; });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
