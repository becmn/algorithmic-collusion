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
  const NT = D.T_DEV + 1;         // pontos do gráfico do desvio: τ = 0 a T_DEV (deviation.js)

  const S = {
    params: null, ref: null, sessions: null, mk: null,
    mode: "recorded",   // "recorded" | "live"
    seed: null, blocks: [], shown: 0, final: null,
    memory: "with",     // "with" (memória de 1 período) | "without" (sessão gravada sem memória)
    k0: null,           // qlearning_memoryless.json
    running: false, paused: false, worker: null, runId: 0,
    dev: null, devShown: NT, devTimer: null, statusKey: null, statusVars: null,
  };

  // ------------------------------------------------------------------ dados
  function loadJSON(name) {
    return fetch("data/" + name).then((r) => {
      if (!r.ok) throw new Error(name + ": " + r.status);
      return r.json();
    });
  }

  // Sessão gravada mostrada: com memória (abre a página) ou sem memória (seletor).
  function recordedSession() {
    const s = S.memory === "without" ? S.k0.session
      : (S.sessions.sessions.find((x) => x.opens_page) || S.sessions.sessions[0]);
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
      s += `<line x1="${x0}" x2="${x1}" y1="${yy}" y2="${yy}" stroke="#e9e7e1"/>`;
      s += `<text x="${x0 - 6}" y="${yy}" dy="0.32em" text-anchor="end" font-size="${fs}" fill="#4a4f58">${esc(num(p, 1))}</text>`;
    }
    return s;
  }

  function refLines(y, x0, x1, fs) {
    const rp = S.params.reference_points;
    const yN = y(rp.price_nash), yM = y(rp.price_monopoly);
    return `<rect x="${x0}" y="${yM}" width="${x1 - x0}" height="${yN - yM}" fill="rgba(29,78,137,0.05)"/>` +
      `<line x1="${x0}" x2="${x1}" y1="${yM}" y2="${yM}" stroke="#17202a" stroke-width="1.5" stroke-dasharray="7 3 2 3"/>` +
      `<line x1="${x0}" x2="${x1}" y1="${yN}" y2="${yN}" stroke="#17202a" stroke-width="1.5" stroke-dasharray="2 3"/>` +
      `<text x="${x1 - 4}" y="${yM - 6}" text-anchor="end" font-size="${fs + 1}" font-family="Newsreader, Georgia, serif" font-style="italic" fill="#17202a" paint-order="stroke" stroke="#fbfbf8" stroke-width="3">${esc(t("line_monopoly"))}</text>` +
      `<text x="${x1 - 4}" y="${yN + fs + 4}" text-anchor="end" font-size="${fs + 1}" font-family="Newsreader, Georgia, serif" font-style="italic" fill="#17202a" paint-order="stroke" stroke="#fbfbf8" stroke-width="3">${esc(t("line_nash"))}</text>`;
  }

  function yDomain() {
    const g = S.params.grid.prices;
    return [g[0] - 0.02, g[g.length - 1] + 0.02];
  }


  // ------------------------------------------------------------------ valores ao passar o mouse
  // Cada gráfico registra um "modelo": quantos pontos tem, qual ponto fica sob o cursor e o que
  // mostrar nele. Mouse, toque e teclado (setas, Esc) usam o mesmo modelo. As tabelas de dados
  // continuam sendo a alternativa em texto (os leitores de tela não leem a caixinha).
  const HOVER = {};
  const tipState = { id: null, i: -1 };
  function tipEls(id) {
    const el = $(id), fig = el.parentElement;
    let tip = fig.querySelector(":scope > .tip");
    if (!tip) {
      tip = document.createElement("div"); tip.className = "tip"; tip.hidden = true; tip.setAttribute("aria-hidden", "true");
      const guide = document.createElement("div"); guide.className = "tip-guide"; guide.hidden = true;
      const dots = document.createElement("div"); dots.className = "tip-dots";
      fig.appendChild(guide); fig.appendChild(dots); fig.appendChild(tip);
    }
    return { el, fig, tip, guide: fig.querySelector(":scope > .tip-guide"), dots: fig.querySelector(":scope > .tip-dots") };
  }
  function hideTip(id) {
    const ids = id ? [id] : Object.keys(HOVER);
    ids.forEach((k) => {
      const e = $(k) && $(k).parentElement;
      if (!e) return;
      e.querySelectorAll(":scope > .tip, :scope > .tip-guide").forEach((n) => { n.hidden = true; });
      const d = e.querySelector(":scope > .tip-dots"); if (d) d.innerHTML = "";
    });
    if (!id || tipState.id === id) { tipState.id = null; tipState.i = -1; }
  }
  function showTip(id, i) {
    const model = HOVER[id];
    if (!model || !model.n) { hideTip(id); return; }
    i = Math.max(0, Math.min(model.n - 1, i));
    const p = model.at(i);
    const { el, fig, tip, guide, dots } = tipEls(id);
    const er = el.getBoundingClientRect(), fr = fig.getBoundingClientRect();
    const k = er.width / model.w, ox = er.left - fr.left, oy = er.top - fr.top;
    tip.innerHTML = `<strong>${esc(p.title)}</strong>` + p.rows.map((r) =>
      `<span class="tip-row">${r.color ? `<i style="background:${r.color}"></i>` : ""}${esc(r.label)}: <b>${esc(r.value)}</b></span>`).join("");
    tip.hidden = false;
    if (p.guideX !== undefined) {
      guide.hidden = false;
      guide.style.left = (ox + p.guideX * k) + "px"; guide.style.top = (oy + model.top * k) + "px";
      guide.style.height = ((model.bottom - model.top) * k) + "px";
    } else guide.hidden = true;
    dots.innerHTML = (p.dots || []).map((d) =>
      `<span class="tip-dot" style="left:${ox + d.x * k}px;top:${oy + d.y * k}px;background:${d.color}"></span>`).join("");
    const ax = ox + p.anchorX * k, ay = oy + p.anchorY * k;
    const tw = tip.offsetWidth, th = tip.offsetHeight;
    let left = p.preferLeft ? ax - 14 - tw : ax + 14;
    if (p.preferLeft && left < 0) left = ax + 14;
    if (left + tw > fr.width) left = ax - 14 - tw;
    tip.style.left = Math.max(0, left) + "px";
    tip.style.top = Math.max(0, Math.min(fr.height - th, ay - th / 2)) + "px";
    tipState.id = id; tipState.i = i;
  }
  function attachHover(id) {
    const el = $(id);
    el.tabIndex = 0;
    const pick = (ev) => {
      const model = HOVER[id];
      if (!model || !model.n) return -1;
      const r = el.getBoundingClientRect(), k = model.w / r.width;
      return model.pick((ev.clientX - r.left) * k, (ev.clientY - r.top) * k);
    };
    el.addEventListener("pointermove", (ev) => { const i = pick(ev); if (i >= 0) showTip(id, i); else hideTip(id); });
    el.addEventListener("pointerdown", (ev) => { const i = pick(ev); if (i >= 0) showTip(id, i); });
    el.addEventListener("pointerleave", (ev) => { if (ev.pointerType === "mouse") hideTip(id); });
    el.addEventListener("focus", () => { const m = HOVER[id]; if (m && m.n) showTip(id, tipState.id === id ? tipState.i : m.n - 1); });
    el.addEventListener("blur", () => hideTip(id));
    el.addEventListener("keydown", (ev) => {
      const m = HOVER[id];
      if (!m || !m.n) return;
      const cur = tipState.id === id ? tipState.i : m.n - 1;
      const step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[ev.key];
      if (step) { ev.preventDefault(); showTip(id, cur + step); }
      else if (ev.key === "Home") { ev.preventDefault(); showTip(id, 0); }
      else if (ev.key === "End") { ev.preventDefault(); showTip(id, m.n - 1); }
      else if (ev.key === "Escape") hideTip(id);
    });
  }
  // depois de redesenhar, a caixinha aberta acompanha os dados novos
  function refreshTip(id) { if (tipState.id === id) showTip(id, Math.min(tipState.i, (HOVER[id] || { n: 0 }).n - 1)); }
  const nearestIndex = (xs, v) => {
    let best = -1, bd = Infinity;
    xs.forEach((x, i) => { const d = Math.abs(x - v); if (d < bd) { bd = d; best = i; } });
    return best;
  };

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
      s += `<line x1="${xx}" x2="${xx}" y1="${f.h - m.b}" y2="${f.h - m.b + 4}" stroke="#4a4f58"/>`;
      s += `<text x="${xx}" y="${f.h - m.b + 16}" text-anchor="middle" font-size="${f.fs}" fill="#4a4f58">${esc(num(v / 1e6, 1))}</text>`;
    }
    s += `<text x="${(m.l + f.w - m.r) / 2}" y="${f.h - 4}" text-anchor="middle" font-size="${f.fs}" fill="#4a4f58">${esc(t("axis_periods_millions"))}</text>`;
    s += refLines(y, m.l, f.w - m.r, f.fs);
    const pts = (key) => blocks.map((b) => `${x(b.t_start + b.n_periods / 2).toFixed(1)},${y(b[key]).toFixed(1)}`).join(" ");
    if (blocks.length) {
      s += `<polyline points="${pts("price_firm2")}" fill="none" stroke="#c2410c" stroke-width="2" stroke-dasharray="6 4" stroke-linejoin="round"/>`;
      s += `<polyline points="${pts("price_firm1")}" fill="none" stroke="#1d4e89" stroke-width="2" stroke-linejoin="round"/>`;
    }
    s += `<line x1="${m.l}" x2="${m.l}" y1="${m.t}" y2="${f.h - m.b}" stroke="#4a4f58"/>`;
    s += `<line x1="${m.l}" x2="${f.w - m.r}" y1="${f.h - m.b}" y2="${f.h - m.b}" stroke="#4a4f58"/>`;
    el.innerHTML = s + "</svg>";
    const xs = blocks.map((b) => x(b.t_start + b.n_periods / 2));
    HOVER["learn-chart"] = {
      n: blocks.length, w: f.w, top: m.t, bottom: f.h - m.b,
      pick: (px) => (px < m.l - 4 || px > f.w - m.r + 4 ? -1 : nearestIndex(xs, px)),
      at: (i) => {
        const b = blocks[i];
        return {
          title: t("tip_periods", { a: num(b.t_start + 1, 0), b: num(b.t_start + b.n_periods, 0) }),
          rows: [{ label: t("firm_a"), value: num(b.price_firm1, 3), color: "#1d4e89" },
                 { label: t("firm_b"), value: num(b.price_firm2, 3), color: "#c2410c" },
                 { label: t("th_delta_block"), value: num(b.delta, 3) }],
          guideX: xs[i], dots: [{ x: xs[i], y: y(b.price_firm1), color: "#1d4e89" }, { x: xs[i], y: y(b.price_firm2), color: "#c2410c" }],
          anchorX: xs[i], anchorY: (y(b.price_firm1) + y(b.price_firm2)) / 2,
        };
      },
    };
    refreshTip("learn-chart");
  }

  // Gráfico do desvio: degraus de τ = 0 a T_DEV, ramo com corte × ramo sem corte.
  function drawDeviation() {
    const el = $("dev-chart");
    const f = frame(el, 0.5, 240, 340);
    const m = { l: 44, r: 12, t: 3 * f.fs + 20, b: 40 };
    const [y0, y1] = yDomain();
    const x = scale(0, NT, m.l, f.w - m.r), y = scale(y0, y1, f.h - m.b, m.t);
    let s = svgOpen(f.w, f.h) + priceAxis(null, y, m.l, f.w - m.r, f.fs);
    for (let k = 0; k < NT; k += (f.w < 420 ? 3 : 1)) {
      s += `<text x="${x(k + 0.5)}" y="${f.h - m.b + 16}" text-anchor="middle" font-size="${f.fs}" fill="#4a4f58">${k}</text>`;
    }
    s += `<text x="${(m.l + f.w - m.r) / 2}" y="${f.h - 4}" text-anchor="middle" font-size="${f.fs}" fill="#4a4f58">${esc(t("axis_tau"))}</text>`;
    s += refLines(y, m.l, f.w - m.r, f.fs);
    const d = S.dev;
    if (d) {
      const n = Math.min(NT, S.devShown);
      const step = (prices, i, upto) => {
        let p = "";
        for (let k = 0; k < upto; k++) {
          const yy = y(prices[k][i]).toFixed(1);
          p += (k === 0 ? "M" : "L") + x(k).toFixed(1) + "," + yy + "L" + x(k + 1).toFixed(1) + "," + yy;
        }
        return p;
      };
      s += `<path d="${step(d.control.price, 0, NT)}" fill="none" stroke="#7a7e86" stroke-width="2" stroke-dasharray="1.5 3.5" stroke-linecap="round"/>`;
      s += `<path d="${step(d.control.price, 1, NT)}" fill="none" stroke="#7a7e86" stroke-width="2" stroke-dasharray="1.5 3.5" stroke-linecap="round"/>`;
      s += `<path d="${step(d.deviation.price, 1, n)}" fill="none" stroke="#c2410c" stroke-width="2.5" stroke-dasharray="6 4"/>`;
      s += `<path d="${step(d.deviation.price, 0, n)}" fill="none" stroke="#1d4e89" stroke-width="2.5"/>`;
      // anotações em sequência
      const ann = (k, row, key) => {
        const xx = x(k).toFixed(1), yy = m.t - 6 - row * (f.fs + 4);
        return `<line x1="${xx}" x2="${xx}" y1="${yy + 4}" y2="${f.h - m.b}" stroke="#17202a" stroke-width="1" stroke-dasharray="3 3"/>` +
          `<text x="${xx}" y="${yy}" font-size="${f.fs + 1}" font-family="Newsreader, Georgia, serif" font-style="italic" fill="#17202a" text-anchor="${k > 11 ? "end" : "start"}" dx="${k > 11 ? -3 : 3}">${esc(t(key))}</text>`;
      };
      if (n > 1 && d.did_deviate) s += ann(1, 2, "ann_cut");
      if (n > 2 && d.did_deviate) s += ann(2, 1, d.R <= -S.ref.deviation.protocol.thetas[0] ? "ann_react" : "ann_noreact");
      if (d.V !== null && n > d.V && d.did_deviate) s += ann(d.V, 0, "ann_back");
    }
    s += `<line x1="${m.l}" x2="${m.l}" y1="${m.t}" y2="${f.h - m.b}" stroke="#4a4f58"/>`;
    s += `<line x1="${m.l}" x2="${f.w - m.r}" y1="${f.h - m.b}" y2="${f.h - m.b}" stroke="#4a4f58"/>`;
    el.innerHTML = s + "</svg>";
    const nShown = d ? Math.min(NT, S.devShown) : 0;
    HOVER["dev-chart"] = {
      n: nShown, w: f.w, top: m.t, bottom: f.h - m.b,
      pick: (px) => (px < m.l - 4 || px > f.w - m.r + 4 ? -1 : Math.max(0, Math.min(nShown - 1, Math.floor((px - m.l) / ((x(1) - x(0)) || 1))))),
      at: (k) => {
        const c = d.control.price[k], v = d.deviation.price[k], xc = x(k + 0.5);
        return {
          title: t("tip_tau", { k: k }),
          rows: [{ label: t("th_a_cut"), value: num(v[0], 3), color: "#1d4e89" }, { label: t("th_b_cut"), value: num(v[1], 3), color: "#c2410c" },
                 { label: t("th_a_nocut"), value: num(c[0], 3), color: "#7a7e86" }, { label: t("th_b_nocut"), value: num(c[1], 3), color: "#7a7e86" }],
          guideX: xc, dots: [{ x: xc, y: y(v[0]), color: "#1d4e89" }, { x: xc, y: y(v[1]), color: "#c2410c" }],
          anchorX: xc, anchorY: (y(v[0]) + y(v[1])) / 2,
        };
      },
    };
    refreshTip("dev-chart");
  }

  // Gráfico do teste de separação: Δ com e sem memória por condição, com IC 95%.
  function drawSeparation() {
    const el = $("sep-chart");
    const rows = S.ref.separation_test.rows;
    const fs = (el.clientWidth || 600) < 420 ? 11.5 : 13;
    const rowH = fs * 2 + 26, top = 6, bottom = 38;
    const w = Math.max(260, Math.round(el.clientWidth || 600)), h = top + rows.length * rowH + bottom;
    const m = { l: 10, r: 14 };
    // eixo de 0,1 em 0,1 cobrindo todos os intervalos de confiança (lido dos dados)
    const lo = Math.max(0, Math.floor(10 * Math.min(...rows.map((r) => Math.min(r.delta_k1_lo, r.delta_k0_lo)))) / 10);
    const hi = Math.ceil(10 * Math.max(...rows.map((r) => Math.max(r.delta_k1_hi, r.delta_k0_hi)))) / 10;
    const x = scale(lo, hi, m.l + 8, w - m.r);
    let s = svgOpen(w, h);
    for (let v = lo; v <= hi + 1e-9; v += 0.1) {
      const xx = x(v).toFixed(1);
      s += `<line x1="${xx}" x2="${xx}" y1="${top}" y2="${h - bottom}" stroke="#e9e7e1"/>`;
      s += `<text x="${xx}" y="${h - bottom + 16}" text-anchor="middle" font-size="${fs}" fill="#4a4f58">${esc(num(v, 1))}</text>`;
    }
    s += `<text x="${(m.l + w - m.r) / 2}" y="${h - 4}" text-anchor="middle" font-size="${fs}" fill="#4a4f58">Δ</text>`;
    rows.forEach((r, i) => {
      const y0 = top + i * rowH;
      s += `<text x="${m.l}" y="${y0 + fs}" font-size="${fs}" fill="#17202a">${esc(t(condKey(r.level)))}</text>`;
      const yW = y0 + fs + 10, yO = yW + 11;
      s += `<line x1="${x(r.delta_k1_lo)}" x2="${x(r.delta_k1_hi)}" y1="${yW}" y2="${yW}" stroke="#1d4e89" stroke-width="2"/>`;
      s += `<circle cx="${x(r.delta_k1)}" cy="${yW}" r="5" fill="#1d4e89"/>`;
      s += `<line x1="${x(r.delta_k0_lo)}" x2="${x(r.delta_k0_hi)}" y1="${yO}" y2="${yO}" stroke="#c2410c" stroke-width="2"/>`;
      s += `<rect x="${x(r.delta_k0) - 4.5}" y="${yO - 4.5}" width="9" height="9" fill="#fbfbf8" stroke="#c2410c" stroke-width="2.5"/>`;
    });
    el.innerHTML = s + "</svg>";
    const ci = (v, a, b) => `${num(v, 3)} [${num(a, 3)}; ${num(b, 3)}]`;
    HOVER["sep-chart"] = {
      n: rows.length, w: w,
      pick: (px, py) => { const i = Math.floor((py - top) / rowH); return i >= 0 && i < rows.length ? i : -1; },
      at: (i) => {
        const r = rows[i], yW = top + i * rowH + fs + 10;
        return {
          title: `${t(condKey(r.level))} (${techLabel(r.level)})`,
          rows: [{ label: t("sep_with"), value: ci(r.delta_k1, r.delta_k1_lo, r.delta_k1_hi), color: "#1d4e89" },
                 { label: t("sep_without"), value: ci(r.delta_k0, r.delta_k0_lo, r.delta_k0_hi), color: "#c2410c" }],
          // caixinha à esquerda dos dois pontos (que ficam quase sempre à direita do gráfico)
          anchorX: Math.min(x(r.delta_k1), x(r.delta_k0)) - 10, anchorY: yW + 5, preferLeft: true,
        };
      },
    };
    refreshTip("sep-chart");
  }
  const condKey = (level) => "cond_" + level.replace(/[-.]/g, "_");
  // Parâmetro técnico de cada condição, montado do nível gravado no JSON (ex.: "beta4e-06").
  function techLabel(level) {
    const sup = (e) => String(e).replace(/-/g, "⁻").replace(/\d/g, (c) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[c]);
    let m = /^beta([\d.]+)e(-?\d+)$/.exec(level);
    if (m) {
      const c = Number(m[1]), e = Number(m[2]);
      return `β = ${c === 1 ? "" : num(c, 0) + " × "}10${sup(e)}`;
    }
    m = /^epsmin([\d.]+)$/.exec(level);
    if (m) return `β = 4 × 10${sup(-6)}, ε ≥ ${num(Number(m[1]) * 100, Number(m[1]) < 0.01 ? 1 : 0)}%`;
    return level;
  }

  // Figura 3 em destaque: duas barras (com e sem memória) na escala de Δ, de 0 (concorrência)
  // a 1 (monopólio), com o valor escrito na ponta (o IC 95%, de ±0,01, fica no texto alternativo e
  // na tabela das 7 condições: no tamanho da barra ele não se vê).
  function drawSepMain() {
    const el = $("sep-main");
    const row = S.ref.separation_test.rows.find((x) => x.level === "beta4e-06");
    const w = Math.max(260, Math.round(el.clientWidth || 600)), fs = w < 420 ? 12 : 14;
    const m = { l: 10, r: 18 }, barH = fs * 2.2, top = 4;
    const bars = [["sep_with", row.delta_k1, row.delta_k1_lo, row.delta_k1_hi, "#1d4e89"],
                  ["sep_without", row.delta_k0, row.delta_k0_lo, row.delta_k0_hi, "#c2410c"]];
    const rowH = fs * 1.6 + barH + 14, h = top + bars.length * rowH + fs * 2.6;
    const x = scale(0, 1, m.l, w - m.r);
    let s = svgOpen(w, h);
    bars.forEach(([key, v, lo, hi, col], i) => {
      const y0 = top + i * rowH;
      s += `<text x="${m.l}" y="${y0 + fs}" font-size="${fs}" fill="#17202a" font-weight="600">${esc(t(key))}</text>`;
      const yb = y0 + fs * 1.6;
      s += `<rect x="${x(0)}" y="${yb}" width="${(x(1) - x(0)).toFixed(1)}" height="${barH}" fill="#efeee9"/>`;
      s += `<rect x="${x(0)}" y="${yb}" width="${(x(v) - x(0)).toFixed(1)}" height="${barH}" fill="${col}"/>`;
      const lab = num(v, 2), inside = x(v) - x(0) > fs * 4;
      s += `<text x="${(inside ? x(v) - 8 : x(v) + 6).toFixed(1)}" y="${yb + barH / 2 + fs * 0.35}" font-size="${fs * 1.15}" font-weight="700"`
        + ` text-anchor="${inside ? "end" : "start"}" fill="${inside ? "#fbfbf8" : "#17202a"}">Δ ${esc(lab)}</text>`;
    });
    const ya = top + bars.length * rowH + 2;
    s += `<line x1="${x(0)}" x2="${x(1)}" y1="${ya}" y2="${ya}" stroke="#4a4f58"/>`;
    for (const v of [0, 0.5, 1]) {
      s += `<line x1="${x(v)}" x2="${x(v)}" y1="${ya}" y2="${ya + 5}" stroke="#4a4f58"/>`;
      s += `<text x="${x(v)}" y="${ya + fs + 4}" font-size="${fs * 0.85}" fill="#4a4f58" text-anchor="${v === 0 ? "start" : v === 1 ? "end" : "middle"}">${esc(num(v, v === 0.5 ? 1 : 0))}</text>`;
    }
    s += `<text x="${x(0)}" y="${ya + fs * 2.3}" font-size="${fs * 0.85}" fill="#4a4f58" font-style="italic">${esc(t("axis_competition"))}</text>`;
    s += `<text x="${x(1)}" y="${ya + fs * 2.3}" font-size="${fs * 0.85}" fill="#4a4f58" font-style="italic" text-anchor="end">${esc(t("axis_monopoly"))}</text>`;
    el.innerHTML = s + "</svg>";
    HOVER["sep-main"] = {
      n: bars.length, w: w,
      pick: (px, py) => { const i = Math.floor((py - top) / rowH); return i >= 0 && i < bars.length ? i : -1; },
      at: (i) => {
        const [key, v, lo, hi, col] = bars[i];
        return {
          title: t(key), rows: [{ label: t("tip_mean"), value: num(v, 3), color: col },
                                { label: t("tip_ci"), value: `${num(lo, 3)} – ${num(hi, 3)}` }],
          anchorX: Math.min(x(v), w * 0.6), anchorY: top + i * rowH + fs * 1.6 + barH / 2,
        };
      },
    };
    refreshTip("sep-main");
  }

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

  // Alternativa em texto da Figura 2 (WCAG 1.1.1): preços dos dois ramos, τ = 0 a T_DEV.
  function renderDevTable() {
    const d = S.dev;
    if (!d) { $("dev-table").innerHTML = ""; return; }
    const rows = d.tau.map((k) => [String(k), num(d.deviation.price[k][0], 3), num(d.deviation.price[k][1], 3),
      num(d.control.price[k][0], 3), num(d.control.price[k][1], 3)]);
    $("dev-table").innerHTML = table([t("th_tau"), t("th_a_cut"), t("th_b_cut"), t("th_a_nocut"), t("th_b_nocut")], rows, 1);
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
    const k0 = S.memory === "without";
    $("source-line").textContent = t(S.mode === "live" ? (k0 ? "source_live_k0" : "source_live")
      : (k0 ? "source_recorded_k0" : "source_recorded"), { seed: S.seed });
    $("badge").textContent = t(S.memory === "without" ? "badge_paper_k0" : "badge_paper");
    // referência: média das 1.000 sessões do mesmo modo (o número grande é só desta sessão)
    const sb = S.ref.separation_test.rows.find((x) => x.level === "beta4e-06");
    const ref = S.memory === "without" ? { n: sb.n_k0, mean: sb.delta_k0 } : { n: S.ref.base.n_sessions, mean: S.ref.base.delta_mean };
    $("delta-mean").textContent = t("delta_mean_ref", { n: num(ref.n, 0), mean: num(ref.mean, 2) });
    $("meter-mean").style.left = (Math.max(0, Math.min(1, ref.mean)) * 100).toFixed(1) + "%";
    let outcome = "";
    if (done && S.final.converged) {
      outcome = S.final.cycle_len > 1 ? t("outcome_cycle", { n: S.final.cycle_len }) : t("outcome_constant");
      // tempo de cálculo da sessão ao vivo (para medir no celular, Fase 2.5)
      if (S.mode === "live" && S.final.ms !== undefined) outcome += " " + t("compute_time", { s: num(S.final.ms / 1000, 2) });
    } else if (!done && shownBlocks > 0) {
      const b = S.blocks[shownBlocks - 1];
      outcome = t("status_learning", { t: num(b.t_start + b.n_periods, 0), eps: num(b.epsilon_start * 100, 1) });
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
    const mi = f.periods / 1e6;
    // funções: o texto é formatado de novo no idioma ativo quando a língua muda
    const vars = { periods: () => num(mi, 2), delta: () => num(f.delta, 2), mi: () => t(mi < 2 ? "million_one" : "million_many") };
    if (!f.converged) return ["status_not_converged", vars];
    return [f.delta >= 0.9 ? "status_converged_high" : f.delta <= 0.1 ? "status_converged_low" : "status_converged_mid", vars];
  }

  function renderButtons() {
    const run = $("btn-run");
    const learning = !(S.final && S.shown >= S.blocks.length);
    run.textContent = t(learning ? (S.paused ? "btn_resume" : "btn_pause") : "btn_run");
    const canDeviate = !!(S.final && S.final.converged && S.shown >= S.blocks.length);
    $("btn-deviate").disabled = !canDeviate;
    $("btn-deviate").textContent = t("btn_deviate");
    $("btn-new").textContent = t("btn_new");
    $("deviate-hint").textContent = canDeviate ? "" : t("btn_deviate_hint");
    $("mem-with").setAttribute("aria-pressed", String(S.memory === "with"));
    $("mem-without").setAttribute("aria-pressed", String(S.memory === "without"));
    $("mem-note").hidden = S.memory !== "without";
    $("dev-mem-note").hidden = S.memory !== "without";
  }

  // Ganho descontado da firma B com o corte de A, na mesma fórmula do G (τ = 1 a 15, peso δ^(τ−1)).
  function rivalGain(d) {
    const m = S.mk.m, PI = S.mk.PI, delta = S.params.learning.delta.value;
    let num_ = 0, den = 0, w = 1;
    for (let k = 1; k < NT; k++) {
      const c = d.control.idx[k], v = d.deviation.idx[k];
      const pc = PI[(c[0] * m + c[1]) * 2 + 1], pv = PI[(v[0] * m + v[1]) * 2 + 1];
      num_ += w * (pv - pc); den += w * pc; w *= delta;
    }
    return num_ / den;
  }

  function renderScore() {
    const d = S.dev;
    if (!d) { ["score-react", "score-back", "score-profit"].forEach((id) => { $(id).textContent = "–"; }); $("verdict").textContent = ""; return; }
    const finished = S.devShown >= NT;
    $("score-react").textContent = S.devShown > 2 ? t("score_react_value", { x: num(d.R * 100, 1) }) : "–";
    $("score-back").textContent = finished ? (d.V === null ? t("score_back_none") : String(d.V)) : "–";
    $("score-profit").textContent = finished ? t("score_profit_value", {
      d: num(d.discounted_profit_firm1_deviation, 2), c: num(d.discounted_profit_firm1_control, 2) }) : "–";
    let v = "";
    if (finished) {
      if (!d.did_deviate) v = t("verdict_none");
      else {
        v = d.G < 0 ? t("verdict_unprofitable", { g: num(-d.G * 100, 1) }) : t("verdict_profitable", { g: num(d.G * 100, 1) });
        // B no mesmo cálculo (lucro descontado em 15 períodos, com × sem corte); descritivo: a
        // pergunta do Protocolo ("o corte compensou?") é só sobre quem cortou (G de A).
        const gb = rivalGain(d);
        v += " " + (Math.abs(gb) < 5e-4 ? t("verdict_b_same")
          : t(gb < 0 ? "verdict_b_less" : "verdict_b_more", { g: num(Math.abs(gb) * 100, 1) }));
      }
    }
    $("verdict").textContent = v;
  }

  function renderStatic() {
    const R = S.ref, P = S.params, base = R.base;
    const bt = R.mandatory_tests.find((x) => x.condition === "beta_4e-06");
    const sepBase = R.separation_test.rows.find((x) => x.level === "beta4e-06");
    $("aggregate").textContent = S.memory === "without"
      ? t("aggregate_k0", { n: num(sepBase.n_k0, 0), mean: num(sepBase.delta_k0, 2),
                            lo: num(sepBase.delta_k0_lo, 2), hi: num(sepBase.delta_k0_hi, 2) })
      : t("aggregate", { n: num(base.n_sessions, 0), mean: num(base.delta_mean, 2),
                         lo: num(bt.delta_ci_lo, 2), hi: num(bt.delta_ci_hi, 2) });
    const cls = R.deviation.classes["theta_0.02"];
    $("dev-aggregate").textContent = t("dev_aggregate", { n: num(base.n_sessions, 0),
      a: num(cls.A / base.n_sessions * 100, 0), u: num(R.deviation.share_unprofitable * 100, 0) });
    $("dev-caveat").innerHTML = esc(t("dev_caveat", { ref: "\u0000" })).replace("\u0000",
      `<a href="${I.LINKS.denboer}" rel="noopener" target="_blank">${esc(t("ref_denboer"))}</a>`);
    // Figura 3: resultado em destaque (parâmetros do artigo) e as condições recolhidas
    const nk = R.separation_test.rows.length;
    const sv = { k: nk, k1: num(sepBase.delta_k1, 2), k0: num(sepBase.delta_k0, 2),
                 k1lo: num(sepBase.delta_k1_lo, 2), k1hi: num(sepBase.delta_k1_hi, 2),
                 k0lo: num(sepBase.delta_k0_lo, 2), k0hi: num(sepBase.delta_k0_hi, 2) };
    $("sep-main-desc").textContent = t("sep_main_desc", sv);
    $("sep-result").textContent = t("sep_result", sv);
    $("sep-more-summary").textContent = t("sep_more", sv);
    $("sep-table").innerHTML = table([t("th_condition"), t("th_param_tech"), t("th_with"), t("th_without")],
      R.separation_test.rows.map((r) => [t(condKey(r.level)), techLabel(r.level),
        `${num(r.delta_k1, 3)} [${num(r.delta_k1_lo, 3)}; ${num(r.delta_k1_hi, 3)}]`,
        `${num(r.delta_k0, 3)} [${num(r.delta_k0_lo, 3)}; ${num(r.delta_k0_hi, 3)}]`]), 2);

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
    $("author-updated").textContent = t("author_updated", { date: I.date(R.meta.generated) });

    // Tabela 1: resultados do projeto × artigo original (tudo lido do JSON, que vem de results/)
    const T = R.targets_calvano, sep0 = R.separation_test.rows.find((x) => x.level === "beta4e-06");
    const rowsT = [
      ["row_delta", num(base.delta_mean, 3), num(T.delta_mean.value, 3)],
      ["row_nash", num(R.best_response.nash_share, 3), num(T.nash_share.value, 3)],
      ["row_nash_any", num(bt.nash_any_dev_share, 3), t("not_reported"), true],
      ["row_react", num(R.deviation.R_mean, 3), num(T.rival_reaction_tau2.value, 3)],
      ["row_unprof", num(R.deviation.share_unprofitable, 3), num(T.unprofitable_deviation_share.value, 3)],
      ["row_punish", num(R.deviation.V_mean_uncensored, 1), num(T.punishment_periods.value, 1)],
      ["row_memoryless", num(sep0.delta_k0, 3), t("not_reported"), true],
    ];
    $("tab1").innerHTML = `<caption><strong>${esc(t("tab1_caption"))}</strong></caption>` +
      `<thead><tr><th scope="col">${esc(t("th_measure"))}</th><th scope="col" class="num">${esc(t("th_project"))}</th><th scope="col" class="num">${esc(t("th_paper"))}</th></tr></thead><tbody>` +
      rowsT.map((r) => `<tr><th scope="row">${esc(t(r[0]))}</th><td class="num${r[3] ? " em" : ""}">${esc(r[1])}</td><td class="num">${esc(r[2])}</td></tr>`).join("") +
      "</tbody>";
    $("tab1-note").textContent = t("tab1_note", {
      proj: num(R.deviation.share_unprofitable, 3), paper: num(T.unprofitable_deviation_share.value, 3),
      anyn: num(Math.round(bt.nash_any_dev_share * bt.n), 0), n: num(bt.n, 0) });
  }

  function renderAll() {
    I.apply();
    const other = I.getLang() === "pt" ? "en" : "pt";
    $("lang-btn-text").setAttribute("lang", other === "en" ? "en" : "pt-BR");
    renderStatic(); renderSide(); renderButtons(); renderScore(); renderLearnTable(); renderDevTable();
    if (S.statusKey) $("status").textContent = t(S.statusKey, S.statusVars);
    $("dev-cycle-note").hidden = !(S.final && S.final.cycle_len > 1);
    drawLearning(); drawDeviation(); drawSeparation(); drawSepMain();
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
      if (S.worker !== w) return;   // mensagem atrasada de um worker já descartado
      const msg = ev.data;
      if (msg.type === "blocks" && msg.runId === S.runId) {
        Array.prototype.push.apply(S.blocks, msg.blocks);
        startAnim();
      } else if (msg.type === "done" && msg.runId === S.runId) {
        const r = msg.result, m = S.mk.m;
        S.final = { converged: r.converged, periods: r.periods, delta: r.delta, cycle_len: r.cycle_len, ms: msg.ms,
                    greedy: Int32Array.from(r.greedy),
                    cycle: r.cycle_idx ? r.cycle_idx.map((p) => p[0] * m + p[1]) : [] };
        startAnim();
      } else if (msg.type === "error" && (msg.runId === undefined || msg.runId === S.runId)) {
        workerFailed(msg.message);
      }
    };
    w.onerror = (e) => { e.preventDefault(); if (S.worker !== w) return; workerFailed(e.message || "Worker"); };
    w.postMessage({ type: "init", params: S.params });
    S.worker = w;
    workerRun = 0;   // o worker novo conta as sessões a partir de 1; a página acompanha
    return w;
  }
  // Worker conta as sessões por runId próprio; a página manda start/stop e acompanha a última.
  let workerRun = 0;

  function workerFailed(message) {
    // descarta o worker quebrado: a próxima "Nova sessão ao vivo" cria outro
    if (S.worker) { try { S.worker.terminate(); } catch (e) { /* já encerrado */ } S.worker = null; }
    if (typeof console !== "undefined") console.error("Worker:", message);
    S.running = false;
    showRecorded();
    setStatus("status_worker_error");
  }

  function startLive(seed) {
    let w;
    try { w = ensureWorker(); } catch (e) { workerFailed(e.message); return; }
    S.mode = "live"; S.seed = seed; S.blocks = []; S.shown = 0; S.final = null;
    S.paused = false; S.running = true;
    workerRun += 1; S.runId = workerRun;
    // sem memória: o mesmo motor com um estado só (validado em test_engine_k0.js)
    w.postMessage({ type: "start", seed: seed, blockSize: S.blockSize, opts: { memory: S.memory === "without" ? 0 : 1 } });
    S.dev = null; S.devShown = NT; clearTimeout(S.devTimer); renderDevTable();
    setStatus("status_computing");
    renderSide(); renderButtons(); renderScore(); drawLearning(); drawDeviation();
    $("dev-cycle-note").hidden = true;
  }

  // Seletor "Com memória | Sem memória": interrompe uma sessão ao vivo e mostra a gravada.
  function setMemory(which) {
    if (S.memory === which) return;
    if (S.worker && S.running) {
      S.worker.postMessage({ type: "stop" });
      workerRun += 1;                   // o worker também conta o "stop" (qlearning.worker.js)
    }
    S.runId = -1;                       // ignora blocos atrasados da sessão ao vivo
    if (rafId !== null) { cancelAnimationFrame(rafId); rafId = null; }
    S.memory = which;
    showRecorded();
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
    S.devShown = animate && !reduceMotion ? 1 : NT;
    renderDevTable();
    const stepAnim = () => {
      drawDeviation(); renderScore();
      if (S.devShown < NT) { S.devShown += 1; S.devTimer = setTimeout(stepAnim, TAU_MS); }
    };
    stepAnim();
  }

  // ------------------------------------------------------------------ botões
  function randomSeed() {
    const a = new Uint32Array(1);
    (window.crypto || window.msCrypto).getRandomValues(a);
    return 1 + (a[0] % 999999);
  }

  // Botão principal: reproduz o aprendizado da sessão mostrada (gravada ou ao vivo), sem
  // recalcular; durante a animação vira Pausar/Continuar. Sessão nova ao vivo: botão "Nova sessão".
  function onRun() {
    const learning = !(S.final && S.shown >= S.blocks.length);
    if (!learning) {
      S.shown = 0; S.paused = false;
      setStatus(null); renderButtons(); renderSide(); startAnim();
      return;
    }
    S.paused = !S.paused;
    if (S.paused) {
      const b = S.blocks[Math.max(0, Math.floor(S.shown) - 1)];
      const per = b ? b.t_start + b.n_periods : 0;
      setStatus("status_paused", { t: () => num(per, 0) });
    } else {
      // replay de sessão já calculada: nada a calcular; sessão ao vivo em cálculo: "Calculando…"
      setStatus(S.final ? null : "status_computing"); startAnim();
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
    Promise.all([loadJSON("qlearning_params.json"), loadJSON("qlearning_reference.json"), loadJSON("qlearning_sessions.json"),
                 loadJSON("qlearning_memoryless.json")])
      .then(([p, r, s, k0]) => {
        S.params = p; S.ref = r; S.sessions = s; S.mk = E.setupMarket(p); S.k0 = k0;
        // tamanho do bloco das sessões gravadas; o worker usa o mesmo nas sessões ao vivo
        S.blockSize = (s.sessions.find((x) => x.opens_page) || s.sessions[0]).learning_blocks_linear.block_size;
        const sup = (e) => String(e).replace(/\d/g, (c) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[c]);
        // números que aparecem nos textos, lidos dos JSON (função: formata no idioma atual)
        I.setGlobals({
          nsess: () => num(r.base.n_sessions, 0),
          block: () => num(S.blockSize, 0),
          stable: () => num(p.learning.convergence.t_stable, 0),
          tmax: () => "10" + sup(Math.round(Math.log10(p.learning.convergence.t_max))),
          m: () => num(p.learning.m.value, 0),
          horizon: () => num(r.deviation.protocol.horizon, 0),
          nrec: () => num(s.sessions.length, 0),
        });
        if (r.deviation.protocol.horizon !== D.T_DEV) console.error("horizonte do desvio: JSON ≠ deviation.js");
        $("btn-run").addEventListener("click", onRun);
        $("btn-deviate").addEventListener("click", onDeviate);
        $("btn-new").addEventListener("click", () => startLive(randomSeed()));
        ["learn-chart", "dev-chart", "sep-main", "sep-chart"].forEach(attachHover);
        document.addEventListener("keydown", (ev) => { if (ev.key === "Escape") hideTip(); });
        $("mem-with").addEventListener("click", () => setMemory("with"));
        $("mem-without").addEventListener("click", () => setMemory("without"));
        // o gráfico das 7 condições fica recolhido: desenha com a largura certa ao abrir
        $("sep-more").addEventListener("toggle", () => { if ($("sep-more").open) drawSeparation(); });
        $("lang-btn").addEventListener("click", onLang);
        let rs = null;
        window.addEventListener("resize", () => { clearTimeout(rs); rs = setTimeout(() => { drawLearning(); drawDeviation(); drawSeparation(); drawSepMain(); }, 120); });
        showRecorded();
        document.body.dataset.ready = "1";
      })
      .catch((e) => { $("status").textContent = "Erro ao carregar os dados / Error loading data: " + e.message; });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
