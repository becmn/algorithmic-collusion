/*
 * Web Worker do simulador (Fase 2.1): roda uma sessão de Q-learning fora da thread da
 * página, para não travar a tela (Pesquisa 9). Calcula em blocos de 10.000 períodos e
 * envia um resumo por bloco; a página decide o ritmo da animação.
 *
 * Mensagens que o Worker recebe:
 *   {type: "init", params}                 params = conteúdo de data/qlearning_params.json
 *   {type: "start", seed, opts?, blockSize?} começa uma sessão nova (interrompe a anterior)
 *   {type: "stop"}                         interrompe a sessão em andamento
 * Mensagens que o Worker envia:
 *   {type: "ready", grid}                  mercado montado
 *   {type: "blocks", runId, seed, blocks}  lote de resumos (formato de learning_blocks_linear)
 *   {type: "done", runId, result, ms}      resultado final (Δ no ciclo-limite, ciclo, estratégia)
 *   {type: "error", message}
 * Código próprio do projeto.
 */
importScripts("engine.js");

let market = null;
let runId = 0;          // muda a cada "start"/"stop"; lotes de sessões antigas são descartados
const SLICE_MS = 30;    // calcula ~30 ms e devolve o controle, para receber "stop"

self.onmessage = function (ev) {
  const msg = ev.data || {};
  try {
    if (msg.type === "init") {
      market = self.QEngine.setupMarket(msg.params);
      self.postMessage({ type: "ready", grid: Array.from(market.prices) });
    } else if (msg.type === "start") {
      if (!market) throw new Error("Worker sem parâmetros: envie 'init' antes de 'start'.");
      // validação das entradas (revisão de 16/10/2026)
      const seed = msg.seed;
      if (!Number.isInteger(seed) || seed < 0 || seed > 0xFFFFFFFF)
        throw new Error("Semente inválida: use um inteiro entre 0 e 4294967295.");
      const blockSize = msg.blockSize === undefined ? 10000 : msg.blockSize;
      if (!Number.isInteger(blockSize) || blockSize < 1)
        throw new Error("blockSize inválido: use um inteiro ≥ 1.");
      const opts = msg.opts || {};
      for (const k of ["tStable", "tMax"])
        if (opts[k] !== undefined && (!Number.isInteger(opts[k]) || opts[k] < 1))
          throw new Error(k + " inválido: use um inteiro ≥ 1.");
      if (opts.memory !== undefined && opts.memory !== 0 && opts.memory !== 1)
        throw new Error("memory inválido: use 0 (sem memória) ou 1 (com memória).");
      runId += 1;
      runLoop(runId, seed, opts, blockSize);
    } else if (msg.type === "stop") {
      runId += 1;
    }
  } catch (err) {
    self.postMessage({ type: "error", message: String(err && err.message ? err.message : err) });
  }
};

function runLoop(id, seed, opts, blockSize) {
  const E = self.QEngine;
  const sess = E.createSession(market, seed, opts);
  const t0 = Date.now();

  function slice() {
    if (id !== runId) return;               // sessão interrompida
    try {
      sliceBody();
    } catch (err) {                         // erros dentro do setTimeout também viram mensagem
      self.postMessage({ type: "error", runId: id,
                         message: String(err && err.message ? err.message : err) });
    }
  }

  function sliceBody() {
    const blocks = [];
    const tSlice = Date.now();
    while (!sess.done && Date.now() - tSlice < SLICE_MS) {
      blocks.push(E.runBlock(sess, blockSize));
    }
    self.postMessage({ type: "blocks", runId: id, seed: seed, blocks: blocks });
    if (sess.done) {
      const result = E.finishSession(sess);
      result.greedy = Array.from(result.greedy);
      self.postMessage({ type: "done", runId: id, result: result, ms: Date.now() - t0 });
    } else {
      setTimeout(slice, 0);
    }
  }
  slice();
}
