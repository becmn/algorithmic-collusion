"""
Fase 3 — Execuções com agentes de LLM: teste de competência, duopólio e teste de desvio.

Protocolo (documento do projeto: Protocolo experimental, Agentes de LLM; Pesquisas 2, 7 e 8):
  - cada período, cada firma recebe o prefixo (system prompt) e o corpo com custo, teto,
    PLANS.txt, INSIGHTS.txt e os últimos 100 períodos; responde raciocínio, PLANS,
    INSIGHTS e preço; PLANS e INSIGHTS são reescritos (vazios no período 1);
  - resposta inválida: até 10 novas tentativas; depois a execução para, com o motivo;
  - limite de uso (429) ou erro do servidor: espera e tenta de novo (não conta como inválida);
  - teto de gasto: para ANTES da chamada que passaria do teto;
  - tudo é salvo a cada período: a execução pode ser retomada do ponto onde parou.

Arquivos de cada execução (pasta results/llm/<run_id>/, ou results/llm_pilot/ no piloto):
  config.json     parâmetros, modelo, hash SHA-256 de cada texto de prompt usado
  state.json      período concluído, notas e histórico de cada firma, situação
  periods.csv     preços, quantidades e lucros por período (com escala)
  calls.jsonl     uma linha por chamada: prompt, resposta, preço extraído, tokens, modelo,
                  versão devolvida, data, temperatura, seed, nível de raciocínio, custo
  deviation/      ramos de controle e de corte e deviation.json (R, G, V, placebo, classes)

Código próprio do projeto.
"""

from __future__ import annotations

import copy
import csv
import datetime as dt
import hashlib
import json
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

from client import (Budget, BudgetExceeded, FatalAPIError, RateLimited, TransientAPIError, call_cost,
                    call_with_backoff, safe_replace)
from market import MAX_RETRIES, PERIODS, Market, competence_summary, deviation_metrics, duopoly_summary
from prompts import InvalidResponse, PromptSet, parse_response, user_prompt

ROOT = Path(__file__).resolve().parent.parent
T_DEV = 15


class RunStopped(RuntimeError):
    """A execução parou de vez (10 novas tentativas inválidas): é reportada com o motivo."""


class RunPaused(RuntimeError):
    """A execução parou mas pode ser retomada (teto de gasto, limite de uso)."""


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def call_seed(run_seed: int, phase: str, period: int, firm: int, attempt: int) -> int:
    """Seed de cada chamada, derivada da semente da execução (registrada no log)."""
    h = hashlib.sha256(f"{run_seed}|{phase}|{period}|{firm}|{attempt}".encode()).hexdigest()
    return int(h[:8], 16) & 0x7FFFFFFF


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class AgentState:
    firm: int
    prefix: str
    plans: str = ""
    insights: str = ""
    history: list = field(default_factory=list)


@dataclass
class Context:
    client: object
    prompts: PromptSet
    market: Market
    budget: Budget
    run_dir: Path
    alpha: float
    seed: int
    sleep: object = time.sleep
    log: object = print
    last_prompt_tokens: int = 0
    last_output_tokens: int = 0

    def max_call_cost(self, chars: int) -> float:
        """Custo máximo estimado da próxima chamada (para o teto): tokens de entrada pela
        última chamada (+20%) ou por 1 token a cada 2 caracteres; saída pelo limite fixado ou
        pelo dobro da última (mínimo 4.000 tokens, folga para o raciocínio)."""
        pin = max(int(self.last_prompt_tokens * 1.2), chars // 2)
        mot = getattr(self.client, "max_output_tokens", None)
        pout = mot or max(4000, 2 * self.last_output_tokens)
        return call_cost(self.client.model, self.client.service_tier, pin, pout, 0)


# ---------------------------------------------------------------------------
# Arquivos
# ---------------------------------------------------------------------------
def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    safe_replace(tmp, path)   # troca atômica: um arquivo nunca fica pela metade


def log_call(run_dir: Path, rec: dict):
    """Uma linha JSON por chamada, em texto simples (se o programa cair no meio de uma
    linha, só ela se perde; as outras continuam legíveis)."""
    p = run_dir / "calls.jsonl"
    lead = ""
    if p.is_file() and p.stat().st_size:
        with open(p, "rb") as f:
            f.seek(-1, 2)
            if f.read(1) != b"\n":
                lead = "\n"          # linha anterior cortada por uma queda: começa numa linha nova
    with open(p, "a", encoding="utf-8") as f:
        f.write(lead + json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()


class OutputLimitHit(RuntimeError):
    """Resposta cortada pelo limite de tokens de saída: repetir não adianta (o limite é o
    mesmo); a execução pausa para aumentar --max-output num plano novo, com registro."""


def read_calls(run_dir: Path, dedupe: bool = False) -> list:
    """Lê o log; ignora uma linha cortada. dedupe=True mantém só o último registro de cada
    (fase, período, firma, tentativa): uma interrupção no meio de um período faz a mesma
    decisão ser pedida de novo na retomada (o custo das duas fica no total)."""
    p = run_dir / "calls.jsonl"
    if not p.is_file():
        return []
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if dedupe:
        last = {}
        for i, c in enumerate(out):
            last[(c["phase"], c["period"], c["firm"], c["attempt"])] = i
        keep = set(last.values())
        out = [c for i, c in enumerate(out) if i in keep]
    return out


PERIOD_COLS = ["period", "price_1", "price_2", "qty_1", "qty_2", "profit_1", "profit_2"]


def write_periods(run_dir: Path, rows: list):
    tmp = run_dir / "periods.csv.tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PERIOD_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in PERIOD_COLS})
    safe_replace(tmp, run_dir / "periods.csv")


def read_periods(run_dir: Path) -> list:
    p = run_dir / "periods.csv"
    if not p.is_file():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return [{k: (float(v) if v not in ("", None) else None) for k, v in r.items()} for r in csv.DictReader(f)]


# ---------------------------------------------------------------------------
# Uma decisão de preço (com novas tentativas)
# ---------------------------------------------------------------------------
def decide(ctx: Context, agent: AgentState, period: int, phase: str, monopoly: bool) -> float:
    mk, ps = ctx.market, ctx.prompts
    system = ps.prefixes[agent.prefix]
    user = user_prompt(ps, cost=mk.cost(ctx.alpha), ceiling=mk.ceiling(ctx.alpha),
                       plans=agent.plans, insights=agent.insights, history=agent.history,
                       monopoly=monopoly)
    errors = []
    for attempt in range(MAX_RETRIES + 1):            # 1 tentativa + até 10 novas
        seed = call_seed(ctx.seed, phase, period, agent.firm, attempt)
        try:
            ctx.budget.check(ctx.max_call_cost(len(system) + len(user)))
        except BudgetExceeded as e:
            raise RunPaused(str(e)) from e
        t0 = time.time()
        try:
            res = call_with_backoff(ctx.client, system, user, seed, sleep=ctx.sleep, log=ctx.log)
        except (RateLimited, TransientAPIError) as e:
            raise RunPaused(f"limite de uso ou erro do servidor: {e}") from e
        except FatalAPIError as e:
            raise RunPaused(f"erro da API (resolver e retomar): {e}") from e
        latency = time.time() - t0
        cost = call_cost(ctx.client.model, ctx.client.service_tier, res.prompt_tokens,
                         res.output_tokens, res.thoughts_tokens)
        ctx.budget.add(cost)
        ctx.last_prompt_tokens, ctx.last_output_tokens = res.prompt_tokens, res.output_tokens + res.thoughts_tokens
        rec = {"time_utc": now(), "phase": phase, "period": period, "firm": agent.firm + 1,
               "attempt": attempt, "prefix": agent.prefix, "alpha": ctx.alpha,
               **{k: v for k, v in ctx.client.settings().items()},
               "seed": seed, "model_version": res.model_version, "response_id": res.response_id,
               "finish_reason": res.finish_reason, "system_prompt": system, "user_prompt": user,
               "response": res.text, **res.extra, "prompt_tokens": res.prompt_tokens,
               "output_tokens": res.output_tokens, "thoughts_tokens": res.thoughts_tokens,
               "cost_usd": cost, "latency_s": round(latency, 3)}
        if str(res.finish_reason or "").upper().endswith("MAX_TOKENS"):
            rec.update(valid=False, error="resposta cortada pelo limite de saída (MAX_TOKENS)", price=None)
            log_call(ctx.run_dir, rec)
            raise RunPaused("resposta cortada pelo limite de tokens de saída (MAX_TOKENS): "
                            "aumentar o limite exige plano novo, registrado em Decisões")
        try:
            parsed = parse_response(ps, res.text, res.finish_reason)
        except InvalidResponse as e:
            rec.update(valid=False, error=str(e), price=None)
            log_call(ctx.run_dir, rec)
            errors.append(str(e))
            continue
        rec.update(valid=True, error=None, price=parsed["price_value"],
                   plans=parsed["plans"], insights=parsed["insights"])
        log_call(ctx.run_dir, rec)
        agent.plans, agent.insights = parsed["plans"], parsed["insights"]   # reescritos (Protocolo)
        return parsed["price_value"]
    raise RunStopped(f"{MAX_RETRIES + 1} respostas inválidas seguidas (firma {agent.firm + 1}, "
                     f"período {period}, fase {phase}); última: {errors[-1]}")


# ---------------------------------------------------------------------------
# Laço de períodos (duopólio, monopolista e ramos do desvio)
# ---------------------------------------------------------------------------
def run_periods(ctx: Context, agents: list, rows: list, start: int, end: int, phase: str,
                state_extra: dict, forced: dict | None = None):
    """Roda os períodos start..end. forced = {(período, firma): preço} impõe um preço
    (o corte do teste de desvio) sem chamar o LLM para aquela firma. Salva tudo a cada período."""
    mk, monopoly = ctx.market, len(agents) == 1
    for t in range(start, end + 1):
        prices = []
        for ag in agents:
            key = (t, ag.firm)
            prices.append(forced[key] if forced and key in forced else decide(ctx, ag, t, phase, monopoly))
        qty, profit = mk.outcome(prices, ctx.alpha)
        for i, ag in enumerate(agents):
            ag.history.append({"period": t, "my_price": prices[i],
                               "rival_price": None if monopoly else prices[1 - i],
                               "qty": float(qty[i]), "profit": float(profit[i])})
        row = {"period": t, "price_1": prices[0], "qty_1": float(qty[0]), "profit_1": float(profit[0])}
        if not monopoly:
            row.update(price_2=prices[1], qty_2=float(qty[1]), profit_2=float(profit[1]))
        rows.append(row)
        write_periods(ctx.run_dir, rows)
        write_json(ctx.run_dir / "state.json", {**state_extra, "period": t, "status": "running",
                                                "agents": [asdict(a) for a in agents], "updated": now()})
    return rows


def _load_or_init(ctx: Context, config: dict, agents: list):
    """Retomada: se já existe state.json, continua dele (e corta periods.csv no mesmo ponto)."""
    st_path = ctx.run_dir / "state.json"
    cfg_path = ctx.run_dir / "config.json"
    if st_path.is_file():
        old = json.loads(cfg_path.read_text(encoding="utf-8"))
        same = {k: old.get(k) for k in ("kind", "prefix", "alpha", "seed", "periods", "model", "prompt_hashes",
                                        "client")}
        new = {k: config.get(k) for k in same}
        if same != new:
            raise RuntimeError(f"{ctx.run_dir} já existe com outra configuração: {same} × {new}")
        st = json.loads(st_path.read_text(encoding="utf-8"))
        if st["status"] in ("done", "stopped"):
            return st, None, None
        if not st.get("agents"):
            # pausou antes de concluir o 1º período: começa do zero (nada foi salvo)
            ctx.log(f"Recomeçando {ctx.run_dir.name} (pausou antes do 1º período: {st.get('reason')})")
            return None, agents, []
        agents = [AgentState(**a) for a in st["agents"]]
        rows = read_periods(ctx.run_dir)[: st["period"] - config.get("first_period", 1) + 1]
        ctx.log(f"Retomando {ctx.run_dir.name} do período {st['period'] + 1}")
        return st, agents, rows
    ctx.run_dir.mkdir(parents=True, exist_ok=True)
    write_json(cfg_path, config)
    return None, agents, []


def _finish(ctx, status: str, reason: str | None, extra: dict | None = None, tb: str | None = None):
    st = json.loads((ctx.run_dir / "state.json").read_text(encoding="utf-8")) \
        if (ctx.run_dir / "state.json").is_file() else {"period": 0, "agents": []}
    if tb:   # erro inesperado: o rastro completo vai para um arquivo, para separar bug de pausa
        with open(ctx.run_dir / "errors.log", "a", encoding="utf-8") as f:
            f.write(f"--- {now()}\n{tb}\n")
    st.update(status=status, reason=reason, updated=now(), **(extra or {}))
    write_json(ctx.run_dir / "state.json", st)
    return st


def base_config(ctx: Context, kind: str, prefixes: list, periods: int, first_period: int = 1) -> dict:
    ps = ctx.prompts
    return {"kind": kind, "prefix": "+".join(prefixes), "alpha": ctx.alpha, "seed": ctx.seed,
            "periods": periods, "first_period": first_period, "created_utc": now(),
            "model": ctx.client.settings()["model"], "client": ctx.client.settings(),
            "prompt_source": ps.source,
            "prompt_hashes": {**{p: sha(ps.prefixes[p]) for p in sorted(set(prefixes))},
                              "template": sha(ps.template),
                              "template_monopoly": sha(ps.template_monopoly) if ps.template_monopoly else None,
                              "format": sha(json.dumps(ps.fmt, sort_keys=True))}}


def run_game(ctx: Context, kind: str, prefixes: list, periods: int = PERIODS) -> dict:
    """kind = 'competence' (1 firma, P0) ou 'duopoly' (2 firmas). Devolve state.json."""
    config = base_config(ctx, kind, prefixes, periods)
    agents = [AgentState(firm=i, prefix=p) for i, p in enumerate(prefixes)]
    st, agents, rows = _load_or_init(ctx, config, agents)
    if agents is None:
        return st
    start = (st["period"] + 1) if st else 1
    try:
        run_periods(ctx, agents, rows, start, periods, kind, {"kind": kind})
    except RunStopped as e:
        return _finish(ctx, "stopped", str(e))
    except RunPaused as e:
        return _finish(ctx, "paused", str(e))
    except Exception as e:  # noqa: BLE001 — erro inesperado: salva o motivo; retoma depois
        return _finish(ctx, "paused", f"erro inesperado (ver errors.log): {type(e).__name__}: {str(e)[:300]}",
                       tb=traceback.format_exc())
    if not rows or len(rows) < periods:   # nada a fazer (já estava completo)
        rows = read_periods(ctx.run_dir)
    p1 = [r["price_1"] for r in rows]
    if kind == "competence":
        summ = competence_summary(ctx.market, p1, ctx.alpha)
    else:
        summ = duopoly_summary(ctx.market, p1, [r["price_2"] for r in rows], ctx.alpha) \
            if periods >= 300 else {"periods": len(rows)}
    return _finish(ctx, "done", None, {"summary": summ})


def run_deviation(ctx: Context, base_dir: Path) -> dict:
    """Teste de desvio comum a partir do estado ao fim do período 300 de um duopólio concluído.
    Ramo de controle: os dois seguem chamando o LLM por 15 períodos. Ramo com corte: em τ = 1 a
    firma 1 cobra a melhor resposta estática contínua ao preço da firma 2 em τ = 0 (sem chamar
    o LLM); a firma 2 e, de τ = 2 a 15, as duas seguem chamando o LLM com a mesma temperatura."""
    st = json.loads((base_dir / "state.json").read_text(encoding="utf-8"))
    cfg = json.loads((base_dir / "config.json").read_text(encoding="utf-8"))
    if cfg["kind"] != "duopoly" or st["status"] != "done":
        raise RuntimeError(f"{base_dir.name}: o desvio exige um duopólio concluído (situação: {st['status']})")
    t0 = st["period"]
    snap = [AgentState(**a) for a in st["agents"]]
    p0 = [snap[0].history[-1]["my_price"], snap[1].history[-1]["my_price"]]
    cut = ctx.market.best_response_scaled(p0[1], ctx.alpha)
    paths = {}
    for branch, forced in (("control", None), ("cut", {(t0 + 1, 0): cut})):
        bctx = copy.copy(ctx)
        bctx.run_dir = base_dir / "deviation" / branch
        phase = f"deviation_{branch}"
        config = base_config(bctx, phase, [a.prefix for a in snap], t0 + T_DEV, first_period=t0 + 1)
        config["base_run"] = base_dir.name
        bst, agents, rows = _load_or_init(bctx, config, copy.deepcopy(snap))
        if agents is None and bst["status"] != "done":
            return {"status": bst["status"], "reason": bst.get("reason")}
        if agents is not None:
            start = max(bst["period"] + 1, t0 + 1) if bst else t0 + 1
            try:
                run_periods(bctx, agents, rows, start, t0 + T_DEV, phase, {"kind": phase}, forced)
            except RunStopped as e:
                _finish(bctx, "stopped", str(e))
                return {"status": "stopped", "reason": str(e)}
            except RunPaused as e:
                _finish(bctx, "paused", str(e))
                return {"status": "paused", "reason": str(e)}
            except Exception as e:  # noqa: BLE001
                reason = f"erro inesperado (ver errors.log): {type(e).__name__}: {str(e)[:300]}"
                _finish(bctx, "paused", reason, tb=traceback.format_exc())
                return {"status": "paused", "reason": reason}
            _finish(bctx, "done", None)
        rows = read_periods(bctx.run_dir)
        paths[branch] = [p0] + [[r["price_1"], r["price_2"]] for r in rows]
    m = deviation_metrics(ctx.market, paths["control"], paths["cut"], ctx.alpha)
    out = {"status": "done", "base_run": base_dir.name, "tau0_prices": p0, "cut_price": cut,
           "control": paths["control"], "cut": paths["cut"], **m}
    write_json(base_dir / "deviation" / "deviation.json", out)
    return out
