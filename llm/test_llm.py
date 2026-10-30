"""
Fase 3 — Conferências da infraestrutura dos LLMs, sem chamar a API (cliente falso).

Uso (na raiz do repositório): python llm/test_llm.py
Código próprio do projeto.
"""

from __future__ import annotations

import gzip
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from client import Budget, FakeClient, RateLimited, call_cost, call_with_backoff  # noqa: E402
from experiment import Context, read_calls, read_periods, run_deviation, run_game  # noqa: E402
from market import ALPHAS, MAX_RETRIES, Market, competence_summary, deviation_metrics, duopoly_summary  # noqa: E402
from prompts import InvalidResponse, load_prompts, market_data, parse_response, user_prompt  # noqa: E402

checks = []


def check(name, ok, detail=None):
    checks.append((name, bool(ok)))
    if not ok and detail is not None:
        print("  ->", detail)


mk = Market()
ps = load_prompts("test")
TMP = Path(tempfile.mkdtemp(prefix="llm_test_"))

# ---------------------------------------------------------------- mercado e escala
ref = json.loads((HERE.parent / "docs" / "data" / "qlearning_params.json").read_text(encoding="utf-8"))["reference_points"]
check("referências iguais às do Q-learning (JSON exportado de results/)",
      abs(mk.pN - ref["price_nash"]) < 1e-6 and abs(mk.pM - ref["price_monopoly"]) < 1e-6
      and abs(mk.piN - ref["profit_nash"]) < 1e-6 and abs(mk.piM - ref["profit_monopoly"]) < 1e-6)
ok = True
for a in ALPHAS:
    ok &= abs(mk.delta(mk.outcome([mk.pN * a] * 2, a)[1].mean(), a)) < 1e-9
    ok &= abs(mk.delta(mk.outcome([mk.pM * a] * 2, a)[1].mean(), a) - 1) < 1e-9
check("Δ = 0 em Nash e 1 no monopólio, nas 3 escalas α", ok)
check("melhor resposta com escala = α × melhor resposta sem escala",
      all(abs(mk.best_response_scaled(1.8 * a, a) - a * mk.best_response_scaled(1.8, 1.0)) < 1e-6 for a in ALPHAS))
check("custo α·c e teto 4,51 × α", mk.cost(3.2) == 3.2 and abs(mk.ceiling(10) - 45.1) < 1e-12)
q, pi = mk.outcome([1.5, 1.6], 1.0)
check("quantidade com β = 100 (fatias × 100)", abs(q.sum() - 100 * (1 - 1 / (1 + np.exp(4 * (2 - 1.5)) + np.exp(4 * (2 - 1.6))))) < 1e-9)

# ---------------------------------------------------------------- prompts e resposta
hist = [{"period": t, "my_price": 1.234 + t, "rival_price": 2.0, "qty": 40.0, "profit": 9.876} for t in range(1, 151)]
md = market_data(ps, hist, monopoly=False)
lines = md.splitlines()
check("histórico: últimos 100 períodos, do mais recente ao mais antigo, 2 casas",
      len(lines) == 100 and lines[0].startswith("Round 150:") and lines[-1].startswith("Round 51:")
      and "my price 151.23" in lines[0] and "profit 9.88" in lines[0])
up = user_prompt(ps, cost=3.2, ceiling=mk.ceiling(3.2), plans="P", insights="I", history=[], monopoly=False)
check("período 1: notas vazias e sem histórico; custo e teto com 2 casas",
      "Cost per unit: $3.20" in up and "$14.43" in up and "(no data yet)" in up)
good = "OBSERVATIONS: a\nPLANS: b\nINSIGHTS: c\nPRICE: $1,234.50"
r = parse_response(ps, good)
check("resposta válida: campos e preço (com $ e separador de milhar)",
      r["plans"] == "b" and r["insights"] == "c" and r["price_value"] == 1234.5)
bad_ok = True
for bad in ["", "OBSERVATIONS: a\nPLANS: b\nINSIGHTS: c", "PLANS: b\nOBSERVATIONS: a\nINSIGHTS: c\nPRICE: 2",
            "OBSERVATIONS: a\nPLANS: b\nINSIGHTS: c\nPRICE: free", "OBSERVATIONS: a\nPLANS: b\nINSIGHTS: c\nPRICE: -3",
            "OBSERVATIONS: a\nPLANS: b\nINSIGHTS: c\nPRICE: 0"]:
    try:
        parse_response(ps, bad)
        bad_ok = False
    except InvalidResponse:
        pass
check("respostas inválidas recusadas (vazia, sem preço, fora de ordem, não numérico, ≤ 0)", bad_ok)

# ---------------------------------------------------------------- políticas falsas
def reply(price, plans="p", insights="i"):
    return f"OBSERVATIONS: x\nPLANS: {plans}\nINSIGHTS: {insights}\nPRICE: {price:.4f}"


def cost_of(user):
    return float(re.search(r"Cost per unit: \$(\d+\.\d+)", user).group(1))


def last(user, what):
    m = re.search(r"Round \d+: my price (\d+\.\d+)(?:, competitor price (\d+\.\d+))?", user)
    if not m:
        return None
    return float(m.group(1 if what == "me" else 2))


def ctx_for(name, client, alpha=1.0, seed=1, budget=100.0):
    d = TMP / name
    return Context(client=client, prompts=ps, market=mk, budget=Budget(TMP / f"{name}_spend.json", budget),
                   run_dir=d, alpha=alpha, seed=seed, sleep=lambda s: None, log=lambda *a: None)


def monopolist(system, user, seed, n):
    a = cost_of(user)                      # custo = α·c com c = 1
    return reply(mk.p1 * a)


for a in ALPHAS:
    st = run_game(ctx_for(f"comp_{a}", FakeClient(monopolist), alpha=a), "competence", ["P0"], 300)
    if not (st["status"] == "done" and st["summary"]["passed"]):
        break
else:
    st = None
check("competência: monopolista no preço ótimo passa nas 3 escalas", st is None, st)
st = run_game(ctx_for("comp_nash", FakeClient(lambda s, u, sd, n: reply(mk.pN * cost_of(u)))), "competence", ["P0"], 300)
check("competência: monopolista no preço de Nash não passa", st["status"] == "done" and not st["summary"]["passed"])

# ---------------------------------------------------------------- novas tentativas
calls = {"n": 0}


def flaky(system, user, seed, n):
    calls["n"] += 1
    return "lixo" if calls["n"] <= 3 else reply(1.7)


st = run_game(ctx_for("retry", FakeClient(flaky)), "duopoly", ["P1", "P1"], 1)
log = read_calls(TMP / "retry")
check("3 respostas inválidas e depois válida: segue, e as 4 tentativas ficam no log",
      st["status"] == "done" and [c["valid"] for c in log[:4]] == [False, False, False, True]
      and [c["attempt"] for c in log[:4]] == [0, 1, 2, 3])
st = run_game(ctx_for("allbad", FakeClient(lambda *a: "lixo")), "duopoly", ["P1", "P1"], 5)
log = read_calls(TMP / "allbad")
check(f"{MAX_RETRIES + 1} respostas inválidas seguidas: a execução para e o motivo fica registrado",
      st["status"] == "stopped" and "inválidas" in st["reason"] and len(log) == MAX_RETRIES + 1)
seeds = [c["seed"] for c in read_calls(TMP / "allbad")]
check("cada tentativa tem seed própria, registrada", len(set(seeds)) == len(seeds))

# ---------------------------------------------------------------- retomada
def follower(system, user, seed, n):
    """Copia o último preço do rival e sobe 1% (determinístico, depende só do prompt)."""
    r = last(user, "rival")
    return reply(1.6 * cost_of(user) if r is None else min(r * 1.01, 1.9 * cost_of(user)))


class Crash(Exception):
    pass


def crashing(after):
    def pol(system, user, seed, n):
        if n > after:
            raise Crash()
        return follower(system, user, seed, n)
    return pol


full = run_game(ctx_for("full", FakeClient(follower)), "duopoly", ["P1", "P1"], 40)
try:
    run_game(ctx_for("resume", FakeClient(crashing(31))), "duopoly", ["P1", "P1"], 40)
except Crash:
    pass
mid = json.loads((TMP / "resume" / "state.json").read_text())
st = run_game(ctx_for("resume", FakeClient(follower)), "duopoly", ["P1", "P1"], 40)
a_rows, b_rows = read_periods(TMP / "full"), read_periods(TMP / "resume")
check("retomada: interrompida no meio, continua do período salvo e chega ao mesmo resultado",
      mid["period"] == 15 and st["status"] == "done" and a_rows == b_rows and len(b_rows) == 40)
try:
    run_game(ctx_for("resume", FakeClient(follower), alpha=3.2), "duopoly", ["P1", "P1"], 40)
    diff_cfg = False
except RuntimeError:
    diff_cfg = True
check("retomada recusa pasta com outra configuração", diff_cfg)

# ---------------------------------------------------------------- limite de uso e teto
fc = FakeClient(follower, rate_limit_calls={1, 2})
waits = []
res = call_with_backoff(fc, "s", "Cost per unit: $1.00", 1, sleep=waits.append, log=lambda *a: None)
check("limite de uso (429): espera e tenta de novo, sem contar como inválida", len(waits) == 2 and fc.calls == 3)
fc = FakeClient(follower, rate_limit_calls=set(range(1, 100)))
st = run_game(ctx_for("rl", fc), "duopoly", ["P1", "P1"], 3)
check("limite de uso persistente: execução pausada (pode ser retomada)", st["status"] == "paused")
one = call_cost("gemini-3.5-flash-lite", None, 2000, 300, 0)
st = run_game(ctx_for("budget", FakeClient(follower), budget=0.10), "duopoly", ["P1", "P1"], 300)
spent = json.loads((TMP / "budget_spend.json").read_text())["spent_usd"]
check("teto de gasto: para antes de passar do teto e fica pausada",
      st["status"] == "paused" and spent <= 0.10 and "teto" in st["reason"], (st.get("reason"), spent))
check("custo: 2.000 de entrada e 300 de saída no Flash-Lite = US$ 0,00135 (Pesquisa 7)", abs(one - 0.00135) < 1e-12)
check("custo do Flex = metade", abs(call_cost("gemini-3.5-flash-lite", "flex", 2000, 300, 0) - one / 2) < 1e-12)

# ---------------------------------------------------------------- resumo e desvio
p1 = np.r_[np.full(250, 1.6), np.full(50, 1.7)]
s = duopoly_summary(mk, p1 * 3.2, p1 * 3.2, 3.2)
pi = mk.outcome([1.7, 1.7], 1.0)[1][0] / 100
check("Δ da execução: média dos períodos 251–300, dividida por α·β",
      abs(s["delta"] - (pi - mk.piN) / (mk.piM - mk.piN)) < 1e-9 and s["stabilized"])
wild = np.r_[np.full(200, 1.6), np.tile([1.5, 1.9], 50)]
check("estabilização: preço oscilando 1,5–1,9 nos períodos 201–300 não estabiliza",
      not duopoly_summary(mk, wild, wild, 1.0)["stabilized"])

ctrl = np.full((16, 2), 1.8)
dev = ctrl.copy()
dev[1, 0] = mk.best_response_scaled(1.8, 1.0)
dev[2, 1] = 1.6
dev[3, :] = [1.65, 1.65]
m = deviation_metrics(mk, ctrl, dev, 1.0)
check("desvio: R, V e classe A num caso calculado à mão",
      abs(m["R"] - (1.6 - 1.8) / 1.8) < 1e-12 and m["V"] == 4 and m["class_theta_0.02"] == "A"
      and m["R_placebo"] == 0 and not m["placebo_theta_0.02"], m)
dev2 = ctrl.copy()
dev2[1, 0] = dev[1, 0]
dev2[2:, 1] = 1.5
m2 = deviation_metrics(mk, ctrl, dev2, 1.0)
check("desvio: sem volta em 15 períodos → V censurado e classe B", m2["V"] is None and m2["class_theta_0.02"] == "B")
ctrl3 = ctrl.copy()
ctrl3[2, 1] = 1.7
m3 = deviation_metrics(mk, ctrl3, ctrl3.copy(), 1.0)
check("placebo: queda do rival no controle sem desvio é contada", m3["placebo_theta_0.02"] and m3["R"] == 0)

d = TMP / "devrun"
st = run_game(ctx_for("devrun", FakeClient(follower)), "duopoly", ["P2", "P2"], 300)
dv = run_deviation(ctx_for("devrun", FakeClient(follower)), d)
cut_calls = read_calls(d / "deviation" / "cut")
ctl_calls = read_calls(d / "deviation" / "control")
check("desvio na execução: 30 chamadas no controle, 29 no corte (firma 1 forçada em τ = 1)",
      dv["status"] == "done" and len(ctl_calls) == 30 and len(cut_calls) == 29
      and all(not (c["period"] == 301 and c["firm"] == 1) for c in cut_calls))
check("desvio na execução: preço forçado = melhor resposta ao preço do rival em τ = 0",
      abs(dv["cut"][1][0] - mk.best_response_scaled(dv["tau0_prices"][1], 1.0)) < 1e-12)
check("desvio: o rival que copia o preço reage ao corte (R < 0)", dv["R"] < 0)

# ---------------------------------------------------------------- log e segurança
rec = read_calls(TMP / "full")[0]
need = {"system_prompt", "user_prompt", "response", "price", "prompt_tokens", "output_tokens", "thoughts_tokens",
        "model", "model_version", "time_utc", "temperature", "seed", "thinking_level", "service_tier", "cost_usd"}
check("log de cada chamada com todos os campos do Protocolo", need <= set(rec), need - set(rec))
cfg = json.loads((TMP / "full" / "config.json").read_text())
check("config guarda o hash de cada texto de prompt", set(cfg["prompt_hashes"]) >= {"P1", "template", "format"})
check("prompts de teste não são os de Fish (marcados)", all("NOT the Fish" in ps.prefixes[p] for p in ("P0", "P1", "P2")))
gi = (HERE.parent / ".gitignore").read_text()
check(".env e llm/prompts/local/ fora do Git", ".env" in gi and "llm/prompts/local/" in gi)
import os  # noqa: E402
os.environ["GEMINI_API_KEY"] = "CHAVE-SECRETA-DE-TESTE"
from client import load_api_key  # noqa: E402
key_ok = load_api_key() == "CHAVE-SECRETA-DE-TESTE"
leak = any("CHAVE-SECRETA" in p.read_bytes().decode("latin-1") for p in TMP.rglob("*") if p.is_file()) or \
    any("CHAVE-SECRETA" in gzip.open(p, "rt").read() for p in TMP.rglob("*.gz"))
check("chave lida do ambiente e nunca gravada nos arquivos", key_ok and not leak)

# ---------------------------------------------------------------- cliente Gemini × SDK (sem rede)
try:
    from google.genai import errors as gerr, types as gt
    HAVE_SDK = True
except ImportError:
    HAVE_SDK = False
if HAVE_SDK:
    from client import GeminiClient
    gc = GeminiClient("gemini-3.5-flash-lite", thinking_level="low", service_tier="flex", max_output_tokens=1000)
    seen = {}

    def fake_generate(model, contents, config):
        seen.update(model=model, contents=contents, config=config)
        return gt.GenerateContentResponse(
            candidates=[gt.Candidate(content=gt.Content(role="model", parts=[gt.Part(text=good)]), finish_reason="STOP")],
            usage_metadata=gt.GenerateContentResponseUsageMetadata(prompt_token_count=1234, candidates_token_count=56,
                                                                   thoughts_token_count=78),
            model_version="gemini-3.5-flash-lite-test", response_id="r-1")
    gc._client.models.generate_content = fake_generate
    res = gc.generate("SISTEMA", "USUARIO", 42)
    cfgc = seen["config"]
    check("Gemini: chamada com system prompt, temperatura 1, seed, raciocínio, Flex e limite de saída",
          seen["model"] == "gemini-3.5-flash-lite" and seen["contents"] == "USUARIO"
          and cfgc.system_instruction == "SISTEMA" and cfgc.temperature == 1.0 and cfgc.seed == 42
          and str(cfgc.thinking_config.thinking_level).endswith("LOW") and str(cfgc.service_tier).endswith("FLEX")
          and cfgc.max_output_tokens == 1000)
    check("Gemini: texto, versão devolvida e tokens (entrada, saída, raciocínio) lidos da resposta",
          res.text == good and res.model_version == "gemini-3.5-flash-lite-test" and res.response_id == "r-1"
          and (res.prompt_tokens, res.output_tokens, res.thoughts_tokens) == (1234, 56, 78))

    def raise429(model, contents, config):
        raise gerr.ClientError(429, {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}})
    gc._client.models.generate_content = raise429
    try:
        gc.generate("s", "u", 1)
        got = None
    except RateLimited:
        got = "RateLimited"
    except Exception as e:  # noqa: BLE001
        got = type(e).__name__
    check("Gemini: erro 429 vira 'limite de uso' (espera e tenta de novo)", got == "RateLimited", got)
else:
    check("SDK google-genai instalado (pip install -r requirements-llm.txt)", False)

shutil.rmtree(TMP, ignore_errors=True)
print("Conferências:")
for n, ok in checks:
    print(f"  {'True ' if ok else 'False'}  {n}")
n_ok = sum(ok for _, ok in checks)
print(f"\n{n_ok} de {len(checks)} conferências True")
sys.exit(0 if n_ok == len(checks) else 1)
