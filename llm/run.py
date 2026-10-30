"""
Fase 3 — Linha de comando dos experimentos com LLM.

Na raiz do repositório, com o venv ativo (passo a passo em llm/README.md):

  python llm/run.py competence --model gemini-3.5-flash-lite --budget 15
      teste de competência (monopolista, P0, 300 períodos, 1 execução por α) → results/llm_pilot/
  python llm/run.py pilot --prompt P1 --alpha 1 --periods 30 --budget 20
      duopólio curto para medir limites, tokens, latência e respostas inválidas → results/llm_pilot/
  python llm/run.py plan --runs-per-prompt 21
      grava results/llm_plan.csv (P1 e P2 alternados, 7 por α); só depois do piloto
  python llm/run.py run --budget 40
      roda o plano: cada execução (300 períodos) e o teste de desvio dela → results/llm/
  python llm/run.py status        situação das execuções e gasto acumulado
  python llm/run.py summarize     results/llm_runs.csv (uma linha por execução)

Tudo pode ser interrompido (Ctrl+C) e retomado: basta rodar o mesmo comando de novo.
--fake usa um cliente falso (sem API) com os prompts de teste: serve só para ver o
programa funcionando; grava em --out (padrão: results/llm_fake/, fora do Git).

Código próprio do projeto.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from client import Budget, FakeClient, GeminiClient  # noqa: E402
from experiment import ConfigMismatch, Context, read_calls, run_deviation, run_game  # noqa: E402
from market import ALPHAS, PERIODS, Market  # noqa: E402
from prompts import PromptFilesMissing, load_prompts  # noqa: E402

ROOT = HERE.parent
RES = ROOT / "results"
MODELS = ("gemini-3.5-flash-lite", "gemini-3.8-flash")


def alpha_tag(a: float) -> str:
    return ("a" + f"{a:g}").replace(".", "_")


def fake_policy(system, user, seed, call_no):
    """Agente falso para --fake: começa no dobro do custo e sobe 2% por período até o teto
    do prompt de teste. Não imita um LLM; só exercita o programa."""
    cost = float(re.search(r"Cost per unit: \$?(\d+\.\d+)", user).group(1))
    m = re.search(r"Round \d+: my price (\d+\.\d+)", user)
    p = float(m.group(1)) * 1.02 if m else 1.5 * cost
    ceiling = float(re.search(r"pay more than \$?(\d+\.\d+)", user).group(1))
    return (f"OBSERVATIONS: fake.\nPLANS: keep raising.\nINSIGHTS: none.\n"
            f"PRICE: {min(p, 0.95 * ceiling):.2f}")


def make_context(args, run_dir: Path, alpha: float, seed: int) -> Context:
    if args.fake:
        client, ps = FakeClient(fake_policy, model=args.model), load_prompts("test")
    else:
        ps = load_prompts("local")
        if ps.source != "local":
            raise SystemExit("Os prompts de teste não podem ser usados com o Gemini de verdade.")
        client = GeminiClient(args.model, temperature=1.0, thinking_level=args.thinking,
                              service_tier="flex" if args.flex else None,
                              max_output_tokens=args.max_output)
    budget = Budget(args.out / "llm_spend.json" if args.fake else RES / "llm_spend.json", args.budget)
    return Context(client=client, prompts=ps, market=Market(), budget=budget, run_dir=run_dir,
                   alpha=alpha, seed=seed)


def report(name, st):
    s = st.get("summary") or {}
    extra = ""
    if "delta" in s:
        extra = f" Δ = {s['delta']:.3f}, estabilizada = {s['stabilized']}"
    if "passed" in s:
        extra = f" P10 {s['p10']:.3f}, P90 {s['p90']:.3f} (alvo {s['target']:.3f}) → {'passou' if s['passed'] else 'não passou'}"
    print(f"{name}: {st['status']}{(' — ' + st['reason']) if st.get('reason') else ''}{extra}")


def cmd_competence(args):
    base = args.out if args.fake else RES / "llm_pilot"
    passed = 0
    for k, a in enumerate(ALPHAS, start=1):
        name = f"competence_{args.model}_{alpha_tag(a)}"
        ctx = make_context(args, base / name, a, seed=k)
        st = run_game(ctx, "competence", ["P0"], PERIODS)
        report(name, st)
        if st["status"] == "paused":
            print("Parado (teto, limite de uso ou erro). Rode o mesmo comando para continuar.")
            return
        passed += bool(st["status"] == "done" and (st.get("summary") or {}).get("passed"))
    # Regra do Protocolo: passa com 3 de 3 (saída válida e convergência ao monopólio de 1 firma);
    # 1 ou 2 de 3 é resultado parcial, registrado e decidido com a Beatriz antes do duopólio.
    print(f"Teste de competência de {args.model}: {passed} de {len(ALPHAS)} "
          f"→ {'PASSOU' if passed == len(ALPHAS) else ('PARCIAL: decidir com registro' if passed else 'NÃO PASSOU')}")


def cmd_pilot(args):
    base = args.out if args.fake else RES / "llm_pilot"
    a = float(args.alpha)
    name = f"duopoly_{args.prompt}_{alpha_tag(a)}_{args.periods}p_{args.model}"
    ctx = make_context(args, base / name, a, seed=args.seed)
    st = run_game(ctx, "duopoly", [args.prompt, args.prompt], args.periods)
    report(name, st)
    calls = read_calls(base / name)
    if calls:
        n = len(calls)
        bad = sum(not c["valid"] for c in calls)
        tok_in = sum(c["prompt_tokens"] for c in calls) / n
        tok_out = sum(c["output_tokens"] + c["thoughts_tokens"] for c in calls) / n
        lat = sum(c["latency_s"] for c in calls) / n
        print(f"  {n} chamadas; inválidas {bad} ({100 * bad / n:.1f}%); tokens por chamada: "
              f"{tok_in:.0f} de entrada, {tok_out:.0f} de saída (com raciocínio); latência média {lat:.1f} s")


def cmd_plan(args):
    path = (args.out if args.fake else RES) / "llm_plan.csv"
    if path.exists() and not args.force:
        raise SystemExit(f"{path} já existe (protocolo fixado). Use --force só com registro em Decisões.")
    n = args.runs_per_prompt
    if n % len(ALPHAS):
        raise SystemExit(f"--runs-per-prompt deve ser múltiplo de {len(ALPHAS)} (mesmo número por α)")
    rng = random.Random(args.seed)               # ordem dos α sorteada com semente registrada
    order = {p: [a for a in ALPHAS for _ in range(n // len(ALPHAS))] for p in ("P1", "P2")}
    for p in order:
        rng.shuffle(order[p])
    rows = []
    for i in range(n):                            # alterna P1 e P2 (Protocolo: "Ordem")
        for p in ("P1", "P2"):
            k = len(rows) + 1
            a = order[p][i]
            # as opções da chamada ficam no plano: todas as execuções usam as mesmas
            rows.append({"order": k, "run_id": f"{p}_{alpha_tag(a)}_s{k:03d}", "prompt": p,
                         "alpha": a, "seed": k, "model": args.model, "thinking": args.thinking or "",
                         "flex": int(bool(args.flex)), "max_output": args.max_output or ""})
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} execuções gravadas em {path} (semente da ordem: {args.seed})")


def read_plan(args):
    path = (args.out if args.fake else RES) / "llm_plan.csv"
    if not path.is_file():
        raise SystemExit("Sem plano: rode primeiro 'python llm/run.py plan --runs-per-prompt N'")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def cmd_run(args):
    base = args.out if args.fake else RES / "llm"
    for r in read_plan(args):
        if args.only and r["run_id"] != args.only:
            continue
        # opções da chamada vêm do plano (fixado antes), não da linha de comando
        args.model = r["model"]
        args.thinking = r.get("thinking") or None
        args.flex = bool(int(r.get("flex") or 0))
        args.max_output = int(r["max_output"]) if r.get("max_output") else None
        a, run_dir = float(r["alpha"]), base / r["run_id"]
        ctx = make_context(args, run_dir, a, int(r["seed"]))
        st = run_game(ctx, "duopoly", [r["prompt"], r["prompt"]], PERIODS)
        report(r["run_id"], st)
        if st["status"] == "paused":
            print("Parado (teto de gasto ou limite de uso). Rode o mesmo comando para continuar.")
            return
        if st["status"] != "done":
            continue                               # parada por respostas inválidas: reportada
        dv = run_deviation(ctx, run_dir)
        if dv["status"] == "done":
            print(f"  desvio: R = {dv['R']:.3f}, G = {dv['G']:.3f}, V = {dv['V']}, "
                  f"classe {dv['class_theta_0.02']}, placebo {dv['placebo_theta_0.02']}")
        else:
            print(f"  desvio: {dv['status']} — {dv.get('reason')}")
            if dv["status"] == "paused":
                return


def run_dirs(base: Path):
    return sorted(p for p in base.iterdir() if (p / "config.json").is_file()) if base.is_dir() else []


def cmd_status(args):
    base = args.out if args.fake else RES
    spend = base / "llm_spend.json"
    if spend.is_file():
        s = json.loads(spend.read_text(encoding="utf-8"))
        print(f"Gasto acumulado (preço pago): US$ {s['spent_usd']:.4f} de {s['ceiling_usd']:.2f}")
    for sub in (["."] if args.fake else ["llm_pilot", "llm"]):
        for d in run_dirs(base / sub):
            st = json.loads((d / "state.json").read_text(encoding="utf-8")) if (d / "state.json").is_file() else {}
            print(f"{sub}/{d.name}: {st.get('status', 'sem estado')}, período {st.get('period', 0)}"
                  f"{(' — ' + st['reason']) if st.get('reason') else ''}")


def cmd_summarize(args):
    base = (args.out if args.fake else RES / "llm")
    rows = []
    for d in run_dirs(base):
        cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
        if cfg["kind"] != "duopoly":
            continue
        st = json.loads((d / "state.json").read_text(encoding="utf-8"))
        dirs = (d, d / "deviation" / "control", d / "deviation" / "cut")
        billed = [c for x in dirs for c in read_calls(x)]              # todas as chamadas pagas
        calls = [c for x in dirs for c in read_calls(x, dedupe=True)]  # sem as repetidas na retomada
        row = {"run_id": d.name, "prompt": cfg["prefix"].split("+")[0], "alpha": cfg["alpha"],
               "seed": cfg["seed"], "model": cfg["model"], "status": st["status"],
               "reason": st.get("reason") or "", "periods_done": st.get("period", 0),
               "calls": len(calls), "calls_billed": len(billed), "invalid_calls": sum(not c["valid"] for c in calls),
               "prompt_tokens": sum(c["prompt_tokens"] for c in billed),
               "output_tokens": sum(c["output_tokens"] for c in billed),
               "thoughts_tokens": sum(c["thoughts_tokens"] for c in billed),
               "cost_usd": round(sum(c["cost_usd"] for c in billed), 6),
               "model_versions": "|".join(sorted({str(c["model_version"]) for c in calls})),
               "first_call_utc": min((c["time_utc"] for c in calls), default=""),
               "last_call_utc": max((c["time_utc"] for c in calls), default="")}
        row.update({k: v for k, v in (st.get("summary") or {}).items()})
        dvp = d / "deviation" / "deviation.json"
        if dvp.is_file():
            dv = json.loads(dvp.read_text(encoding="utf-8"))
            row.update({k: dv[k] for k in dv if k.startswith(("R", "G", "V", "class_", "placebo_", "did_"))})
        rows.append(row)
    if not rows:
        print("Nenhuma execução de duopólio encontrada.")
        return
    cols = list(dict.fromkeys(k for r in rows for k in r))
    out = (args.out if args.fake else RES) / "llm_runs.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} execuções em {out}")


def cmd_preview(args):
    """Mostra o prompt montado com os textos locais (período 1 e período 3, com histórico de
    exemplo) e confere se uma resposta com os rótulos de format.json é lida. Não chama a API."""
    from prompts import FIELDS, parse_response, user_prompt
    ps = load_prompts("test" if args.fake else "local")
    mk, a = Market(), float(args.alpha)
    hist = [{"period": 1, "my_price": 1.80 * a, "rival_price": 1.75 * a, "qty": 30.12, "profit": 24.10 * a},
            {"period": 2, "my_price": 1.70 * a, "rival_price": 1.72 * a, "qty": 34.56, "profit": 24.19 * a}]
    mono = args.prompt == "P0"
    if mono:
        hist = [{**h, "rival_price": None} for h in hist]
    for title, h, plans, ins in (("PERÍODO 1 (notas vazias, sem histórico)", [], "", ""),
                                 ("PERÍODO 3 (com 2 períodos de exemplo)", hist,
                                  "[exemplo de PLANS]", "[exemplo de INSIGHTS]")):
        print("=" * 70 + f"\n{title} — prefixo {args.prompt}, α = {a:g}\n" + "=" * 70)
        print("[SYSTEM]\n" + ps.prefixes[args.prompt] + "\n\n[USER]")
        print(user_prompt(ps, cost=mk.cost(a), ceiling=mk.ceiling(a), plans=plans, insights=ins,
                          history=h, monopoly=mono))
    labels = ps.fmt["labels"]
    sample = "\n".join(f"{labels[f]} {'1.85' if f == 'price' else 'texto'}" for f in FIELDS)
    r = parse_response(ps, sample)
    print("=" * 70 + f"\nResposta de exemplo lida: preço = {r['price_value']} (rótulos: {list(labels.values())})")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Experimentos com agentes de LLM (Fase 3)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, budget=True):
        p.add_argument("--model", default=MODELS[0], choices=MODELS)
        p.add_argument("--thinking", default=None, choices=["minimal", "low", "medium", "high"],
                       help="nível de raciocínio (padrão: o do modelo; registrado no log)")
        p.add_argument("--max-output", type=int, default=None, help="limite de tokens de saída")
        p.add_argument("--flex", action="store_true", help="modo Flex (metade do preço; Pesquisa 7)")
        if budget:
            p.add_argument("--budget", type=float, required=True, help="teto de gasto em US$ (preço pago)")
        p.add_argument("--fake", action="store_true", help="cliente falso, sem API (só demonstração)")
        p.add_argument("--out", type=Path, default=RES / "llm_fake")

    p = sub.add_parser("competence"); common(p); p.set_defaults(f=cmd_competence)
    p = sub.add_parser("pilot"); common(p)
    p.add_argument("--prompt", choices=["P1", "P2"], default="P1")
    p.add_argument("--alpha", type=float, choices=ALPHAS, default=1.0)
    p.add_argument("--periods", type=int, default=30)
    p.add_argument("--seed", type=int, default=1)
    p.set_defaults(f=cmd_pilot)
    p = sub.add_parser("plan"); common(p, budget=False)
    p.add_argument("--runs-per-prompt", type=int, required=True)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--force", action="store_true")
    p.set_defaults(f=cmd_plan)
    p = sub.add_parser("run"); common(p)
    p.add_argument("--only", default=None, help="roda só esta execução do plano")
    p.set_defaults(f=cmd_run)
    p = sub.add_parser("status"); common(p, budget=False); p.set_defaults(f=cmd_status)
    p = sub.add_parser("preview"); common(p, budget=False)
    p.add_argument("--prompt", choices=["P0", "P1", "P2"], default="P1")
    p.add_argument("--alpha", type=float, choices=ALPHAS, default=1.0)
    p.set_defaults(f=cmd_preview)
    p = sub.add_parser("summarize"); common(p, budget=False); p.set_defaults(f=cmd_summarize)

    args = ap.parse_args(argv)
    # Δ, →, — e acentos sem erro no PowerShell, mesmo com a saída gravada num arquivo
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    try:
        args.f(args)
    except (PromptFilesMissing, ConfigMismatch) as e:
        raise SystemExit(str(e))


if __name__ == "__main__":
    main()
