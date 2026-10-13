"""
Fase 1.5, parte 3b: por que o controle sem memória com δ = 0,95 dá Δ alto (0,957 na parte 2).
Sem memória o agente não consegue punir, então o preço alto não vem de punição. Quatro blocos,
1.000 sessões cada condição (sementes 1 a 1.000), α = 0,15 e β = 4e-6 do caso base, k = 0:
  q0       — o Q inicial importa? Q inicial da eq. 8 (como na parte 2), zero e lucro de Nash/(1 − δ).
  delta    — o desconto importa? δ = 0; 0,25; 0,5; 0,75; 0,9; 0,95.
  mecanismo — com δ = 0,95: a nota do preço jogado é exata e a da melhor resposta está defasada?
  epsmin   — exploração que não some resolve? ε = max(e^(−βt), ε mínimo), ε mínimo 0,1%, 1% e 5%,
             4 milhões de períodos sem parada antecipada (cerca do dobro do tempo médio até convergir).
Δ = ciclo-limite da estratégia gulosa no momento em que a sessão para (igual ao Δ usual nas
sessões que convergem). Hipótese de referência: Xu e Zhao (Pesquisa 5), "o desconto infla os
valores Q". Diagnóstico exploratório, fora da família confirmatória.
Uso: python qlearning/run_memoryless_diagnosis.py BLOCO [N]
     BLOCO = q0 | delta | mecanismo | epsmin | resumo
     N = número de sessões (padrão 1000; com menos de 1000 não salva)
     "resumo" junta os blocos já salvos em results/qlearning_memoryless_diagnosis.json.
"""

import csv
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numba
import numpy as np
import scipy

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from base_case import PARAMS, setup
from controls import run_many_mem_diag
from session import initial_q

BLOCK = sys.argv[1] if len(sys.argv) > 1 else "resumo"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
SAVE = N == 1000
OUT = HERE.parent / "results"
FILE = OUT / "qlearning_memoryless_diagnosis.json"
P = PARAMS
b = setup()
PI = b["PI"]
seeds = np.arange(1, N + 1, dtype=np.int64)
DELTAS = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95)
EPS_MINS = (0.001, 0.01, 0.05)
T_FIXED = 4_000_000


def q0(delta, kind="eq8"):
    Q0 = initial_q(PI, delta)[:, :1, :].copy()
    if kind == "zero":
        Q0[:] = 0.0
    elif kind == "nash":
        Q0[:] = b["piN"] / (1 - delta)
    return Q0


def run(delta, Q0, eps_min=0.0, t_stable=P["t_stable"], t_max=P["t_max"]):
    t0 = time.perf_counter()
    r = run_many_mem_diag(seeds, PI, Q0, P["alpha"], delta, P["beta"], t_stable, t_max,
                          b["piN"], b["piM"], 0, eps_min)
    return r, time.perf_counter() - t0


def stats(r, secs):
    conv, per, dg = r[0], r[1], r[2]
    se = dg.std(ddof=1) / np.sqrt(len(dg))
    return {"delta_mean": float(dg.mean()), "delta_sd": float(dg.std(ddof=1)),
            "delta_ci95": [float(dg.mean() - 1.96 * se), float(dg.mean() + 1.96 * se)],
            "delta_median": float(np.median(dg)), "share_converged": float(conv.mean()),
            "periods_mean": float(per.mean()), "seconds": secs}


def line(name, s):
    lo, hi = s["delta_ci95"]
    return (f"  {name:34s} Δ {s['delta_mean']:.3f} [IC 95% {lo:.3f}; {hi:.3f}]  "
            f"mediana {s['delta_median']:.3f}  convergiram {s['share_converged']:.2f}  "
            f"({s['seconds']:.0f} s)")


def load():
    if FILE.exists():
        with open(FILE, encoding="utf-8") as f:
            return json.load(f)
    return {"description": "Diagnóstico do controle sem memória com δ = 0,95, sementes 1 a 1000",
            "params": {"memory": 0, "alpha": P["alpha"], "beta": P["beta"],
                       "t_stable": P["t_stable"], "t_max": P["t_max"]},
            "blocks": {}}


print(f"Núcleos usados pelo numba: {numba.get_num_threads()}; bloco: {BLOCK}; sessões: {N}")
t0 = time.perf_counter()
run_many_mem_diag(np.array([1], dtype=np.int64), PI, q0(0.95), P["alpha"], 0.95, P["beta"],
                  P["t_stable"], P["t_max"], b["piN"], b["piM"], 0, 0.0)
print(f"Compilação: {time.perf_counter() - t0:.1f} s")
result = None

if BLOCK == "q0":
    result = {}
    for kind, name in (("eq8", "Q inicial eq. 8 (parte 2)"), ("zero", "Q inicial zero"),
                       ("nash", "Q inicial = lucro de Nash/(1 − δ)")):
        r, secs = run(0.95, q0(0.95, kind))
        result[kind] = stats(r, secs)
        print(line(name, result[kind]))
        if kind == "eq8":
            f = OUT / "qlearning_controls_sessions.csv"
            if f.exists():
                with open(f, encoding="utf-8") as fh:
                    c = {int(x["seed"]): float(x["delta"]) for x in csv.DictReader(fh)
                         if x["condition"] == "sem_memoria_d095"}
                same = all(abs(c[int(s)] - d) < 1e-12 for s, d in zip(seeds, r[2]))
                result["eq8"]["same_as_part_2"] = same
                print(f"    igual ao controle 1 da parte 2: {same}")

elif BLOCK == "delta":
    result = {}
    for d in DELTAS:
        r, secs = run(d, q0(d))
        result[str(d)] = stats(r, secs)
        print(line(f"δ = {d}", result[str(d)]))

elif BLOCK == "mecanismo":
    r, secs = run(0.95, q0(0.95))
    conv, per, dg, clen, p1, p2, qp, qf, qb, vb = r
    ok = ~np.isnan(qp[:, 0])
    qp, qf, qb, vb = qp[ok], qf[ok], qb[ok], vb[ok]
    eps_end = np.exp(-P["beta"] * per[ok].astype(float))
    m = PI.shape[0]
    stale = (qb < qp) & (vb > qp)          # cortar daria mais, mas a nota guardada diz que não
    result = {
        "n_sessions_constant_price": int(ok.sum()),
        "share_firms_played_note_within_0.1pct": float((np.abs(qp / qf - 1) < 1e-3).mean()),
        "share_firms_br_true_value_above_played": float((vb > qp).mean()),
        "share_firms_br_note_below_played": float((qb < qp).mean()),
        "share_firms_stale_pattern": float(stale.mean()),
        "share_sessions_stale_both_firms": float(stale.all(axis=1).mean()),
        "br_note_gap_pct_mean": float((100 * (vb - qb) / vb).mean()),
        "true_gain_pct_mean": float((100 * (vb - qp) / qp).mean()),
        "eps_end_mean": float(eps_end.mean()),
        "periods_between_tests_of_one_price_mean": float((m / eps_end).mean()),
        "seconds": secs,
    }
    s = result
    print(f"  sessões com preço constante: {s['n_sessions_constant_price']} de {N}")
    print(f"  nota do preço jogado a menos de 0,1% de π/(1 − δ): {s['share_firms_played_note_within_0.1pct']:.3f} das firmas")
    print(f"  cortar 1 vez para a melhor resposta daria mais: {s['share_firms_br_true_value_above_played']:.3f} "
          f"(ganho médio verdadeiro {s['true_gain_pct_mean']:.2f}%)")
    print(f"  mas a nota guardada da melhor resposta é menor que a do preço jogado: "
          f"{s['share_firms_br_note_below_played']:.3f}")
    print(f"  padrão completo (cortar daria mais, nota diz que não): {s['share_firms_stale_pattern']:.3f} "
          f"das firmas; nas duas firmas da sessão: {s['share_sessions_stale_both_firms']:.3f}")
    print(f"  nota da melhor resposta abaixo do valor verdadeiro: {s['br_note_gap_pct_mean']:.2f}% em média")
    print(f"  ε no fim: {s['eps_end_mean']:.5f}; cada preço é testado 1 vez a cada "
          f"{s['periods_between_tests_of_one_price_mean']:,.0f} períodos, em média")

elif BLOCK == "epsmin":
    result = {"t_max": T_FIXED}
    for e in EPS_MINS:
        r, secs = run(0.95, q0(0.95), eps_min=e, t_stable=T_FIXED + 1, t_max=T_FIXED)
        result[str(e)] = stats(r, secs)
        print(line(f"ε mínimo {e}", result[str(e)]))
    print("  (aqui nenhuma sessão para antes: todas jogam {:,} períodos)".format(T_FIXED))

elif BLOCK == "resumo":
    data = load()
    print(json.dumps(data["blocks"], ensure_ascii=False, indent=1)[:6000])

else:
    sys.exit(f"bloco desconhecido: {BLOCK}")

if SAVE and result is not None:
    OUT.mkdir(exist_ok=True)
    data = load()
    data["blocks"][BLOCK] = {"date": datetime.now().isoformat(timespec="seconds"), **result}
    data["versions"] = {"python": platform.python_version(), "numpy": np.__version__,
                        "numba": numba.__version__, "scipy": scipy.__version__,
                        "platform": platform.platform()}
    data["numba_threads"] = numba.get_num_threads()
    with open(FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\nBloco '{BLOCK}' salvo em {FILE}")
