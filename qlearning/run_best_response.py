"""
Fase 1.5, parte 1: teste de melhor resposta nas 1.000 sessões do caso base (sementes 1 a 1.000).
Refaz as sessões (mesmo código e sementes da Fase 1.3), confere que o Δ de cada sessão é igual
ao de results/qlearning_base_sessions.csv e salva em results/:
  - qlearning_bestresponse_sessions.csv: resultado do teste por sessão
  - qlearning_bestresponse_meta.json: método, resumo e comparação com Calvano (Tabela 1)
Duas versões (ver best_response.py): "calvano" (desvio de um período; é a que se compara
com a Tabela 1) e "otima" (qualquer desvio; exploratória).
Uso: python qlearning/run_best_response.py          (1.000 sessões)
     python qlearning/run_best_response.py 20       (teste rápido, não salva)
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
from best_response import TOL_BR, TOL_VI, run_many_best_response

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
SAVE = N == 1000
OUT = HERE.parent / "results"
VERSIONS = ("calvano", "otima")
TYPES = ("constante_simetrico", "constante_assimetrico", "ciclo_2", "ciclo_3_ou_mais")

# Alvos: Calvano et al. (2020), Tabela 1, experimento representativo (Pesquisa 1)
TARGET = {
    "nash_share": {"todas": 0.505, "constante_simetrico": 0.686, "constante_assimetrico": 0.661,
                   "constante_total": 0.672, "ciclo_2": 0.294, "ciclo_3_ou_mais": 0.025},
    "qloss_path": 0.002, "qloss_all": 0.018,
    "type_share": {"constante_simetrico": 0.277, "constante_assimetrico": 0.366,
                   "ciclo_2": 0.238, "ciclo_3_ou_mais": 0.119},
}

b = setup()
P = PARAMS
args = (b["PI"], b["Q0"], P["alpha"], P["delta"], P["beta"], P["t_stable"], P["t_max"],
        b["piN"], b["piM"])
seeds = np.arange(1, N + 1, dtype=np.int64)

print(f"Núcleos usados pelo numba: {numba.get_num_threads()}")
t0 = time.perf_counter()
run_many_best_response(np.array([1], dtype=np.int64), *args)
print(f"Compilação (1 sessão): {time.perf_counter() - t0:.1f} s")

t0 = time.perf_counter()
conv, dlt, clen, sym, brp, bra, qlp, qla = run_many_best_response(seeds, *args)
secs = time.perf_counter() - t0
print(f"{N} sessões + teste de melhor resposta em {secs:.1f} s")

# Conferência com a Fase 1.3
base_csv = OUT / "qlearning_base_sessions.csv"
same_delta = None
if base_csv.exists():
    with open(base_csv, encoding="utf-8") as f:
        base = {int(r["seed"]): float(r["delta"]) for r in csv.DictReader(f)}
    same_delta = all(abs(base[int(s)] - d) < 1e-12 for s, d in zip(seeds, dlt) if int(s) in base)
    print(f"Δ igual ao da Fase 1.3 em todas as sessões: {same_delta}")


def kind(k):
    if clen[k] == 1:
        return "constante_simetrico" if sym[k] else "constante_assimetrico"
    return "ciclo_2" if clen[k] == 2 else "ciclo_3_ou_mais"


def wilson(x, n, z=1.96):
    p = x / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def mean_or_none(x):
    return float(x.mean()) if len(x) else None


ok = conv
n_ok = int(ok.sum())
kinds = np.array([kind(k) if ok[k] else "" for k in range(N)])
nash = brp[:, :, 0] & brp[:, :, 1]          # [sessão, versão]
one = brp[:, :, 0] | brp[:, :, 1]
spe = bra[:, :, 0] & bra[:, :, 1]

rows = []
for k in range(N):
    row = {"seed": int(seeds[k]), "converged": bool(conv[k]), "delta": float(dlt[k]),
           "type": kinds[k], "cycle_len": int(clen[k])}
    for v, name in enumerate(VERSIONS):
        row[f"nash_on_path_{name}"] = bool(nash[k, v])
        row[f"br_path_firm1_{name}"] = bool(brp[k, v, 0])
        row[f"br_path_firm2_{name}"] = bool(brp[k, v, 1])
        row[f"spe_{name}"] = bool(spe[k, v])
        row[f"qloss_path_{name}"] = float(qlp[k, v])
        row[f"qloss_all_{name}"] = float(qla[k, v])
    rows.append(row)

summary = {"n_sessions": N, "n_tested": n_ok, "seconds": secs,
           "same_delta_as_phase_1_3": same_delta, "by_version": {}}
for v, name in enumerate(VERSIONS):
    nv = nash[ok, v]
    d_ok = dlt[ok]
    by_type = {}
    for t in TYPES:
        sel = kinds[ok] == t
        by_type[t] = {"n": int(sel.sum()), "share_of_sessions": float(sel.mean()),
                      "nash_share": mean_or_none(nv[sel]),
                      "qloss_path_mean": mean_or_none(qlp[ok, v][sel])}
    sel_c = clen[ok] == 1
    by_type["constante_total"] = {"n": int(sel_c.sum()), "nash_share": mean_or_none(nv[sel_c])}
    n_nash = int(nv.sum())
    summary["by_version"][name] = {
        "nash_share": n_nash / n_ok, "nash_share_wilson95": wilson(n_nash, n_ok),
        "at_least_one_br_share": float(one[ok, v].mean()),
        "spe_share": float(spe[ok, v].mean()),
        "qloss_path_mean": float(qlp[ok, v].mean()), "qloss_path_sd": float(qlp[ok, v].std(ddof=1)),
        "qloss_all_mean": float(qla[ok, v].mean()), "qloss_all_sd": float(qla[ok, v].std(ddof=1)),
        "delta_mean_nash_sessions": mean_or_none(d_ok[nv]),
        "delta_sd_nash_sessions": float(d_ok[nv].std(ddof=1)) if n_nash > 1 else None,
        "corr_delta_nash": (float(np.corrcoef(d_ok, nv.astype(float))[0, 1])
                            if 0 < n_nash < n_ok else None),
        "by_type": by_type,
    }


def f3(x):
    return "—" if x is None else f"{x:.3f}"


print(f"\nResumo (sessões testadas: {n_ok})")
for name in VERSIONS:
    s = summary["by_version"][name]
    lo, hi = s["nash_share_wilson95"]
    tag = "comparar com a Tabela 1" if name == "calvano" else "exploratória"
    print(f"\n[{name}] ({tag})")
    print(f"  fração em Nash no caminho: {s['nash_share']:.3f} [IC 95% {lo:.3f}; {hi:.3f}]"
          f"   alvo {TARGET['nash_share']['todas']}")
    print(f"  pelo menos 1 agente em melhor resposta: {s['at_least_one_br_share']:.3f}")
    print(f"  equilíbrio perfeito em subjogos: {s['spe_share']:.3f}   (Calvano: raro)")
    print(f"  Q-loss no caminho: {s['qloss_path_mean']:.4f} (DP {s['qloss_path_sd']:.4f})"
          f"   alvo {TARGET['qloss_path']} (DP 0,004)")
    print(f"  Q-loss em todos os estados: {s['qloss_all_mean']:.4f} (DP {s['qloss_all_sd']:.4f})"
          f"   alvo {TARGET['qloss_all']} (DP 0,006)")
    print(f"  Δ médio das sessões em Nash: {f3(s['delta_mean_nash_sessions'])}"
          f"   alvo 0,854 (DP 0,108)")
    print(f"  correlação Δ × Nash: {f3(s['corr_delta_nash'])}   (Calvano, na grade α × β: 0,12)")
    print("  por tipo (fração das sessões; fração em Nash) | alvo:")
    for t in TYPES:
        v = s["by_type"][t]
        print(f"    {t:22s} {v['share_of_sessions']:.3f}; {f3(v['nash_share'])}"
              f"   | {TARGET['type_share'][t]}; {TARGET['nash_share'][t]}")
    print(f"    {'constante_total':22s}        {f3(s['by_type']['constante_total']['nash_share'])}"
          f"   |        {TARGET['nash_share']['constante_total']}")

if SAVE:
    OUT.mkdir(exist_ok=True)
    with open(OUT / "qlearning_bestresponse_sessions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    meta = {
        "description": "Teste de melhor resposta, Q-learning caso base, sementes 1 a 1000",
        "date": datetime.now().isoformat(timespec="seconds"),
        "method": {
            "rival": "estratégia final gulosa do rival, fixa (Q congelada, ε = 0)",
            "calvano": "Q(s,a) = π(a,b) + δ Q(s', σ(s')): continuação pela própria estratégia "
                       "final (sistema linear da eq. 3); testa desvios de um período; "
                       "versão comparada com a Tabela 1",
            "otima": "Q*(s,a) = π(a,b) + δ max Q*(s',·): qualquer desvio; exploratória",
            "solver": f"iteração até diferença < {TOL_VI}",
            "nash_on_path": "os dois agentes jogam a melhor ação em todos os estados do ciclo-limite",
            "spe": "os dois agentes jogam a melhor ação nos 225 estados",
            "qloss": "[max Q(s,·) − Q(s,σ(s))] / max Q(s,·), média sobre estados e agentes "
                     "(leitura adotada pelo projeto; o artigo diz só que a perda é em %)",
            "tie_tolerance_relative": TOL_BR,
        },
        "targets_calvano_table_1": TARGET,
        "summary": summary,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "numba": numba.__version__, "scipy": scipy.__version__,
                     "platform": platform.platform()},
        "numba_threads": numba.get_num_threads(),
    }
    with open(OUT / "qlearning_bestresponse_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\nSalvo em {OUT}")
