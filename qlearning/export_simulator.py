"""
Fase 1.7: exporta em JSON, para o simulador web (pasta docs/, Fase 2), os parâmetros,
os resultados de referência do Q-learning e 3 sessões completas.

Grava em docs/data/:
  - qlearning_params.json      parâmetros da Ficha (com a fonte), pontos de referência,
                               grade, tabela de lucros e Q inicial
  - qlearning_reference.json   resultados de referência das 1.000 sessões, lidos de results/
                               (Δ, teste de desvio, melhor resposta, trajetória média,
                               tabela dos testes obrigatórios) e alvos de Calvano
  - qlearning_sessions.json    3 sessões completas: aprendizado em blocos, ciclo-limite,
                               estratégia final e teste de desvio a partir de cada ponto do ciclo

Regra de escolha das sessões (fixada antes, a mesma da Fase 1.6, sem escolher a dedo):
as sessões de Δ mais próximo dos percentis 25, 50 e 75 das 1.000 sessões da Fase 1.3
(empate: menor semente). A página abre com a do percentil 50 (a mediana).

As 3 sessões são refeitas com as mesmas sementes e o mesmo código (trajectories.py, igual
a session.py); o script confere que períodos, Δ, ciclo, teste de desvio e trajetória em
blocos logarítmicos são iguais aos já salvos em results/. Os demais números são só lidos
de results/: nada novo é calculado sobre as 1.000 sessões.
Números com ponto flutuante são arredondados a 6 casas decimais no JSON.
Código próprio do projeto.
Uso: python qlearning/export_simulator.py
"""

import csv
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import numba
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from base_case import PARAMS, setup
from deviation import T_DEV, classify, deviation_paths, metrics_one_start
from session import limit_cycle, run_session
from trajectories import log_block_edges, run_session_traced

ROOT = HERE.parent
RES = ROOT / "results"
OUT = ROOT / "docs" / "data"
DEC = 6                       # casas decimais no JSON
LIN_BLOCK = 10_000            # bloco linear do modo reprodução (igual ao do Worker, Fase 2)
LIN_T_END = 4_000_000         # acima do máximo de períodos das 1.000 sessões (3,18 milhões)
LOG_T_END, LOG_N = 10_000_000, 241   # mesmos blocos logarítmicos da Fase 1.6
THETAS = (0.02, 0.05)


def r(x):
    """Arredonda floats (também dentro de listas e dicionários)."""
    if isinstance(x, float):
        return round(x, DEC)
    if isinstance(x, (np.floating,)):
        return round(float(x), DEC)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, list):
        return [r(v) for v in x]
    if isinstance(x, tuple):
        return [r(v) for v in x]
    if isinstance(x, dict):
        return {k: r(v) for k, v in x.items()}
    return x


def read_csv(name):
    with open(RES / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(name):
    with open(RES / name, encoding="utf-8") as f:
        return json.load(f)


def num(s):
    """Texto do CSV para número (vazio vira None)."""
    if s in ("", None):
        return None
    if s in ("True", "False"):
        return s == "True"
    try:
        v = float(s)
    except ValueError:
        return s
    return int(v) if v.is_integer() and "." not in s and "e" not in s.lower() else v


# ---------------------------------------------------------------- dados já salvos
b = setup()
P = PARAMS
prices = b["prices"]
PI = b["PI"]
piN, piM = b["piN"], b["piM"]
m = len(prices)
step = float(prices[1] - prices[0])

base_meta = read_json("qlearning_base_meta.json")
dev_meta = read_json("qlearning_deviation_meta.json")
br_meta = read_json("qlearning_bestresponse_meta.json")
traj_meta = read_json("qlearning_trajectories_meta.json")
base_rows = {int(x["seed"]): x for x in read_csv("qlearning_base_sessions.csv")}
dev_rows = {int(x["seed"]): x for x in read_csv("qlearning_deviation_sessions.csv")}
br_rows = {int(x["seed"]): x for x in read_csv("qlearning_bestresponse_sessions.csv")}
blocks_all = read_csv("qlearning_trajectory_blocks.csv")
ex_rows = read_csv("qlearning_trajectory_examples.csv")
dev_paths = read_csv("qlearning_deviation_paths.csv")
tests_table = read_csv("qlearning_mandatory_tests_table.csv")

checks = {}

# ---------------------------------------------------------------- regra de escolha
seeds_all = np.array(sorted(base_rows))
d_all = np.array([float(base_rows[s]["delta"]) for s in seeds_all])
chosen = {}
for role, q in (("p25", 25), ("p50", 50), ("p75", 75)):
    target = float(np.percentile(d_all, q))
    dist = np.abs(d_all - target)
    chosen[role] = {"seed": int(seeds_all[np.flatnonzero(dist == dist.min())[0]]),
                    "delta_target": target}
checks["sementes_iguais_fase16"] = all(chosen[k]["seed"] == traj_meta["examples"][k]["seed"]
                                       for k in chosen)

# ---------------------------------------------------------------- 3 sessões completas
lin_edges = np.arange(0, LIN_T_END + LIN_BLOCK, LIN_BLOCK, dtype=np.int64)
log_edges = log_block_edges(LOG_T_END, LOG_N)
targs = (PI, b["Q0"], P["alpha"], P["delta"], P["beta"], P["t_stable"], P["t_max"], piN, piM,
         prices)


def blocks(edges, sp1, sp2, spi, cnt):
    """Colunas por bloco (só blocos com períodos): início, fim, preços médios, Δ e ε."""
    keep = np.flatnonzero(cnt > 0)
    t0 = edges[keep]
    n = cnt[keep]
    return {
        "t_start": [int(x) for x in t0],
        "n_periods": [int(x) for x in n],
        "price_firm1": [float(x) for x in sp1[keep] / n],
        "price_firm2": [float(x) for x in sp2[keep] / n],
        "delta": [float(x) for x in ((spi[keep] / n) - piN) / (piM - piN)],
        "epsilon_start": [float(np.exp(-P["beta"] * x)) for x in t0],
    }


sessions = []
for role in ("p50", "p25", "p75"):
    seed = chosen[role]["seed"]
    # sessão original (Fase 1.3) e as duas versões com registro em blocos
    ok0, t0_, d0, nc0, pidx0, Q0f, greedy0, s0 = run_session(seed, *targs[:-1])
    okL, tL, dL, ncL, greedyL, sL, sp1L, sp2L, spiL, cntL = run_session_traced(
        seed, *targs, lin_edges)
    okG, tG, dG, ncG, greedyG, sG, sp1G, sp2G, spiG, cntG = run_session_traced(
        seed, *targs, log_edges)
    br = base_rows[seed]
    checks[f"s{seed}_igual_fase13"] = (
        ok0 and okL and okG and int(br["periods"]) == t0_ == tL == tG
        and abs(float(br["delta"]) - d0) < 1e-12 and d0 == dL == dG
        and int(br["cycle_len"]) == nc0 == ncL == ncG
        and np.array_equal(greedy0, greedyL) and np.array_equal(greedy0, greedyG)
        and s0 == sL == sG)
    checks[f"s{seed}_blocos_somam_periodos"] = (int(cntL.sum()) == t0_ == int(cntG.sum()))

    # trajetória em blocos logarítmicos = a da Fase 1.6
    ex = [x for x in ex_rows if int(x["seed"]) == seed]
    lg = blocks(log_edges, sp1G, sp2G, spiG, cntG)
    checks[f"s{seed}_blocos_log_iguais_fase16"] = (
        len(ex) == len(lg["t_start"])
        and all(int(e["t_start"]) == a and int(e["n_periods"]) == n
                and abs(float(e["price_firm1"]) - p1) < 1e-12
                and abs(float(e["price_firm2"]) - p2) < 1e-12
                and abs(float(e["delta_block"]) - d) < 1e-12
                for e, a, n, p1, p2, d in zip(ex, lg["t_start"], lg["n_periods"],
                                              lg["price_firm1"], lg["price_firm2"], lg["delta"])))

    # ciclo-limite a partir do estado em que a sessão parou
    cyc = limit_cycle(greedy0, s0, m)
    L = len(cyc)
    in_cycle = np.zeros(m * m, dtype=np.bool_)
    in_cycle[cyc] = True
    cycle_idx = [[int(s // m), int(s % m)] for s in cyc]

    # teste de desvio a partir de cada ponto do ciclo (protocolo da Fase 1.4)
    starts = []
    acc = {"R": 0.0, "G": 0.0, "V": 0.0, "Vp": 0.0}
    cens = cens_p = 0
    for j, s_start in enumerate(cyc):
        ctrl, dev = deviation_paths(greedy0, PI, s_start)
        R, G, V, Vp, didv, Rc, curve = metrics_one_start(greedy0, PI, prices, P["delta"], step,
                                                         s_start, in_cycle, L > 1)
        w = P["delta"] ** np.arange(T_DEV)
        pi1c = np.array([PI[ctrl[t, 0], ctrl[t, 1], 0] for t in range(T_DEV + 1)])
        pi1d = np.array([PI[dev[t, 0], dev[t, 1], 0] for t in range(T_DEV + 1)])
        pi2c = np.array([PI[ctrl[t, 0], ctrl[t, 1], 1] for t in range(T_DEV + 1)])
        pi2d = np.array([PI[dev[t, 0], dev[t, 1], 1] for t in range(T_DEV + 1)])
        disc_c = float((w * pi1c[1:]).sum())
        disc_d = float((w * pi1d[1:]).sum())
        starts.append({
            "cycle_position": j,
            "tau": list(range(T_DEV + 1)),
            "control": {"idx_firm1": [int(x) for x in ctrl[:, 0]],
                        "idx_firm2": [int(x) for x in ctrl[:, 1]],
                        "price_firm1": [float(prices[x]) for x in ctrl[:, 0]],
                        "price_firm2": [float(prices[x]) for x in ctrl[:, 1]],
                        "profit_firm1": [float(x) for x in pi1c],
                        "profit_firm2": [float(x) for x in pi2c]},
            "deviation": {"idx_firm1": [int(x) for x in dev[:, 0]],
                          "idx_firm2": [int(x) for x in dev[:, 1]],
                          "price_firm1": [float(prices[x]) for x in dev[:, 0]],
                          "price_firm2": [float(prices[x]) for x in dev[:, 1]],
                          "profit_firm1": [float(x) for x in pi1d],
                          "profit_firm2": [float(x) for x in pi2d]},
            "did_deviate": bool(didv),
            "R": float(R), "G": float(G),
            "V": int(V) if V >= 0 else None,
            "V_phase": int(Vp) if Vp >= 0 else None,
            "discounted_profit_firm1_control": disc_c,
            "discounted_profit_firm1_deviation": disc_d,
            "rival_price_ratio_minus_1": [float(x) for x in curve],
        })
        acc["R"] += R
        acc["G"] += G
        if V < 0:
            cens += 1
        else:
            acc["V"] += V
        if Vp < 0:
            cens_p += 1
        else:
            acc["Vp"] += Vp
    R_s, G_s = acc["R"] / L, acc["G"] / L
    V_s = acc["V"] / L if cens == 0 else -1.0
    Vp_s = acc["Vp"] / L if cens_p == 0 else -1.0
    dr = dev_rows[seed]
    cls = {f"theta_{th}": classify(R_s, G_s, V_s, th) for th in THETAS}
    checks[f"s{seed}_desvio_igual_fase14"] = (
        abs(float(dr["R"]) - R_s) < 1e-12 and abs(float(dr["G"]) - G_s) < 1e-12
        and ((dr["V"] == "" and V_s < 0) or (dr["V"] != "" and abs(float(dr["V"]) - V_s) < 1e-12))
        and cls["theta_0.02"] == dr["class_theta2"] and cls["theta_0.05"] == dr["class_theta5"])

    sessions.append({
        "role": role,
        "opens_page": role == "p50",
        "seed": seed,
        "selection": f"Δ mais próximo do percentil {role[1:]} das 1.000 sessões",
        "delta_percentile_target": chosen[role]["delta_target"],
        "converged": bool(ok0),
        "periods": int(t0_),
        "delta": float(d0),
        "outcome": br["outcome"],
        "cycle_len": int(L),
        "cycle_idx": cycle_idx,
        "cycle_prices": [[float(prices[a]), float(prices[c])] for a, c in cycle_idx],
        "cycle_profits": [[float(PI[a, c, 0]), float(PI[a, c, 1])] for a, c in cycle_idx],
        "final_state_idx": [int(s0 // m), int(s0 % m)],
        "price_mean_firm1": float(br["p1_mean"]),
        "price_mean_firm2": float(br["p2_mean"]),
        "nash_on_path": num(br_rows[seed]["nash_on_path_calvano"]),
        "greedy_strategy": {
            "state_index": "s = idx_firm1 * 15 + idx_firm2 (preços do período anterior, "
                           "índices 0 a 14)",
            "firm1": [int(x) for x in greedy0[0]],
            "firm2": [int(x) for x in greedy0[1]],
        },
        "learning_blocks_linear": dict(block_size=LIN_BLOCK,
                                       **blocks(lin_edges, sp1L, sp2L, spiL, cntL)),
        "learning_blocks_log": lg,
        "deviation_test": {
            "default_start": 0,
            "starts": starts,
            "session_mean": {"R": R_s, "G": G_s,
                             "V": V_s if V_s >= 0 else None,
                             "V_phase": Vp_s if Vp_s >= 0 else None,
                             "class_theta_2pct": cls["theta_0.02"],
                             "class_theta_5pct": cls["theta_0.05"]},
        },
    })

# ---------------------------------------------------------------- parâmetros
params = {
    "about": "Q-learning, caso base de Calvano, Calzolari, Denicolò e Pastorello (2020), "
             "AER 110(10). Valores da Ficha de parâmetros do projeto (Pesquisa 1); páginas "
             "do artigo em 'source'.",
    "badge": "Parâmetros do artigo",
    "market": {
        "n": {"value": P["n"], "label": "Número de firmas", "source": "p. 3274"},
        "c": {"value": P["c"], "label": "Custo marginal", "source": "p. 3274"},
        "a": {"value": P["a"], "label": "Qualidade do produto", "source": "p. 3274"},
        "a0": {"value": P["a0"], "label": "Bem externo", "source": "p. 3274"},
        "mu": {"value": P["mu"], "label": "Diferenciação horizontal", "source": "p. 3274"},
        "demand": "q_i = exp((a − p_i)/μ) / [Σ_j exp((a − p_j)/μ) + exp(a0/μ)]; "
                  "π_i = (p_i − c) q_i (eq. 5, p. 3273)",
    },
    "learning": {
        "delta": {"value": P["delta"], "label": "Fator de desconto (δ; γ no roteiro de estudo)",
                  "source": "p. 3274; eq. 4"},
        "alpha": {"value": P["alpha"], "label": "Taxa de aprendizado", "source": "p. 3280"},
        "beta": {"value": P["beta"], "label": "Decaimento da exploração, ε = exp(−β t) "
                 "para cada agente", "source": "p. 3274 e 3280"},
        "k": {"value": P["k"], "label": "Memória (períodos)", "source": "p. 3274"},
        "m": {"value": P["m"], "label": "Preços na grade", "source": "p. 3274"},
        "xi": {"value": P["xi"], "label": "Extensão da grade além de Nash e monopólio",
               "source": "p. 3274"},
        "q0": {"rule": "lucro médio contra rival que sorteia preços uniformemente, "
                       "dividido por (1 − δ); igual em todos os estados", "source": "eq. 8, p. 3275"},
        "initial_state": {"rule": "sorteado em cada sessão", "source": "p. 3275"},
        "tie_break": {"rule": "menor preço", "source": "p. 3274"},
        "convergence": {"t_stable": P["t_stable"], "t_max": P["t_max"],
                        "rule": "ação ótima de cada agente em cada estado inalterada por "
                                "t_stable períodos; teto de t_max", "source": "p. 3276"},
        "sessions": {"value": 1000, "seeds": "1 a 1000", "source": "p. 3273; Pesquisa 8"},
    },
    "reference_points": {
        "price_nash": float(b["pN"]), "price_monopoly": float(b["pM"]),
        "profit_nash": float(piN), "profit_monopoly": float(piM),
        "note": "valores contínuos recalculados (scipy); o artigo não os publica [I]",
    },
    "grid": {"prices": [float(x) for x in prices], "step": step,
             "index_base": 0},
    "profit_table": {"layout": "profit[idx_firm1][idx_firm2] = [lucro firma 1, lucro firma 2]",
                     "values": [[[float(PI[i, j, 0]), float(PI[i, j, 1])] for j in range(m)]
                                for i in range(m)]},
    "q0": {"layout": "q0[firma][idx_preço], igual em todos os 225 estados",
           "values": [[float(x) for x in b["Q0"][0, 0]], [float(x) for x in b["Q0"][1, 0]]]},
    "delta_index": {"formula": "Δ = (π̄ − πN) / (πM − πN), π̄ = lucro médio por firma",
                    "measured_on": "ciclo-limite (Q congelada, ε = 0); média sobre um ciclo completo"},
    "rng_note": "As sessões gravadas usam o gerador do numba (np.random, semente = número da "
                "sessão). O motor em JavaScript usa outro gerador: a mesma semente não "
                "reproduz a mesma sessão; a comparação é entre distribuições.",
}
checks["grade_igual_fase13"] = bool(np.max(np.abs(prices - np.array(base_meta["price_grid"])))
                                    < 1e-12)
checks["referencias_iguais_fase13"] = all(
    abs(params["reference_points"][k] - base_meta["reference_points"][s]) < 1e-12
    for k, s in (("price_nash", "pN"), ("price_monopoly", "pM"), ("profit_nash", "piN"),
                 ("profit_monopoly", "piM")))

# ---------------------------------------------------------------- referência (1.000 sessões)
bs = base_meta["summary"]
dsum = dev_meta["summary"]
brs = br_meta["summary"]["by_version"]["calvano"]
d_sorted = np.sort(d_all)
cls2 = [dev_rows[s]["class_theta2"] for s in seeds_all]
full_blocks = [x for x in blocks_all if int(x["n_sessions"]) == bs["n_sessions"]]
b300 = min(full_blocks, key=lambda x: abs(np.log(float(x["t_mid_geom"]) / 300)))
reference = {
    "about": "Resultados de referência do Q-learning, caso base, 1.000 sessões (sementes 1 a "
             "1.000), lidos de results/. [I] = cálculo do projeto.",
    "sources": ["results/qlearning_base_meta.json", "results/qlearning_base_sessions.csv",
                "results/qlearning_deviation_meta.json",
                "results/qlearning_deviation_sessions.csv",
                "results/qlearning_deviation_paths.csv",
                "results/qlearning_bestresponse_meta.json",
                "results/qlearning_bestresponse_sessions.csv",
                "results/qlearning_trajectory_blocks.csv",
                "results/qlearning_trajectories_meta.json",
                "results/qlearning_mandatory_tests_table.csv"],
    "targets_calvano": {
        "delta_mean": {"value": 0.849, "sd": 0.112, "source": "Tabela 1, p. 3277"},
        "nash_share": {"value": 0.505, "source": "Tabela 1"},
        "rival_reaction_tau2": {"value": -0.127, "source": "Apêndice, Tabela A5"},
        "unprofitable_deviation_share": {"value": 0.936, "source": "Apêndice, Tabela A5"},
        "punishment_periods": {"value": 5.7, "source": "Apêndice, Tabela A5"},
    },
    "base": {
        "n_sessions": bs["n_sessions"], "converged": bs["converged"],
        "delta_mean": bs["delta_mean"], "delta_sd": bs["delta_sd"],
        "delta_mc_error_95": bs["delta_mc_error_95"],
        "delta_median": float(np.median(d_all)),
        "delta_p25": float(np.percentile(d_all, 25)),
        "delta_p75": float(np.percentile(d_all, 75)),
        "share_delta_above_0_5": float((d_all > 0.5).mean()),
        "share_delta_below_0": float((d_all < 0).mean()),
        "periods_mean": bs["periods_mean"], "periods_min": bs["periods_min"],
        "periods_max": bs["periods_max"], "share_cycle": bs["share_cycle"],
    },
    "all_sessions": {
        "seed": [int(s) for s in seeds_all],
        "delta": [float(base_rows[s]["delta"]) for s in seeds_all],
        "periods": [int(base_rows[s]["periods"]) for s in seeds_all],
        "cycle_len": [int(base_rows[s]["cycle_len"]) for s in seeds_all],
        "class_theta_2pct": cls2,
    },
    "deviation": {
        "protocol": dev_meta["protocol"],
        "R_mean": dsum["R_mean"], "G_mean": dsum["G_mean"],
        "share_unprofitable": dsum["share_unprofitable_G_lt_0"],
        "share_unprofitable_wilson95": dsum["share_unprofitable_wilson95"],
        "V_mean_uncensored": dsum["V_mean_uncensored"],
        "V_median_uncensored": dsum["V_median_uncensored"],
        "classes": dsum["classes"],
        "mean_paths": {k: [num(x[k]) for x in dev_paths] for k in dev_paths[0]},
    },
    "best_response": {
        "nash_share": brs["nash_share"], "nash_share_wilson95": brs["nash_share_wilson95"],
        "qloss_path_mean": brs["qloss_path_mean"], "qloss_all_mean": brs["qloss_all_mean"],
    },
    "learning_trajectory": {
        "rule": traj_meta["blocks"]["rule"],
        "blocks_with_all_sessions": {k: [num(x[k]) for x in full_blocks]
                                     for k in ("t_start", "t_end", "t_mid_geom", "delta_mean",
                                               "delta_p10", "delta_p90", "price_mean",
                                               "epsilon_start")},
        "around_period_300": {"t_start": num(b300["t_start"]), "t_end": num(b300["t_end"]),
                              "delta_mean": num(b300["delta_mean"])},
        "delta_random_prices": {"value": 0.4974, "source": "Ficha de parâmetros, teste de sanidade "
                                "(qlearning/reference_points.py, Fase 1.1) [I]"},
    },
    "mandatory_tests": [{k: num(v) for k, v in x.items()} for x in tests_table],
}
checks["referencia_igual_results"] = (
    abs(np.mean(reference["all_sessions"]["delta"]) - bs["delta_mean"]) < 1e-12
    and len(reference["all_sessions"]["seed"]) == 1000
    and cls2.count("A") == dsum["classes"]["theta_0.02"]["A"]
    and abs(reference["learning_trajectory"]["around_period_300"]["delta_mean"] - 0.497) < 0.0005)

# ---------------------------------------------------------------- gravação
meta = {"generated": datetime.now().isoformat(timespec="seconds"),
        "script": "qlearning/export_simulator.py",
        "decimals": DEC,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "numba": numba.__version__}}
OUT.mkdir(parents=True, exist_ok=True)
files = {
    "qlearning_params.json": {"meta": meta, **params},
    "qlearning_reference.json": {"meta": meta, **reference},
    "qlearning_sessions.json": {
        "meta": meta,
        "about": "3 sessões completas do caso base, gravadas no Python (Fase 1.7). Regra fixada "
                 "antes: Δ mais próximo dos percentis 25, 50 e 75 das 1.000 sessões (empate: "
                 "menor semente); a página abre com a do percentil 50.",
        "deviation_note": "Teste de desvio da Fase 1.4: Q congelada, ε = 0; em τ = 1 a firma 1 "
                          "joga a melhor resposta estática (na grade) ao preço da firma 2 em "
                          "τ = 0; τ = 0 a 15. Em ciclos, um teste por ponto do ciclo; "
                          "'session_mean' é a média (1 observação). Lucro descontado: soma de "
                          "0,95^(τ−1) × lucro da firma 1, τ = 1 a 15.",
        "sessions": sessions,
    },
}
for name, obj in files.items():
    with open(OUT / name, "w", encoding="utf-8", newline="\n") as f:
        json.dump(r(obj), f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)

# conferência de leitura: o JSON gravado abre e bate com os números de origem
back = {n: json.load(open(OUT / n, encoding="utf-8")) for n in files}
checks["json_relido_ok"] = (
    abs(back["qlearning_reference.json"]["base"]["delta_mean"] - bs["delta_mean"]) < 1e-6
    and [s["seed"] for s in back["qlearning_sessions.json"]["sessions"]]
    == [chosen[k]["seed"] for k in ("p50", "p25", "p75")]
    and abs(back["qlearning_params.json"]["profit_table"]["values"][9][9][0]
            - PI[9, 9, 0]) < 1e-6)

# ---------------------------------------------------------------- saída na tela
print("Sessões exportadas (regra: Δ mais próximo dos percentis 25, 50 e 75):")
for s in sessions:
    sm = s["deviation_test"]["session_mean"]
    st = s["deviation_test"]["starts"][0]
    print(f"  {s['role']}: semente {s['seed']}, Δ {s['delta']:.3f}, {s['periods']:,} períodos, "
          f"ciclo {s['cycle_len']} {s['cycle_idx']}, abre a página: {s['opens_page']}")
    print(f"       desvio (média do ciclo): R {sm['R']:+.3f}, G {sm['G']:+.4f}, "
          f"V {sm['V']:.2f}, classe {sm['class_theta_2pct']}; "
          f"lucro descontado da firma 1, ponto 0: sem desvio "
          f"{st['discounted_profit_firm1_control']:.4f}, com desvio "
          f"{st['discounted_profit_firm1_deviation']:.4f}")
    print(f"       blocos: {len(s['learning_blocks_linear']['t_start'])} lineares de "
          f"{LIN_BLOCK:,}, {len(s['learning_blocks_log']['t_start'])} logarítmicos")
print(f"\nReferência: Δ médio {bs['delta_mean']:.3f} (alvo 0,849); desvio não lucrativo "
      f"{dsum['share_unprofitable_G_lt_0']:.3f}; Nash {brs['nash_share']:.3f}; "
      f"Δ perto do período 300: {reference['learning_trajectory']['around_period_300']['delta_mean']:.3f}")
print("\nArquivos em docs/data/:")
for n in files:
    print(f"  {n}: {(OUT / n).stat().st_size / 1024:.1f} KB")
print("\nConferências:")
for k, v in checks.items():
    print(f"  {k}: {v}")
print(f"\nTodas True: {all(checks.values())}")
