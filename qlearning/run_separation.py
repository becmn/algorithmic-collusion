"""
Fase 1.8: teste de separação (memória × falha de aprendizado). Ver separation.py.
Regra de decisão fixada ANTES de rodar as condições novas (16/10/2026; Decisões do documento):
  "mais aprendizado" = β = 1e-6 (exploração decrescente mais longa) e ε mínimo = 5%
  (exploração permanente mais alta). D = Δ(k = 1) − Δ(k = 0), IC 95% de Welch.
  - favorável à leitura de punição: nas DUAS condições, IC de D inteiro acima de 0;
  - desfavorável: nas DUAS, IC de D inteiro abaixo de 0;
  - inconclusivo: qualquer outro caso.
Uso:
  python qlearning/run_separation.py CONDIÇÃO [N]
     CONDIÇÃO = k0_beta1e-05 | k0_beta4e-06 | k0_beta2e-06 | k0_beta1e-06
              | k1_beta1e-05 | k1_beta4e-06 | k1_beta2e-06 | k1_beta1e-06
              | k0_epsmin0.001 | k0_epsmin0.01 | k0_epsmin0.05
              | k1_epsmin0.001 | k1_epsmin0.01 | k1_epsmin0.05
     N = número de sessões (padrão 1000; com menos de 1000 só confere, não salva)
  python qlearning/run_separation.py resumo      (tabela e decisão, a partir do que foi salvo)
Código próprio do projeto.
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
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from base_case import PARAMS, setup
from deviation import classify
from separation import run_many_sep

OUT = HERE.parent / "results"
CSV_FILE = OUT / "qlearning_separation_sessions.csv"
META = OUT / "qlearning_separation_meta.json"
P = PARAMS
THETA = 0.02
T_FIXED = 4_000_000
Z = float(stats.norm.ppf(0.975))
BETAS = (1e-5, 4e-6, 2e-6, 1e-6)
EPS_MINS = (0.001, 0.01, 0.05)
DECISION_LEVELS = ("beta1e-06", "epsmin0.05")
RULE = ("Regra fixada em 16/10/2026, antes de rodar as condições novas: D = Δ(k=1) − Δ(k=0), "
        "IC 95% de Welch, nas condições de mais aprendizado (β = 1e-6 e ε mínimo = 5%). "
        "Favorável à leitura de punição se o IC de D ficar inteiro acima de 0 nas duas; "
        "desfavorável se ficar inteiro abaixo de 0 nas duas; inconclusivo nos demais casos.")

CONDS = {}
for mem in (0, 1):
    for bt in BETAS:
        CONDS[f"k{mem}_beta{bt:g}"] = {"memory": mem, "beta": bt, "eps_min": 0.0,
                                       "t_stable": P["t_stable"], "t_max": P["t_max"],
                                       "level": f"beta{bt:g}"}
    for e in EPS_MINS:
        CONDS[f"k{mem}_epsmin{e:g}"] = {"memory": mem, "beta": P["beta"], "eps_min": e,
                                        "t_stable": T_FIXED + 1, "t_max": T_FIXED,
                                        "level": f"epsmin{e:g}"}

b = setup()
PI, prices = b["PI"], b["prices"]
step = prices[1] - prices[0]


def wilson(x, n):
    p = x / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def t_ci(x):
    x = np.asarray(x, dtype=float)
    se = x.std(ddof=1) / np.sqrt(len(x))
    q = stats.t.ppf(0.975, len(x) - 1)
    return [float(x.mean() - q * se), float(x.mean() + q * se)]


def q0_for(memory):
    Q0 = b["Q0"]                      # eq. 8 com δ = 0,95, igual em todos os estados
    return Q0[:, :1, :].copy() if memory == 0 else Q0.copy()


def load_meta():
    if META.exists():
        with open(META, encoding="utf-8") as f:
            return json.load(f)
    return {"description": "Fase 1.8, teste de separação: k = 0 × k = 1 com a mesma exploração, "
                           "δ = 0,95, sementes 1 a 1000",
            "rule": RULE, "theta": THETA, "t_fixed_eps_min": T_FIXED, "conditions": {}}


def read_csv(name, cond_key=None, cond_val=None):
    f = OUT / name
    if not f.exists():
        return None
    with open(f, encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh)
                if cond_key is None or r[cond_key] == cond_val]
    return {int(r["seed"]): r for r in rows}


def summarize(name, c, r, secs):
    conv, per, dg, clen, nash1, nopt, qlp, R, G, V = r
    n = len(dg)
    s = {"memory": c["memory"], "beta": c["beta"], "eps_min": c["eps_min"],
         "t_max": c["t_max"], "n_sessions": n, "seconds": secs,
         "share_converged": float(conv.mean()), "periods_mean": float(per.mean()),
         "delta_mean": float(dg.mean()), "delta_sd": float(dg.std(ddof=1)),
         "delta_ci95_t": t_ci(dg), "delta_median": float(np.median(dg)),
         "share_cycle": float((clen > 1).mean()),
         "nash1_share": float(nash1.mean()), "nash1_wilson95": wilson(int(nash1.sum()), n),
         "nash_opt_share": float(nopt.mean()), "nash_opt_wilson95": wilson(int(nopt.sum()), n),
         "qloss_path_mean": float(np.nanmean(qlp))}
    if c["memory"] == 1:
        cls = [classify(R[k], G[k], V[k], THETA) for k in range(n)]
        nA = cls.count("A")
        s.update({"R_mean": float(R.mean()), "share_G_lt_0": float((G < 0).mean()),
                  "classes": {k: cls.count(k) for k in ("A", "B", "C", "outro")},
                  "classA_wilson95": wilson(nA, n)})
    return s, (cls if c["memory"] == 1 else [""] * n)


def checks(name, c, seeds, r):
    """Conferências contra fases anteriores (devem dar True)."""
    conv, per, dg, clen, nash1, nopt, qlp, R, G, V = r
    out = {}
    if c["memory"] == 1 and c["eps_min"] == 0.0:
        ref = read_csv("qlearning_exploration_sessions.csv", "beta", f"{c['beta']:g}")
        if ref:
            out["Δ = Fase 1.5 parte 3"] = all(abs(float(ref[int(s)]["delta"]) - d) < 1e-12
                                              for s, d in zip(seeds, dg))
            out["Nash 1 período = Fase 1.5 parte 3"] = all(
                (ref[int(s)]["nash_on_path_calvano"] == "True") == bool(x)
                for s, x in zip(seeds, nash1))
            # no CSV da Fase 1.5, V censurado foi gravado vazio; aqui é −1
            vref = lambda x: -1.0 if x == "" else float(x)
            out["R, G, V = Fase 1.5 parte 3"] = all(
                abs(float(ref[int(s)]["R"]) - R[k]) < 1e-12 and
                abs(float(ref[int(s)]["G"]) - G[k]) < 1e-12 and
                abs(vref(ref[int(s)]["V"]) - V[k]) < 1e-12 for k, s in enumerate(seeds))
        if abs(c["beta"] - P["beta"]) < 1e-15:
            br = read_csv("qlearning_bestresponse_sessions.csv")
            if br:
                out["Nash ótimo = Fase 1.5 parte 1"] = all(
                    (br[int(s)]["nash_on_path_otima"] == "True") == bool(x)
                    for s, x in zip(seeds, nopt))
    if c["memory"] == 0 and c["eps_min"] == 0.0 and abs(c["beta"] - P["beta"]) < 1e-15:
        ref = read_csv("qlearning_controls_sessions.csv", "condition", "sem_memoria_d095")
        if ref:
            out["Δ = controle sem memória (Fase 1.5 parte 2)"] = all(
                abs(float(ref[int(s)]["delta"]) - d) < 1e-12 for s, d in zip(seeds, dg))
            out["Nash = controle sem memória (Fase 1.5 parte 2)"] = all(
                (ref[int(s)]["nash_on_path_calvano"] == "True") == bool(x)
                for s, x in zip(seeds, nash1))
    if c["memory"] == 0 and c["eps_min"] > 0 and len(seeds) == 1000:
        f = OUT / "qlearning_memoryless_diagnosis.json"
        if f.exists():
            with open(f, encoding="utf-8") as fh:
                ref = json.load(fh)["blocks"]["epsmin"][str(c["eps_min"])]["delta_mean"]
            out["Δ médio = diagnóstico ε mínimo (Fase 1.5 parte 3)"] = abs(ref - dg.mean()) < 1e-12
    return out


def save(name, c, seeds, r, cls, summary, chk):
    OUT.mkdir(exist_ok=True)
    conv, per, dg, clen, nash1, nopt, qlp, R, G, V = r
    header = ["condition", "memory", "beta", "eps_min", "seed", "converged", "periods", "delta",
              "cycle_len", "nash_on_path_1period", "nash_on_path_optimal", "qloss_path",
              "R", "G", "V", "class_theta2"]
    old = []
    if CSV_FILE.exists():
        with open(CSV_FILE, encoding="utf-8") as fh:
            old = [row for row in csv.DictReader(fh) if row["condition"] != name]
    tmp_csv = CSV_FILE.with_suffix(".tmp")
    with open(tmp_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        for row in old:
            w.writerow(row)
        for k, s in enumerate(seeds):
            w.writerow({"condition": name, "memory": c["memory"], "beta": f"{c['beta']:g}",
                        "eps_min": f"{c['eps_min']:g}", "seed": int(s),
                        "converged": bool(conv[k]), "periods": int(per[k]),
                        "delta": repr(float(dg[k])), "cycle_len": int(clen[k]),
                        "nash_on_path_1period": bool(nash1[k]),
                        "nash_on_path_optimal": bool(nopt[k]),
                        "qloss_path": repr(float(qlp[k])),
                        "R": "" if np.isnan(R[k]) else repr(float(R[k])),
                        "G": "" if np.isnan(G[k]) else repr(float(G[k])),
                        "V": "" if np.isnan(V[k]) else repr(float(V[k])),
                        "class_theta2": cls[k]})
    tmp_csv.replace(CSV_FILE)
    meta = load_meta()
    meta["conditions"][name] = {"date": datetime.now().isoformat(timespec="seconds"),
                                "checks": {k: bool(v) for k, v in chk.items()}, **summary}
    meta["versions"] = {"python": platform.python_version(), "numpy": np.__version__,
                        "numba": numba.__version__, "scipy": scipy.__version__,
                        "platform": platform.platform()}
    meta["numba_threads"] = numba.get_num_threads()
    write_meta(meta)


def write_meta(meta):
    """Grava num arquivo temporário e só então substitui o antigo (nunca deixa o JSON pela metade)."""
    tmp = META.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    tmp.replace(META)


def resumo():
    meta = load_meta()
    if not CSV_FILE.exists():
        sys.exit("nada salvo ainda")
    with open(CSV_FILE, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    by = {}
    for r in rows:
        by.setdefault(r["condition"], []).append(float(r["delta"]))
    levels = [f"beta{bt:g}" for bt in BETAS] + [f"epsmin{e:g}" for e in EPS_MINS]
    print("\n Exploração       Δ sem memória (k=0)     Δ com memória (k=1)     D = k1 − k0 [IC 95% Welch]"
          "   Nash 1p k1  Nash ótimo k1  classe A k1")
    table = {}
    for lv in levels:
        a0, a1 = by.get(f"k0_{lv}"), by.get(f"k1_{lv}")
        if a0 is None or a1 is None:
            print(f" {lv:15s} (falta rodar)")
            continue
        a0, a1 = np.array(a0), np.array(a1)
        tt = stats.ttest_ind(a1, a0, equal_var=False)
        ci = tt.confidence_interval(0.95)
        s1 = meta["conditions"][f"k1_{lv}"]
        table[lv] = {"delta_k0": float(a0.mean()), "delta_k1": float(a1.mean()),
                     "D": float(a1.mean() - a0.mean()), "D_ci95_welch": [float(ci.low), float(ci.high)],
                     "p_welch": float(tt.pvalue), "nash1_k1": s1["nash1_share"],
                     "nash_opt_k1": s1["nash_opt_share"],
                     "classA_k1": s1["classes"]["A"] / s1["n_sessions"]}
        t = table[lv]
        print(f" {lv:15s} {t['delta_k0']:.3f} {str([round(x, 3) for x in t_ci(a0)]):20s}"
              f"   {t['delta_k1']:.3f} {str([round(x, 3) for x in t_ci(a1)]):20s}"
              f"   {t['D']:+.3f} [{ci.low:+.3f}; {ci.high:+.3f}]"
              f"        {t['nash1_k1']:.3f}       {t['nash_opt_k1']:.3f}        {t['classA_k1']:.3f}")
    verdict = None
    if all(lv in table for lv in DECISION_LEVELS):
        lows = [table[lv]["D_ci95_welch"][0] for lv in DECISION_LEVELS]
        highs = [table[lv]["D_ci95_welch"][1] for lv in DECISION_LEVELS]
        if all(x > 0 for x in lows):
            verdict = "favoravel"
        elif all(x < 0 for x in highs):
            verdict = "desfavoravel"
        else:
            verdict = "inconclusivo"
    print(f"\n{RULE}\nDecisão: {verdict}")
    meta["comparison"] = table
    meta["decision"] = verdict
    write_meta(meta)


if __name__ == "__main__":
    ARG = sys.argv[1] if len(sys.argv) > 1 else "resumo"
    if ARG == "resumo":
        resumo()
        sys.exit()
    if ARG not in CONDS:
        sys.exit(f"condição desconhecida: {ARG}")
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    c = CONDS[ARG]
    seeds = np.arange(1, N + 1, dtype=np.int64)
    args = (PI, q0_for(c["memory"]), P["alpha"], P["delta"], c["beta"], c["t_stable"],
            c["t_max"], b["piN"], b["piM"], prices, step, c["memory"], c["eps_min"])
    print(f"Núcleos usados pelo numba: {numba.get_num_threads()}; condição {ARG}; sessões: {N}")
    t0 = time.perf_counter()
    run_many_sep(np.array([1], dtype=np.int64), *args)
    print(f"Compilação (1 sessão): {time.perf_counter() - t0:.1f} s")
    t0 = time.perf_counter()
    r = run_many_sep(seeds, *args)
    secs = time.perf_counter() - t0
    summary, cls = summarize(ARG, c, r, secs)
    chk = checks(ARG, c, seeds, r)
    lo, hi = summary["delta_ci95_t"]
    print(f"{N} sessões em {secs:.1f} s; convergiram {summary['share_converged']:.3f}; "
          f"períodos {summary['periods_mean']:,.0f}")
    print(f"Δ {summary['delta_mean']:.3f} [IC 95% {lo:.3f}; {hi:.3f}]; Nash 1 período "
          f"{summary['nash1_share']:.3f}; Nash ótimo {summary['nash_opt_share']:.3f}")
    if c["memory"] == 1:
        print(f"R {summary['R_mean']:+.3f}; G < 0 {summary['share_G_lt_0']:.3f}; "
              f"classes {summary['classes']}")
    for k, v in chk.items():
        print(f"  conferência — {k}: {v}")
    if chk and not all(chk.values()):
        sys.exit("ERRO: alguma conferência deu False; nada foi salvo")
    if N == 1000:
        save(ARG, c, seeds, r, cls, summary, chk)
        print(f"Salvo em {CSV_FILE.name} e {META.name}")
    else:
        print("(menos de 1000 sessões: só conferência, nada salvo)")
