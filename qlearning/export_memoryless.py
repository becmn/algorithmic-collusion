"""
Fase 2 (revisão de 30/10/2026) — Sessão SEM memória gravada para o simulador.

Escolha (mesma regra da sessão com memória que abre a página): entre as 1.000 sessões sem
memória do teste de separação com os parâmetros do artigo (condição k0_beta4e-06 de
results/qlearning_separation_sessions.csv), a de Δ mais próximo da mediana.

A sessão é refeita com um laço igual ao de controls.run_session_mem (memória 0, ε mínimo 0),
que também soma os preços em blocos de 10.000 períodos (os mesmos blocos do Worker).
Conferências: períodos, Δ e estratégia final iguais aos de run_session_mem e do CSV.

Teste do corte: a estratégia sem memória tem um estado só; ela é copiada para os 225 estados
da memória de 1 período, e o teste de desvio comum (deviation.py) roda sem mudança: como a
estratégia não depende do passado, a rival não pode reagir ao corte.

Saída: docs/data/qlearning_memoryless.json. Uso: python qlearning/export_memoryless.py
Código próprio do projeto.
"""

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "qlearning"))

from base_case import PARAMS, setup  # noqa: E402
from controls import _argmax_max, run_session_mem  # noqa: E402
from deviation import T_DEV, classify, deviation_paths, metrics_one_start  # noqa: E402

COND = "k0_beta4e-06"
BLOCK = 10_000
T_END = 4_000_000
OUT = ROOT / "docs" / "data" / "qlearning_memoryless.json"


@njit(cache=True)
def run_traced_k0(seed, PI, Q0, alpha, delta, beta, t_stable, t_max, edges):
    """Cópia de run_session_mem com memory = 0 e eps_min = 0, somando preços e lucros por bloco.
    A ordem dos sorteios é a mesma, então a sessão é idêntica (conferido em main)."""
    np.random.seed(seed)
    m = PI.shape[0]
    Q = Q0.copy()
    greedy = np.empty((2, 1), dtype=np.int64)
    maxq = np.empty((2, 1))
    for i in range(2):
        g, v = _argmax_max(Q[i, 0])
        greedy[i, 0] = g
        maxq[i, 0] = v
    s = np.random.randint(0, 1)
    nb = edges.shape[0] - 1
    sp1 = np.zeros(nb)
    sp2 = np.zeros(nb)
    spi = np.zeros(nb)
    cnt = np.zeros(nb, dtype=np.int64)
    stable = 0
    t = 0
    converged = False
    act = np.empty(2, dtype=np.int64)
    b = 0
    while t < t_max:
        eps = np.exp(-beta * t)
        for i in range(2):
            if np.random.random() < eps:
                act[i] = np.random.randint(0, m)
            else:
                act[i] = greedy[i, s]
        changed = False
        for i in range(2):
            target = PI[act[0], act[1], i] + delta * maxq[i, 0]
            Q[i, s, act[i]] = (1.0 - alpha) * Q[i, s, act[i]] + alpha * target
            g, v = _argmax_max(Q[i, s])
            if g != greedy[i, s]:
                changed = True
                greedy[i, s] = g
            maxq[i, s] = v
        while b < nb - 1 and t >= edges[b + 1]:
            b += 1
        sp1[b] += act[0]
        sp2[b] += act[1]
        spi[b] += 0.5 * (PI[act[0], act[1], 0] + PI[act[0], act[1], 1])
        cnt[b] += 1
        if changed:
            stable = 0
        else:
            stable += 1
        s = 0
        t += 1
        if stable >= t_stable:
            converged = True
            break
    return converged, t, greedy, sp1, sp2, spi, cnt


def main():
    P = PARAMS
    b = setup()
    PI, prices, piN, piM = b["PI"], b["prices"], b["piN"], b["piM"]
    m = len(prices)
    Q0 = b["Q0"][:, :1, :].copy()                      # igual a q0_for(0) de run_separation.py
    with open(ROOT / "results" / "qlearning_separation_sessions.csv", newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["condition"] == COND]
    assert len(rows) == 1000, len(rows)
    d = np.array([float(r["delta"]) for r in rows])
    target = float(np.median(d))
    k = int(np.argmin(np.abs(d - target)))
    row = rows[k]
    seed = int(row["seed"])

    args = (PI, Q0, P["alpha"], P["delta"], P["beta"], P["t_stable"], P["t_max"])
    ok, t, dlt, nc, pidx, greedy_ref, s_end, _ = run_session_mem(seed, *args, piN, piM, 0, 0.0)
    edges = np.arange(0, T_END + BLOCK, BLOCK, dtype=np.int64)
    okT, tT, greedy, sp1, sp2, spi, cnt = run_traced_k0(seed, *args, edges)
    a1, a2 = int(greedy[0, 0]), int(greedy[1, 0])
    delta_final = (0.5 * (PI[a1, a2, 0] + PI[a1, a2, 1]) - piN) / (piM - piN)
    checks = {
        "sessao_igual_run_session_mem": bool(ok and okT and t == tT and np.array_equal(greedy, greedy_ref)
                                             and abs(delta_final - dlt) < 1e-12),
        "sessao_igual_csv": bool(int(row["periods"]) == t and abs(float(row["delta"]) - dlt) < 1e-12
                                 and row["converged"] == "True"),
        "blocos_somam_periodos": int(cnt.sum()) == t,
    }

    # estratégia constante copiada para os 225 estados da memória de 1 período
    g_full = np.empty((2, m * m), dtype=np.int64)
    g_full[0, :], g_full[1, :] = a1, a2
    s0 = a1 * m + a2
    in_cycle = np.zeros(m * m, dtype=np.bool_)
    in_cycle[s0] = True
    ctrl, dev = deviation_paths(g_full, PI, s0)
    R, G, V, Vp, didv, Rc, curve = metrics_one_start(g_full, PI, prices, P["delta"],
                                                     float(prices[1] - prices[0]), s0, in_cycle, False)
    checks["rival_nao_reage"] = bool(np.all(dev[1:, 1] == ctrl[1:, 1]) and R == 0.0)
    w = P["delta"] ** np.arange(T_DEV)
    pi1c = np.array([PI[ctrl[x, 0], ctrl[x, 1], 0] for x in range(T_DEV + 1)])
    pi1d = np.array([PI[dev[x, 0], dev[x, 1], 0] for x in range(T_DEV + 1)])

    keep = np.flatnonzero(cnt > 0)
    n = cnt[keep]
    out = {
        "meta": {"generated": datetime.now().isoformat(timespec="seconds"),
                 "script": "qlearning/export_memoryless.py", "condition": COND, "checks": checks,
                 "selection": "Δ mais próximo da mediana das 1.000 sessões sem memória (parâmetros do artigo)",
                 "median_delta": target},
        "session": {
            "role": "memoryless_p50", "memory": 0, "seed": seed, "converged": bool(ok), "periods": int(t),
            "delta": float(dlt), "cycle_len": 1, "cycle_idx": [[a1, a2]],
            "price_firm1_final": float(prices[a1]), "price_firm2_final": float(prices[a2]),
            "greedy_strategy": {"note": "estratégia sem memória (1 estado) copiada para os 225 estados",
                                "firm1": [int(x) for x in g_full[0]], "firm2": [int(x) for x in g_full[1]]},
            "learning_blocks_linear": {
                "block_size": BLOCK,
                "t_start": [int(x) for x in edges[keep]], "n_periods": [int(x) for x in n],
                "price_firm1": [float(x) for x in prices[0] + (sp1[keep] / n) * (prices[1] - prices[0])],
                "price_firm2": [float(x) for x in prices[0] + (sp2[keep] / n) * (prices[1] - prices[0])],
                "delta": [float(x) for x in ((spi[keep] / n) - piN) / (piM - piN)],
                "epsilon_start": [float(np.exp(-P["beta"] * x)) for x in edges[keep]],
            },
            "deviation_test": {
                "control_price_firm1": [float(prices[x]) for x in ctrl[:, 0]],
                "control_price_firm2": [float(prices[x]) for x in ctrl[:, 1]],
                "deviation_price_firm1": [float(prices[x]) for x in dev[:, 0]],
                "deviation_price_firm2": [float(prices[x]) for x in dev[:, 1]],
                "R": float(R), "G": float(G), "V": int(V) if V >= 0 else None,
                "class_theta_0.02": classify(R, G, V, 0.02),
                "discounted_profit_firm1_control": float((w * pi1c[1:]).sum()),
                "discounted_profit_firm1_deviation": float((w * pi1d[1:]).sum()),
            },
        },
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print("Conferências:")
    for kname, v in checks.items():
        print(f"  {v!s:<5}  {kname}")
    print(f"\n{sum(checks.values())} de {len(checks)} conferências True; semente {seed}, Δ {dlt:.4f}, "
          f"{t:,} períodos, preços finais {prices[a1]:.4f} / {prices[a2]:.4f}")


if __name__ == "__main__":
    main()
