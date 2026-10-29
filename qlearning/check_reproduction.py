"""
Teste rápido de reprodutibilidade do Q-learning (usado pelo CI, .github/workflows/ci.yml).

Refaz as sessões do caso base das sementes 1 a N (padrão 40) com o código atual e exige
igualdade EXATA com o que está salvo em results/qlearning_base_sessions.csv (Fase 1.3):
convergiu, períodos até convergir, Δ (até 1e-12) e tamanho do ciclo. Se alguém mudar o laço,
os parâmetros ou o gerador sem querer, este teste falha.

Uso (na raiz do repositório): python qlearning/check_reproduction.py [N]
Sai com código 0 se tudo bater e 1 se algo diferir.
Código próprio do projeto.
"""

import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from base_case import PARAMS as P, setup
from batch import run_many

CSV_FILE = HERE.parent / "results" / "qlearning_base_sessions.csv"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    with open(CSV_FILE, newline="", encoding="utf-8") as f:
        saved = {int(r["seed"]): r for r in csv.DictReader(f)}

    st = setup()
    seeds = np.arange(1, n + 1, dtype=np.int64)
    ok, periods, deltas, cyc_len, _ = run_many(
        seeds, st["PI"], st["Q0"], P["alpha"], P["delta"], P["beta"],
        P["t_stable"], P["t_max"], st["piN"], st["piM"])

    bad = []
    for k, s in enumerate(seeds):
        r = saved[int(s)]
        same = (bool(ok[k]) == (r["converged"] == "True")
                and int(periods[k]) == int(r["periods"])
                and abs(float(deltas[k]) - float(r["delta"])) < 1e-12
                and int(cyc_len[k]) == int(r["cycle_len"]))
        if not same:
            bad.append((int(s), int(periods[k]), r["periods"], float(deltas[k]), r["delta"]))

    print(f"Sementes 1 a {n}: {n - len(bad)} de {n} sessões idênticas a results/qlearning_base_sessions.csv")
    for b in bad:
        print(f"  DIFERENTE semente {b[0]}: períodos {b[1]} × {b[2]}; Δ {b[3]:.12f} × {b[4]}")
    print("True " if not bad else "False", " reprodução exata do caso base")
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
