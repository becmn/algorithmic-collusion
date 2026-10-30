"""
Conferências da análise Q-learning × LLM (analysis/llm_compare.py) com dados sintéticos.
Uso: python analysis/test_llm_compare.py
Código próprio do projeto.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm_compare as L  # noqa: E402

checks = []


def check(name, ok, detail=None):
    checks.append((name, bool(ok)))
    if not ok and detail is not None:
        print("  ->", detail)


rng = np.random.default_rng(1)
Q = rng.normal(0.85, 0.11, 1000)

# Welch: IC 95% conferido com a fórmula de Welch–Satterthwaite feita à parte
x, y = rng.normal(0.6, 0.3, 21), rng.normal(0.5, 0.25, 21)
d, ci, p, df = L.welch(x, y, 0.95)
se = np.sqrt(x.var(ddof=1) / 21 + y.var(ddof=1) / 21)
check("Welch: p igual ao do scipy e IC = d ± t(df)·EP",
      abs(p - stats.ttest_ind(x, y, equal_var=False).pvalue) < 1e-12
      and abs(ci[1] - d - stats.t.ppf(0.975, df) * se) < 1e-12)

# Holm: exemplo clássico p = (0,01; 0,04; 0,03) → ajustados (0,03; 0,06; 0,06)
h = L.holm({"a": 0.01, "b": 0.04, "c": 0.03})
check("Holm: p ajustados monótonos (0,03; 0,06; 0,06) e decisão a 5%",
      abs(h["a"]["p_holm"] - 0.03) < 1e-12 and abs(h["c"]["p_holm"] - 0.06) < 1e-12
      and abs(h["b"]["p_holm"] - 0.06) < 1e-12 and h["a"]["reject"] and not h["b"]["reject"])

# Decisões
far = L.compare({"Q": Q, "P1": rng.normal(0.30, 0.2, 21), "P2": rng.normal(0.31, 0.2, 21)})
check("diferença grande (0,85 × 0,30) → 'diferente' em H1 e H2", far["H1"]["decision"] == "diferente"
      and far["H2"]["decision"] == "diferente", far["H1"])
close = L.compare({"Q": Q, "P1": rng.normal(0.85, 0.03, 60), "P2": rng.normal(0.85, 0.03, 60)})
check("médias iguais com pouca variação → 'equivalente' (IC 90% dentro de ±0,10)",
      close["H1"]["decision"] == "equivalente" and close["H3"]["decision"] == "equivalente", close["H1"])
wide = L.compare({"Q": Q, "P1": rng.normal(0.80, 0.45, 12), "P2": rng.normal(0.85, 0.45, 12)})
check("muita variação e diferença pequena → 'inconclusivo' (nunca 'igual')",
      wide["H1"]["decision"] == "inconclusivo" and wide["H3"]["decision"] == "inconclusivo", wide["H1"])
check("IC da família no nível de Bonferroni (98,3%)", abs(far["H1"]["ci_family_level"] - (1 - 0.05 / 3)) < 1e-12)
check("BCa calculado, com semente fixa (mesmo resultado em duas rodadas)",
      L.compare({"Q": Q, "P1": x, "P2": y})["H3"]["bca_family"] == L.compare({"Q": Q, "P1": x, "P2": y})["H3"]["bca_family"])

# Família secundária e placebo
llm = pd.DataFrame({"prompt": ["P1"] * 21 + ["P2"] * 21,
                    "class_theta_0.02": ["A"] * 19 + ["C"] * 2 + ["A"] * 5 + ["C"] * 16,
                    "placebo_theta_0.02": [False] * 20 + [True] + [True] * 6 + [False] * 15})
s = L.secondary(llm, 0.903)
check("secundária: fração A, binomial exato contra a do Q-learning, Holm e regra do placebo",
      s["P1"]["share_A"][0] == 19 / 21 and s["P1"]["punishment_pattern"] and not s["P2"]["punishment_pattern"]
      and abs(s["P2"]["p"] - stats.binomtest(5, 21, 0.903).pvalue) < 1e-12 and "p_holm" in s["P1"], s)

print("Conferências:")
for n, ok in checks:
    print(f"  {'True ' if ok else 'False'}  {n}")
n_ok = sum(ok for _, ok in checks)
print(f"\n{n_ok} de {len(checks)} conferências True")
sys.exit(0 if n_ok == len(checks) else 1)
