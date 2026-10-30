"""
Fase 3 — Textos dos prompts e leitura da resposta do LLM.

Os textos exatos de Fish et al. (arXiv 2404.00806 v6, Apêndice G.1 e G.2.1) NÃO estão no
repositório: o artigo tem a licença padrão do arXiv, não Creative Commons, e a decisão em
aberto do projeto é conferir a licença ou pedir permissão aos autores antes de publicá-los
(documento do projeto, Decisões em aberto; Pesquisas 2 e 11). Por isso:

  llm/prompts/local/   (fora do Git, ver .gitignore) recebe os textos colados do artigo;
  llm/prompts/test/    tem textos PRÓPRIOS, curtos, só para os testes sem API. O programa se
                       recusa a usá-los com o Gemini de verdade.

Arquivos de cada pasta:
  P0.txt, P1.txt, P2.txt  prefixos (vão como system prompt; Protocolo)
  template.txt            corpo do prompt do duopólio, com os marcadores {cost}, {ceiling},
                          {plans}, {insights} e {market_data}
  template_monopoly.txt   corpo do teste de competência (opcional; se faltar, usa template.txt)
  format.json             rótulos dos 4 campos da resposta e o formato de cada linha do
                          histórico (ver llm/prompts/README.md)

Código próprio do projeto.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

PROMPTS = Path(__file__).resolve().parent / "prompts"
PLACEHOLDERS = ("cost", "ceiling", "plans", "insights", "market_data")
FIELDS = ("observations", "plans", "insights", "price")   # ordem da resposta (Protocolo)


class PromptFilesMissing(RuntimeError):
    pass


@dataclass
class PromptSet:
    source: str                         # "local" (textos do artigo) ou "test"
    prefixes: dict = field(default_factory=dict)
    template: str = ""
    template_monopoly: str = ""
    fmt: dict = field(default_factory=dict)

    def body(self, monopoly: bool) -> str:
        return self.template_monopoly if (monopoly and self.template_monopoly) else self.template


def load_prompts(source: str = "local") -> PromptSet:
    """Lê os textos de llm/prompts/<source>/; diz exatamente o que falta, se faltar."""
    d = PROMPTS / source
    need = ["P0.txt", "P1.txt", "P2.txt", "template.txt", "format.json"]
    missing = [n for n in need if not (d / n).is_file()]
    if missing:
        raise PromptFilesMissing(
            f"Faltam em {d}: {', '.join(missing)}. Veja llm/prompts/README.md (passo a passo).")
    ps = PromptSet(source=source)
    for name in ("P0", "P1", "P2"):
        ps.prefixes[name] = (d / f"{name}.txt").read_text(encoding="utf-8").strip()
    ps.template = (d / "template.txt").read_text(encoding="utf-8")
    tm = d / "template_monopoly.txt"
    ps.template_monopoly = tm.read_text(encoding="utf-8") if tm.is_file() else ""
    ps.fmt = json.loads((d / "format.json").read_text(encoding="utf-8"))
    for tpl in filter(None, (ps.template, ps.template_monopoly)):
        absent = [p for p in PLACEHOLDERS if "{" + p + "}" not in tpl]
        if absent:
            raise PromptFilesMissing(f"O modelo de prompt em {d} não tem os marcadores: "
                                     + ", ".join("{" + p + "}" for p in absent))
    labels = ps.fmt.get("labels", {})
    if sorted(labels) != sorted(FIELDS) or not all(labels.values()):
        raise PromptFilesMissing(f"format.json em {d} precisa de 'labels' com {FIELDS}.")
    for key in ("row_duopoly", "row_monopoly"):
        if key not in ps.fmt:
            raise PromptFilesMissing(f"format.json em {d} precisa de '{key}'.")
    return ps


def money(x: float) -> str:
    """Valores com 2 casas decimais (Protocolo: histórico com 2 casas)."""
    return f"{x:.2f}"


def market_data(ps: PromptSet, history: list, monopoly: bool) -> str:
    """Últimos 100 períodos, do mais recente para o mais antigo (Fish, G.2.3; Pesquisa 2).
    Cada item de history: dict com period, my_price, rival_price (None no monopólio), qty, profit."""
    from market import HISTORY
    rows = []
    row_t = ps.fmt["row_monopoly" if monopoly else "row_duopoly"]
    for h in reversed(history[-HISTORY:]):
        vals = {"round": str(h["period"]), "my_price": money(h["my_price"]),
                "qty": money(h["qty"]), "profit": money(h["profit"])}
        if not monopoly:
            vals["rival_price"] = money(h["rival_price"])
        rows.append(_fill(row_t, vals))
    if not rows:
        return ps.fmt.get("empty_history", "")
    return ps.fmt.get("row_separator", "\n").join(rows)


def _fill(text: str, vals: dict) -> str:
    """Troca só os marcadores conhecidos ({nome}); chaves literais do texto ficam intactas."""
    for k, v in vals.items():
        text = text.replace("{" + k + "}", v)
    return text


def user_prompt(ps: PromptSet, *, cost: float, ceiling: float, plans: str, insights: str,
                history: list, monopoly: bool) -> str:
    return _fill(ps.body(monopoly), {
        "cost": money(cost), "ceiling": money(ceiling), "plans": plans, "insights": insights,
        "market_data": market_data(ps, history, monopoly)})


# ---------------------------------------------------------------------------
# Leitura da resposta
# ---------------------------------------------------------------------------
NUM = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?|[-+]?\.\d+")


class InvalidResponse(ValueError):
    pass


def parse_response(ps: PromptSet, text: str) -> dict:
    """Separa os 4 campos pela ordem dos rótulos e extrai o preço (primeiro número depois do
    rótulo do preço). Resposta inválida: falta rótulo, ordem errada, preço ausente, não
    numérico ou ≤ 0. Preço acima do monopólio é válido (é só sinalizado na análise)."""
    if not isinstance(text, str) or not text.strip():
        raise InvalidResponse("resposta vazia")
    labels = ps.fmt["labels"]
    low = text.lower()
    pos, start = {}, 0
    for f in FIELDS:
        i = low.find(labels[f].lower(), start)
        if i < 0:
            raise InvalidResponse(f"rótulo ausente ou fora de ordem: {labels[f]!r}")
        pos[f] = i
        start = i + len(labels[f])
    out = {}
    for k, f in enumerate(FIELDS):
        a = pos[f] + len(labels[f])
        b = pos[FIELDS[k + 1]] if k + 1 < len(FIELDS) else len(text)
        out[f] = text[a:b].strip().lstrip(":").strip()
    m = NUM.search(out["price"].replace("$", ""))
    if not m:
        raise InvalidResponse("preço ausente")
    try:
        price = float(m.group(0).replace(",", ""))
    except ValueError as e:
        raise InvalidResponse(f"preço ilegível: {m.group(0)!r}") from e
    if not math.isfinite(price) or price <= 0:
        raise InvalidResponse(f"preço inválido: {price}")
    out["price_value"] = price
    return out
