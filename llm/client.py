"""
Fase 3 — Cliente do Gemini, cliente falso para testes, custo e teto de gasto.

GeminiClient usa o SDK oficial google-genai (versão fixada em requirements-llm.txt).
A chave fica só no arquivo .env da raiz do repositório (GEMINI_API_KEY=...), que está no
.gitignore; nunca é impressa nem gravada no log.

Preços por 1M de tokens (US$), Pesquisa 7 (consulta em 07/10/2026) e documento do projeto
(Tecnologias e custos). Os tokens de raciocínio ("thoughts") são cobrados como saída.
O contador conta sempre o preço PAGO PADRÃO, mesmo na camada gratuita e mesmo pedindo Flex
(nada confirma que a chamada foi atendida como Flex): é o custo "se fosse pago", o lado
conservador do teto.

Código próprio do projeto.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# US$ por 1M de tokens: (entrada, saída). Flex = metade (Pesquisa 7, seção 1).
PRICES = {
    "gemini-3.5-flash-lite": [((None, None), (0.30, 2.50))],
    # 3.8 Flash: 0,75 / 3,75 até 31/12/2026; 1,50 / 7,50 a partir de 1/1/2027
    "gemini-3.8-flash": [((None, dt.date(2026, 12, 31)), (0.75, 3.75)),
                         ((dt.date(2027, 1, 1), None), (1.50, 7.50))],
}


def price_per_mtok(model: str, service_tier: str | None, today: dt.date | None = None):
    """Preço padrão do dia. service_tier fica na assinatura para registro, mas o contador
    não aplica o desconto do Flex (lado conservador)."""
    today = today or dt.datetime.now(dt.timezone.utc).date()
    for (lo, hi), (pin, pout) in PRICES[model]:
        if (lo is None or today >= lo) and (hi is None or today <= hi):
            return pin, pout
    raise KeyError(f"sem preço para {model} em {today}")


def call_cost(model, service_tier, prompt_tokens, output_tokens, thoughts_tokens, today=None):
    pin, pout = price_per_mtok(model, service_tier, today)
    return (prompt_tokens * pin + (output_tokens + thoughts_tokens) * pout) / 1e6


class RateLimited(RuntimeError):
    """HTTP 429 (limite por minuto ou por dia)."""
    def __init__(self, msg, retry_after=None):
        super().__init__(msg)
        self.retry_after = retry_after


class TransientAPIError(RuntimeError):
    """Erro temporário: servidor (5xx), tempo esgotado (408) ou falha de rede."""


class FatalAPIError(RuntimeError):
    """Erro que esperar não resolve (chave inválida, modelo inexistente, saldo pré-pago zerado,
    parâmetro recusado): a execução pausa com o motivo, para a Beatriz resolver e retomar."""


# Frases de saldo/cobrança de verdade. NÃO usar "billing" sozinho: a mensagem comum de limite
# de uso do Gemini diz "...please check your plan and billing details" e deve esperar e repetir.
BILLING_WORDS = ("prepay", "prepayment", "credit balance", "insufficient balance", "billing account",
                 "payment required", "billing is not enabled", "billing has been disabled",
                 "spending cap", "spend cap", "limit: 0")
HTTP_TIMEOUT_MS = 180_000   # 3 min por chamada; sem isso o SDK espera para sempre


@dataclass
class CallResult:
    text: str
    model_version: str | None = None
    response_id: str | None = None
    prompt_tokens: int = 0
    output_tokens: int = 0
    thoughts_tokens: int = 0
    finish_reason: str | None = None
    extra: dict = field(default_factory=dict)


def load_api_key(env_file: Path = ROOT / ".env") -> str:
    """GEMINI_API_KEY do ambiente ou do arquivo .env (formato NOME=valor)."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key and env_file.is_file():
        raw = env_file.read_bytes()
        # o Bloco de Notas pode gravar com BOM; o PowerShell 5 (echo >) grava em UTF-16
        text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8-sig", errors="replace")
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("GEMINI_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        raise RuntimeError("Chave não encontrada: crie o arquivo .env na raiz com GEMINI_API_KEY=sua_chave")
    return key


class GeminiClient:
    """Uma chamada por decisão de preço. Registra modelo pedido e versão devolvida."""
    real = True

    def __init__(self, model: str, temperature: float = 1.0, thinking_level: str | None = None,
                 service_tier: str | None = None, max_output_tokens: int | None = None):
        from google import genai  # importado só aqui: os testes sem API não precisam do SDK
        from google.genai import types
        self.model, self.temperature = model, temperature
        self.thinking_level, self.service_tier = thinking_level, service_tier
        self.max_output_tokens = max_output_tokens
        self._client = genai.Client(api_key=load_api_key(),
                                    http_options=types.HttpOptions(timeout=HTTP_TIMEOUT_MS))

    def settings(self) -> dict:
        return {"model": self.model, "temperature": self.temperature,
                "thinking_level": self.thinking_level, "service_tier": self.service_tier,
                "max_output_tokens": self.max_output_tokens}

    def generate(self, system: str, user: str, seed: int) -> CallResult:
        from google.genai import errors, types
        cfg = dict(system_instruction=system, temperature=self.temperature, seed=seed)
        if self.max_output_tokens:
            cfg["max_output_tokens"] = self.max_output_tokens
        if self.thinking_level:
            cfg["thinking_config"] = types.ThinkingConfig(thinking_level=self.thinking_level.upper())
        if self.service_tier:
            cfg["service_tier"] = self.service_tier.upper()
        import httpx
        try:
            r = self._client.models.generate_content(
                model=self.model, contents=user, config=types.GenerateContentConfig(**cfg))
        except errors.APIError as e:
            full, msg = str(e), str(e)[:300]
            if e.code == 429 and any(w in full.lower() for w in BILLING_WORDS):
                raise FatalAPIError(f"saldo ou cobrança: {msg}") from e
            if e.code == 429:
                raise RateLimited(msg) from e
            if e.code in (408, 499) or (e.code is not None and e.code >= 500):
                raise TransientAPIError(msg) from e
            raise FatalAPIError(f"HTTP {e.code}: {msg}") from e
        except (httpx.TimeoutException, httpx.TransportError) as e:
            raise TransientAPIError(f"rede: {type(e).__name__}: {str(e)[:200]}") from e
        u = r.usage_metadata
        fin, text = None, ""
        if r.candidates:
            c0 = r.candidates[0]
            fin = getattr(c0.finish_reason, "value", c0.finish_reason) if c0.finish_reason else None
            text = "".join(p.text for p in (c0.content.parts if c0.content and c0.content.parts else [])
                           if getattr(p, "text", None) and not getattr(p, "thought", False))
        extra = {}
        if r.prompt_feedback and r.prompt_feedback.block_reason:
            extra["block_reason"] = str(getattr(r.prompt_feedback.block_reason, "value", r.prompt_feedback.block_reason))
        if u and getattr(u, "traffic_type", None):
            extra["traffic_type"] = str(getattr(u.traffic_type, "value", u.traffic_type))
        return CallResult(
            text=text, model_version=r.model_version, response_id=r.response_id,
            prompt_tokens=(u.prompt_token_count or 0) if u else 0,
            output_tokens=(u.candidates_token_count or 0) if u else 0,
            thoughts_tokens=(u.thoughts_token_count or 0) if u else 0,
            finish_reason=fin, extra=extra)


class FakeClient:
    """Cliente falso para testes: policy(system, user, seed, call_no) devolve o texto.
    Pode simular limite de uso (lista de chamadas que levantam RateLimited)."""
    real = False

    def __init__(self, policy, model="gemini-3.5-flash-lite", rate_limit_calls=(), tokens=(2000, 300, 0)):
        self.policy, self.model = policy, model
        self.temperature, self.thinking_level, self.service_tier = 1.0, None, None
        self.max_output_tokens = None
        self.calls = 0
        self.rate_limit_calls = set(rate_limit_calls)
        self.tokens = tokens

    def settings(self) -> dict:
        return {"model": self.model, "temperature": self.temperature, "thinking_level": None,
                "service_tier": None, "max_output_tokens": None, "fake": True}

    def generate(self, system, user, seed):
        self.calls += 1
        if self.calls in self.rate_limit_calls:
            raise RateLimited("429 simulado", retry_after=1)
        pt, ot, tt = self.tokens
        return CallResult(text=self.policy(system, user, seed, self.calls), model_version="fake-1",
                          response_id=f"fake-{self.calls}", prompt_tokens=pt, output_tokens=ot,
                          thoughts_tokens=tt, finish_reason="STOP")


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    """Teto de gasto em US$, somado entre execuções num arquivo JSON local.
    Para ANTES de uma chamada se o gasto + o custo máximo estimado dela passar do teto."""

    def __init__(self, path: Path, ceiling_usd: float):
        self.path, self.ceiling = Path(path), float(ceiling_usd)
        self.spent = 0.0
        self._reload()

    def _reload(self):
        """Relê o gasto do disco (outro terminal pode ter gastado); nunca diminui."""
        if self.path.is_file():
            self.spent = max(self.spent, float(json.loads(self.path.read_text(encoding="utf-8"))["spent_usd"]))

    def check(self, max_call_cost: float):
        self._reload()
        if self.spent + max_call_cost > self.ceiling:
            raise BudgetExceeded(
                f"teto de US$ {self.ceiling:.2f}: gasto {self.spent:.4f} + próxima chamada "
                f"até {max_call_cost:.4f} passaria do teto")

    def add(self, cost: float):
        self._reload()
        self.spent += cost
        tmp = self.path.with_suffix(".tmp")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps({"spent_usd": self.spent, "ceiling_usd": self.ceiling,
                                   "updated": dt.datetime.now(dt.timezone.utc).isoformat()}), encoding="utf-8")
        safe_replace(tmp, self.path)


def safe_replace(src: Path, dst: Path, tries: int = 20, sleep=time.sleep):
    """Troca atômica com novas tentativas: no Windows, um arquivo aberto (Excel, OneDrive,
    antivírus) bloqueia a troca por alguns instantes."""
    for k in range(tries):
        try:
            src.replace(dst)
            return
        except PermissionError:
            if k == tries - 1:
                raise
            sleep(0.5)


def call_with_backoff(client, system, user, seed, *, sleep=time.sleep, max_waits=60,
                      first_wait=10.0, max_wait=600.0, log=print):
    """Espera e tenta de novo ao atingir limite de uso (429) ou erro temporário (5xx).
    Essas esperas NÃO contam como resposta inválida. Depois de max_waits esperas seguidas,
    desiste (a execução para e pode ser retomada depois, por exemplo no dia seguinte)."""
    wait = first_wait
    for k in range(max_waits + 1):
        try:
            return client.generate(system, user, seed)
        except (RateLimited, TransientAPIError) as e:
            if k == max_waits:
                raise
            w = min(max_wait, getattr(e, "retry_after", None) or wait)
            log(f"  {type(e).__name__}: esperando {w:.0f} s ({k + 1}/{max_waits})")
            sleep(w)
            wait = min(max_wait, wait * 2)
