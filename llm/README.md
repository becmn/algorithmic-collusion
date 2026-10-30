# Fase 3 — Agentes de LLM

Protocolo: documento do projeto, seção "Protocolo experimental e plano estatístico — Agentes de LLM"
(Pesquisas 2, 7 e 8). Resumo: dois agentes Gemini (`gemini-3.5-flash-lite`), temperatura 1, 300
períodos, prompts P1 e P2 de Fish et al., escalas α ∈ {1; 3,2; 10}, Δ nos períodos 251–300, teste de
desvio comum com controle e placebo. O teste de competência (monopolista, P0) vem antes.

## Arquivos

| Arquivo | O que faz |
|---|---|
| `market.py` | mercado com as escalas de Fish (α, β = 100), Δ, estabilização, teste de competência, métricas do desvio |
| `prompts.py` | lê os textos de `prompts/local/`, monta o prompt (histórico de 100 períodos, notas) e lê a resposta |
| `client.py` | cliente do Gemini (SDK `google-genai`), custo por chamada, teto de gasto, espera em limite de uso |
| `experiment.py` | execuções: salva a cada período (retomada), log de cada chamada, novas tentativas, desvio |
| `run.py` | linha de comando (abaixo) |
| `test_llm.py` | 34 conferências sem API (roda no CI) |

## Passo a passo (no seu computador, PowerShell, na raiz do repositório, venv ativo)

1. Instale o SDK: `pip install -r requirements-llm.txt`
2. Crie a chave no Google AI Studio e grave no arquivo `.env` da raiz: `GEMINI_API_KEY=sua_chave`
   (o `.env` está no `.gitignore`; nunca cole a chave no chat nem no código).
3. Cole os textos de Fish em `llm/prompts/local/` (ver `llm/prompts/README.md`) e confira com
   `python llm/run.py preview --prompt P1` e `--prompt P0`.
4. Teste de competência (piloto; 900 chamadas por modelo):
   `python llm/run.py competence --model gemini-3.5-flash-lite --budget 5`
   e, para comparação, `--model gemini-3.8-flash`.
5. Duopólio curto (piloto): `python llm/run.py pilot --prompt P1 --alpha 1 --periods 30 --budget 5`
   (mostra chamadas, respostas inválidas, tokens e latência; anote o limite diário da conta no AI Studio).
6. Com os resultados do piloto, decidir e registrar em Decisões: número de execuções (21; mínimo 12),
   teto de gasto, nível de raciocínio e modelo. Só então: `python llm/run.py plan --runs-per-prompt 21`
   e commit do plano (protocolo fixado antes da 1ª execução completa).
7. Experimento: `python llm/run.py run --budget <teto>` (roda cada execução e o desvio dela; pode
   parar e voltar: é só rodar o mesmo comando). `python llm/run.py status` mostra onde está.
8. `python llm/run.py summarize` gera `results/llm_runs.csv` (uma linha por execução).

Sem chave, `--fake` roda tudo com um cliente falso e os prompts de teste (grava em `results/llm_fake/`,
fora do Git): serve só para ver o programa funcionando.

## Regras que o código garante

- Até 10 novas tentativas por resposta inválida; depois a execução para e é registrada com o motivo.
- Limite de uso (429) e erro do servidor: espera e tenta de novo, sem contar como resposta inválida.
- Teto de gasto: o contador soma o custo a preço pago (mesmo na camada gratuita) e para **antes** da
  chamada que passaria do teto. O teto da conta (Prepay sem recarga e teto do projeto no AI Studio)
  é a 2ª e a 3ª camada (Pesquisa 7).
- Log de cada chamada: prompt, resposta, preço extraído, tokens (entrada, saída e raciocínio),
  modelo, versão devolvida, data, temperatura, seed, nível de raciocínio e custo.
- Pilotos ficam em `results/llm_pilot/` e não entram na análise.
- Teto anunciado: 4,51 × α (Fish, G.2.2; Protocolo). Observação: 2,34 × pM dá 4,50; o 4,51 publicado
  sugere um multiplicador de ≈ 2,343 [I].
