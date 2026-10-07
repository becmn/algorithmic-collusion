# CLAUDE.md — algorithmic-collusion

Contexto para o Claude Code. O planejamento e as decisões do projeto ficam num documento no claude.ai que o Claude Code não acessa; este arquivo é o resumo dele. Se algo aqui contradizer o que a Beatriz disser, vale o que ela disser.

## O projeto
Experimento de portfólio: IAs de precificação aprendem sozinhas a manter preços acima do competitivo (colusão tácita)? Compara agentes de **Q-learning** (tabular) e agentes de **LLM** num **duopólio de Bertrand repetido** com produtos diferenciados e demanda logit (Calvano et al., 2020).

- **Métrica principal:** índice de ganho de lucro Δ = (π − πN) / (πM − πN). 0 = lucro competitivo (Nash), 1 = lucro de monopólio.
- **Base científica:** Calvano, Calzolari, Denicolò e Pastorello (2020, AER 110(10)); replicação em Python de Schildknecht (2026); Fish, Gonczarowski e Shorrer, *Algorithmic Collusion by Large Language Models* (arXiv 2404.00806).
- **Entregáveis:** simulador interativo no navegador (HTML/JS), experimento de Q-learning (Python), experimento com LLMs, análise comparativa, artigo curto, README, divulgação.
- **Fora do escopo:** afirmar que empresas reais coludem; prever preços reais; deep RL.

## Estrutura de pastas
| Pasta | Conteúdo |
|---|---|
| `qlearning/` | Simulações de Q-learning |
| `llm/` | Agentes de LLM |
| `analysis/` | Notebooks de análise |
| `results/` | Dados gerados (resumos por sessão; trajetórias completas só de algumas sessões) |
| `docs/` | Simulador web (pasta publicada pelo GitHub Pages) |
| `paper/` | Artigo e figuras |

## Ambiente
- Windows, PowerShell, Python 3.11, ambiente virtual em `venv/`.
- Pacotes: numpy, pandas, scipy, matplotlib, numba, jupyter (`requirements.txt`).
- LLM (proposta, a confirmar após piloto): API Gemini, modelo Flash-Lite.

## Regras
- Toda simulação tem **semente registrada**; todo número do artigo vem de um arquivo em `results/`.
- **Nenhum parâmetro sem fonte**: valores vêm da ficha de parâmetros (artigo + página), nunca de memória.
- Chave da API só em `.env`, **nunca** no Git. O simulador web **nunca** chama a API do LLM; mostra resultados pré-calculados em JSON.
- Commit ao fim de cada sessão de trabalho.
- A Beatriz não é programadora profissional: código claro e comentado, passos explicados quando ela pedir.

## Status
Fase 0 (preparação): ambiente configurado, repositório criado. Próximo: conceitos, leitura dirigida dos artigos, ficha de parâmetros e localizar o código de Schildknecht (conferir licença).