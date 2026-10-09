# CLAUDE.md — algorithmic-collusion

Contexto para o Claude Code. O planejamento, as decisões e o status do projeto ficam num documento no claude.ai que o Claude Code não acessa; este arquivo é o resumo estável dele. Se algo aqui contradizer o que a Beatriz disser, vale o que ela disser.

## O projeto
Replicação comparativa reprodutível: agentes que definem preços chegam sozinhos a preços acima do competitivo? Compara pares de agentes de **Q-learning** tabular (Calvano et al., 2020) e pares de agentes de **LLM** (protocolo de Fish et al.), cada par no seu próprio mercado (mercado A: 2 Q-learning; mercado B: 2 LLMs), num **duopólio de Bertrand repetido** com produtos diferenciados e demanda logit.

- **Contribuição:** a mesma métrica Δ e o mesmo teste de desvio forçado para os dois tipos de agente; Q-learning validado contra Calvano; LLM barato ainda não estudado (Gemini 3.5 Flash-Lite); código e protocolo abertos; simulador público. A comparação Q-learning × LLM em si já existe (Keppo et al., 2026): nunca descrever o projeto como "primeiro estudo".
- **Métrica principal:** Δ = (π̄ − πN) / (πM − πN), com πN ≈ 0,2229 e πM ≈ 0,3375 (lucros contínuos, não os da grade). 0 = Nash, 1 = monopólio. Secundária: % do preço médio sobre o preço de Nash (1,4729).
- **Base científica:** Calvano, Calzolari, Denicolò e Pastorello (2020, AER 110(10)); Schildknecht (2026, replicação em Python; pacote sem licença, usado só como referência para validar resultados); Fish, Gonczarowski e Shorrer (arXiv 2404.00806, v6); Keppo, Li, Tsoukalas e Yuan (arXiv 2603.20281); crítica principal: den Boer, Meylahn e Schinkel (Management Science, 2026).
- **Entregáveis:** simulador (HTML/JS), replicação do Q-learning (Python + numba), experimento com LLMs, análise estatística, artigo em inglês (8–12 páginas), README, divulgação.
- **Fora do escopo:** afirmar que empresas reais coludem; prever preços reais; deep RL; grade α × β completa; LLM × Q-learning no mesmo mercado e hub-and-spoke (trabalho futuro).

## Parâmetros do caso base (Calvano et al.; fonte e página na Ficha de parâmetros do documento)
| Parâmetro | Valor |
|---|---|
| n, c, a, a₀, μ | 2; 1; 2; 0; 0,25 |
| δ (desconto; é o γ do roteiro de estudo) | 0,95 |
| Grade de preços | m = 15 preços igualmente espaçados em [pN − ξ(pM − pN), pM + ξ(pM − pN)], ξ = 0,1: de 1,4277 a 1,9702 |
| Memória | k = 1: estado = preços das duas firmas no período anterior (225 estados × 15 ações) |
| α, β | 0,15; 4 × 10⁻⁶, com ε = e^(−βt) para cada agente |
| Q inicial | lucro descontado contra rival que sorteia preços uniformemente |
| Desempate | menor preço |
| Convergência | ação ótima de cada agente em cada estado inalterada por 100.000 períodos; teto de 10⁹ |
| Sessões | 1.000, sementes 1 a 1.000 |
| Referências | Nash 1,4729 (lucro 0,2229); monopólio conjunto 1,9250 (0,3375); monopólio de 1 firma 1,802 |
| Alvos da replicação | Δ 0,849 (DP 0,112); fração em Nash 0,505; reação do rival −0,127; desvio não lucrativo 0,936; punição ~5,7 períodos |

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
- LLM: API Gemini, `gemini-3.5-flash-lite` (principal); `gemini-3.8-flash` só no teste de competência e como reserva. Temperatura 1.

## Regras de implementação
- **Código de partida do Q-learning:** Courthoud, Algorithmic-Collusion-Replication (MIT, © 2021 Matteo Courthoud). Corrigir a grade (ξ = 0,1), a convergência (contar 1 vez por período) e o teto (10⁹, lido da chave certa); escrever Δ, várias sessões, sementes, salvamento e teste de desvio. Manter o aviso MIT em `THIRD_PARTY_LICENSES.md` e um comentário de origem no topo dos arquivos derivados. Nada é copiado do pacote de Schildknecht (sem licença).
- **Sessão de Q-learning:** compilada com numba (`@njit(cache=True)`) num arquivo de `qlearning/` importado pelo notebook; semente passada como argumento e aplicada dentro da função compilada; guardar a estratégia gulosa e recalcular só a linha da Q que mudou; todos os parâmetros como argumentos (nada global); medir o tempo sem a 1ª chamada (compilação).
- **Várias sessões:** paralelizar com `prange`, não com `multiprocessing`; rodar duas vezes e conferir que os resultados são idênticos.
- **Δ do Q-learning:** após a convergência, Q congelada e ε = 0; jogar a estratégia gulosa até o estado se repetir e tirar a média do lucro das duas firmas sobre um ciclo completo.
- **Sessões que não convergem:** contar, reportar e incluir com a média dos últimos 100.000 períodos.
- **Teste de desvio (igual para Q-learning e LLM):** ramo de controle sem desvio e ramo em que a firma 1 desvia 1 período para a melhor resposta estática ao preço do rival; 15 períodos; Q congelada e ε = 0. R = [p₂ desvio(2) − p₂ controle(2)] / p₂(0). G = soma de 0,95^(τ−1) × [π₁ desvio(τ) − π₁ controle(τ)] dividida pela soma de 0,95^(τ−1) × π₁ controle(τ), com τ de 1 a 15. V = primeiro τ ≥ 2 a partir do qual os dois preços ficam a até 1 degrau da grade (0,0387) do controle por 3 períodos seguidos (censurado em 15). Classes com θ = 2%: A (R ≤ −θ, G < 0, V ≤ 15), B (R ≤ −θ, V censurado), C (R > −θ). Em ciclos, desviar a partir de cada ponto do ciclo e contar a média como 1 observação.
- **LLM:** chave só em `.env`, **nunca** no Git; registrar em cada chamada prompt, resposta, preço extraído, tokens, modelo, versão devolvida, data, temperatura e seed; até 10 novas tentativas por resposta inválida, depois a execução para e é registrada; retomada do ponto onde parou; contador de custo que para antes do teto.
- **Simulador:** **nunca** chama a API do LLM; mostra resultados pré-calculados em JSON. Motor em JavaScript puro num Web Worker, com gerador aleatório com semente.

## Regras gerais
- Toda simulação tem **semente registrada**; todo número do artigo vem de um arquivo em `results/`.
- **Nenhum parâmetro sem fonte**: valores vêm da Ficha de parâmetros do documento, nunca de memória.
- Protocolo e número de execuções fixados antes da 1ª execução completa; pilotos não entram na análise.
- Código de terceiros só com licença lida e créditos no README.
- Em README, artigo e comentários: nunca "IA forma cartel", "provamos colusão" ou "primeiro estudo"; usar "preços acima do competitivo" e "compatível com colusão tácita" (só com os testes favoráveis).
- Commit ao fim de cada sessão de trabalho.
- A Beatriz não é programadora profissional: código claro e comentado, passos explicados quando ela pedir.
