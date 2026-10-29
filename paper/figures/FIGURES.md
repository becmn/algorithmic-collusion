# Figuras e tabelas do artigo

Plano decidido em 13/10/2026 (Fase 1.6) e revisto em 16/10/2026 (revisão da Fase 1): o teste de separação (Fase 1.8) passa a ser a figura 4 do texto principal, porque é o teste que decide a leitura dos preços altos; a figura dos testes obrigatórios vai para o apêndice e continua no texto principal como tabela. Todas as figuras são geradas por código a partir de `results/`; nenhuma é copiada de outro artigo.

## Texto principal (no máximo 4 figuras)

| Nº | Arquivo | Conteúdo | Script | Na Fase 4 |
|---|---|---|---|---|
| 1 | `fig_delta_distribution` | Distribuição de Δ entre sessões | `analysis/fig_delta_base.py` | Vira Δ por sessão/execução, com média e IC, por agente e prompt |
| 2 | `fig_learning_trajectory` | Trajetória de Δ em eixo logarítmico, com o período 300 marcado | `analysis/fig_trajectories_deviation.py` | Recebe as trajetórias dos LLMs |
| 3 | `fig_deviation_response` | Preços após o desvio forçado e reação do rival em relação ao controle | `analysis/fig_trajectories_deviation.py` | Curvas do LLM lado a lado |
| 4 | `fig_separation` | Δ com e sem memória sob a mesma exploração (teste de separação, Fase 1.8) | `analysis/fig_separation.py` | Sem mudança |

Tabelas no texto principal: `paper/tables/table_mandatory_tests.tex` (testes obrigatórios, Fase 1.5) e `paper/tables/table_separation.tex` (teste de separação, Fase 1.8).

## Apêndice de robustez

| Arquivo | Conteúdo | Script |
|---|---|---|
| `fig_robustness_tests` | Testes obrigatórios: Δ, Nash (1 período e qualquer desvio) e classe A por condição | `analysis/fig_tests_robustness.py` |
| `fig_delta_vs_periods` | Δ × períodos até convergir (componente oficial de 10/10/2026) | `analysis/fig_delta_base.py` |
| `fig_price_trajectories` | Preços de 3 sessões (Δ mais próximo dos percentis 25, 50 e 75) | `analysis/fig_trajectories_deviation.py` |
| `fig_memoryless_discount` | Δ sem memória por fator de desconto δ | `analysis/fig_tests_robustness.py` |

As trajetórias dos LLMs, se feitas, vão para o apêndice.
