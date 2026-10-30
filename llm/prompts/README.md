# Textos dos prompts (Fase 3)

Os textos exatos de Fish, Gonczarowski e Shorrer (arXiv 2404.00806, v6, Apêndice G) **não ficam no
repositório público**: o artigo tem a licença padrão do arXiv (não Creative Commons). A decisão em
aberto do projeto é conferir a licença ou pedir permissão aos autores antes de publicá-los
(documento do projeto, Decisões em aberto; Pesquisas 2 e 11). Por isso eles ficam na pasta
`llm/prompts/local/`, que o `.gitignore` deixa fora do Git.

`llm/prompts/test/` tem textos **próprios do projeto**, só para os testes sem API. O programa não
deixa usá-los com o Gemini de verdade.

## Como preencher `llm/prompts/local/` (uma vez)

1. Abra o PDF da v6 do artigo (https://arxiv.org/abs/2404.00806v6) no Apêndice G.
2. Crie a pasta `llm/prompts/local/` e, dentro dela, os arquivos abaixo (UTF-8, no VS Code):

| Arquivo | O que colar | De onde |
|---|---|---|
| `P0.txt` | o prefixo P0, inteiro | G.1 |
| `P1.txt` | o prefixo P1, inteiro (P0 + a frase do P1) | G.1 |
| `P2.txt` | o prefixo P2, inteiro (P0 + a frase do P2) | G.1 |
| `template.txt` | o modelo geral do prompt, **sem** o prefixo | G.2.1 |
| `template_monopoly.txt` | (opcional) o modelo do monopolista, se o artigo tiver um diferente | Apêndice C / G |
| `format.json` | rótulos da resposta e formato de cada linha do histórico | G.2.1 e exemplos G.2.2–G.2.3 |

3. No `template.txt`, troque cada valor variável por um marcador (as chaves fazem parte do marcador):

| Marcador | Onde fica no modelo de Fish |
|---|---|
| `{cost}` | o custo por unidade (só o número; o `$` fica no texto) |
| `{ceiling}` | o preço máximo que um cliente pagaria (só o número) |
| `{plans}` | o conteúdo de PLANS.txt (entre as linhas de `+`) |
| `{insights}` | o conteúdo de INSIGHTS.txt (entre as linhas de `+`) |
| `{market_data}` | os dados de mercado dos últimos períodos |

4. `format.json` (copie o de `llm/prompts/test/format.json` e troque os valores):
   - `labels`: o texto exato de cada um dos 4 campos que o modelo deve preencher, na ordem
     raciocínio (`observations`), `plans`, `insights` e `price`, como aparecem no G.2.1;
   - `row_duopoly`: uma linha (ou bloco) de um período, como nos exemplos G.2.2–G.2.3, com os
     marcadores `{round}`, `{my_price}`, `{rival_price}`, `{qty}` e `{profit}`;
   - `row_monopoly`: o mesmo sem o preço do concorrente;
   - `row_separator`: o que separa dois períodos (`"\n"` = quebra de linha);
   - `empty_history`: o que aparece no período 1, sem dados.
5. Confira: `python llm/run.py preview --prompt P1` (e `--prompt P0`) mostra o prompt montado nos
   períodos 1 e 3. Compare com o artigo, linha por linha. Nenhuma chamada à API é feita.

O hash SHA-256 de cada texto usado fica gravado em `config.json` de cada execução: prova qual texto
foi usado sem publicá-lo.
