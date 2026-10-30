/*
 * Textos do simulador em português e inglês (Fase 2.2).
 * Toda frase visível da página está aqui; o HTML só tem as chaves (data-i18n).
 * {nome} é trocado por um valor na hora de mostrar (números já formatados no idioma).
 * Regra do projeto: o resultado do Q-learning nunca é chamado de "colusão" (Fase 1.8).
 * test_i18n.js confere que as duas línguas têm as mesmas chaves e os mesmos {nomes}.
 * Código próprio do projeto.
 */
(function (root) {
  "use strict";

  const LINKS = {
    repo: "https://github.com/becmn/algorithmic-collusion",
    calvano: "https://www.aeaweb.org/doi/10.1257/aer.20190623",
    denboer: "https://doi.org/10.1287/mnsc.2024.08557",
    cade: "https://www.gov.br/cade/pt-br/assuntos/noticias/cade-celebra-acordo-em-investigacao-sobre-uso-de-algoritmo-de-precificacao",
    lefosse: "https://lefosse.com/newsletter/o-uso-de-algoritmos-de-precificacao-no-radar-do-cade-intelprice-celebra-tcc-no-caso-do-software-aprix/",
    realpage: "https://www.fenwick.com/insights/publications/dojs-realpage-settlement-a-blueprint-for-safer-algorithmic-pricing",
  };
  const a = (href, text) => `<a href="${href}" rel="noopener" target="_blank">${text}</a>`;

  const STR = {
    pt: {
      doc_title: "Preços algorítmicos · simulador de Q-learning",
      doc_description: "Simulador no navegador: dois algoritmos de preço que nunca se comunicaram aprendem a cobrar acima do preço competitivo (duopólio de Calvano et al., 2020).",
      skip_link: "Pular para o conteúdo",
      site_name: "Preços algorítmicos",
      nav_deviation: "Teste do desvio",
      dek: "Mas, sem memória, os mesmos algoritmos cobraram preços ainda mais altos. Este simulador mostra as duas coisas, com os parâmetros do artigo original.",
      nav_results: "Resultados",
      author_name: "Beatriz Cury Mac Nicol",
      author_affil: "Administração, Insper",
      author_updated: "Atualizado em {date} · código e dados abertos",
      fig_1: "Figura 1.",
      fig_2: "Figura 2.",
      fig_3: "Figura 3.",
      sep_rule_note: "A regra que definiu esta frase foi escrita antes de rodar o teste: a expressão “compatível com colusão tácita” só seria usada se os algoritmos com memória ficassem acima dos sem memória.",
      tab1_h2: "Resultados principais e o artigo original",
      tab1_caption: "Tabela 1. Caso base, {nsess} sessões (sementes 1 a {nsess})",
      th_measure: "Medida",
      th_project: "Este projeto",
      th_paper: "Calvano et al. (2020)",
      row_delta: "Δ médio (índice de lucro)",
      row_nash: "Sessões em equilíbrio de Nash no caminho (só desvios de 1 período, leitura do artigo)",
      row_nash_any: "Sessões em equilíbrio de Nash no caminho (qualquer desvio)",
      row_react: "Reação da rival no período seguinte ao corte",
      row_unprof: "Cortes de preço que não compensaram",
      row_punish: "Períodos até os preços voltarem (média das sessões em que voltaram)",
      row_memoryless: "Δ médio sem memória, com a mesma exploração",
      not_reported: "não reportado",
      tab1_note: "A diferença em cortes que não compensaram ({proj} contra {paper}) não foi explicada: o artigo não descreve a conta exata. Na linha “qualquer desvio”, a empresa pode desviar por qualquer número de períodos (melhor resposta ótima); com essa regra mais exigente, {anyn} das {n} sessões ficam em equilíbrio de Nash no caminho. Todo número da coluna “Este projeto” vem de um arquivo em results/ no repositório; a outra coluna vem do artigo.",
      about_h2: "Sobre este projeto",
      about_p1: "Este simulador é a primeira parte de um projeto que replica o experimento de Calvano et al. (2020) e, na próxima fase, aplica as mesmas medidas, o índice Δ e o teste do desvio, a agentes de linguagem (LLMs).",
      about_p2: "As escolhas de método foram registradas antes de rodar os testes, inclusive a regra que decidiu a frase da conclusão. Código, dados e testes automáticos estão no repositório.",
      about_stack: "Python e numba para as {nsess} sessões · JavaScript no navegador, validado contra o Python.",
      footer_fonts: "Fontes Newsreader e Public Sans (licença SIL OFL).",
      nav_method: "Método",
      nav_code: "Código",
      lang_switch_text: "English",
      lang_switch_aria: "Ler esta página em inglês",

      kicker: "Replicação · Calvano, Calzolari, Denicolò e Pastorello (2020)",
      title: "Dois algoritmos de preço que nunca se comunicaram aprenderam a cobrar acima do preço competitivo",
      subtitle: "Cada linha é o preço de uma empresa controlada por um algoritmo que aprende sozinho, por tentativa e erro. Perto da linha de baixo é concorrência; perto da de cima, monopólio.",
      badge_paper: "Parâmetros do artigo",
      badge_paper_k0: "Parâmetros do artigo, sem memória",
      mem_switch_label: "Algoritmos",
      mem_with: "Com memória",
      mem_without: "Sem memória",
      mem_note: "Sem memória, cada algoritmo escolhe o preço sem saber o que a rival cobrou antes. Esta é uma sessão gravada no Python, com os parâmetros do artigo; a sessão ao vivo existe só com memória.",
      source_recorded_k0: "Sessão sem memória gravada no Python · semente {seed}",
      live_only_memory: "A sessão ao vivo existe só com memória.",
      aggregate_k0: "Em {n} sessões sem memória com os parâmetros do artigo, o Δ médio foi {mean} (IC 95%: {lo} a {hi}).",
      dev_mem_note: "Sem memória, B não sabe que A cortou o preço: não tem como reagir, e os dois voltam ao preço de antes já no período seguinte.",

      chart_learning_title: "Preço de cada empresa ao longo do aprendizado",
      chart_learning_desc: "Gráfico de linhas: preço médio das empresas A e B em blocos de {block} períodos, com as linhas do preço competitivo e do preço de monopólio.",
      chart_learning_note: "Cada ponto é a média de um bloco de {block} períodos.",
      axis_periods_millions: "Períodos (milhões)",
      axis_price: "Preço",
      line_nash: "Preço competitivo (Nash)",
      line_monopoly: "Preço de monopólio",
      firm_a: "Empresa A",
      firm_b: "Empresa B",

      delta_label: "Índice de lucro Δ desta sessão",
      delta_mean_ref: "Média de {n} sessões: {mean} (traço pontilhado na régua)",
      delta_scale: "0 = lucro competitivo · 1 = lucro de monopólio",
      delta_final: "depois da convergência",
      delta_during: "durante o aprendizado (estratégias ainda mudando)",
      source_recorded: "Sessão gravada no Python · semente {seed}",
      source_live: "Ao vivo no seu navegador · semente {seed}",
      outcome_constant: "Resultado final: as duas empresas mantêm um preço constante.",
      outcome_cycle: "Resultado final: os preços se repetem num ciclo de {n} períodos.",
      compute_time: "Calculado em {s} s no seu navegador.",
      million_one: "milhão",
      million_many: "milhões",

      status_converged_mid: "As duas empresas pararam de mudar de estratégia depois de {periods} {mi} de períodos. O lucro ficou em Δ = {delta}: acima do nível competitivo e abaixo do de monopólio.",
      status_converged_high: "As duas empresas pararam de mudar de estratégia depois de {periods} {mi} de períodos. O lucro ficou em Δ = {delta}, perto do lucro de monopólio.",
      status_converged_low: "As duas empresas pararam de mudar de estratégia depois de {periods} {mi} de períodos. O lucro ficou em Δ = {delta}, perto do lucro competitivo.",
      status_not_converged: "A sessão parou em {periods} {mi} de períodos sem convergir; Δ = {delta} é a média dos últimos {stable} períodos.",
      status_learning: "Aprendendo: período {t}. As empresas ainda testam um preço ao acaso em {eps}% das vezes.",
      status_paused: "Pausado no período {t}.",
      status_computing: "Calculando no seu navegador…",
      status_deviation_done: "Corte de preço aplicado: veja a reação no painel “O teste do desvio”.",
      status_worker_error: "Não foi possível rodar a simulação neste navegador. O resultado gravado continua disponível.",

      btn_run: "Reproduzir o aprendizado",
      btn_pause: "Pausar",
      btn_resume: "Continuar",
      btn_deviate: "Forçar corte de preço",
      btn_new: "Nova sessão ao vivo",
      btn_deviate_hint: "Disponível depois que as empresas param de aprender.",
      controls_label: "Controles da simulação",

      aggregate: "Em {n} sessões com os parâmetros do artigo, o Δ médio foi {mean} (IC 95%: {lo} a {hi}), em que 0 é o lucro competitivo e 1, o de monopólio.",
      data_table_summary: "Ver os dados do gráfico",
      chart_hint: "Passe o mouse ou toque nos gráficos para ver os valores exatos.",
      tip_periods: "Períodos {a} a {b}",
      tip_tau: "Período {k} depois do corte",
      tip_mean: "Δ médio",
      tip_ci: "IC 95%",
      th_periods: "Período (milhões)",
      th_price_a: "Preço A",
      th_price_b: "Preço B",
      th_delta_block: "Δ do bloco",

      dev_h2: "O teste do desvio",
      dev_intro: "E se uma empresa cortar o preço? Depois que as duas param de aprender, a empresa A é forçada a cobrar, por um único período, o preço que mais lucra contra o preço atual da B. Depois disso, as duas voltam a seguir o que aprenderam. A linha cinza mostra o que teria acontecido sem o corte.",
      dev_cycle_note: "Nesta sessão os preços se repetem num ciclo; o corte parte do primeiro ponto do ciclo.",
      chart_dev_title: "Preços antes e depois do corte",
      chart_dev_desc: "Gráfico em degraus: preços das empresas A e B nos {horizon} períodos depois do corte forçado, comparados com o caminho sem corte.",
      axis_tau: "Períodos depois do corte (τ)",
      line_control: "Sem corte",
      th_tau: "τ",
      th_a_cut: "A, com corte",
      th_b_cut: "B, com corte",
      th_a_nocut: "A, sem corte",
      th_b_nocut: "B, sem corte",
      ann_cut: "A corta o preço",
      ann_react: "B reage",
      ann_noreact: "B não reage",
      ann_back: "Voltam ao padrão",
      score_react: "Reação de B no período seguinte",
      score_react_value: "{x}% no preço",
      score_back: "Períodos até os preços voltarem",
      score_back_none: "não voltaram em {horizon}",
      score_profit: "Lucro de A em {horizon} períodos (descontado)",
      score_profit_value: "com corte {d} · sem corte {c}",
      verdict_unprofitable: "O corte não compensou: A lucrou {g}% a menos do que sem o corte.",
      verdict_profitable: "O corte compensou: A lucrou {g}% a mais do que sem o corte.",
      verdict_none: "Nesta sessão, A já cobrava o preço que mais lucra contra a B: não houve corte de fato.",
      dev_aggregate: "Nas {n} sessões, a rival reagiu e os preços voltaram em {a}% delas, e o corte não compensou em {u}%.",
      dev_caveat: "Esse padrão de reação e volta é o que a literatura chama de punição. Sozinho, ele não prova colusão, e essa leitura é contestada ({ref}).",
      ref_denboer: "den Boer, Meylahn e Schinkel, 2026",

      sep_h3: "A punição explica os preços altos?",
      sep_predict: "O teste: algoritmos sem memória não sabem o que a rival cobrou ontem, então não conseguem punir um corte de preço. Se a ameaça de punição explicasse os preços altos, sem memória o lucro deveria cair.",
      sep_main_title: "Lucro com e sem memória, com os parâmetros do artigo ({nsess} sessões cada)",
      sep_main_desc: "Barras horizontais: Δ médio com memória {k1} (IC 95%: {k1lo} a {k1hi}) e sem memória {k0} (IC 95%: {k0lo} a {k0hi}). 0 é o lucro de concorrência e 1, o de monopólio.",
      sep_result: "Resultado: aconteceu o contrário. Sem memória, o lucro ficou ainda mais perto do monopólio (Δ {k0} contra {k1}). O mesmo aconteceu nas {k} formas de exploração testadas: a possibilidade de punir não eleva os preços.",
      sep_more: "Ver as {k} formas de exploração testadas",
      axis_competition: "Concorrência",
      axis_monopoly: "Monopólio",
      th_param_tech: "Parâmetro técnico",
      sep_conclusion: "Conclusão: preços acima do competitivo; na maioria das sessões, um corte forçado foi seguido de reação da rival e não compensou. Mas não é possível descartar que parte dos preços altos venha de aprendizado incompleto, e a possibilidade de punir não eleva os preços sob a mesma exploração.",
      chart_sep_title: "Lucro (Δ) com e sem memória em cada forma de exploração, {nsess} sessões cada",
      chart_sep_desc: "Gráfico de pontos: para cada nível de exploração, o Δ médio dos algoritmos com memória e sem memória, com intervalo de confiança de 95%.",
      sep_with: "Com memória (podem punir)",
      sep_without: "Sem memória (não podem punir)",
      sep_note: "Cada linha muda por quanto tempo os algoritmos testam preços ao acaso (a exploração). Ponto azul: com memória; quadrado laranja: sem memória; o traço em volta é o intervalo de confiança de 95%. Mais à direita, lucro mais perto do monopólio. Em todas as linhas, o quadrado fica à direita do ponto.",
      cond_beta1e_05: "Aprendizado curto",
      cond_beta4e_06: "Parâmetros do artigo",
      cond_beta2e_06: "Aprendizado longo",
      cond_beta1e_06: "Aprendizado muito longo",
      cond_epsmin0_001: "Exploração que nunca para (0,1%)",
      cond_epsmin0_01: "Exploração que nunca para (1%)",
      cond_epsmin0_05: "Exploração que nunca para (5%)",
      th_condition: "Condição",
      th_with: "Δ com memória",
      th_without: "Δ sem memória",

      why_h2: "Por que isso importa",
      why_p1: "Empresas já usam algoritmos para definir preços no comércio eletrônico, em postos de combustível, em aluguéis e em passagens aéreas, e autoridades de concorrência do Brasil, dos Estados Unidos e da Europa investigam esse uso.",
      why_p2: "Mas quase todos os casos reais são de outro tipo: concorrentes que usam o mesmo fornecedor de software, que reúne dados de todos. Este experimento testa uma situação diferente: algoritmos que aprendem sozinhos, sem trocar informação.",
      why_table_caption: "Dois mecanismos diferentes",
      why_col_tacit: "Algoritmos que aprendem sozinhos (o que este experimento testa)",
      why_col_hub: "Hub-and-spoke (o mecanismo da maioria dos casos reais)",
      why_row_how: "Como funciona",
      why_how_tacit: "Cada empresa tem o próprio algoritmo e só vê os preços do mercado.",
      why_how_hub: "Um intermediário, como o fornecedor do software, reúne dados de vários concorrentes.",
      why_row_info: "Troca de informação",
      why_info_tacit: "Nenhuma.",
      why_info_hub: "Sim, por meio do intermediário.",
      why_row_examples: "Exemplos",
      why_examples_tacit: "Experimentos de laboratório, como o de Calvano et al. (2020).",
      why_examples_hub: "Aprix (CADE, Brasil) e RealPage (Departamento de Justiça dos EUA).",
      why_aprix: `No caso Aprix, um software de preços para postos de combustível, o CADE homologou em 8/4/2026 um Termo de Compromisso de Cessação com a Intelprice. Segundo a ${a(LINKS.cade, "nota do CADE")}, havia suspeita de que a ferramenta pudesse promover “coordenação e alinhamento de preços”; o acordo prevê confidencialidade de dados sensíveis e segregação de preços e estratégias dos clientes. Segundo o escritório ${a(LINKS.lefosse, "Lefosse")}, o software usava dados compartilhados pelos postos que o utilizavam. Acordo não é condenação.`,
      why_realpage: `Nos Estados Unidos, o acordo de 24/11/2025 entre o Departamento de Justiça e a RealPage, um software de aluguéis, restringe o uso de dados não públicos de concorrentes (${a(LINKS.realpage, "resumo do escritório Fenwick")}).`,
      why_not: "O que este experimento não mostra: que empresas reais coludem, nem que preços elas cobram.",

      method_h2: "Método, dados e limitações",
      method_open: "Ver método, parâmetros e limitações",
      method_model: "Modelo: duopólio de Bertrand repetido, com produtos diferenciados e demanda logit (Calvano et al., 2020, eq. 5). Cada empresa usa Q-learning com memória de um período: escolhe o preço de hoje olhando os dois preços de ontem.",
      method_params_caption: "Parâmetros (todos do artigo; a fonte indica a página)",
      th_param: "Parâmetro",
      th_symbol: "Símbolo",
      th_value: "Valor",
      th_source: "Fonte",
      p_n: "Número de empresas",
      p_c: "Custo marginal",
      p_a: "Qualidade do produto",
      p_a0: "Bem externo (não comprar)",
      p_mu: "Diferenciação dos produtos",
      p_delta: "Fator de desconto (peso do futuro)",
      p_alpha: "Taxa de aprendizado",
      p_beta: "Decaimento da exploração, ε = e^(−βt)",
      p_k: "Memória (períodos)",
      p_m: "Preços possíveis na grade",
      p_xi: "Extensão da grade além de Nash e monopólio",
      p_conv: "Convergência",
      p_conv_value: "estratégia sem mudar por {stable} períodos (teto de {tmax})",
      p_sessions: "Sessões",
      p_sessions_value: "{n} (sementes 1 a {n})",
      method_measure: "Medida: Δ = (π̄ − πN) / (πM − πN), em que π̄ é o lucro médio por empresa depois da convergência, πN = {piN} é o lucro competitivo (Nash) e πM = {piM} é o lucro de monopólio. Preços de referência: Nash {pN}; monopólio {pM}.",
      method_engine: "No navegador, o mesmo algoritmo roda em JavaScript com outro gerador de números aleatórios, então a mesma semente não repete a sessão do Python. O motor foi validado contra o Python pela distribuição de Δ, e o teste do desvio foi conferido ponto a ponto nas sessões gravadas (testes no repositório).",
      lim_h3: "Limitações",
      lim_1: "Mercado simulado e simétrico, com {m} preços possíveis.",
      lim_2: "Os resultados dependem do fator de desconto, da exploração e da forma como o algoritmo atualiza o que aprendeu.",
      lim_3: "Preços sorteados ao acaso na grade já dariam Δ ≈ {rand}: Δ alto, sozinho, não indica estratégia.",
      lim_4: "O teste do desvio é evidência indireta, e a sua leitura é contestada.",
      lim_5: "Algoritmos sem memória chegam a lucros ainda mais altos: parte dos preços altos pode vir de aprendizado incompleto.",
      lim_6: "Não há uso documentado de Q-learning tabular para definir preços em empresas.",
      data_h3: "Dados",
      data_params: "Parâmetros, grade de preços e tabela de lucros (JSON)",
      data_reference: "Resultados de referência das {nsess} sessões (JSON)",
      data_sessions: "{nrec} sessões completas, com o teste do desvio (JSON)",
      data_repo: "Resultados completos e código-fonte (GitHub)",
      cite_h3: "Como citar",
      cite_text: "Mac Nicol, Beatriz Cury (2026). Preços algorítmicos: simulador de Q-learning no duopólio de Calvano et al. (2020). Código e dados: {repo}",
      cite_original: "Artigo original: Calvano, E., Calzolari, G., Denicolò, V. e Pastorello, S. (2020). Artificial Intelligence, Algorithmic Pricing, and Collusion. American Economic Review, 110(10), 3267–3297.",
      updated: "Dados atualizados em {date}.",

      footer_author: "Feito por Beatriz Cury Mac Nicol.",
      footer_license: "Código sob licença MIT.",
      footer_credits: "Modelo e parâmetros: Calvano, Calzolari, Denicolò e Pastorello (2020). O laço de Q-learning foi adaptado do código de Matteo Courthoud (licença MIT).",
      footer_no_ai: "Esta página não chama nenhum serviço externo: tudo roda no seu navegador ou vem de resultados gravados.",
    },

    en: {
      doc_title: "Algorithmic pricing · Q-learning simulator",
      doc_description: "In-browser simulator: two pricing algorithms that never communicated learn to charge above the competitive price (the Calvano et al., 2020 duopoly).",
      skip_link: "Skip to content",
      site_name: "Algorithmic pricing",
      nav_deviation: "Deviation test",
      dek: "But without memory, the same algorithms charged even higher prices. This simulator shows both, with the original paper's parameters.",
      nav_results: "Results",
      author_name: "Beatriz Cury Mac Nicol",
      author_affil: "Business Administration, Insper",
      author_updated: "Updated {date} · open code and data",
      fig_1: "Figure 1.",
      fig_2: "Figure 2.",
      fig_3: "Figure 3.",
      sep_rule_note: "The rule that set this sentence was written before the test was run: the phrase “consistent with tacit collusion” would be used only if the algorithms with memory ended up above those without it.",
      tab1_h2: "Main results and the original paper",
      tab1_caption: "Table 1. Base case, {nsess} sessions (seeds 1 to {nsess})",
      th_measure: "Measure",
      th_project: "This project",
      th_paper: "Calvano et al. (2020)",
      row_delta: "Mean Δ (profit index)",
      row_nash: "Sessions in Nash equilibrium on path (one-period deviations only, as in the paper)",
      row_nash_any: "Sessions in Nash equilibrium on path (any deviation)",
      row_react: "Rival's response in the period after the cut",
      row_unprof: "Price cuts that did not pay off",
      row_punish: "Periods until prices return (mean over sessions where they returned)",
      row_memoryless: "Mean Δ without memory, with the same exploration",
      not_reported: "not reported",
      tab1_note: "The gap in cuts that did not pay off ({proj} versus {paper}) remains unexplained: the paper does not describe the exact calculation. In the “any deviation” row, a firm may deviate for any number of periods (optimal best response); under this stricter rule, {anyn} of the {n} sessions are in Nash equilibrium on path. Every number in the “This project” column comes from a file in results/ in the repository; the other column comes from the paper.",
      about_h2: "About this project",
      about_p1: "This simulator is the first part of a project that replicates the Calvano et al. (2020) experiment and, in the next phase, applies the same measures, the Δ index and the deviation test, to language-model (LLM) agents.",
      about_p2: "Methodological choices were recorded before the tests were run, including the rule that set the conclusion's wording. Code, data and automated tests are in the repository.",
      about_stack: "Python and numba for the {nsess} sessions · JavaScript in the browser, validated against Python.",
      footer_fonts: "Newsreader and Public Sans typefaces (SIL Open Font License).",
      nav_method: "Method",
      nav_code: "Code",
      lang_switch_text: "Português",
      lang_switch_aria: "Read this page in Portuguese",

      kicker: "Replication · Calvano, Calzolari, Denicolò and Pastorello (2020)",
      title: "Two pricing algorithms that never communicated learned to charge above the competitive price",
      subtitle: "Each line is the price of a firm run by an algorithm that learns on its own, by trial and error. Prices near the bottom line mean competition; near the top line, monopoly.",
      badge_paper: "Parameters from the paper",
      badge_paper_k0: "Parameters from the paper, without memory",
      mem_switch_label: "Algorithms",
      mem_with: "With memory",
      mem_without: "Without memory",
      mem_note: "Without memory, each algorithm picks its price without knowing what the rival charged before. This is a session recorded in Python with the paper's parameters; the live session exists only with memory.",
      source_recorded_k0: "Memoryless session recorded in Python · seed {seed}",
      live_only_memory: "The live session exists only with memory.",
      aggregate_k0: "Across {n} memoryless sessions with the paper's parameters, mean Δ was {mean} (95% CI: {lo} to {hi}).",
      dev_mem_note: "Without memory, B does not know that A cut its price: it cannot respond, and both return to their earlier price in the very next period.",

      chart_learning_title: "Each firm's price during learning",
      chart_learning_desc: "Line chart: average price of firms A and B in blocks of {block} periods, with lines for the competitive price and the monopoly price.",
      chart_learning_note: "Each point is the average of a block of {block} periods.",
      axis_periods_millions: "Periods (millions)",
      axis_price: "Price",
      line_nash: "Competitive price (Nash)",
      line_monopoly: "Monopoly price",
      firm_a: "Firm A",
      firm_b: "Firm B",

      delta_label: "Profit index Δ, this session",
      delta_mean_ref: "Mean of {n} sessions: {mean} (dotted mark on the scale)",
      delta_scale: "0 = competitive profit · 1 = monopoly profit",
      delta_final: "after convergence",
      delta_during: "during learning (strategies still changing)",
      source_recorded: "Session recorded in Python · seed {seed}",
      source_live: "Live in your browser · seed {seed}",
      outcome_constant: "Final outcome: both firms keep a constant price.",
      outcome_cycle: "Final outcome: prices repeat in a cycle of {n} periods.",
      compute_time: "Computed in {s} s in your browser.",
      million_one: "million",
      million_many: "million",

      status_converged_mid: "Both firms stopped changing strategy after {periods} {mi} periods. Profit settled at Δ = {delta}: above the competitive level and below the monopoly level.",
      status_converged_high: "Both firms stopped changing strategy after {periods} {mi} periods. Profit settled at Δ = {delta}, close to monopoly profit.",
      status_converged_low: "Both firms stopped changing strategy after {periods} {mi} periods. Profit settled at Δ = {delta}, close to competitive profit.",
      status_not_converged: "The session stopped at {periods} {mi} periods without converging; Δ = {delta} is the average of the last {stable} periods.",
      status_learning: "Learning: period {t}. The firms still try a random price {eps}% of the time.",
      status_paused: "Paused at period {t}.",
      status_computing: "Computing in your browser…",
      status_deviation_done: "Price cut applied: see the response in “The deviation test” below.",
      status_worker_error: "The simulation could not run in this browser. The recorded result is still available.",

      btn_run: "Replay the learning",
      btn_pause: "Pause",
      btn_resume: "Resume",
      btn_deviate: "Force a price cut",
      btn_new: "New live session",
      btn_deviate_hint: "Available once the firms stop learning.",
      controls_label: "Simulation controls",

      aggregate: "Across {n} sessions with the paper's parameters, mean Δ was {mean} (95% CI: {lo} to {hi}), where 0 is the competitive profit and 1 the monopoly profit.",
      data_table_summary: "Show the chart data",
      chart_hint: "Hover over or tap the charts to see the exact values.",
      tip_periods: "Periods {a} to {b}",
      tip_tau: "Period {k} after the cut",
      tip_mean: "Mean Δ",
      tip_ci: "95% CI",
      th_periods: "Period (millions)",
      th_price_a: "Price A",
      th_price_b: "Price B",
      th_delta_block: "Block Δ",

      dev_h2: "The deviation test",
      dev_intro: "What if one firm cuts its price? Once both have stopped learning, firm A is forced to charge, for a single period, the price that earns the most against B's current price. After that, both go back to what they learned. The grey line shows what would have happened without the cut.",
      dev_cycle_note: "In this session prices repeat in a cycle; the cut starts from the first point of the cycle.",
      chart_dev_title: "Prices before and after the cut",
      chart_dev_desc: "Step chart: prices of firms A and B over the {horizon} periods after the forced cut, compared with the path without the cut.",
      axis_tau: "Periods after the cut (τ)",
      line_control: "No cut",
      th_tau: "τ",
      th_a_cut: "A, with the cut",
      th_b_cut: "B, with the cut",
      th_a_nocut: "A, without the cut",
      th_b_nocut: "B, without the cut",
      ann_cut: "A cuts its price",
      ann_react: "B responds",
      ann_noreact: "B does not respond",
      ann_back: "Back to the pattern",
      score_react: "B's response in the next period",
      score_react_value: "{x}% in price",
      score_back: "Periods until prices return",
      score_back_none: "did not return within {horizon}",
      score_profit: "A's profit over {horizon} periods (discounted)",
      score_profit_value: "with the cut {d} · without {c}",
      verdict_unprofitable: "The cut did not pay off: A earned {g}% less than without it.",
      verdict_profitable: "The cut paid off: A earned {g}% more than without it.",
      verdict_none: "In this session, A was already charging the price that earns the most against B: there was no actual cut.",
      dev_aggregate: "Across the {n} sessions, the rival responded and prices returned in {a}% of them, and the cut did not pay off in {u}%.",
      dev_caveat: "This pattern of response and return is what the literature calls punishment. On its own it does not prove collusion, and that reading is disputed ({ref}).",
      ref_denboer: "den Boer, Meylahn and Schinkel, 2026",

      sep_h3: "Does punishment explain the high prices?",
      sep_predict: "The test: memoryless algorithms do not know what the rival charged yesterday, so they cannot punish a price cut. If the threat of punishment explained the high prices, profit should fall without memory.",
      sep_main_title: "Profit with and without memory, with the paper's parameters ({nsess} sessions each)",
      sep_main_desc: "Horizontal bars: mean Δ with memory {k1} (95% CI: {k1lo} to {k1hi}) and without memory {k0} (95% CI: {k0lo} to {k0hi}). 0 is the competitive profit and 1 the monopoly profit.",
      sep_result: "Result: the opposite happened. Without memory, profit came even closer to monopoly (Δ {k0} versus {k1}). The same happened in all {k} exploration settings tested: the ability to punish does not raise prices.",
      sep_more: "See all {k} exploration settings tested",
      axis_competition: "Competition",
      axis_monopoly: "Monopoly",
      th_param_tech: "Technical parameter",
      sep_conclusion: "Conclusion: prices above the competitive level; in most sessions, a forced cut was followed by a response from the rival and did not pay off. But it cannot be ruled out that part of the high prices comes from incomplete learning, and the ability to punish does not raise prices under the same exploration.",
      chart_sep_title: "Profit (Δ) with and without memory in each exploration setting, {nsess} sessions each",
      chart_sep_desc: "Dot chart: for each exploration level, the mean Δ of algorithms with and without memory, with 95% confidence intervals.",
      sep_with: "With memory (can punish)",
      sep_without: "Without memory (cannot punish)",
      sep_note: "Each row changes how long the algorithms keep trying random prices (exploration). Blue dot: with memory; orange square: without memory; the line around each mark is the 95% confidence interval. Further right means profit closer to monopoly. In every row, the square is to the right of the dot.",
      cond_beta1e_05: "Short learning",
      cond_beta4e_06: "Paper's parameters",
      cond_beta2e_06: "Long learning",
      cond_beta1e_06: "Very long learning",
      cond_epsmin0_001: "Exploration that never stops (0.1%)",
      cond_epsmin0_01: "Exploration that never stops (1%)",
      cond_epsmin0_05: "Exploration that never stops (5%)",
      th_condition: "Condition",
      th_with: "Δ with memory",
      th_without: "Δ without memory",

      why_h2: "Why it matters",
      why_p1: "Firms already use algorithms to set prices in e-commerce, at fuel stations, in rental housing and in air travel, and competition authorities in Brazil, the United States and Europe are investigating this use.",
      why_p2: "But almost all real cases are of a different kind: competitors that use the same software vendor, which pools everyone's data. This experiment tests a different situation: algorithms that learn on their own, without sharing information.",
      why_table_caption: "Two different mechanisms",
      why_col_tacit: "Algorithms that learn on their own (what this experiment tests)",
      why_col_hub: "Hub-and-spoke (the mechanism in most real cases)",
      why_row_how: "How it works",
      why_how_tacit: "Each firm has its own algorithm and sees only market prices.",
      why_how_hub: "An intermediary, such as the software vendor, pools data from several competitors.",
      why_row_info: "Information sharing",
      why_info_tacit: "None.",
      why_info_hub: "Yes, through the intermediary.",
      why_row_examples: "Examples",
      why_examples_tacit: "Lab experiments such as Calvano et al. (2020).",
      why_examples_hub: "Aprix (CADE, Brazil) and RealPage (US Department of Justice).",
      why_aprix: `In the Aprix case, a pricing tool for fuel stations, Brazil's competition authority (CADE) approved a cease-and-desist settlement with Intelprice on 8 April 2026. According to ${a(LINKS.cade, "CADE's statement")} (in Portuguese), there were concerns that the tool could promote “price coordination and alignment” (free translation); the settlement requires confidentiality of sensitive data and separation of clients' prices and strategies. According to the law firm ${a(LINKS.lefosse, "Lefosse")}, the software used data shared by the stations that used it. A settlement is not a conviction.`,
      why_realpage: `In the United States, the 24 November 2025 settlement between the Department of Justice and RealPage, a rental-housing software company, restricts the use of competitors' non-public data (${a(LINKS.realpage, "summary by the law firm Fenwick")}).`,
      why_not: "What this experiment does not show: that real firms collude, or what prices they charge.",

      method_h2: "Method, data and limitations",
      method_open: "Show method, parameters and limitations",
      method_model: "Model: a repeated Bertrand duopoly with differentiated products and logit demand (Calvano et al., 2020, eq. 5). Each firm uses Q-learning with one period of memory: it picks today's price by looking at both of yesterday's prices.",
      method_params_caption: "Parameters (all from the paper; the source gives the page)",
      th_param: "Parameter",
      th_symbol: "Symbol",
      th_value: "Value",
      th_source: "Source",
      p_n: "Number of firms",
      p_c: "Marginal cost",
      p_a: "Product quality",
      p_a0: "Outside good (not buying)",
      p_mu: "Product differentiation",
      p_delta: "Discount factor (weight of the future)",
      p_alpha: "Learning rate",
      p_beta: "Exploration decay, ε = e^(−βt)",
      p_k: "Memory (periods)",
      p_m: "Possible prices on the grid",
      p_xi: "Grid extension beyond Nash and monopoly",
      p_conv: "Convergence",
      p_conv_value: "strategy unchanged for {stable} periods (cap of {tmax})",
      p_sessions: "Sessions",
      p_sessions_value: "{n} (seeds 1 to {n})",
      method_measure: "Measure: Δ = (π̄ − πN) / (πM − πN), where π̄ is the average profit per firm after convergence, πN = {piN} is the competitive (Nash) profit and πM = {piM} is the monopoly profit. Reference prices: Nash {pN}; monopoly {pM}.",
      method_engine: "In the browser, the same algorithm runs in JavaScript with a different random number generator, so the same seed does not repeat the Python session. The engine was validated against Python on the distribution of Δ, and the deviation test was checked point by point on the recorded sessions (tests in the repository).",
      lim_h3: "Limitations",
      lim_1: "A simulated, symmetric market with {m} possible prices.",
      lim_2: "Results depend on the discount factor, on exploration and on how the algorithm updates what it has learned.",
      lim_3: "Prices drawn at random on the grid would already give Δ ≈ {rand}: a high Δ alone does not indicate strategy.",
      lim_4: "The deviation test is indirect evidence, and its interpretation is disputed.",
      lim_5: "Memoryless algorithms reach even higher profits: part of the high prices may come from incomplete learning.",
      lim_6: "There is no documented use of tabular Q-learning for setting prices in real firms.",
      data_h3: "Data",
      data_params: "Parameters, price grid and profit table (JSON)",
      data_reference: "Reference results for the {nsess} sessions (JSON)",
      data_sessions: "{nrec} complete sessions, with the deviation test (JSON)",
      data_repo: "Full results and source code (GitHub)",
      cite_h3: "How to cite",
      cite_text: "Mac Nicol, Beatriz Cury (2026). Algorithmic pricing: a Q-learning simulator of the Calvano et al. (2020) duopoly. Code and data: {repo}",
      cite_original: "Original paper: Calvano, E., Calzolari, G., Denicolò, V. and Pastorello, S. (2020). Artificial Intelligence, Algorithmic Pricing, and Collusion. American Economic Review, 110(10), 3267–3297.",
      updated: "Data updated on {date}.",

      footer_author: "Made by Beatriz Cury Mac Nicol.",
      footer_license: "Code under the MIT license.",
      footer_credits: "Model and parameters: Calvano, Calzolari, Denicolò and Pastorello (2020). The Q-learning loop was adapted from code by Matteo Courthoud (MIT license).",
      footer_no_ai: "This page calls no external service: everything runs in your browser or comes from recorded results.",
    },
  };

  // en-GB: datas no formato "29 October 2026", igual às datas escritas nos textos.
  const LOCALE = { pt: "pt-BR", en: "en-GB" };
  let lang = "pt";

  // Valores que vêm dos JSON e aparecem em vários textos (ex.: {nsess}, {horizon}); a página os
  // define com setGlobals depois de carregar os dados, assim nenhum número fica digitado no texto.
  let globals = null;
  function setGlobals(g) { globals = g || {}; }

  function t(key, vars) {
    let s = STR[lang][key];
    if (s === undefined) { console.error("i18n: chave ausente", lang, key); s = STR.pt[key] || key; }
    const fill = (k) => {
      const v = vars && vars[k] !== undefined ? vars[k] : (globals ? globals[k] : undefined);
      return typeof v === "function" ? v() : v;
    };
    // sem dados ainda (globals null): "–" no lugar do valor; depois, chave sem valor fica visível (os testes pegam)
    s = s.replace(/\{(\w+)\}/g, (m0, k) => { const v = fill(k); return v !== undefined ? v : (globals ? m0 : "–"); });
    return s;
  }

  // Número no formato do idioma (vírgula decimal em português).
  function num(x, digits) {
    return new Intl.NumberFormat(LOCALE[lang], {
      minimumFractionDigits: digits, maximumFractionDigits: digits,
    }).format(x);
  }

  function date(iso) {
    return new Intl.DateTimeFormat(LOCALE[lang], { day: "numeric", month: "long", year: "numeric" })
      .format(new Date(iso));
  }

  // Aplica os textos a todo elemento com data-i18n (texto), data-i18n-html (com links)
  // ou data-i18n-attr="atributo:chave;atributo:chave".
  function apply(rootEl) {
    const r = rootEl || document;
    r.querySelectorAll("[data-i18n]").forEach((el) => { el.textContent = t(el.dataset.i18n); });
    r.querySelectorAll("[data-i18n-html]").forEach((el) => { el.innerHTML = t(el.dataset.i18nHtml); });
    r.querySelectorAll("[data-i18n-attr]").forEach((el) => {
      el.dataset.i18nAttr.split(";").forEach((pair) => {
        const [attr, key] = pair.split(":");
        el.setAttribute(attr.trim(), t(key.trim()));
      });
    });
    document.documentElement.lang = LOCALE[lang];
    document.title = t("doc_title");
    const meta = document.querySelector('meta[name="description"]');
    if (meta) meta.setAttribute("content", t("doc_description"));
  }

  function setLang(l) { if (STR[l]) lang = l; }
  function getLang() { return lang; }

  // Idioma inicial: ?lang= na URL, depois a escolha salva, depois o idioma do navegador.
  function initialLang() {
    try {
      const q = new URLSearchParams(location.search).get("lang");
      if (q && STR[q]) return q;
    } catch (e) { /* sem URL */ }
    try {
      const saved = localStorage.getItem("lang");
      if (saved && STR[saved]) return saved;
    } catch (e) { /* armazenamento bloqueado */ }
    const nav = (navigator.language || "pt").toLowerCase();
    return nav.startsWith("pt") ? "pt" : "en";
  }

  const api = { STR: STR, LINKS: LINKS, t: t, num: num, date: date, apply: apply,
                setLang: setLang, getLang: getLang, initialLang: initialLang, setGlobals: setGlobals };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.I18N = api;
})(typeof self !== "undefined" ? self : this);
