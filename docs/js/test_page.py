"""
Fase 2.2: teste da página do simulador num navegador de verdade (Chromium, Playwright).
Uso (na raiz do repositório): python docs/js/test_page.py [pasta_para_capturas]
Sobe um servidor local para docs/, abre a página e confere, em português e em inglês:
carregamento sem erros, textos completos (nenhuma chave nem {valor} sobrando), troca de
idioma (lang do HTML, URL e escolha lembrada), sessão ao vivo até convergir, desvio forçado,
celular a 320 px sem rolagem lateral, alvos de toque de 44 px, teclado e movimento reduzido.
Requer: pip install playwright (e o Chromium do Playwright). Código próprio do projeto.
"""
import functools, http.server, json, re, socketserver, sys, threading
from pathlib import Path
from playwright.sync_api import sync_playwright

DOCS = Path(__file__).resolve().parents[1]
SHOTS = Path(sys.argv[1]) if len(sys.argv) > 1 else None
I18N = None
checks = []
def check(name, ok, detail=""):
    checks.append((name, bool(ok)))
    if not ok and detail: print("  ->", detail)

Handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(DOCS))
class Quiet(Handler.func):
    def log_message(self, *a): pass
srv = socketserver.TCPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(DOCS)))
threading.Thread(target=srv.serve_forever, daemon=True).start()
URL = f"http://127.0.0.1:{srv.server_address[1]}/index.html"

def texts_ok(page, lang):
    # compara cada elemento com o texto da língua pedida, já com os valores globais preenchidos
    bad = page.evaluate("""(l) => (I18N.getLang() !== l ? ['língua ativa ' + I18N.getLang()] :
        [...document.querySelectorAll('[data-i18n]')]
        .filter(e => e.textContent !== I18N.t(e.dataset.i18n)).map(e => e.dataset.i18n))""", lang)
    body = page.inner_text("body")
    left = re.findall(r"\{\w+\}|undefined|NaN", body)
    return bad, left, body

with sync_playwright() as p:
    b = p.chromium.launch()
    for lang in ("pt", "en"):
        ctx = b.new_context(viewport={"width": 1200, "height": 900})
        pg = ctx.new_page(); errors = []
        pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto(f"{URL}?lang={lang}"); pg.wait_for_selector("body[data-ready='1']", timeout=20000)
        check(f"[{lang}] carrega sem erros no console", not errors, errors)
        check(f"[{lang}] lang do HTML", pg.evaluate("document.documentElement.lang") == ("pt-BR" if lang == "pt" else "en-GB"))
        bad, left, body = texts_ok(pg, lang)
        check(f"[{lang}] todo texto com data-i18n traduzido", not bad, bad)
        check(f"[{lang}] nenhum {{valor}}, undefined ou NaN na página", not left, left)
        if lang == "en":
            pt_words = [w for w in re.findall(r"\S+", body) if re.search(r"[ãõçáéíóúâêôà]", w, re.I)
                        and not any(a in w for a in ("Denicolò", "Português"))]
            check("[en] nenhuma palavra em português na página em inglês", not pt_words, pt_words[:10])
        check(f"[{lang}] gráficos desenhados", pg.evaluate("['learn-chart','dev-chart','sep-chart'].every(i => document.querySelector('#'+i+' svg polyline, #'+i+' svg path, #'+i+' svg circle'))"))
        check(f"[{lang}] Δ da sessão gravada mostrado", re.match(r"^0[.,]8\d$", pg.inner_text("#delta-value")), pg.inner_text("#delta-value"))
        check(f"[{lang}] placar do desvio preenchido", pg.inner_text("#verdict").strip() != "")
        if lang == "pt":
            # resultado de B igual ao calculado com os lucros gravados no Python (semente 48, ponto 0)
            import json as _j
            ss = _j.loads((DOCS / "data" / "qlearning_sessions.json").read_text(encoding="utf-8"))
            s0 = [x for x in ss["sessions"] if x["opens_page"]][0]["deviation_test"]["starts"][0]
            pc, pv = s0["control"]["profit_firm2"], s0["deviation"]["profit_firm2"]
            wts = [0.95 ** (k - 1) for k in range(1, len(pc))]
            gb = sum(w_ * (pv[k] - pc[k]) for w_, k in zip(wts, range(1, len(pc)))) / sum(w_ * pc[k] for w_, k in zip(wts, range(1, len(pc))))
            exp_txt = f"B lucrou {abs(gb) * 100:.1f}".replace(".", ",") + (" % a menos" if gb < 0 else " % a mais").replace(" %", "%")
            check("resultado de B no placar igual ao do Python", exp_txt in pg.inner_text("#verdict"), (exp_txt, pg.inner_text("#verdict")))
        check(f"[{lang}] tabela 1 com 7 linhas de resultados", pg.evaluate("document.querySelectorAll('#tab1 tbody tr').length") == 7)
        check(f"[{lang}] tabela 1 traz as duas versões do teste de Nash",
              ("qualquer desvio" if lang == "pt" else "any deviation") in pg.inner_text("#tab1"))
        check(f"[{lang}] fontes do site carregadas (Newsreader e Public Sans)",
              pg.evaluate("document.fonts.ready.then(() => document.fonts.check('600 20px Newsreader') && document.fonts.check('16px \"Public Sans\"'))"))
        if SHOTS: pg.screenshot(path=str(SHOTS / f"desktop_{lang}.png"), full_page=True)
        ctx.close()

    # troca de idioma pelo botão, URL e escolha lembrada
    ctx = b.new_context(viewport={"width": 1200, "height": 900}); pg = ctx.new_page()
    pg.goto(URL + "?lang=pt"); pg.wait_for_selector("body[data-ready='1']")
    pg.click("#lang-btn")
    st = pg.inner_text("#status")
    check("status reformatado ao trocar de idioma (inglês, ponto decimal)",
          re.search(r"after \d+\.\d\d million periods", st) and "milh" not in st, st)
    check("botão troca para inglês", pg.evaluate("document.documentElement.lang") == "en-GB" and "lang=en" in pg.url)
    bad, left, _ = texts_ok(pg, "en")
    check("depois da troca, todo texto em inglês", not bad and not left, bad + left)
    check("título da aba em inglês", "Algorithmic" in pg.title())
    pg.goto(URL); pg.wait_for_selector("body[data-ready='1']")
    check("escolha do idioma lembrada ao recarregar", pg.evaluate("document.documentElement.lang") == "en-GB")
    pg.click("#lang-btn")
    check("botão volta para português", pg.evaluate("document.documentElement.lang") == "pt-BR")

    # sessão ao vivo e desvio (teclado: Tab até o botão e Enter)
    pg.focus("#btn-new"); pg.keyboard.press("Enter")
    pg.wait_for_function("document.getElementById('btn-deviate').disabled === false", timeout=90000)
    src = pg.inner_text("#source-line")
    check("sessão ao vivo converge e libera o desvio", "navegador" in src, src)
    st = pg.inner_text("#status")
    check("status da sessão ao vivo (milhão/milhões concorda com o número)",
          re.search(r"(^|\D)1,\d\d milhão de períodos|[2-9],\d\d milhões de períodos|\d\d+,\d\d milhões de períodos", st), st)
    oc = pg.inner_text("#outcome")
    check("tempo de cálculo mostrado na sessão ao vivo", re.search(r"Calculado em \d+,\d\d s", oc), oc)
    pg.click("#btn-deviate")
    pg.wait_for_function("document.getElementById('score-back').textContent !== '–'", timeout=20000)
    check("desvio forçado na sessão ao vivo preenche o placar", pg.inner_text("#verdict").strip() != "")
    check("tabela de dados da Figura 2 (desvio) com τ = 0 a 15",
          pg.evaluate("document.querySelectorAll('#dev-table tbody tr').length") == pg.evaluate("QDeviation.T_DEV + 1"))
    pg.click("#lang-btn")
    bad, left, _ = texts_ok(pg, "en")
    check("troca de idioma depois de rodar ao vivo", not bad and not left and "browser" in pg.inner_text("#source-line"), bad + left)
    # reproduzir o aprendizado da sessão já calculada (mesma sessão, sem recalcular)
    pg.click("#btn-run")
    check("reproduzir: botão vira Pausar", pg.inner_text("#btn-run") == "Pause")
    pg.wait_for_function("document.getElementById('btn-run').textContent === 'Replay the learning'", timeout=30000)
    check("reproduzir não muda a sessão", "browser" in pg.inner_text("#source-line"))
    # pausar e continuar
    pg.click("#btn-new"); pg.wait_for_timeout(600); pg.click("#btn-run")
    check("pausar", pg.inner_text("#btn-run") == "Resume")
    pg.click("#btn-run")
    check("continuar", pg.inner_text("#btn-run") == "Pause")
    ctx.close()

    # celular a 320 px
    for lang in ("pt", "en"):
        ctx = b.new_context(viewport={"width": 320, "height": 640}, has_touch=True, is_mobile=True, device_scale_factor=2)
        pg = ctx.new_page(); pg.goto(f"{URL}?lang={lang}"); pg.wait_for_selector("body[data-ready='1']")
        sw = pg.evaluate("document.documentElement.scrollWidth")
        check(f"[{lang}] 320 px sem rolagem lateral", sw <= 320, sw)
        small = pg.evaluate("""[...document.querySelectorAll('button, .topnav a, summary')]
            .filter(e => e.offsetParent).map(e => [e.id || e.textContent.trim().slice(0, 20), e.getBoundingClientRect().height,
              e.matches('.topnav a') ? e.getBoundingClientRect().width : 99])
            .filter(x => x[1] < 44 || x[2] < 44)""")
        check(f"[{lang}] alvos de toque com 44 px de altura", not small, small)
        if SHOTS: pg.screenshot(path=str(SHOTS / f"mobile_{lang}.png"), full_page=True)
        ctx.close()

    # valores ao passar o mouse, tocar ou usar o teclado (revisão de 30/10/2026)
    ctx = b.new_context(viewport={"width": 1200, "height": 900}); pg = ctx.new_page()
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto(URL + "?lang=pt"); pg.wait_for_selector("body[data-ready='1']")
    box = pg.locator("#learn-chart").bounding_box()
    pg.mouse.move(box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.5)
    tip = pg.locator("#learn-chart + .tip, #learn-chart ~ .tip").first
    txt = tip.inner_text() if tip.is_visible() else ""
    check("Figura 1: passar o mouse mostra período, preços de A e B e Δ do bloco",
          "Períodos" in txt and "Empresa A" in txt and "Empresa B" in txt and re.search(r"\d,\d{3}", txt), txt)
    pg.mouse.move(5, 5)
    check("Figura 1: a caixinha some quando o mouse sai", not tip.is_visible())
    pg.focus("#dev-chart"); pg.keyboard.press("Home"); pg.keyboard.press("ArrowRight")
    dtip = pg.locator("#dev-chart ~ .tip").first
    check("Figura 2: teclado (Home, seta) mostra o período 1 depois do corte",
          dtip.is_visible() and "Período 1 depois do corte" in dtip.inner_text(), dtip.inner_text() if dtip.is_visible() else "")
    pg.keyboard.press("Escape")
    check("Esc fecha a caixinha", not dtip.is_visible())
    pg.locator("#sep-main").scroll_into_view_if_needed()
    box = pg.locator("#sep-main").bounding_box()
    pg.mouse.move(box["x"] + box["width"] * 0.3, box["y"] + box["height"] * 0.45)
    stip = pg.locator("#sep-main ~ .tip").first
    check("Figura 3: barras mostram Δ médio e IC 95%", stip.is_visible() and "IC 95%" in stip.inner_text())
    check("caixinhas sem erros no console", not errs, errs)
    ctx.close()
    ctx = b.new_context(viewport={"width": 320, "height": 640}, has_touch=True, is_mobile=True, device_scale_factor=2)
    pg = ctx.new_page(); pg.goto(URL + "?lang=en"); pg.wait_for_selector("body[data-ready='1']")
    pg.locator("#learn-chart").scroll_into_view_if_needed()
    box = pg.locator("#learn-chart").bounding_box()
    pg.touchscreen.tap(box["x"] + box["width"] * 0.9, box["y"] + box["height"] * 0.5)
    tip = pg.locator("#learn-chart ~ .tip").first
    tb = tip.bounding_box() if tip.is_visible() else None
    check("celular: tocar mostra a caixinha, dentro da tela (320 px)",
          tb is not None and tb["x"] >= 0 and tb["x"] + tb["width"] <= 321 and "Firm A" in tip.inner_text(), tb)
    ctx.close()

    # seletor "Com memória | Sem memória" e Figura 3 (revisão de 30/10/2026)
    ctx = b.new_context(viewport={"width": 1200, "height": 900}); pg = ctx.new_page()
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto(URL + "?lang=pt"); pg.wait_for_selector("body[data-ready='1']")
    check("Figura 3: barras desenhadas e resultado escrito com números",
          pg.evaluate("document.querySelectorAll('#sep-main svg rect').length") >= 4
          and re.search(r"Δ 0,9\d contra 0,8\d", pg.inner_text("#sep-result")), pg.inner_text("#sep-result"))
    pg.click("#sep-more summary")
    check("Figura 3: as 7 condições aparecem ao abrir, com tabela e parâmetro técnico",
          pg.evaluate("document.querySelectorAll('#sep-chart svg circle').length") == 7
          and pg.evaluate("document.querySelectorAll('#sep-table tbody tr').length") == 7
          and "β = 4 × 10⁻⁶" in pg.inner_text("#sep-table"))
    pg.click("#mem-without")
    src = pg.inner_text("#source-line")
    check("sem memória: sessão gravada sem memória, aviso e botão ao vivo desligado",
          "sem memória" in src and pg.is_visible("#mem-note") and pg.is_disabled("#btn-new")
          and pg.get_attribute("#mem-without", "aria-pressed") == "true", src)
    check("sem memória: Δ da sessão gravada e média das 1.000 sem memória",
          re.match(r"^0,9\d$", pg.inner_text("#delta-value")) and "sem memória" in pg.inner_text("#aggregate"))
    pg.click("#btn-deviate")
    pg.wait_for_function("document.getElementById('score-back').textContent !== '–'", timeout=20000)
    check("sem memória: no corte a rival não reage (0,0%) e a nota explica",
          pg.inner_text("#score-react").startswith("0,0") and pg.is_visible("#dev-mem-note"), pg.inner_text("#score-react"))
    pg.click("#lang-btn")
    check("sem memória em inglês", "Memoryless" in pg.inner_text("#source-line") and "without memory" in pg.inner_text("#badge").lower())
    pg.click("#lang-btn")
    pg.click("#mem-with")
    pg.click("#btn-new")
    pg.wait_for_timeout(300)
    pg.click("#mem-without")             # troca no meio de uma sessão ao vivo
    pg.click("#mem-with")
    pg.click("#btn-new")
    try:
        pg.wait_for_function("document.getElementById('btn-deviate').disabled === false && document.getElementById('source-line').textContent.includes('navegador')", timeout=90000)
        ok_live = True
    except Exception:
        ok_live = False
    check("trocar de modo no meio de uma sessão ao vivo não trava a próxima", ok_live and not errs, errs)
    ctx.close()

    # falha do worker: volta à sessão gravada com aviso, e um 2º clique não trava em "Calculando"
    ctx = b.new_context(viewport={"width": 1000, "height": 800}); pg = ctx.new_page()
    pg.route("**/qlearning.worker.js", lambda route: route.fulfill(status=404, body=""))
    pg.goto(URL + "?lang=en"); pg.wait_for_selector("body[data-ready='1']")
    ok = True
    for _ in range(2):
        pg.click("#btn-new")
        try:
            pg.wait_for_function("document.getElementById('status').textContent.includes('could not run')", timeout=10000)
        except Exception:
            ok = False
        pg.evaluate("document.getElementById('status').textContent = ''")
    check("falha do worker: aviso nas duas tentativas, sem travar", ok and "Python" in pg.inner_text("#source-line"))
    ctx.close()
    # falha só na 1ª carga do worker: a 2ª tentativa cria outro worker e a sessão ao vivo termina
    ctx = b.new_context(viewport={"width": 1000, "height": 800}); pg = ctx.new_page()
    loads = {"n": 0}
    def flaky(route):
        loads["n"] += 1
        if loads["n"] == 1: route.fulfill(status=404, body="")
        else: route.continue_()
    pg.route("**/qlearning.worker.js", flaky)
    pg.goto(URL + "?lang=en"); pg.wait_for_selector("body[data-ready='1']")
    pg.click("#btn-new")
    pg.wait_for_function("document.getElementById('status').textContent.includes('could not run')", timeout=10000)
    pg.click("#btn-new")
    try:
        pg.wait_for_function("document.getElementById('btn-deviate').disabled === false && document.getElementById('source-line').textContent.includes('browser')", timeout=90000)
        rec = True
    except Exception:
        rec = False
    check("depois de uma falha, um worker novo roda a sessão ao vivo até o fim", rec)
    ctx.close()

    # movimento reduzido: sem animação, resultado direto
    ctx = b.new_context(viewport={"width": 1000, "height": 800}, reduced_motion="reduce")
    pg = ctx.new_page(); pg.goto(URL + "?lang=pt"); pg.wait_for_selector("body[data-ready='1']")
    pg.click("#btn-deviate")
    check("movimento reduzido: desvio aparece inteiro na hora", pg.inner_text("#score-back") != "–")
    ctx.close()
    b.close()
srv.shutdown()

print("Conferências:")
for n, ok in checks: print(f"  {'True ' if ok else 'False'}  {n}")
n_ok = sum(ok for _, ok in checks)
print(f"\n{n_ok} de {len(checks)} conferências True")
sys.exit(0 if n_ok == len(checks) else 1)
