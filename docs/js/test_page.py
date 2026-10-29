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
    d = page.evaluate("(l) => I18N.STR[l]", lang)
    bad = page.evaluate("""(d) => [...document.querySelectorAll('[data-i18n]')]
        .filter(e => e.textContent !== d[e.dataset.i18n]).map(e => e.dataset.i18n)""", d)
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
        check(f"[{lang}] lang do HTML", pg.evaluate("document.documentElement.lang") == ("pt-BR" if lang == "pt" else "en"))
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
        check(f"[{lang}] tabela 1 com 6 linhas de resultados", pg.evaluate("document.querySelectorAll('#tab1 tbody tr').length") == 6)
        check(f"[{lang}] fontes do site carregadas (Newsreader e Public Sans)",
              pg.evaluate("document.fonts.ready.then(() => document.fonts.check('600 20px Newsreader') && document.fonts.check('16px \"Public Sans\"'))"))
        if SHOTS: pg.screenshot(path=str(SHOTS / f"desktop_{lang}.png"), full_page=True)
        ctx.close()

    # troca de idioma pelo botão, URL e escolha lembrada
    ctx = b.new_context(viewport={"width": 1200, "height": 900}); pg = ctx.new_page()
    pg.goto(URL + "?lang=pt"); pg.wait_for_selector("body[data-ready='1']")
    pg.click("#lang-btn")
    check("botão troca para inglês", pg.evaluate("document.documentElement.lang") == "en" and "lang=en" in pg.url)
    bad, left, _ = texts_ok(pg, "en")
    check("depois da troca, todo texto em inglês", not bad and not left, bad + left)
    check("título da aba em inglês", "Algorithmic" in pg.title())
    pg.goto(URL); pg.wait_for_selector("body[data-ready='1']")
    check("escolha do idioma lembrada ao recarregar", pg.evaluate("document.documentElement.lang") == "en")
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
          pg.evaluate("document.querySelectorAll('#dev-table tbody tr').length") == 16)
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
            .filter(e => e.offsetParent).map(e => [e.id || e.textContent.trim().slice(0, 20), e.getBoundingClientRect().height])
            .filter(x => x[1] < 44)""")
        check(f"[{lang}] alvos de toque com 44 px de altura", not small, small)
        if SHOTS: pg.screenshot(path=str(SHOTS / f"mobile_{lang}.png"), full_page=True)
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
