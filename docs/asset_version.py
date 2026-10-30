"""
Versão dos arquivos do simulador (cache do navegador).

O GitHub Pages deixa o navegador reaproveitar arquivos por até 10 minutos. Sem cuidado, logo
depois de um merge o navegador pode juntar o HTML novo com CSS ou JavaScript antigos. Por isso
o index.html pede cada arquivo com ?v=<versão>, e a versão é calculada do CONTEÚDO dos arquivos
(SHA-256): mudou qualquer arquivo, muda a versão, e o navegador baixa tudo de novo.

Uso (na raiz do repositório):
  python docs/asset_version.py          grava a versão nova no index.html
  python docs/asset_version.py --check  só confere (a CI usa: falha se esquecerem de rodar)
Código próprio do projeto.
"""

import hashlib
import re
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent
FILES = sorted([DOCS / "css" / "style.css", *(DOCS / "js").glob("*.js"), *(DOCS / "data").glob("*.json")])
FILES = [f for f in FILES if not f.name.startswith("test_")]


def version() -> str:
    h = hashlib.sha256()
    for f in FILES:
        h.update(f.relative_to(DOCS).as_posix().encode())
        h.update(f.read_bytes().replace(b"\r\n", b"\n"))   # igual no Windows e no Linux
    return h.hexdigest()[:10]


def main():
    html_path = DOCS / "index.html"
    html = html_path.read_text(encoding="utf-8")
    v = version()
    found = set(re.findall(r'\?v=([\w-]+)"', html))
    if "--check" in sys.argv:
        ok = found == {v}
        print(f"versão dos arquivos: index.html {sorted(found)} × calculada {v} → {'OK' if ok else 'DESATUALIZADA: rode python docs/asset_version.py'}")
        sys.exit(0 if ok else 1)
    new = re.sub(r'\?v=[\w-]+"', f'?v={v}"', html)
    html_path.write_text(new, encoding="utf-8", newline="\n")
    print(f"index.html com a versão {v}")


if __name__ == "__main__":
    main()
