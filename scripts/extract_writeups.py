"""
extract_writeups.py — Extrae el texto de los writeups (.docx y .pdf) a .txt planos,
para poder leerlos y convertirlos en dataset de la v9.

.docx: son zip + XML (word/document.xml), se extrae sin librerias externas.
.pdf:  usa pypdf si esta instalado; si no, avisa para instalarlo.

Salida: por cada documento, un .txt con el mismo nombre en data/archivo/_txt/

Uso (desde F:\\vertex-lab, con el venv):
    .\\venv\\Scripts\\python.exe scripts\\extract_writeups.py
"""
import html
import re
import zipfile
from pathlib import Path

SRC = Path(r"F:\vertex-lab\data\archivo")
OUT = SRC / "_txt"
OUT.mkdir(exist_ok=True)


def docx_to_text(path: Path) -> str:
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8", "ignore")
    # Parrafos y saltos de linea -> newlines
    xml = xml.replace("</w:p>", "\n").replace("<w:br/>", "\n").replace("<w:br />", "\n")
    text = re.sub(r"<[^>]+>", "", xml)
    text = html.unescape(text)
    # Limpia lineas vacias repetidas
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def pdf_to_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader  # fallback
        except ImportError:
            return "[ERROR: instala pypdf -> pip install pypdf]"
    reader = PdfReader(str(path))
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n\n".join(parts).strip()


def main():
    docs = list(SRC.glob("*.docx")) + list(SRC.glob("*.pdf"))
    print(f"[*] {len(docs)} documentos encontrados en {SRC}")
    for d in docs:
        try:
            if d.suffix.lower() == ".docx":
                text = docx_to_text(d)
            else:
                text = pdf_to_text(d)
            out = OUT / (d.stem + ".txt")
            out.write_text(text, encoding="utf-8")
            print(f"  [OK] {d.name:55s} -> {len(text):6d} chars -> _txt/{out.name}")
        except Exception as exc:  # noqa: BLE001
            print(f"  [ERR] {d.name}: {exc}")
    print(f"\n[i] Listo. Los .txt estan en {OUT}")


if __name__ == "__main__":
    main()
