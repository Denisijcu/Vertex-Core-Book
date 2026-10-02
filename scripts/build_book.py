"""
build_book.py — Une el libro completo con formato profesional:
front matter (preliminares) + cuerpo (capitulos) + back matter (finales),
con tabla de contenidos, y salida en Markdown + Word (.docx) para KDP.

Uso (desde F:\\vertex-lab):
    .\\venv\\Scripts\\python.exe scripts\\build_book.py
Si falta python-docx:
    .\\venv\\Scripts\\python.exe -m pip install python-docx
"""
import re
from pathlib import Path

MAN = Path(r"F:\vertex-lab\docs\manuscrito")
OUT_MD = Path(r"F:\vertex-lab\docs\Vertex-Core-LIBRO.md")
OUT_DOCX = Path(r"F:\vertex-lab\docs\Vertex-Core-LIBRO.docx")

# FRONT MATTER (preliminares) -> INDICE (generado) -> CUERPO -> BACK MATTER
FRONT = [
    "front-00-portada-creditos.md",
    "front-01-dedicatoria.md",
    "front-02-prefacio.md",
    "front-03-para-quien-es.md",
    "front-04-como-leer.md",
]
BODY = [
    "cap-01-por-que-un-modelo-propio.md",
    "cap-02-el-heredero-equivocado.md",
    "cap-03-montar-el-entorno.md",
    "cap-04-los-datos-son-tuyos.md",
    "cap-05-ensenarle-quien-es.md",
    "cap-06-entrenar-de-verdad-qlora.md",
    "cap-07-del-adaptador-a-ollama.md",
    "cap-08-el-examen-de-estres.md",
    "cap-09-cuando-miente-e-inventa.md",
    "cap-09b-el-pendulo.md",
    "cap-10-del-modelo-al-agente.md",
    "cap-11-sacarlo-al-mundo.md",
    "cap-12-ensenarle-a-usar-las-manos.md",
    "cap-13-el-portero.md",
]
BACK = [
    "back-01-conclusion.md",
    "apendice-B-glosario.md",
    "apendice-C-pipeline.md",
    "apendice-D-servidores-mcp.md",
    "apendice-E-configuracion.md",
    "apendice-F-el-portero.md",
    "back-02-sobre-el-autor.md",
]


def limpiar(texto: str) -> str:
    """Quita los bloques de nota '> Borrador. Fuente: ...' del inicio de los capitulos."""
    lineas = texto.splitlines()
    out, i = [], 0
    while i < len(lineas):
        l = lineas[i]
        if l.strip().startswith(">"):
            i += 1
            if i < len(lineas) and not lineas[i].strip():
                i += 1
            continue
        out.append(l)
        i += 1
    return "\n".join(out).strip()


def titulo_de(texto: str) -> str:
    """Extrae el primer encabezado '# ...' de un archivo para la tabla de contenidos."""
    for l in texto.splitlines():
        if l.startswith("# "):
            return l[2:].strip()
    return "(sin titulo)"


def seccion(files):
    bloques = []
    titulos = []
    for fn in files:
        p = MAN / fn
        if not p.exists():
            print(f"  [!] falta {fn}")
            continue
        txt = limpiar(p.read_text(encoding="utf-8"))
        titulos.append(titulo_de(txt))
        bloques.append(txt)
        bloques.append("\n\n\\newpage\n")  # marca de salto de pagina
        print(f"  [OK] {fn}")
    return "\n".join(bloques), titulos


def main():
    print("== FRONT MATTER ==")
    front_md, _ = seccion(FRONT)
    print("== CUERPO ==")
    body_md, body_titulos = seccion(BODY)
    print("== BACK MATTER ==")
    back_md, back_titulos = seccion(BACK)

    # Tabla de contenidos (indice) a partir de los titulos del cuerpo + back
    toc = ["# Índice\n"]
    for t in body_titulos:
        toc.append(f"- {t}")
    for t in back_titulos:
        toc.append(f"- {t}")
    toc_md = "\n".join(toc) + "\n\n\\newpage\n"

    libro = "\n".join([front_md, toc_md, body_md, back_md])
    OUT_MD.write_text(libro, encoding="utf-8")
    words = len(libro.split())
    print(f"\n[OK] Markdown -> {OUT_MD}  (~{words} palabras, ~{max(1, words//300)} paginas de texto)")

    # DOCX
    try:
        from docx import Document
        from docx.enum.text import WD_BREAK
    except ImportError:
        print("\n[i] python-docx no instalado; no se genero .docx.")
        print("    .\\venv\\Scripts\\python.exe -m pip install python-docx")
        return

    doc = Document()
    primer_h1 = True
    for raw in libro.splitlines():
        line = raw.rstrip()
        if line.strip() == "\\newpage":
            continue  # el salto lo fuerza cada titulo de capitulo (page_break_before), mas limpio
        elif line.startswith("#### "):
            doc.add_heading(line[5:].strip(), level=4)
        elif line.startswith("### "):
            doc.add_heading(line[4:].strip(), level=3)
        elif line.startswith("## "):
            doc.add_heading(line[3:].strip(), level=2)
        elif line.startswith("# "):
            h = doc.add_heading(line[2:].strip(), level=1)
            # Cada capitulo/apendice empieza en pagina nueva (excepto el primero)
            if not primer_h1:
                h.paragraph_format.page_break_before = True
            primer_h1 = False
        elif line.startswith("- "):
            doc.add_paragraph(line[2:].strip(), style="List Bullet")
        elif line.startswith("|"):
            doc.add_paragraph(line)  # tablas: quedan como texto, se formatean en Word
        elif not line.strip():
            continue
        else:
            # negritas **..** -> texto plano
            doc.add_paragraph(re.sub(r"\*\*(.+?)\*\*", r"\1", line))
    doc.save(str(OUT_DOCX))
    print(f"[OK] Word     -> {OUT_DOCX}")
    print("\n[i] Cada capitulo/apendice empieza en pagina nueva (page_break_before). Las secciones")
    print("    internas (##) fluyen seguidas, para no desperdiciar paginas con secciones cortas.")
    print("[i] En Word: Referencias -> Tabla de contenido para el indice con numeros de pagina reales;")
    print("    anade portada de disenno y revisa margenes para KDP.")


if __name__ == "__main__":
    main()
