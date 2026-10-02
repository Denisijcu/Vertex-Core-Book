"""
01h_csv_techniques.py — Convierte los CSV de maquinas en ejemplos de asociacion
"maquina/SO/dificultad -> que tecnicas de ataque considerar".

Enfoque generalizado (como el razonamiento): el modelo aprende QUE tecnicas aparecen
segun el tipo de maquina, no la solucion paso a paso de una maquina concreta. Las
"Tecnicas Vistas" de cada CSV son justo esa asociacion.

Arregla la codificacion rota de los CSV (acentos: "DifA-cil" -> "Dificil", etc.).

Fuentes: data/archivo/htb-machines.csv, Vulnhub.csv, web-hacking.csv
Salida:  data/raw/csv_techniques.jsonl   (formato {"messages": [...]})

Uso (desde F:\\vertex-lab):
    .\\venv\\Scripts\\python.exe scripts\\01h_csv_techniques.py
"""
import csv
import json
import random
from pathlib import Path

ARCH = Path(r"F:\vertex-lab\data\archivo")
OUT = Path(r"F:\vertex-lab\data\raw\csv_techniques.jsonl")
rng = random.Random(42)

SYSTEM = (
    "Eres Vertex-Core, analista de seguridad ofensiva de Vertex Coders para pentesting "
    "AUTORIZADO. Cuando describes una maquina o un servicio, enumeras las tecnicas de ataque "
    "que conviene considerar, explicando brevemente cada una. Respondes en espanol, tecnico, "
    "con listas limpias. Solo objetivos autorizados."
)

# Mapa de reparacion de mojibake comun (UTF-8 leido como latin-1 u otros)
FIXES = {
    "A-": "i", "Ã­": "i", "Ã¡": "a", "Ã©": "e", "Ã³": "o", "Ãº": "u", "Ã±": "n",
    "DifA cil": "Dificil", "DifA-cil": "Dificil", "FA cil": "Facil", "FA-cil": "Facil",
    "TAccnicas": "Tecnicas", "MA quina": "Maquina", "MA-quina": "Maquina",
    "Ã‘": "N", "Â": "",
}


def fix(s: str) -> str:
    if not s:
        return s
    for bad, good in FIXES.items():
        s = s.replace(bad, good)
    return s.strip()


def read_rows(path: Path):
    """Lee un CSV tolerante: busca filas con nombre de maquina y 'Tecnicas Vistas'."""
    rows = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        header = None
        for parts in reader:
            if not any(parts):
                continue
            joined = fix(" ".join(parts))
            # Detecta la cabecera (tiene 'quina' y 'cnicas')
            if header is None and ("quina" in joined or "Vulnerabilidad" in joined) and "cnicas" in joined.replace("Acc", "ec"):
                header = [fix(p) for p in parts]
                continue
            if header is None:
                continue
            # fila de datos: primera celda = nombre; busca la celda larga = tecnicas
            name = fix(parts[0]) if parts else ""
            tech_cell = max((fix(p) for p in parts), key=len, default="")
            if name and len(tech_cell) > 15 and tech_cell != name:
                rows.append((name, tech_cell))
    return rows


def techniques_list(cell: str) -> list[str]:
    """Las 'Tecnicas Vistas' vienen multilinea; las separa en items limpios."""
    items = [fix(t) for t in cell.replace("\r", "\n").split("\n")]
    return [t for t in items if t and len(t) > 3][:10]


def main():
    all_rows = []
    for fn in ["htb-machines.csv", "Vulnhub.csv", "web-hacking.csv"]:
        r = read_rows(ARCH / fn)
        print(f"  {fn:22s} -> {len(r)} filas con tecnicas")
        all_rows += r

    examples = []
    for name, cell in all_rows:
        techs = techniques_list(cell)
        if len(techs) < 2:
            continue
        tech_md = "\n".join(f"- {t}" for t in techs)
        # Plantilla generalizada: no da la solucion, da las tecnicas a considerar.
        user = rng.choice([
            f"En un pentest autorizado me enfrento a una maquina tipo '{name}'. "
            f"¿Que clases de tecnicas suelo tener que considerar en objetivos asi?",
            f"¿Que vectores de ataque son tipicos en una maquina como '{name}' (lab autorizado)?",
        ])
        assistant = (f"En objetivos de ese estilo, las tecnicas que conviene tener en el radar son:\n"
                     f"{tech_md}\n\n"
                     f"Recuerda: enumera a fondo cada servicio antes de atacar, y encadena las "
                     f"vulnerabilidades (una suele habilitar la siguiente). Solo en objetivos autorizados.")
        examples.append([
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ])

    rng.shuffle(examples)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        for ex in examples:
            f.write(json.dumps({"messages": ex}, ensure_ascii=False) + "\n")
    print(f"\n[OK] {len(examples)} ejemplos de tecnicas por maquina -> {OUT}")
    print("[i] Generalizados (tecnicas a considerar, no soluciones). Revisa un par de lineas por si la codificacion quedo limpia.")


if __name__ == "__main__":
    main()
