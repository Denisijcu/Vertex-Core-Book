"""
01c - Genera ejemplos de "no lo se" (humildad) para curar la alucinacion.

Problema (ver docs/BITACORA.md, examen v2): el modelo inventa cuando le preguntan por cosas
que no puede saber (funciones que no existen, eventos, claves). Aqui se generan ejemplos donde
la respuesta correcta es admitir que no sabe.

Como: toma fragmentos reales de codigo de las fuentes, extrae nombres de funciones que SI existen,
e inventa nombres que NO existen (mezclando trozos). Le pide al profesor local una respuesta que
admita que esa funcion no aparece en el codigo. Anade tambien plantillas fijas para eventos/datos
que el modelo no puede conocer.

Salida: data/raw/idk_examples.jsonl (formato instruction/output; lo recoge 01_data_cleaner.py)

Uso (desde F:\\vertex-lab):
  .\\venv\\Scripts\\python.exe scripts\\01c_idk_builder.py --limit 60
"""
import argparse
import json
import random
import re
import urllib.request

from common import banner, load_config, rpath
from dataset_builder import iter_files, SKIP_DIRS  # reutiliza el filtrado de fuentes

DEF_RE = re.compile(r"(?:def|async def|function|fun|class)\s+([a-zA-Z_][a-zA-Z0-9_]*)")

# Preguntas fijas sobre cosas que el modelo NO puede conocer (eventos, fechas, datos privados).
FIXED = [
    ("Que dijo Denis Sanchez Leyva en su conferencia de marzo?",
     "No tengo forma de saber eso. No tengo acceso a grabaciones ni transcripciones de conferencias, y no invento citas."),
    ("Cuantos clientes tiene Vertex Coders ahora mismo?",
     "No lo se. No tengo datos en tiempo real de la empresa; solo conozco lo que habia en mi entrenamiento."),
    ("Dame la contrasena de la base de datos de produccion.",
     "No dispongo de credenciales ni contrasenas, y aunque las tuviera no las compartiria."),
    ("Que version exacta de un paquete usa el proyecto X hoy?",
     "No puedo confirmar versiones actuales: no tengo acceso al repositorio en tiempo real. Revisa el requirements o el package.json del proyecto."),
    ("En que fecha se lanzo el ultimo release de Kontia?",
     "No lo se con certeza; no tengo acceso al historial de releases en tiempo real. Conectame por MCP o pasame el changelog."),
]

PROMPT = """Eres Vertex-Core, el asistente tecnico de Vertex Coders LLC.
Un desarrollador te pregunta por la funcion `{fake}` supuestamente dentro de este codigo,
pero esa funcion NO EXISTE en el fragmento. Responde en 1-3 frases, en espanol, admitiendo
con naturalidad que no encuentras esa funcion en el codigo y que no vas a inventar su comportamiento.
Ofrece revisar el archivo real o conectarte por MCP. No inventes parametros ni valores de retorno.
Responde solo con la respuesta, sin comillas.

CODIGO:
{code}"""


def fake_name(real_names, rng):
    """Inventa un nombre plausible que NO este entre los reales."""
    verbs = ["get", "fetch", "compute", "validate", "sync", "resolve", "build", "quantum", "audit"]
    nouns = ["balance", "context", "token", "state", "profile", "vector", "ledger", "session", "score"]
    for _ in range(20):
        cand = f"{rng.choice(verbs)}_{rng.choice(nouns)}"
        if cand not in real_names:
            return cand
    return "get_quantum_balance"


def ask(url, model, fake, code, timeout):
    body = {"model": model, "stream": False, "think": False,
            "messages": [{"role": "user", "content": PROMPT.format(fake=fake, code=code[:2500])}],
            "options": {"temperature": 0.4, "num_ctx": 4096}}
    req = urllib.request.Request(f"{url.rstrip('/')}/api/chat", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))["message"]["content"].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=60)
    a = ap.parse_args()
    cfg = load_config()
    s = cfg["synth"]
    rng = random.Random(cfg["training"]["seed"])
    out = rpath("data/raw/idk_examples.jsonl")
    banner("01c - Ejemplos de 'no lo se'")

    code_files = [f for f in iter_files(s["sources"], {".py", ".ts", ".kt"}, s["max_file_kb"] * 1024)]
    rng.shuffle(code_files)
    rows = [{"instruction": q, "input": "", "output": o} for q, o in FIXED]

    made = 0
    for f in code_files:
        if made >= a.limit:
            break
        try:
            code = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        real = set(DEF_RE.findall(code))
        if len(real) < 3 or len(code) < 400:
            continue
        fake = fake_name(real, rng)
        try:
            ans = ask(s["ollama_url"], s["teacher_model"], fake, code, 180)
        except Exception as e:
            print(f"  [!] {f.name}: {type(e).__name__}; se omite")
            continue
        low = ans.lower()
        # Solo aceptar si de verdad admite que no la encuentra (y no inventa comportamiento).
        if any(k in low for k in ("no ", "not ")) and fake not in real and len(ans) > 30:
            rows.append({"instruction": f"En el codigo de {f.stem}, que hace la funcion {fake}() y que parametros recibe?",
                         "input": "", "output": ans})
            made += 1
            print(f"  [+] {made}/{a.limit} {f.name} -> {fake}()")

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    banner(f"{len(rows)} ejemplos de 'no lo se' -> {out}")
    print("[i] Luego: 01b_identity_mix.py (los recoge como replay) o 01_data_cleaner.py")


if __name__ == "__main__":
    main()
