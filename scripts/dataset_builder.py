"""
Generador de dataset LOCAL para Vertex-Core.

Lee TU material (manuscritos de libros, READMEs, docs y código de tus repos), lo trocea
y le pide a un modelo abierto que corre en tu Ollama (licencia Apache 2.0, p. ej. qwen3:8b)
que genere pares pregunta/respuesta fundamentados SOLO en cada fragmento.

Salida: data/raw/synth_from_sources.jsonl (formato instruction/output que entiende 01_data_cleaner.py)
         + data/raw/synth_review.md para que revises las muestras antes de entrenar.
Reanudable: guarda qué fragmentos ya procesó (Ctrl+C y vuelves a lanzarlo sin perder nada).

IMPORTANTE: no lo corras mientras se entrena (ambos usan la GPU de 6 GB).

Uso (desde F:\\vertex-lab):
  ollama pull qwen3:8b                                                         # una vez
  .\\venv\\Scripts\\python.exe scripts\\dataset_builder.py --dry-run             # ver qué archivos/fragmentos entran
  .\\venv\\Scripts\\python.exe scripts\\dataset_builder.py --limit 10            # prueba con 10 fragmentos
  .\\venv\\Scripts\\python.exe scripts\\dataset_builder.py                       # todo
"""
import argparse
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from common import banner, load_config, rpath

SKIP_DIRS = {".git", "node_modules", "venv", ".venv", "__pycache__", "dist", "build", ".hf_cache",
             ".angular", ".gradle", ".idea", "outputs", "target", "bin", "obj", ".next", "coverage",
             "site-packages", "env", "generated", "intermediates", "tests", "test",
             # código de terceros incluido en los repos (no es material de Vertex Coders)
             "liboqs", "vendor", "third_party", "third-party", "external", "deps", "migrations", "alembic"}
SKIP_DIR_PATTERNS = [re.compile(p, re.I) for p in (r"respaldo", r"backup", r"^_?old$", r"\.bak$")]
SKIP_FILE_PATTERNS = [re.compile(p, re.I) for p in (
    r"^\.env", r"secret", r"credential", r"\.lock$", r"package-lock\.json$",
    r"borrador", r"^construir_epub\.py$", r"^tabla_de_contenido", r"^PLAN\.md$", r"^test_.*\.py$",
    # material editorial/marketing: no es conocimiento técnico
    r"kdp", r"ficha", r"youtube", r"portada", r"^FRAMEWORK\.md$", r"descripcion", r"marketing",
    r"^CHANGELOG", r"^TODO", r"^LICENSE", r"agradecimiento", r"sobre-el-autor", r"creditos", r"indice",
    # build/deps y notas de estado o negocio (caducan o llevan datos de terceros)
    r"^CMakeLists\.txt$", r"^requirements.*\.txt$", r"briefing", r"retomar", r"^STATUS", r"handoff",
    r"^backfill_", r"seed", r"fixture", r"^_.*\.txt$", r"report\.txt$",
)]

SECRET_PATTERNS = [re.compile(p) for p in (
    r"sk-[A-Za-z0-9_\-]{20,}",                       # OpenAI / Anthropic style keys
    r"AKIA[0-9A-Z]{16}",                             # AWS
    r"ghp_[A-Za-z0-9]{30,}",                         # GitHub
    r"AIza[0-9A-Za-z_\-]{30,}",                      # Google
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    r"(?i)(password|passwd|api[_-]?key|secret|token)\s*[:=]\s*['\"][^'\"]{6,}['\"]",
)]
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")

SCHEMA = {
    "type": "object",
    "properties": {"pairs": {"type": "array", "items": {
        "type": "object",
        "properties": {"instruction": {"type": "string"}, "output": {"type": "string"}},
        "required": ["instruction", "output"]}}},
    "required": ["pairs"],
}

TEACHER_PROMPT = """Eres un generador de datos de entrenamiento para "Vertex-Core", el asistente técnico de Vertex Coders LLC.
Recibirás un FRAGMENTO del material técnico propio de la empresa (libros técnicos, documentación o código).

GLOSARIO FIJO (nunca lo cambies ni lo traduzcas distinto):
- MCP = Model Context Protocol (protocolo abierto para conectar modelos de lenguaje con herramientas y datos).
- LLM = modelo de lenguaje grande. GraphQL, FastAPI, Angular, Jetpack Compose, Kotlin, PowerShell: nombres propios.

Genera hasta {n} pares pregunta/respuesta basados en el fragmento:
- La "instruction" es lo que preguntaría un desarrollador real al trabajar: "¿cómo hago X?", "¿por qué falla Y?",
  "revisa este código", "¿A o B y por qué?". Pregunta por riesgos de seguridad SOLO si el fragmento describe un
  riesgo real y concreto; nunca fuerces el tema. Si el fragmento tiene código, al menos una pregunta debe incluir
  un trozo de ese código y pedir explicarlo, mejorarlo o depurarlo.
- La "output" es Vertex-Core hablando DIRECTAMENTE al desarrollador, como un ingeniero senior: afirmaciones
  concretas, el porqué técnico, pasos o código en bloques con lenguaje, y el riesgo de seguridad solo si existe.
  Respuestas completas: nunca cortes una frase ni dejes un bloque de código sin cerrar.
  En español (o en inglés si el fragmento está en inglés).

PROHIBIDO:
- Voz de resumen o de terceros: "se señala", "se menciona", "se recomienda", "el autor", "el libro", "el documento",
  "el fragmento", "según".
- Estados de proyecto o cosas que caducan: fechas, "aún no se ha", "está pendiente", "estado actual", planes, ventas.
- Inventar APIs, nombres de funciones, cifras o soluciones que no estén en el fragmento ni sean conocimiento estándar.
- Nombres de personas o clientes, correos, IPs, rutas personales, credenciales.
- Temas editoriales o de marketing (portadas, KDP, descripciones de producto, YouTube).
- Preguntas sobre la estructura del material (partes, capítulos, secciones, "qué pasa si me salto X"):
  pregunta por la TÉCNICA, como si el desarrollador nunca hubiera visto el material.
- Riesgos de seguridad inventados o exagerados.

Si el fragmento no permite preguntas técnicas de calidad, devuelve "pairs": []. Menos pares buenos > muchos flojos.
Responde solo con el JSON."""


def iter_files(sources, exts, max_bytes):
    for src in sources:
        root = Path(src)
        if not root.exists():
            print(f"[!] No existe la fuente: {root}")
            continue
        files = [root] if root.is_file() else root.rglob("*")
        for f in files:
            if not f.is_file() or f.suffix.lower() not in exts:
                continue
            if any(part in SKIP_DIRS for part in f.parts):
                continue
            if any(p.search(part) for part in f.parts[:-1] for p in SKIP_DIR_PATTERNS):
                continue
            if any(p.search(f.name) for p in SKIP_FILE_PATTERNS):
                continue
            if f.stat().st_size > max_bytes:
                continue
            yield f


def chunk_text(text, size):
    paras, buf = re.split(r"\n\s*\n", text), ""
    for p in paras:
        if len(buf) + len(p) > size and buf:
            yield buf.strip()
            buf = ""
        buf += p + "\n\n"
        while len(buf) > size * 1.5:          # párrafos gigantes (código largo)
            yield buf[:size].strip()
            buf = buf[size:]
    if buf.strip():
        yield buf.strip()


def is_clean(text, blocklist):
    if any(p.search(text) for p in SECRET_PATTERNS):
        return False
    low = text.lower()
    return not any(b.lower() in low for b in blocklist)


def scrub(text):
    return UUID_RE.sub("<uuid>", IPV4_RE.sub("<IP>", EMAIL_RE.sub("<email>", text)))


def ask_teacher(url, model, chunk, n, timeout):
    body = {
        "model": model,
        "messages": [{"role": "system", "content": TEACHER_PROMPT.format(n=n)},
                     {"role": "user", "content": f"FRAGMENTO:\n\n{chunk}"}],
        "stream": False,
        "think": False,
        "format": SCHEMA,
        "keep_alive": "10m",
        "options": {"temperature": 0.5, "num_ctx": 8192},
    }
    data = json.dumps(body).encode("utf-8")
    last = None
    for intento in range(2):   # una petición + un reintento
        try:
            req = urllib.request.Request(f"{url.rstrip('/')}/api/chat", data=data,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                content = json.loads(r.read().decode("utf-8"))["message"]["content"]
            return json.loads(content).get("pairs", [])
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last = e
            if intento == 0:
                time.sleep(5)   # dale un respiro a Ollama y reintenta una vez
    raise last


def looks_complete(out):
    """Descarta respuestas truncadas: bloques/backticks sin cerrar o frase cortada."""
    if out.count("```") % 2:
        return False
    prose = re.sub(r"```.*?```", "", out, flags=re.S)
    if prose.count("`") % 2:
        return False
    tail = out.rstrip()
    return tail.endswith("```") or tail[-1:] in ".!?)»\"'`*:|]"


def valid_pair(p, blocklist):
    ins, out = str(p.get("instruction", "")).strip(), str(p.get("output", "")).strip()
    if len(ins) < 10 or len(out) < 60:
        return None
    bad = ("el fragmento", "el texto proporcionado", "como modelo de lenguaje", "as an ai language model",
           "se señala", "se menciona", "el autor", "el libro", "el documento", "según el",
           "aún no se ha", "está pendiente", "se pendiente", "estado actual", "modelo de contenido",
           "kdp", "portada", "el capítulo", "este capítulo")
    low = (ins + " " + out).lower()
    if any(b in low for b in bad) or not is_clean(ins + out, blocklist):
        return None
    if re.search(r"\b(la )?parte (i{1,3}|iv|v|vi{1,3})\b", low):   # "Parte III", no "la parte importante"
        return None
    if not looks_complete(out):
        return None
    return {"instruction": scrub(ins), "input": "", "output": scrub(out)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="Máximo de fragmentos a procesar en esta corrida")
    ap.add_argument("--reset", action="store_true", help="Archiva la salida y el estado actuales (.bak) y empieza de cero")
    a = ap.parse_args()

    cfg = load_config()
    s = cfg["synth"]
    out_file, review_file = rpath(s["out_file"]), rpath(s["review_file"])
    state_file = rpath("data/.synth_state.json")   # fuera de data/raw para que 01 no lo lea
    if a.reset:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        bak_dir = rpath("data/synth_bak")
        bak_dir.mkdir(parents=True, exist_ok=True)
        for f in (out_file, review_file, state_file):
            if f.exists():
                f.rename(bak_dir / f"{stamp}_{f.name}")
                print(f"[i] Archivado: {f.name} -> data\\synth_bak\\{stamp}_{f.name}")
    done = set(json.loads(state_file.read_text(encoding="utf-8"))) if state_file.exists() else set()
    exts = {e.lower() for e in s["extensions"]}
    banner(f"Dataset builder | profesor: {s['teacher_model']} | fuentes: {len(s['sources'])}")

    chunks, seen_now = [], set()
    for f in iter_files(s["sources"], exts, s["max_file_kb"] * 1024):
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for c in chunk_text(text, s["chunk_chars"]):
            if len(c) < 300 or not is_clean(c, s["blocklist"]):
                continue
            h = hashlib.sha1(c.encode("utf-8")).hexdigest()
            if h not in done and h not in seen_now:
                seen_now.add(h)
                chunks.append((f, h, c))

    print(f"[*] Fragmentos pendientes: {len(chunks)} (ya procesados: {len(done)})")
    if a.dry_run:
        by_file = {}
        for f, _, _ in chunks:
            by_file[str(f)] = by_file.get(str(f), 0) + 1
        for f, n in sorted(by_file.items(), key=lambda x: -x[1])[:40]:
            print(f"    {n:4d}  {f}")
        print(f"[i] Estimado: ~{len(chunks) * s['pairs_per_chunk']} pares")
        return
    if a.limit:
        chunks = chunks[: a.limit]

    out_file.parent.mkdir(parents=True, exist_ok=True)
    total, t0 = 0, time.time()
    try:
        with open(out_file, "a", encoding="utf-8", newline="\n") as out, \
             open(review_file, "a", encoding="utf-8", newline="\n") as rev:
            for i, (f, h, c) in enumerate(chunks, 1):
                try:
                    pairs = ask_teacher(s["ollama_url"], s["teacher_model"], c, s["pairs_per_chunk"], s["timeout_s"])
                except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as e:
                    print(f"  [!] {i}/{len(chunks)} {f.name}: {type(e).__name__}: {e} (se reintenta en la próxima corrida)")
                    continue
                kept = [p for p in (valid_pair(x, s["blocklist"]) for x in pairs) if p]
                for p in kept:
                    out.write(json.dumps({**p, "source": str(f)}, ensure_ascii=False) + "\n")
                    rev.write(f"### {p['instruction']}\n_fuente: {f.name}_\n\n{p['output']}\n\n---\n\n")
                out.flush(); rev.flush()
                done.add(h)
                total += len(kept)
                state_file.write_text(json.dumps(sorted(done)), encoding="utf-8")
                rate = (time.time() - t0) / i
                print(f"  [+] {i}/{len(chunks)} {f.name}: {len(kept)} pares | total {total} | ~{rate * (len(chunks) - i) / 60:.0f} min restantes")
    except KeyboardInterrupt:
        print("\n[i] Pausado. Vuelve a lanzarlo y continúa donde se quedó.")

    banner(f"{total} pares nuevos -> {out_file}")
    print(f"[i] Revisa {review_file} y borra del .jsonl lo que no te guste. Luego: scripts\\01_data_cleaner.py")


if __name__ == "__main__":
    main()
