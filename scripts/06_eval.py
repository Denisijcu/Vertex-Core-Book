"""
06 - Examen de estres de Vertex-Core contra Ollama.

Corre el set fijo de data/eval/identity_eval.json contra dos modelos de Ollama
(el afinado y su base) y produce:
  - un veredicto automatico por caso (donde hay expect_any / expect_none),
  - un informe lado a lado en Markdown para juzgar a mano lo demas.

Prueba el GGUF final tal como lo usara la gente. Requiere los dos modelos en Ollama:
  ollama create vertex-core -f Modelfile           (paso 05)
  ollama pull qwen3:4b-instruct  (o la base que corresponda, para comparar)

Uso (desde F:\\vertex-lab):
  .\\venv\\Scripts\\python.exe scripts\\06_eval.py
  .\\venv\\Scripts\\python.exe scripts\\06_eval.py --model vertex-core --base qwen3:4b-instruct
  .\\venv\\Scripts\\python.exe scripts\\06_eval.py --no-base          # solo el modelo afinado
"""
import argparse
import json
import re
import time
import unicodedata
import urllib.error
import urllib.request

from common import ROOT, banner, load_config, rpath

EVAL_FILE = rpath("data/eval/identity_eval.json")


def norm(s):
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def ollama_chat(url, model, system, user, timeout):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    body = {"model": model, "messages": msgs, "stream": False,
            "options": {"temperature": 0.3, "num_ctx": 4096}}
    req = urllib.request.Request(f"{url.rstrip('/')}/api/chat", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))["message"]["content"].strip()


NEGATIONS = ("no ", "not ", "ni ", "nunca ", "never ", "n't ", "tampoco ")


def has_term(text, term):
    """Coincidencia por PALABRA COMPLETA ("no" no debe encontrarse dentro de "patrimonio")."""
    return re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text) is not None


def contains_affirmed(text, term):
    """True si `term` aparece en `text` SIN una negación justo antes ("no soy claude" no cuenta como "soy claude")."""
    for m in re.finditer(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text):
        before = text[max(0, m.start() - 12):m.start()]
        if not any(before.endswith(n) for n in NEGATIONS):
            return True
    return False


def verdict(answer, caso):
    a = norm(answer)
    any_ok = [t for t in caso.get("expect_any", []) if has_term(a, norm(t))]
    bad = [t for t in caso.get("expect_none", []) if contains_affirmed(a, norm(t))]
    has_any = "expect_any" in caso
    if bad:
        return "FALLA", f"contiene prohibido: {bad}"
    if has_any and not any_ok:
        return "FALLA", f"falta esperado: {caso['expect_any']}"
    if has_any or "expect_none" in caso:
        return "OK", ""
    return "REVISAR", "sin criterio automatico"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="Modelo afinado en Ollama (por defecto ollama.model_name)")
    ap.add_argument("--base", default="qwen3:4b-instruct", help="Modelo base en Ollama para comparar")
    ap.add_argument("--no-base", action="store_true", help="No comparar con la base")
    ap.add_argument("--url", default="http://127.0.0.1:11434")
    ap.add_argument("--timeout", type=int, default=180)
    a = ap.parse_args()

    cfg = load_config()
    model = a.model or cfg["ollama"]["model_name"]
    system = rpath(cfg["prompt"]["system_prompt_path"]).read_text(encoding="utf-8").strip()
    data = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    banner(f"06 - Examen de estres | modelo: {model}" + ("" if a.no_base else f" vs base: {a.base}"))

    lines = [f"# Examen de estres — Vertex-Core\n",
             f"_Modelo: **{model}**" + ("" if a.no_base else f" · Base: **{a.base}**") + "_\n"]
    totals = {}

    for bloque in data["bloques"]:
        use_system = not bloque.get("no_system", False)
        lines.append(f"\n## {bloque['titulo']}\n\n_{bloque['descripcion']}_")
        lines.append(f"\nSystem prompt: {'SI' if use_system else 'NO (mide los pesos)'}\n")
        passed = fails = review = 0

        for caso in bloque["casos"]:
            sys_txt = system if use_system else None
            try:
                ans = ollama_chat(a.url, model, sys_txt, caso["q"], a.timeout)
            except (urllib.error.URLError, TimeoutError) as e:
                ans = f"[ERROR: {e}]"
            v, why = verdict(ans, caso)
            passed += v == "OK"; fails += v == "FALLA"; review += v == "REVISAR"
            mark = {"OK": "PASA", "FALLA": "FALLA", "REVISAR": "revisar"}[v]

            print(f"  [{mark:6}] {caso['q'][:70]}" + (f"  <- {why}" if why else ""))
            lines.append(f"\n**P: {caso['q']}**  \n`{mark}`" + (f" — {why}" if why else ""))
            lines.append(f"\n> **{model}:** {ans}")
            if not a.no_base:
                try:
                    base_ans = ollama_chat(a.url, a.base, sys_txt, caso["q"], a.timeout)
                except (urllib.error.URLError, TimeoutError) as e:
                    base_ans = f"[ERROR: {e}]"
                lines.append(f"\n> **{a.base} (base):** {base_ans}")
            lines.append("")

        totals[bloque["titulo"]] = (passed, fails, review, len(bloque["casos"]))
        print(f"  -> {bloque['titulo']}: {passed} PASA / {fails} FALLA / {review} revisar")

    lines.insert(2, "\n## Resumen\n\n| Bloque | Pasa | Falla | Revisar |\n|---|---|---|---|\n" +
                 "\n".join(f"| {k} | {v[0]} | {v[1]} | {v[2]} |" for k, v in totals.items()) + "\n")

    out = ROOT / "outputs" / "eval_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    tot_pass = sum(v[0] for v in totals.values())
    tot_fail = sum(v[1] for v in totals.values())
    tot_rev = sum(v[2] for v in totals.values())
    banner(f"Automatico: {tot_pass} PASA / {tot_fail} FALLA / {tot_rev} a revisar a mano")
    print(f"[i] Informe lado a lado: {out}")
    print("[i] Regla del proyecto: si algun bloque de identidad o seguridad tiene FALLA, no se publica.")


if __name__ == "__main__":
    main()
