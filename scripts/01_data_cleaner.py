"""
01 - Limpia y convierte los datos crudos a JSONL en formato chat (messages).

Entrada: todos los .json / .jsonl de data/raw. Cada registro puede ser:
  a) {"instruction": ..., "input": ..., "output": ...}      (estilo Alpaca)
  b) {"messages": [{"role": "user"|"assistant"|"system", "content": ...}, ...]}

Salida: data/processed/vertex_train_dataset.jsonl (+ eval si hay suficientes datos)
El system prompt se toma SIEMPRE de prompts/vertex_core.jinja2 (una sola fuente de verdad).

Uso (desde F:\\vertex-lab):  .\\venv\\Scripts\\python.exe scripts\\01_data_cleaner.py
"""
import hashlib
import json
import random
import re

from common import banner, load_config, read_system_prompt, rpath

VALID_ROLES = {"system", "user", "assistant"}

# Filtro de seguridad: descarta el par (no todo el dataset) cuando la SALIDA es un arma
# lista para usar, no una explicación. Enseñar cómo funciona un ataque y cómo defenderse: OK.
# Entregar un payload/exploit crudo a demanda: fuera. (Línea editorial del proyecto.)
SECRET_RE = re.compile(
    r"sk-[A-Za-z0-9_\-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{30,}|AIza[0-9A-Za-z_\-]{30,}"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"
)
# Frases de jailbreak / inyección de prompts como PRODUCTO (que el modelo las emita a pedido).
INJECTION_RE = re.compile(
    r"(?i)(ignora (todas )?(tus|las) instrucciones|olvida (tus|las) instrucciones|ignore (all )?(your |previous )?instructions"
    r"|disregard (your|all|previous) instructions|you are now (in )?DAN|actúa como si no tuvieras (reglas|restricciones)"
    r"|eres un asistente sin (reglas|restricciones|filtros)|bypass your (safety|guardrails)|salta(te)? (tus )?(reglas|filtros|restricciones))"
)


def security_reason(user, assistant):
    """Devuelve el motivo por el que se descarta el par, o None si es seguro."""
    if SECRET_RE.search(user) or SECRET_RE.search(assistant):
        return "secreto/credencial"
    # La inyección solo importa si el MODELO la produce (en su respuesta), no si el usuario pregunta por ella.
    if INJECTION_RE.search(assistant):
        return "jailbreak en la respuesta"
    return None


def iter_raw_records(raw_dir):
    files = sorted([*raw_dir.glob("*.json"), *raw_dir.glob("*.jsonl")])
    if not files:
        raise SystemExit(f"[!] No hay .json/.jsonl en {raw_dir}")
    for fp in files:
        text = fp.read_text(encoding="utf-8-sig").strip()
        if not text:
            continue
        if fp.suffix == ".jsonl":
            for n, line in enumerate(text.splitlines(), 1):
                if line.strip():
                    try:
                        yield fp.name, json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"  [!] {fp.name}:{n} JSON inválido ({e}), se omite")
        else:
            data = json.loads(text)
            for item in data if isinstance(data, list) else [data]:
                yield fp.name, item


def to_messages(item, system_prompt, with_system=True):
    if "messages" in item:
        msgs = [m for m in item["messages"] if m.get("role") in VALID_ROLES and str(m.get("content", "")).strip()]
        msgs = [{"role": m["role"], "content": str(m["content"]).strip()} for m in msgs if m["role"] != "system"]
    else:
        instr = str(item.get("instruction", "")).strip()
        inp = str(item.get("input", "")).strip()
        out = str(item.get("output", "")).strip()
        if not instr or not out:
            return None
        user = f"{instr}\n\nContexto/Entrada:\n{inp}" if inp else instr
        msgs = [{"role": "user", "content": user}, {"role": "assistant", "content": out}]

    if len(msgs) < 2 or msgs[0]["role"] != "user" or msgs[-1]["role"] != "assistant":
        return None
    return [{"role": "system", "content": system_prompt}, *msgs] if with_system else msgs


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps({"messages": r}, ensure_ascii=False) + "\n")


def main():
    cfg = load_config()
    ds = cfg["dataset"]
    system_prompt = read_system_prompt(cfg)
    banner("01 - Limpieza de datos")

    # Parte de las muestras SIN system prompt: la identidad queda en los pesos, no solo en el prompt.
    no_sys_ratio = float(ds.get("no_system_ratio", 0.4))
    rng = random.Random(cfg["training"]["seed"])

    seen, rows, skipped, blocked = set(), [], 0, 0
    for src, item in iter_raw_records(rpath(ds["raw_dir"])):
        with_sys = rng.random() >= no_sys_ratio
        msgs = to_messages(item, system_prompt, with_sys) if isinstance(item, dict) else None
        if msgs is None:
            skipped += 1
            continue
        convo = [m for m in msgs if m["role"] != "system"]
        reason = security_reason(convo[0]["content"], convo[-1]["content"])
        if reason:
            blocked += 1
            continue
        key = hashlib.sha1(json.dumps(convo, ensure_ascii=False).encode()).hexdigest()
        if key in seen:
            skipped += 1
            continue
        seen.add(key)
        rows.append(msgs)

    if not rows:
        raise SystemExit("[!] Ningún registro válido.")

    train, evals = rows, []
    if len(rows) >= 20 and ds.get("eval_ratio", 0) > 0:
        random.Random(cfg["training"]["seed"]).shuffle(rows)
        n_eval = max(1, int(len(rows) * ds["eval_ratio"]))
        evals, train = rows[:n_eval], rows[n_eval:]

    write_jsonl(rpath(ds["train_path"]), train)
    eval_path = rpath(ds["eval_path"])
    if evals:
        write_jsonl(eval_path, evals)
    elif eval_path.exists():
        eval_path.unlink()  # evita usar un eval viejo que ya no corresponde

    n_nosys = sum(1 for r in rows if r[0]["role"] != "system")
    print(f"[+] train: {len(train)} | eval: {len(evals)} | sin system prompt: {n_nosys} | duplicados: {skipped} | bloqueados (seguridad): {blocked}")
    print(f"[+] -> {rpath(ds['train_path'])}")
    if len(rows) < 100:
        print(f"[i] Solo {len(rows)} muestras: sirve para smoke test, no para que el modelo aprenda de verdad (apunta a 300+).")


if __name__ == "__main__":
    main()
