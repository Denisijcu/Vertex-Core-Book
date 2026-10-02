"""
01f_v8_mix.py — Arma el dataset UNICO de la v8 (desde la base, todo mezclado).

La v8 anade a lo que ya funcionaba (identidad, codigo, no-lo-se, CVEs) los ejemplos de
TOOL CALLING (formato Qwen3 ya renderizado), que es lo que faltaba en v1-v7.

Composicion:
  - Tool calling: data/raw/toolcall_examples.jsonl (formato {"text": ...}), repetido N veces.
  - Identidad + no-lo-se + codigo: los mismos ficheros raw de la v7, en formato {"messages": ...}.
  - Tecnico (replay): muestra de data/processed/vertex_train_dataset.jsonl.

Sin repetir en exceso (leccion v5: no memorizar). Mezcla y baraja.

Salida: data/processed/vertex_v8_dataset.jsonl

Uso (desde F:\\vertex-lab):
  .\\venv\\Scripts\\python.exe scripts\\01f_v8_mix.py
"""
import json
import random

from common import banner, load_config, read_system_prompt, rpath
from importlib import import_module

_mix = import_module("01b_identity_mix")  # reutiliza _iter_records / load_identity


def main():
    cfg = load_config()
    v = cfg["stages"]["v8"]
    rng = random.Random(cfg["training"]["seed"])
    system = read_system_prompt(cfg)
    banner("01f - Mezcla unica v8 (tool calling + identidad + no-se + codigo + tecnico)")

    rows = []

    # 1) TOOL CALLING (ya renderizado, formato text). Repetido para peso suficiente sin memorizar.
    tc_path = rpath("data/raw/toolcall_examples.jsonl")
    tc = [json.loads(l) for l in tc_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    for _ in range(int(v["toolcall_repeat"])):
        for r in tc:
            rows.append({"text": r["text"]})
    n_tc = len(rows)

    # 2) Identidad + no-lo-se + codigo (formato messages), con upsampling moderado.
    identity = _mix.load_identity(v["identity_files"])
    id_users = {u for u, _ in identity}
    for user, out in identity:
        for _ in range(int(v["identity_repeat"])):
            msgs = [{"role": "user", "content": user}, {"role": "assistant", "content": out}]
            if rng.random() < v["identity_system_ratio"]:
                msgs.insert(0, {"role": "system", "content": system})
            rows.append({"messages": msgs})
    n_id = len(rows) - n_tc

    # 3) Tecnico (replay): muestra del dataset limpio, con system prompt casi siempre.
    tech_path = rpath(cfg["dataset"]["train_path"])
    tech = [json.loads(l)["messages"] for l in tech_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    def user_of(m):
        return next((x["content"] for x in m if x["role"] == "user"), "")
    tech = [m for m in tech if user_of(m) not in id_users]
    rng.shuffle(tech)
    tech = tech[: int(v["replay_samples"])]
    for m in tech:
        convo = [x for x in m if x["role"] != "system"]
        if rng.random() < float(v.get("tech_system_ratio", 0.9)):
            convo = [{"role": "system", "content": system}, *convo]
        rows.append({"messages": convo})
    n_tech = len(rows) - n_tc - n_id

    rng.shuffle(rows)
    out_path = rpath(v["train_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    bs = cfg["training"]["per_device_train_batch_size"] * cfg["training"]["gradient_accumulation_steps"]
    banner(f"{len(rows)} muestras -> {out_path}")
    print(f"[+] tool calling: {n_tc} ({n_tc/len(rows):.0%})")
    print(f"[+] identidad/codigo/no-se: {n_id} ({n_id/len(rows):.0%})")
    print(f"[+] tecnico (replay): {n_tech} ({n_tech/len(rows):.0%})")
    print(f"[i] Pasos aprox. por epoca: {len(rows)//bs}")


if __name__ == "__main__":
    main()
