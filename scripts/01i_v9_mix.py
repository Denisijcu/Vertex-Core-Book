"""
01i_v9_mix.py — Arma el dataset UNICO de la v9 (desde la base, todo mezclado).

La v9 anade a la v8 el RAZONAMIENTO DE PENTESTING (generalizado, sin flags) y las
TECNICAS POR MAQUINA (de los CSV), y REFUERZA el tool calling (que en la v8 salio
inconsistente). Objetivo: que el modelo razone como pentester y use tools de forma
consistente, sin perder identidad/codigo/etica.

Composicion (balance acordado):
  - Tool calling (reforzado):      data/raw/toolcall_examples.jsonl   {"text": ...}
  - Razonamiento de pentesting:    data/raw/pentest_reasoning.jsonl   {"messages": ...}
  - Tecnicas por maquina (CSV):    data/raw/csv_techniques.jsonl      {"messages": ...}
  - Identidad + no-se + codigo:    ficheros raw de identidad          {"messages": ...}
  - Tecnico (replay):              muestra del dataset limpio         {"messages": ...}

Sin repetir en exceso (leccion v5). Mezcla y baraja.
Salida: data/processed/vertex_v9_dataset.jsonl

Uso (desde F:\\vertex-lab):
  .\\venv\\Scripts\\python.exe scripts\\01i_v9_mix.py
"""
import json
import random

from common import banner, load_config, read_system_prompt, rpath
from importlib import import_module

_mix = import_module("01b_identity_mix")


def _load_messages(path):
    p = rpath(path)
    if not p.exists():
        return []
    return [json.loads(l)["messages"] for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    cfg = load_config()
    v = cfg["stages"]["v9"]
    rng = random.Random(cfg["training"]["seed"])
    system = read_system_prompt(cfg)
    banner("01i - Mezcla unica v9 (pentest + tecnicas + tool calling + identidad + tecnico)")

    rows = []

    # 1) TOOL CALLING reforzado (formato text ya renderizado).
    tc_path = rpath("data/raw/toolcall_examples.jsonl")
    tc = [json.loads(l) for l in tc_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    for _ in range(int(v["toolcall_repeat"])):
        for r in tc:
            rows.append({"text": r["text"]})
    n_tc = len(rows)

    # 2) RAZONAMIENTO DE PENTESTING (generalizado), repetido para peso.
    pent = _load_messages("data/raw/pentest_reasoning.jsonl")
    for _ in range(int(v["pentest_repeat"])):
        for m in pent:
            rows.append({"messages": m})
    n_pent = len(rows) - n_tc

    # 3) TECNICAS POR MAQUINA (CSV): muestra (son 362; no todas, para no dominar).
    csvt = _load_messages("data/raw/csv_techniques.jsonl")
    rng.shuffle(csvt)
    csvt = csvt[: int(v["csv_samples"])]
    for m in csvt:
        rows.append({"messages": m})
    n_csv = len(rows) - n_tc - n_pent

    # 4) Identidad + no-lo-se + codigo, con upsampling moderado.
    identity = _mix.load_identity(v["identity_files"])
    id_users = {u for u, _ in identity}
    for user, out in identity:
        for _ in range(int(v["identity_repeat"])):
            msgs = [{"role": "user", "content": user}, {"role": "assistant", "content": out}]
            if rng.random() < v["identity_system_ratio"]:
                msgs.insert(0, {"role": "system", "content": system})
            rows.append({"messages": msgs})
    n_id = len(rows) - n_tc - n_pent - n_csv

    # 5) Tecnico (replay): muestra del dataset limpio.
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
    n_tech = len(rows) - n_tc - n_pent - n_csv - n_id

    rng.shuffle(rows)
    out_path = rpath(v["train_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    bs = cfg["training"]["per_device_train_batch_size"] * cfg["training"]["gradient_accumulation_steps"]
    total = len(rows)
    banner(f"{total} muestras -> {out_path}")
    print(f"[+] tool calling:          {n_tc}  ({n_tc/total:.0%})")
    print(f"[+] razonamiento pentest:  {n_pent}  ({n_pent/total:.0%})")
    print(f"[+] tecnicas por maquina:  {n_csv}  ({n_csv/total:.0%})")
    print(f"[+] identidad/codigo/no-se:{n_id}  ({n_id/total:.0%})")
    print(f"[+] tecnico (replay):      {n_tech}  ({n_tech/total:.0%})")
    print(f"[i] Pasos aprox. por epoca: {total//bs}")


if __name__ == "__main__":
    main()
