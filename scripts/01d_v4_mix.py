"""
01d - Arma el dataset UNICO y completo para la v4 (una sola etapa, desde el modelo base).

Por que (ver docs/BITACORA.md): entrenar identidad y humildad en etapas encadenadas hace que
cada etapa pise a la anterior (v2 arreglo identidad pero inventaba; v3 curo el invento pero
volvio a dudar de quien es). La v4 mezcla TODO desde el principio y entrena una sola vez sobre
la base, para que aprenda el equilibrio de una vez.

Composicion (equilibrada a mano para que ni la identidad domine ni se diluya):
  - Tecnico: todo data/processed/vertex_train_dataset.jsonl (lo que ya paso el 01_data_cleaner)
  - Identidad: data/raw/identity.json, repetida N veces
  - "No lo se": data/raw/idk_examples.jsonl, repetida N veces
  - Fraccion sin system prompt controlada, para que la identidad viva en los pesos SIN contaminar
    las respuestas tecnicas (la clave del equilibrio).

Salida: data/processed/vertex_v4_dataset.jsonl

Uso (desde F:\\vertex-lab):
  .\\venv\\Scripts\\python.exe scripts\\01_data_cleaner.py        # asegura el tecnico actualizado
  .\\venv\\Scripts\\python.exe scripts\\01d_v4_mix.py
  .\\venv\\Scripts\\python.exe scripts\\02_trainer.py --stage v4
"""
import argparse
import json
import random

from common import banner, load_config, read_system_prompt, rpath
from importlib import import_module

_mix = import_module("01b_identity_mix")  # reutiliza _iter_records / load_identity


def to_msgs(user, out, system, with_system, rng):
    msgs = [{"role": "user", "content": user}, {"role": "assistant", "content": out}]
    if with_system:
        msgs.insert(0, {"role": "system", "content": system})
    return msgs


def main():
    ap = argparse.ArgumentParser()
    a = ap.parse_args()
    cfg = load_config()
    v = cfg["stages"]["v4"]
    rng = random.Random(cfg["training"]["seed"])
    system = read_system_prompt(cfg)
    banner("01d - Mezcla unica v4 (identidad + no-se + tecnico, una sola etapa)")

    rows = []

    # 1) Identidad, con upsampling. Fraccion alta SIN system prompt: la identidad debe vivir en los pesos.
    identity = _mix.load_identity(v["identity_files"])
    id_users = {u for u, _ in identity}
    for user, out in identity:
        for _ in range(int(v["identity_repeat"])):
            rows.append(to_msgs(user, out, system, rng.random() < v["identity_system_ratio"], rng))
    n_id = len(rows)

    # 2) "No lo se", con upsampling. Aqui el system prompt SI ayuda (son preguntas de uso real).
    idk = _mix.load_identity(v["idk_files"])
    idk_users = {u for u, _ in idk}
    n_before = len(rows)
    for user, out in idk:
        for _ in range(int(v["idk_repeat"])):
            rows.append(to_msgs(user, out, system, rng.random() < v["idk_system_ratio"], rng))
    n_idk = len(rows) - n_before

    # 3) Tecnico: todo el dataset ya limpio, sin duplicar lo que sea identidad/idk.
    tech_path = rpath(cfg["dataset"]["train_path"])
    tech = [json.loads(l)["messages"] for l in tech_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    def user_of(m):
        return next((x["content"] for x in m if x["role"] == "user"), "")
    tech = [m for m in tech if user_of(m) not in id_users and user_of(m) not in idk_users]
    # Reescribe el system prompt del tecnico segun tech_system_ratio (el dataset base trae 40% sin
    # prompt; para la v4 el tecnico casi siempre lo lleva, para que el "no lo se" no se pegue a todo).
    tech_ratio = float(v.get("tech_system_ratio", 0.9))
    fixed = []
    for m in tech:
        convo = [x for x in m if x["role"] != "system"]
        if rng.random() < tech_ratio:
            convo = [{"role": "system", "content": system}, *convo]
        fixed.append(convo)
    rows.extend(fixed)
    n_tech = len(fixed)

    rng.shuffle(rows)
    out_path = rpath(v["train_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps({"messages": r}, ensure_ascii=False) + "\n")

    n_nosys = sum(1 for r in rows if r[0]["role"] != "system")
    bs = cfg["training"]["per_device_train_batch_size"] * cfg["training"]["gradient_accumulation_steps"]
    banner(f"{len(rows)} muestras -> {out_path}")
    print(f"[+] identidad: {len(identity)} x {v['identity_repeat']} = {n_id}  ({n_id/len(rows):.0%})")
    print(f"[+] no-lo-se:  {len(idk)} x {v['idk_repeat']} = {n_idk}  ({n_idk/len(rows):.0%})")
    print(f"[+] tecnico:   {n_tech}  ({n_tech/len(rows):.0%})")
    print(f"[+] sin system prompt: {n_nosys} ({n_nosys/len(rows):.0%})")
    print(f"[i] Pasos aprox. por epoca: {len(rows)//bs}")


if __name__ == "__main__":
    main()
