"""
01b - Arma el dataset de la etapa de REFUERZO DE IDENTIDAD.

Problema que resuelve (ver docs/BITACORA.md, 2026-09-25): con 45 ejemplos de identidad entre
~3,600 pares técnicos (1.2%) y una sola época, el modelo perdió la identidad de Qwen sin aprender
la suya. Aquí la identidad se REPITE (upsampling) y se mezcla con una muestra de pares técnicos
ya vistos (replay) para que no olvide lo técnico mientras refuerza quién es.

Uso (desde F:\\vertex-lab):
  .\\venv\\Scripts\\python.exe scripts\\01b_identity_mix.py
  .\\venv\\Scripts\\python.exe scripts\\02_trainer.py --stage identity
"""
import argparse
import json
import random

from common import banner, load_config, read_system_prompt, rpath


def _iter_records(path):
    """Acepta .json (lista) y .jsonl (una linea por objeto)."""
    text = rpath(path).read_text(encoding="utf-8-sig").strip()
    if str(path).endswith(".jsonl"):
        for line in text.splitlines():
            if line.strip():
                yield json.loads(line)
    else:
        data = json.loads(text)
        yield from (data if isinstance(data, list) else [data])


def load_identity(paths):
    items = []
    for p in paths:
        for it in _iter_records(p):
            ins, out = str(it.get("instruction", "")).strip(), str(it.get("output", "")).strip()
            inp = str(it.get("input", "")).strip()
            if ins and out:
                if "AQUÍ TU RESPUESTA" in out.upper() or out.strip().upper().startswith("PLACEHOLDER") or out.strip().lower() in {"tu respuesta", "tu última respuesta"}:
                    raise SystemExit(f"[!] Texto de relleno en {p}: '{ins}'. Escribe la respuesta real antes de entrenar.")
                user = f"{ins}\n\nContexto/Entrada:\n{inp}" if inp else ins
                items.append((user, out))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="identity", help="Etapa de 'stages:' del YAML (identity, honesty...)")
    a = ap.parse_args()
    cfg = load_config()
    st = cfg["stages"][a.stage]
    seed = cfg["training"]["seed"]
    rng = random.Random(seed)
    system = read_system_prompt(cfg)
    banner(f"01b - Mezcla de la etapa '{a.stage}'")

    identity = load_identity(st["identity_files"])
    if not identity:
        raise SystemExit("[!] No hay ejemplos de identidad.")

    rows = []
    rep, sys_ratio = int(st["identity_repeat"]), float(st["identity_system_ratio"])
    for user, out in identity:
        for r in range(rep):
            msgs = [{"role": "user", "content": user}, {"role": "assistant", "content": out}]
            if rng.random() < sys_ratio:
                msgs.insert(0, {"role": "system", "content": system})
            rows.append(msgs)
    n_identity = len(rows)

    train_path = rpath(cfg["dataset"]["train_path"])
    replay_pool = [json.loads(l)["messages"] for l in train_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    # Solo pares técnicos: los de identidad ya están arriba (se detectan por aparecer en identity).
    id_users = {u for u, _ in identity}
    replay_pool = [m for m in replay_pool if next(x["content"] for x in m if x["role"] == "user") not in id_users]

    # La v1 aprendió a responder demasiado corto (el profesor escribía respuestas breves).
    # El replay se toma de las respuestas MÁS COMPLETAS: con bloque de código y/o explicación larga.
    def richness(m):
        ans = m[-1]["content"]
        return ("```" in ans) * 2 + (len(ans) >= int(st.get("replay_min_chars", 400)))

    rich = [m for m in replay_pool if richness(m) >= 2]
    medium = [m for m in replay_pool if richness(m) == 1]
    print(f"[i] replay disponible: {len(rich)} con código, {len(medium)} largas sin código, {len(replay_pool)} total")
    ordered = rich + medium if len(rich) + len(medium) >= int(st["replay_samples"]) else replay_pool
    rng.shuffle(rich)
    rng.shuffle(medium)
    n_replay = min(int(st["replay_samples"]), len(ordered))
    picked = (rich + medium)[:n_replay] if ordered is not replay_pool else rng.sample(replay_pool, n_replay)
    rows.extend(picked)

    rng.shuffle(rows)
    out_path = rpath(st["train_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps({"messages": r}, ensure_ascii=False) + "\n")

    n_nosys = sum(1 for r in rows[:] if r[0]["role"] != "system")
    banner(f"{len(rows)} muestras -> {out_path}")
    print(f"[+] identidad: {len(identity)} ejemplos x {rep} = {n_identity} ({n_identity / len(rows):.0%} de la mezcla)")
    print(f"[+] replay técnico: {n_replay} | sin system prompt (total): {n_nosys}")
    print(f"[i] Pasos aprox.: {len(rows) // (cfg['training']['per_device_train_batch_size'] * cfg['training']['gradient_accumulation_steps'])}")


if __name__ == "__main__":
    main()
