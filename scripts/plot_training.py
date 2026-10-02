"""
Genera las gráficas de entrenamiento para el libro y la bitácora a partir de los
registros REALES del trainer (trainer_state.json de un checkpoint), no de capturas.

Salida en docs/graficas/:
  <nombre>.png       imagen 300 dpi (apta para KDP / impresión)
  data/<nombre>.json copia del historial (loss, learning rate, eval_loss) para regenerar

Requisito una vez (desde F:\\vertex-lab):  .\\venv\\Scripts\\python.exe -m pip install matplotlib

Uso (desde F:\\vertex-lab):
  .\\venv\\Scripts\\python.exe scripts\\plot_training.py                    # todas las etapas conocidas
  .\\venv\\Scripts\\python.exe scripts\\plot_training.py --state outputs\\checkpoints\\checkpoint-410\\trainer_state.json --name v1
"""
import argparse
import json
import math
from pathlib import Path

from common import ROOT, banner, rpath

OUT_DIR = ROOT / "docs" / "graficas"

# Etapas conocidas: se usa el checkpoint con más pasos de cada carpeta.
ETAPAS = [
    ("v1_entrenamiento_principal", "outputs/checkpoints",
     "Vertex-Core v1 — entrenamiento principal (Qwen3-4B, 3,278 muestras, 1 época)",
     # El eval final se calculó después del último checkpoint; el inicial viene del smoke test.
     # Valores tomados del log de consola (docs/BITACORA.md).
     [(0, 1.542, "antes de entrenar, smoke test"), (410, 1.110, "al terminar")]),
    ("v2_refuerzo_identidad", "outputs/checkpoints_identity",
     "Vertex-Core v2 — refuerzo de identidad (1,180 muestras, parte de la v1)", []),
]


def latest_state(folder):
    folder = rpath(folder)
    cks = sorted(folder.glob("checkpoint-*/trainer_state.json"),
                 key=lambda p: int(p.parent.name.split("-")[-1]))
    return cks[-1] if cks else None


def moving_avg(vals, w=10):
    out = []
    for i in range(len(vals)):
        s = vals[max(0, i - w + 1): i + 1]
        out.append(sum(s) / len(s))
    return out


def plot(state_path, name, title, alarm=0.2, extra_evals=()):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hist = json.loads(Path(state_path).read_text(encoding="utf-8"))["log_history"]
    train = [h for h in hist if "loss" in h and "step" in h]
    evals = [h for h in hist if "eval_loss" in h]
    seen = {e["step"] for e in evals}
    evals += [{"step": s, "eval_loss": v, "nota": n} for s, v, n in extra_evals if s not in seen]
    steps = [h["step"] for h in train]
    loss = [float(h["loss"]) for h in train]
    lr = [float(h.get("learning_rate", 0)) for h in train]
    ma = moving_avg(loss)

    (OUT_DIR / "data").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "data" / f"{name}.json").write_text(
        json.dumps({"fuente": str(state_path), "log_history": hist}, indent=1), encoding="utf-8")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.2), height_ratios=[2.2, 1], sharex=True)

    ax1.scatter(steps, loss, s=9, color="#b4b2a9", label="Loss de cada paso", zorder=2)
    ax1.plot(steps, ma, color="#2a78d6", lw=2, label="Tendencia (media de 10 pasos)", zorder=3)
    ax1.axhline(alarm, color="#d03b3b", lw=1.5, ls="--", label=f"Alarma de memorización ({alarm})")
    if evals:
        ax1.scatter([e["step"] for e in evals], [e["eval_loss"] for e in evals], marker="D", s=40,
                    color="#eb6834", label="Loss de evaluación (datos no vistos)", zorder=4)
        for e in evals:
            ax1.annotate(f"{e['eval_loss']:.3f}" + (f" ({e['nota']})" if e.get("nota") else ""),
                         (e["step"], e["eval_loss"]), textcoords="offset points",
                         ha="right" if e["step"] > 0.7 * max(steps) else "left",
                         xytext=(-6, 8) if e["step"] > 0.7 * max(steps) else (6, 8), fontsize=9, color="#eb6834",
                         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9), zorder=5)
    ax1.set_ylabel("Loss")
    ax1.set_ylim(0, max(max(loss) * 1.08, 0.5))
    ax1.grid(axis="y", color="#e1e0d9", lw=0.8)
    ax1.legend(loc="upper right", fontsize=8.5, frameon=False)
    ax1.set_title(title, fontsize=11, loc="left")

    ax2.plot(steps, lr, color="#1baf7a", lw=2)
    ax2.set_ylabel("Learning rate")
    ax2.set_xlabel("Paso")
    ax2.grid(axis="y", color="#e1e0d9", lw=0.8)
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    first, last = sum(loss[:10]) / min(10, len(loss)), sum(loss[-10:]) / min(10, len(loss))
    fig.text(0.01, 0.005, f"Pasos: {steps[-1]} · media primeros 10: {first:.2f} · media últimos 10: {last:.2f}"
             f" · mínimo: {min(loss):.2f} (e^-loss = {math.exp(-last):.0%} de probabilidad media a la palabra correcta al final)",
             fontsize=8, color="#52514e")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = OUT_DIR / f"{name}.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[+] {out}  ({steps[-1]} pasos, {len(evals)} evaluaciones)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", help="trainer_state.json concreto")
    ap.add_argument("--name", help="nombre del archivo de salida (sin extensión)")
    ap.add_argument("--title", default="Entrenamiento de Vertex-Core")
    a = ap.parse_args()
    banner("Gráficas de entrenamiento")

    if a.state:
        plot(rpath(a.state), a.name or Path(a.state).parent.name, a.title)
        return
    for name, folder, title, extra in ETAPAS:
        st = latest_state(folder)
        if st is None:
            print(f"[i] Sin checkpoints en {folder}; se omite {name}")
            continue
        plot(st, name, title, extra_evals=extra)


if __name__ == "__main__":
    main()
