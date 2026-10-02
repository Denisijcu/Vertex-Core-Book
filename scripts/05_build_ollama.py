"""
05 - Genera el Modelfile y registra el modelo en Ollama como `vertex-core`.

Detecta la familia del modelo (config.json del modelo fusionado) y usa la plantilla
de chat correcta: ChatML para Qwen2/Qwen3, formato Llama 3 para Llama.

Uso (desde F:\\vertex-lab):
  .\\venv\\Scripts\\python.exe scripts\\05_build_ollama.py            # Modelfile + ollama create
  .\\venv\\Scripts\\python.exe scripts\\05_build_ollama.py --only-modelfile
Después:  ollama run vertex-core
"""
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from common import ROOT, banner, load_config, read_system_prompt, rpath

TEMPLATES = {
    "chatml": {
        "template": """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>
""",
        "stops": ["<|im_start|>", "<|im_end|>"],
    },
    "llama3": {
        "template": """{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>""",
        "stops": ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"],
    },
}
FAMILY_BY_MODEL_TYPE = {"qwen2": "chatml", "qwen3": "chatml", "llama": "llama3"}


def detect_family(merged_dir: Path) -> str:
    cfg_file = merged_dir / "config.json"
    if not cfg_file.exists():
        print("[i] No encontré config.json del modelo fusionado; asumo ChatML (Qwen).")
        return "chatml"
    model_type = json.loads(cfg_file.read_text(encoding="utf-8")).get("model_type", "")
    fam = FAMILY_BY_MODEL_TYPE.get(model_type)
    if fam is None:
        raise SystemExit(f"[!] model_type '{model_type}' sin plantilla definida. Añádela en TEMPLATES.")
    print(f"[*] model_type={model_type} -> plantilla {fam}")
    return fam


def find_ollama():
    exe = shutil.which("ollama")
    if exe:
        return exe
    cand = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
    return str(cand) if cand.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-modelfile", action="store_true")
    a = ap.parse_args()

    cfg = load_config()
    o = cfg["ollama"]
    gguf = rpath(cfg["outputs"]["gguf_dir"]) / f"{o['model_name']}-{cfg['outputs']['quantization']}.gguf"
    if not gguf.exists():
        raise SystemExit(f"[!] No existe {gguf}. Corre 04_export_gguf.py")

    tpl = TEMPLATES[detect_family(rpath(cfg["outputs"]["merged_dir"]))]
    system = read_system_prompt(cfg).replace('"""', "'''")
    stops = "".join(f'PARAMETER stop "{s}"\n' for s in tpl["stops"])

    modelfile = ROOT / "Modelfile"
    modelfile.write_text(
        f"FROM {gguf.as_posix()}\n\n"
        f'TEMPLATE """{tpl["template"]}"""\n\n'
        f'SYSTEM """{system}"""\n\n'
        f"PARAMETER temperature {o['temperature']}\n"
        f"PARAMETER num_ctx {o['num_ctx']}\n"
        f"{stops}",
        encoding="utf-8",
    )
    banner(f"Modelfile escrito en {modelfile}")
    if a.only_modelfile:
        return

    ollama = find_ollama()
    if not ollama:
        print("[!] Ollama no está instalado o no está en el PATH.")
        print("    Instálalo:  winget install Ollama.Ollama   y luego, desde F:\\vertex-lab:")
        print(f"    ollama create {o['model_name']} -f Modelfile")
        return
    subprocess.run([ollama, "create", o["model_name"], "-f", str(modelfile)], check=True, cwd=ROOT)
    banner(f"Listo:  ollama run {o['model_name']}")


if __name__ == "__main__":
    main()
