"""
04 - Exporta el modelo fusionado a GGUF con las herramientas oficiales de llama.cpp.

Paso 1: convert_hf_to_gguf.py  -> vertex-core-f16.gguf
Paso 2: llama-quantize.exe     -> vertex-core-q4_k_m.gguf

No compila nada: usa los binarios precompilados que baja tools\\setup_llamacpp.ps1.
(El save_pretrained_gguf de Unsloth intentaba compilar llama.cpp y en Windows fallaba.)

Uso (desde F:\\vertex-lab):
  .\\venv\\Scripts\\python.exe scripts\\04_export_gguf.py
  .\\venv\\Scripts\\python.exe scripts\\04_export_gguf.py --quant q8_0
"""
import argparse
import os
import subprocess
import sys

from common import banner, load_config, rpath


def find_quantize(llama_dir):
    for name in ("llama-quantize.exe", "llama-quantize"):
        hits = list(llama_dir.rglob(name))
        if hits:
            return hits[0]
    return None


def run(cmd, env=None):
    print("[>] " + " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd))
    r = subprocess.run([str(c) for c in cmd], env=env)
    if r.returncode != 0:
        raise SystemExit(f"[!] Falló (código {r.returncode})")


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", default=cfg["outputs"]["quantization"])
    a = ap.parse_args()

    merged = rpath(cfg["outputs"]["merged_dir"])
    gguf_dir = rpath(cfg["outputs"]["gguf_dir"])
    llama_dir = rpath(cfg["llama_cpp"]["dir"])
    name = cfg["ollama"]["model_name"]
    banner(f"04 - Export GGUF ({a.quant})")

    if not (merged / "config.json").exists():
        raise SystemExit(f"[!] No hay modelo fusionado en {merged}. Corre 03_merge_lora.py")
    convert = llama_dir / "convert_hf_to_gguf.py"
    if not convert.exists():
        raise SystemExit(f"[!] Falta {convert}. Corre tools\\setup_llamacpp.ps1 (desde F:\\vertex-lab)")

    gguf_dir.mkdir(parents=True, exist_ok=True)
    f16 = gguf_dir / f"{name}-f16.gguf"
    final = gguf_dir / f"{name}-{a.quant}.gguf"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(llama_dir / "gguf-py") + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    # El conversor usa el temporal del sistema (suele estar en C:). Si C: va justo de espacio,
    # se llena a mitad de escribir el f16 (~8 GB) y falla con OSError. Se apunta el temporal al
    # mismo disco de salida, que tiene sitio.
    tmp = gguf_dir / "_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    for var in ("TMPDIR", "TEMP", "TMP"):
        env[var] = str(tmp)

    run([sys.executable, convert, merged, "--outfile", f16, "--outtype", "f16"], env=env)

    if a.quant.lower() == "f16":
        final = f16
    else:
        q = find_quantize(llama_dir)
        if q is None:
            raise SystemExit(f"[!] No encuentro llama-quantize.exe dentro de {llama_dir}. Re-corre tools\\setup_llamacpp.ps1")
        run([q, f16, final, a.quant.upper()])
        f16.unlink()  # el f16 intermedio pesa ~6 GB

    banner(f"GGUF listo: {final} ({final.stat().st_size / 1024**3:.2f} GB)")
    print("[i] Siguiente: scripts\\05_build_ollama.py")


if __name__ == "__main__":
    main()
