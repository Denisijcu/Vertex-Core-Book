"""
00 - Verifica que el entorno sirve para entrenar en esta máquina.
Uso (desde F:\\vertex-lab):  .\\venv\\Scripts\\python.exe scripts\\00_check_env.py
"""
import importlib
import platform
import sys

from common import ROOT, banner, load_config

ok = True


def check(name):
    global ok
    try:
        m = importlib.import_module(name)
        print(f"  [+] {name:<14} {getattr(m, '__version__', '?')}")
        return m
    except Exception as e:
        ok = False
        print(f"  [!] {name:<14} NO disponible -> {type(e).__name__}: {e}")
        return None


banner("Vertex Lab - chequeo de entorno")
print(f"  Python   {sys.version.split()[0]}  ({sys.executable})")
print(f"  SO       {platform.platform()}")
print(f"  Repo     {ROOT}")

print("\nLibrerías:")
torch = check("torch")
for lib in ("transformers", "peft", "accelerate", "datasets", "bitsandbytes", "safetensors", "yaml"):
    check(lib)

for bad in ("unsloth", "triton"):
    try:
        importlib.import_module(bad)
        print(f"  [i] {bad} está instalado pero este pipeline no lo usa (en Windows+Turing suele romper el import).")
    except Exception:
        pass

if torch is not None:
    print("\nGPU:")
    if not torch.cuda.is_available():
        ok = False
        print("  [!] CUDA no disponible. ¿Instalaste torch desde el index cuXXX? (ver setup_env.ps1)")
    else:
        p = torch.cuda.get_device_properties(0)
        cap = torch.cuda.get_device_capability(0)
        print(f"  [+] {p.name} | VRAM {p.total_memory / 1024**3:.1f} GB | sm_{cap[0]}{cap[1]} | CUDA torch {torch.version.cuda}")
        print(f"  [i] bf16 soportado: {torch.cuda.is_bf16_supported()}  (en Turing debe usarse fp16)")
        a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
        _ = (a @ a).sum().item()
        print("  [+] matmul fp16 en GPU OK")

        cfg = load_config()
        if cfg["training"].get("bf16") and not torch.cuda.is_bf16_supported():
            ok = False
            print("  [!] El YAML tiene bf16: true y esta GPU no lo soporta.")

    try:
        import bitsandbytes as bnb  # noqa
        import torch.nn as nn
        lin = bnb.nn.Linear4bit(64, 64, compute_dtype=torch.float16, quant_type="nf4").cuda()
        lin(torch.randn(2, 64, device="cuda", dtype=torch.float16))
        print("  [+] bitsandbytes 4-bit (nf4) en GPU OK")
    except Exception as e:
        ok = False
        print(f"  [!] bitsandbytes 4-bit falló: {e}")

banner("RESULTADO: " + ("LISTO para entrenar" if ok else "HAY PROBLEMAS (arriba)"))
sys.exit(0 if ok else 1)
