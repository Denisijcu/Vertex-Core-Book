"""
03 - Fusiona el adaptador LoRA con el modelo base en fp16 (en CPU/RAM, no en la GPU).

Se hace sobre los pesos ORIGINALES (no 4-bit): fusionar sobre un modelo cuantizado
degrada la calidad. Un 3B en fp16 ocupa ~6-7 GB de RAM (tienes 32 GB, sobra).

Uso (desde F:\\vertex-lab):  .\\venv\\Scripts\\python.exe scripts\\03_merge_lora.py
"""
import argparse
import json
import shutil

from common import banner, load_config, rpath

import torch
import transformers
from packaging.version import Version
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="Carpeta del adaptador (por defecto outputs.adapter_dir)")
    a = ap.parse_args()
    cfg = load_config()
    adapter_dir = rpath(a.adapter or cfg["outputs"]["adapter_dir"])
    merged_dir = rpath(cfg["outputs"]["merged_dir"])
    if not (adapter_dir / "adapter_config.json").exists():
        raise SystemExit(f"[!] No hay adaptador en {adapter_dir}. Corre 02_trainer.py primero.")

    base = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))["base_model_name_or_path"]
    banner(f"03 - Merge LoRA -> {base}")

    dtype_kw = "dtype" if Version(transformers.__version__) >= Version("4.56.0") else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(base, device_map={"": "cpu"}, low_cpu_mem_usage=True,
                                                 **{dtype_kw: torch.float16})
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    print("[*] Fusionando pesos...")
    model = model.merge_and_unload()

    if merged_dir.exists():
        shutil.rmtree(merged_dir)
    merged_dir.mkdir(parents=True)
    model.save_pretrained(str(merged_dir), safe_serialization=True, max_shard_size="2GB")
    AutoTokenizer.from_pretrained(str(adapter_dir)).save_pretrained(str(merged_dir))

    size = sum(f.stat().st_size for f in merged_dir.glob("*.safetensors")) / 1024**3
    banner(f"Modelo fusionado en {merged_dir} ({size:.1f} GB)")
    print("[i] Siguiente: scripts\\04_export_gguf.py")


if __name__ == "__main__":
    main()
