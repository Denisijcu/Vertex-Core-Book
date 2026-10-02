"""
Prueba rápida del modelo: base 4-bit + adaptador LoRA, o el modelo fusionado.

Uso (desde F:\\vertex-lab):
  .\\venv\\Scripts\\python.exe scripts\\chat_test.py                         # adaptador, prompts por defecto
  .\\venv\\Scripts\\python.exe scripts\\chat_test.py -p "¿Quién te creó?"
  .\\venv\\Scripts\\python.exe scripts\\chat_test.py --base                  # base SIN adaptador (para comparar)
  .\\venv\\Scripts\\python.exe scripts\\chat_test.py --merged                # modelo fusionado (tras 03)
  .\\venv\\Scripts\\python.exe scripts\\chat_test.py -i                      # modo interactivo
"""
import argparse

from common import load_config, read_system_prompt, rpath

import torch
import transformers
from packaging.version import Version
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DEFAULT_PROMPTS = ["¿Cuál es tu nombre y quién te creó?", "Saluda al equipo de Vertex Coders."]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prompt", action="append")
    ap.add_argument("-i", "--interactive", action="store_true")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--base", action="store_true")
    g.add_argument("--merged", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--adapter", default=None, help="Carpeta del adaptador (por defecto outputs.adapter_dir), p. ej. outputs/adapters/vertex-core-lora-v2")
    ap.add_argument("--no-system", action="store_true", help="Sin system prompt: mide lo que aprendió el modelo, no lo que le dice el prompt")
    a = ap.parse_args()

    cfg = load_config()
    dtype_kw = "dtype" if Version(transformers.__version__) >= Version("4.56.0") else "torch_dtype"
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)

    if a.merged:
        src = str(rpath(cfg["outputs"]["merged_dir"]))
        tok = AutoTokenizer.from_pretrained(src)
        model = AutoModelForCausalLM.from_pretrained(src, quantization_config=q, device_map={"": 0}, **{dtype_kw: torch.float16})
    else:
        base = cfg["model"]["base_model_name"]
        tok = AutoTokenizer.from_pretrained(base)
        model = AutoModelForCausalLM.from_pretrained(base, quantization_config=q, device_map={"": 0}, **{dtype_kw: torch.float16})
        if not a.base:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(rpath(a.adapter or cfg["outputs"]["adapter_dir"])))
    model.eval()
    system = read_system_prompt(cfg)
    label = ("BASE" if a.base else ("MERGED" if a.merged else "LoRA")) + (" sin-system" if a.no_system else "")

    def ask(user):
        msgs = [{"role": "user", "content": user}]
        if not a.no_system:
            msgs.insert(0, {"role": "system", "content": system})
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=a.max_new_tokens, do_sample=True, temperature=0.3,
                                 top_p=0.9, repetition_penalty=1.05, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    if a.interactive:
        print(f"[{label}] Modo interactivo. Línea vacía para salir.")
        while (u := input("\ntú > ").strip()):
            print(f"vertex-core > {ask(u)}")
        return

    for p in a.prompt or DEFAULT_PROMPTS:
        print(f"\n[{label}] >>> {p}\n{ask(p)}")


if __name__ == "__main__":
    main()
