"""
02 - Entrenamiento QLoRA (transformers + PEFT + bitsandbytes), sin Unsloth.

Por qué sin Unsloth: en Windows nativo depende de triton-windows con versión
casada exacta con torch, y en una GTX 1660 Ti (Turing) la ganancia es mínima.
Este camino es estable y usa la misma técnica (base 4-bit NF4 + adaptadores LoRA).

Uso (desde F:\\vertex-lab):
  .\\venv\\Scripts\\python.exe scripts\\02_trainer.py                  # entrenamiento normal
  .\\venv\\Scripts\\python.exe scripts\\02_trainer.py --max_steps 5    # smoke test rápido
  .\\venv\\Scripts\\python.exe scripts\\02_trainer.py --resume         # continuar desde el último checkpoint

Etapa 2 (refuerzo de identidad sobre un adaptador ya entrenado):
  .\\venv\\Scripts\\python.exe scripts\\02_trainer.py --stage identity
"""
import argparse
import json
import math
import time

from common import banner, filter_kwargs, load_config, rpath

import torch
import transformers
from datasets import load_dataset
from packaging.version import Version
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

TF_VER = Version(transformers.__version__)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--max_steps", type=int, default=-1, help="Limita pasos (smoke test)")
    ap.add_argument("--resume", action="store_true", help="Reanuda desde el último checkpoint")
    ap.add_argument("--stage", default=None, help="Nombre de una etapa definida en 'stages:' del YAML (p. ej. identity)")
    return ap.parse_args()


# ---------------------------------------------------------------- datos
def build_tokenize_fn(tokenizer, max_len, assistant_only):
    def render(msgs, gen_prompt):
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen_prompt)

    def tok(text):
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    def fn(example):
        # Dos formatos admitidos:
        #  - {"messages": [...]}  -> conversacion normal (identidad, codigo, CVEs).
        #  - {"text": "..."}      -> ya renderizado con el chat template (tool calling, v8).
        if example.get("text"):
            # Ya trae el formato Qwen3 completo (con <tool_call>/<tool_response>). Se tokeniza tal cual.
            # No se enmascara el prompt: es texto ya armado; se entrena sobre todo (incluye la respuesta).
            full_ids = tok(example["text"])
            labels = list(full_ids)
            full_ids, labels = full_ids[:max_len], labels[:max_len]
            return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}

        msgs = example["messages"]
        full_ids = tok(render(msgs, False))
        if assistant_only:
            prompt_ids = tok(render(msgs[:-1], True))
            n_prompt = len(prompt_ids) if full_ids[: len(prompt_ids)] == prompt_ids else 0
            labels = [-100] * n_prompt + full_ids[n_prompt:]
        else:
            labels = list(full_ids)
        full_ids, labels = full_ids[:max_len], labels[:max_len]
        return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}

    return fn


def load_split(path, tokenize_fn):
    if path is None or not path.exists():
        return None
    ds = load_dataset("json", data_files=str(path), split="train")
    # Normaliza columnas: si faltan 'messages' o 'text', se anaden vacias para que .map no falle
    # al mezclar ficheros con formatos distintos (messages vs text).
    cols = ds.column_names
    if "text" not in cols:
        ds = ds.add_column("text", [None] * len(ds))
    if "messages" not in cols:
        ds = ds.add_column("messages", [None] * len(ds))
    ds = ds.map(tokenize_fn, remove_columns=ds.column_names)
    before = len(ds)
    ds = ds.filter(lambda e: any(l != -100 for l in e["labels"]))
    if len(ds) < before:
        print(f"[i] {before - len(ds)} muestras descartadas (el prompt ocupaba todo max_seq_length)")
    return ds


# ---------------------------------------------------------------- modelo
def load_model(cfg_model, lora_cfg, tokenizer, init_adapter=None):
    dtype_kw = "dtype" if TF_VER >= Version("4.56.0") else "torch_dtype"
    kwargs = {
        dtype_kw: torch.float16,
        "device_map": {"": 0},
        "attn_implementation": cfg_model.get("attn_implementation", "sdpa"),
    }
    if cfg_model.get("load_in_4bit", True):
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(cfg_model["base_model_name"], **kwargs)
    model.config.use_cache = False
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    if init_adapter:
        # Continúa entrenando un adaptador existente (no parte de cero).
        from peft import PeftModel
        print(f"[*] Partiendo del adaptador: {init_adapter}")
        model = PeftModel.from_pretrained(model, str(init_adapter), is_trainable=True)
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                task_type="CAUSAL_LM",
                r=lora_cfg["r"],
                lora_alpha=lora_cfg["lora_alpha"],
                lora_dropout=lora_cfg["lora_dropout"],
                bias=lora_cfg["bias"],
                target_modules=lora_cfg["target_modules"],
            ),
        )
    # Con fp16 + GradScaler los parámetros ENTRENABLES deben ser fp32
    # (si no: "Attempting to unscale FP16 gradients").
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()
    return model


class NanGuard(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kw):
        loss = (logs or {}).get("loss")
        if loss is not None and (math.isnan(loss) or math.isinf(loss)):
            print("\n[!] Loss NaN/Inf: se detiene. Prueba attn_implementation: 'eager' o un learning_rate menor.")
            control.should_training_stop = True


# ---------------------------------------------------------------- main
def main():
    args = parse_args()
    cfg = load_config(args.config)
    m, lora, ds_cfg, tr, out = cfg["model"], cfg["lora"], cfg["dataset"], cfg["training"], cfg["outputs"]

    # Una etapa sobreescribe dataset/entrenamiento/salida sin tocar la configuración principal.
    init_adapter = None
    if args.stage:
        st = (cfg.get("stages") or {}).get(args.stage)
        if not st:
            raise SystemExit(f"[!] No existe la etapa '{args.stage}' en 'stages:' del YAML")
        ds_cfg = {**ds_cfg, "train_path": st["train_path"], "eval_path": st.get("eval_path")}
        tr = {**tr, **st.get("training", {})}
        out = {**out, "adapter_dir": st["adapter_dir"]}
        if "max_seq_length" in st:
            m = {**m, "max_seq_length": st["max_seq_length"]}
        init_adapter = rpath(st["init_adapter"]) if st.get("init_adapter") else None
        banner(f"Etapa: {args.stage}")

    banner(f"02 - QLoRA | {m['base_model_name']}")
    if not torch.cuda.is_available():
        raise SystemExit("[!] CUDA no disponible. Corre scripts/00_check_env.py")
    if tr.get("bf16") and not torch.cuda.is_bf16_supported():
        raise SystemExit("[!] bf16: true pero la GPU no lo soporta. Usa fp16: true / bf16: false en el YAML.")

    tokenizer = AutoTokenizer.from_pretrained(m["base_model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    tokenize_fn = build_tokenize_fn(tokenizer, m["max_seq_length"], ds_cfg.get("train_on_assistant_only", True))
    train_ds = load_split(rpath(ds_cfg["train_path"]), tokenize_fn)
    if train_ds is None or len(train_ds) == 0:
        raise SystemExit(f"[!] Dataset vacío o inexistente: {ds_cfg['train_path']}. Corre 01_data_cleaner.py")
    eval_ds = load_split(rpath(ds_cfg.get("eval_path")), tokenize_fn)
    n_eval_max = ds_cfg.get("max_eval_samples")
    if eval_ds is not None and n_eval_max and len(eval_ds) > n_eval_max:
        eval_ds = eval_ds.shuffle(seed=tr["seed"]).select(range(n_eval_max))
    print(f"[*] train: {len(train_ds)} muestras | eval: {len(eval_ds) if eval_ds else 0}")

    model = load_model(m, lora, tokenizer, init_adapter)

    targs = {
        "output_dir": str(rpath(tr["output_dir"])),
        "per_device_train_batch_size": tr["per_device_train_batch_size"],
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": tr["gradient_accumulation_steps"],
        "num_train_epochs": tr["num_train_epochs"],
        "max_steps": args.max_steps,
        "learning_rate": tr["learning_rate"],
        "lr_scheduler_type": tr["lr_scheduler_type"],
        "weight_decay": tr["weight_decay"],
        "fp16": tr["fp16"],
        "bf16": tr["bf16"],
        "optim": tr["optim"],
        "logging_steps": tr["logging_steps"],
        "save_strategy": tr["save_strategy"],
        "save_steps": tr.get("save_steps", 500),
        "save_total_limit": tr["save_total_limit"],
        "report_to": tr["report_to"],
        "seed": tr["seed"],
        "gradient_checkpointing": False,   # ya activado a mano con use_reentrant=False
        "remove_unused_columns": False,
        "dataloader_num_workers": 0,        # Windows: sin multiprocessing en el DataLoader
        "dataloader_pin_memory": True,
    }
    allowed = filter_kwargs(TrainingArguments, {"warmup_ratio": 0, "eval_strategy": 0}, label="probe")
    if "warmup_ratio" in allowed:
        targs["warmup_ratio"] = tr["warmup_ratio"]
    else:
        targs["warmup_steps"] = tr["warmup_ratio"]  # transformers v5: float < 1 = ratio
    if eval_ds is not None:
        # Evaluar solo al final de cada época, aunque los checkpoints se guarden por pasos.
        targs["eval_strategy" if "eval_strategy" in allowed else "evaluation_strategy"] = tr.get("eval_strategy", "epoch")

    training_args = TrainingArguments(**filter_kwargs(TrainingArguments, targs))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100, pad_to_multiple_of=8),
        callbacks=[NanGuard()],
    )

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    print("[*] Entrenando... (Ctrl+C para abortar; --resume para continuar)")
    trainer.train(resume_from_checkpoint=True if args.resume else None)
    mins = (time.time() - t0) / 60
    peak = torch.cuda.max_memory_allocated() / 1024**3

    adapter_dir = rpath(out["adapter_dir"])
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    (adapter_dir / "vertex_meta.json").write_text(
        json.dumps({"base_model_name": m["base_model_name"], "max_seq_length": m["max_seq_length"],
                    "train_samples": len(train_ds), "minutes": round(mins, 1), "peak_vram_gb": round(peak, 2)},
                   indent=2),
        encoding="utf-8",
    )
    # Historial completo (loss, learning rate, eval) para gráficas del libro: scripts/plot_training.py --state ...
    (adapter_dir / "trainer_state.json").write_text(
        json.dumps({"log_history": trainer.state.log_history}, indent=1), encoding="utf-8")
    banner(f"Adaptador LoRA guardado en {adapter_dir}  |  {mins:.1f} min  |  VRAM pico {peak:.2f} GB")
    print("[i] Siguiente: scripts\\chat_test.py (probar)  ->  scripts\\03_merge_lora.py (fusionar)")


if __name__ == "__main__":
    main()
