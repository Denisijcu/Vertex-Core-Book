# 🛡️ Vertex Lab — Fine-tuning local de Vertex-Core

Pipeline para entrenar, fusionar, cuantizar y servir en Ollama un modelo propio de **Vertex Coders LLC**
en una **GTX 1660 Ti (6 GB, Turing) con Windows 11**, sin WSL y sin compilar nada.

```
data/raw/*.json ─► 01 limpieza ─► 02 QLoRA ─► chat_test ─► 03 merge fp16 ─► 04 GGUF q4_k_m ─► 05 Ollama
```

## Estado

| Fase | Estado |
|---|---|
| Entorno (torch CUDA + bitsandbytes, sin Unsloth) | ✅ `setup_env.ps1` + `00_check_env.py` |
| Datos → ChatML JSONL (system prompt único desde `prompts/`) | ✅ `01_data_cleaner.py` |
| Entrenamiento QLoRA (loss solo en respuestas del asistente) | ✅ `02_trainer.py` |
| Prueba base vs LoRA vs merged | ✅ `chat_test.py` |
| Merge LoRA → fp16 | ✅ `03_merge_lora.py` |
| Export GGUF (binarios oficiales llama.cpp) | ✅ `04_export_gguf.py` + `tools/setup_llamacpp.ps1` |
| Registro en Ollama | ✅ `05_build_ollama.py` |
| Dataset real (300+ muestras) | ⏳ ahora mismo hay 2 (solo smoke test) |
| Conexión a servidores MCP / evaluación | 🔜 Fase 3 |

## Ejecución (todo desde `F:\vertex-lab`)

```powershell
cd F:\vertex-lab

# 1) Una sola vez: venv limpio (el viejo se renombra a venv_bak_<fecha>, no se borra)
powershell -ExecutionPolicy Bypass -File .\setup_env.ps1

# 2) Una sola vez: llama.cpp precompilado para el export GGUF
powershell -ExecutionPolicy Bypass -File .\tools\setup_llamacpp.ps1

# 3) Pipeline
.\venv\Scripts\python.exe scripts\01_data_cleaner.py
.\venv\Scripts\python.exe scripts\02_trainer.py --max_steps 5     # smoke test (~minutos)
.\venv\Scripts\python.exe scripts\02_trainer.py                   # entrenamiento completo
.\venv\Scripts\python.exe scripts\chat_test.py                    # ¿aprendió?  (--base para comparar)
.\venv\Scripts\python.exe scripts\03_merge_lora.py
.\venv\Scripts\python.exe scripts\04_export_gguf.py
.\venv\Scripts\python.exe scripts\05_build_ollama.py
ollama run vertex-core
```

La primera corrida de `02` descarga el modelo base (~6 GB para Qwen2.5-3B) a `F:\vertex-lab\.hf_cache`.

## Datos

Pon `.json` o `.jsonl` en `data/raw/`. Formatos aceptados por registro:

```json
{"instruction": "...", "input": "", "output": "..."}
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

El system prompt **no** va en los datos: se inyecta desde `prompts/vertex_core.jinja2`, y el mismo texto
se usa en el Modelfile de Ollama. Con ≥20 muestras se separa un 10% para evaluación.

## Configuración clave (`configs/qlora_train.yaml`)

| Parámetro | Valor | Motivo |
|---|---|---|
| `base_model_name` | `Qwen/Qwen2.5-3B-Instruct` | 7B no cabe para entrenar en 6 GB. `1.5B` para pruebas rápidas |
| `max_seq_length` | 1024 | Sube a 2048 si la VRAM pico (se imprime al final) lo permite |
| `fp16` / `bf16` | `true` / `false` | Turing (sm_75) no tiene bf16 |
| batch × accum | 1 × 8 | batch efectivo 8 sin reventar VRAM |
| `optim` | `adamw_torch` | los params LoRA son pocos; 8-bit no aporta aquí |

## Por qué sin Unsloth (lo que rompía el pipeline anterior)

1. **Entorno incompatible**: torch 2.5.1+cu121 con triton-windows 3.8 y transformers 5.x. Unsloth en Windows
   exige triton-windows casado con la versión exacta de torch; con ese desfase `import torch/unsloth` se colgaba.
2. **`bf16: true`** en una GPU Turing → error inmediato.
3. **Qwen2.5-7B + seq 4096 en 6 GB** → OOM garantizado.
4. **`reporting_to`** (no existe; es `report_to`) y API vieja de `SFTTrainer` (`tokenizer=`, `dataset_text_field="messages"`).
5. **`save_pretrained_gguf`** intenta compilar llama.cpp (no hay cmake/compilador) y el merge se hacía sobre el modelo 4-bit.

El camino actual (transformers + PEFT + bitsandbytes) hace exactamente lo mismo — QLoRA NF4 — y en esta GPU
la diferencia de velocidad con Unsloth es pequeña. Si algún día entrenas en WSL2 o en una GPU Ampere+,
Unsloth vale la pena.

## Problemas comunes

| Síntoma | Solución |
|---|---|
| `CUDA out of memory` | `max_seq_length: 768`, o el modelo 1.5B; cierra apps que usen GPU |
| Loss `nan` (el trainer se detiene solo) | `attn_implementation: "eager"` o `learning_rate: 1.0e-4` |
| El modelo no "aprende" la identidad | Más datos (300+), variaciones de la misma pregunta, `num_train_epochs: 5` |
| `convert_hf_to_gguf.py` pide un módulo | `.\venv\Scripts\python.exe -m pip install <módulo>` |
| `ollama` no encontrado | `winget install Ollama.Ollama` y reabre la terminal |

## Estructura

```text
vertex-lab/
├── setup_env.ps1            # venv limpio + torch CUDA + deps
├── configs/qlora_train.yaml # única fuente de configuración
├── prompts/vertex_core.jinja2
├── data/{raw,processed}/
├── scripts/
│   ├── common.py            # rutas relativas al repo, caché HF, utilidades
│   ├── 00_check_env.py
│   ├── 01_data_cleaner.py
│   ├── 02_trainer.py
│   ├── chat_test.py
│   ├── 03_merge_lora.py
│   ├── 04_export_gguf.py
│   └── 05_build_ollama.py
├── tools/setup_llamacpp.ps1 # baja llama.cpp → tools/llama.cpp/
└── outputs/{checkpoints,adapters,models}/
```

*Diseñado y dirigido por Denis Sanchez Leyva | Vertex Coders LLC.*
