"""
Utilidades compartidas del pipeline Vertex Lab.

- Resuelve todas las rutas relativas a la raíz del repo (F:\\vertex-lab),
  así los scripts funcionan sin importar desde qué directorio se lancen.
- Fija la caché de Hugging Face dentro del repo para no llenar C:.
"""
from __future__ import annotations

import dataclasses
import inspect
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "qlora_train.yaml"

# Caché de modelos dentro del repo (se puede sobreescribir con la variable HF_HOME)
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Consola de Windows con UTF-8 (tildes y emojis en los logs)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def rpath(p: str | os.PathLike | None) -> Path | None:
    """Ruta absoluta relativa a la raíz del repo."""
    if p is None:
        return None
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def load_config(path: str | os.PathLike | None = None) -> dict:
    cfg_path = rpath(path) if path else DEFAULT_CONFIG
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_system_prompt(cfg: dict) -> str:
    return rpath(cfg["prompt"]["system_prompt_path"]).read_text(encoding="utf-8").strip()


def filter_kwargs(target, kwargs: dict, label: str = "") -> dict:
    """
    Quita los argumentos que la versión instalada de una librería no acepta
    (transformers/peft cambian nombres entre versiones). Avisa de lo descartado.
    """
    if dataclasses.is_dataclass(target):
        allowed = {f.name for f in dataclasses.fields(target)}
    else:
        allowed = set(inspect.signature(target).parameters)
    dropped = [k for k in kwargs if k not in allowed]
    if dropped:
        print(f"[i] {label or getattr(target, '__name__', target)}: ignorando args no soportados por esta versión: {dropped}")
    return {k: v for k, v in kwargs.items() if k in allowed}


def banner(txt: str) -> None:
    print("\n" + "=" * 60 + f"\n  {txt}\n" + "=" * 60)
