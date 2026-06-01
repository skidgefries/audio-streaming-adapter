"""Load ``.env`` from the package root and read typed training variables."""

from __future__ import annotations

import os
from pathlib import Path

_DEFAULT_HF_ENDPOINT = "https://huggingface.co"


def package_root(start: str | None = None) -> str:
    """Directory containing ``pyproject.toml`` (``audio-streaming-adapter/``)."""
    cur = Path(start or Path(__file__).resolve()).parent
    for _ in range(8):
        if (cur / "pyproject.toml").is_file():
            return str(cur)
        if cur.parent == cur:
            break
        cur = cur.parent
    return str(Path(__file__).resolve().parents[2])


def load_project_env(root: str | None = None, *, override: bool = False) -> str | None:
    """
    Parse ``.env`` into ``os.environ``.

    Uses ``setdefault`` unless ``override=True``. Returns the path loaded, or ``None``.
    """
    env_path = Path(root or package_root()) / ".env"
    if not env_path.is_file():
        return None

    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if override or key == "HF_ENDPOINT":
            os.environ[key] = value
        else:
            os.environ.setdefault(key, value)

    return str(env_path)


def apply_hf_hub_endpoint(root: str | None = None) -> str:
    """
    Point Hugging Face Hub at the official endpoint (or ``HF_ENDPOINT`` in ``.env``).

    Some hosts install ``site-packages/hf_config.pth`` that forces an unstable mirror;
    call this after ``load_project_env`` and before importing ``transformers`` / ``llm``.
    """
    env_path = Path(root or package_root()) / ".env"
    endpoint = _DEFAULT_HF_ENDPOINT
    if env_path.is_file():
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[7:].strip()
            key, _, value = line.partition("=")
            if key.strip() != "HF_ENDPOINT":
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if value:
                endpoint = value
            break
    endpoint = endpoint.rstrip("/")
    os.environ["HF_ENDPOINT"] = endpoint
    return endpoint


def env_str(key: str, default: str | None = None) -> str | None:
    value = os.environ.get(key)
    if value is None or not str(value).strip():
        return default
    return str(value).strip()


def env_bool(key: str, default: bool = False) -> bool:
    value = env_str(key)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def env_int(key: str, default: int) -> int:
    value = env_str(key)
    if value is None:
        return default
    return int(value)


def env_float(key: str, default: float) -> float:
    value = env_str(key)
    if value is None:
        return default
    return float(value)


def env_optional_int(key: str) -> int | None:
    value = env_str(key)
    if value is None:
        return None
    return int(value)
