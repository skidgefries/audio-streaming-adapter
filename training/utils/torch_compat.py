"""
Detect GPU/CUDA driver support and install matching ``torch`` / ``torchvision`` / ``torchaudio``.

Used by ``setup_remote_training.sh`` and optionally at trainer startup when
``CHECK_TORCH_COMPAT=1``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Iterable

from training.utils.env import env_str, load_project_env, package_root

DEFAULT_TORCH_VERSION = "2.11.0"
DEFAULT_FALLBACK_CUDA_TAGS = ("cu128", "cu126", "cu124", "cu118")


def _has_nvidia_gpu() -> bool:
    try:
        out = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return out.returncode == 0 and bool(out.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def detect_pytorch_cuda_index() -> str:
    """Map driver/CUDA version to a PyTorch wheel index tag (``cpu`` if no GPU)."""
    if not _has_nvidia_gpu():
        return "cpu"

    cuda_ver = ""
    try:
        out = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        for line in out.stdout.splitlines():
            if "CUDA Version:" in line:
                part = line.split("CUDA Version:")[1].strip().split()[0]
                cuda_ver = part
                break
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    if not cuda_ver:
        try:
            out = subprocess.run(
                ["nvcc", "--version"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            for line in out.stdout.splitlines():
                if "release" in line:
                    cuda_ver = line.split("release")[1].strip().split(",")[0]
                    break
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    if not cuda_ver:
        return "cpu"

    major_s, _, rest = cuda_ver.partition(".")
    major = int(major_s)
    minor = int(rest.split(".")[0]) if rest else 0

    if major >= 12 and minor >= 8:
        return "cu128"
    if major == 12 and minor >= 6:
        return "cu126"
    if major == 12:
        return "cu124"
    if major == 11 and minor >= 8:
        return "cu118"
    return "cpu"


def _unique_tags(tags: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for tag in tags:
        tag = tag.strip()
        if not tag or tag in seen:
            continue
        seen.add(tag)
        ordered.append(tag)
    return ordered


def candidate_cuda_tags() -> list[str]:
    override = env_str("TORCH_CUDA_INDEX")
    retry = env_str("TORCH_COMPAT_RETRY_INDEXES")
    retry_tags = (
        [t.strip() for t in retry.split(",") if t.strip()]
        if retry
        else list(DEFAULT_FALLBACK_CUDA_TAGS)
    )
    detected = detect_pytorch_cuda_index()
    return _unique_tags([override or "", detected, *retry_tags, "cpu"])


def install_pytorch(*, cuda_tag: str, torch_version: str) -> None:
    base = ["uv", "pip", "install", "--upgrade"]
    packages = [
        f"torch=={torch_version}",
        "torchvision",
        f"torchaudio=={torch_version}",
    ]
    if cuda_tag == "cpu":
        cmd = [*base, *packages]
    else:
        index = f"https://download.pytorch.org/whl/{cuda_tag}"
        cmd = [*base, "--index-url", index, *packages]
    print(f"[torch_compat] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=package_root())


def check_torch_cuda_compatible(*, require_gpu: bool | None = None) -> tuple[bool, str]:
    """
    Return ``(ok, message)`` after a CUDA smoke test.

    ``require_gpu`` defaults to whether ``nvidia-smi`` reports a GPU.
    """
    import torch

    if require_gpu is None:
        require_gpu = _has_nvidia_gpu()

    if require_gpu and not torch.cuda.is_available():
        return False, (
            f"CUDA not available (torch {torch.__version__}, "
            f"built cuda={getattr(torch.version, 'cuda', None)})"
        )

    if not require_gpu:
        return True, f"CPU torch OK ({torch.__version__})"

    try:
        device = torch.device("cuda:0")
        x = torch.zeros(1, device=device)
        x += 1
        torch.cuda.synchronize(device)
        name = torch.cuda.get_device_name(0)
        return True, (
            f"torch {torch.__version__}, cuda {torch.version.cuda}, device {name!r}"
        )
    except Exception as exc:
        return False, f"CUDA smoke test failed: {exc}"


def ensure_torch_compatible(*, force_reinstall: bool = False) -> str:
    """
    Install PyTorch wheels until the CUDA smoke test passes.

    Sets ``TORCH_CUDA_INDEX`` in the environment to the tag that worked.
    Returns the tag used (``cpu`` when no GPU).
    """
    load_project_env()
    torch_version = env_str("TORCH_VERSION", DEFAULT_TORCH_VERSION) or DEFAULT_TORCH_VERSION
    require_gpu = _has_nvidia_gpu()

    if not force_reinstall:
        ok, msg = check_torch_cuda_compatible(require_gpu=require_gpu)
        if ok:
            print(f"[torch_compat] Compatible: {msg}", flush=True)
            tag = env_str("TORCH_CUDA_INDEX") or detect_pytorch_cuda_index()
            os.environ.setdefault("TORCH_CUDA_INDEX", tag)
            return tag

    last_error = ""
    for cuda_tag in candidate_cuda_tags():
        if cuda_tag != "cpu" and not require_gpu:
            continue
        try:
            install_pytorch(cuda_tag=cuda_tag, torch_version=torch_version)
        except subprocess.CalledProcessError as exc:
            last_error = f"pip install failed for {cuda_tag}: {exc}"
            print(f"[torch_compat] {last_error}", flush=True)
            continue

        ok, msg = check_torch_cuda_compatible(require_gpu=require_gpu)
        if ok:
            os.environ["TORCH_CUDA_INDEX"] = cuda_tag
            print(f"[torch_compat] Installed {cuda_tag}: {msg}", flush=True)
            return cuda_tag
        last_error = msg
        print(f"[torch_compat] Incompatible after {cuda_tag}: {msg}", flush=True)

    raise RuntimeError(
        "PyTorch CUDA compatibility check failed after trying: "
        f"{', '.join(candidate_cuda_tags())}. Last error: {last_error}"
    )


def main() -> None:
    force = "--force" in sys.argv
    tag = ensure_torch_compatible(force_reinstall=force)
    print(tag)


if __name__ == "__main__":
    main()
