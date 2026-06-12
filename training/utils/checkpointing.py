from __future__ import annotations

import os
import re
from dataclasses import dataclass, fields, replace

import torch

from training.utils.config import GateConfig

STAGE_EPOCH_FILENAME = re.compile(r"^adapter_stage(\d+)_epoch(\d+)\.pt$")


@dataclass(frozen=True)
class TrainingCheckpoint:
    stage: int
    epoch: int
    global_step: int
    adapter_state_dict: dict
    gate_state_dict: dict | None = None
    optimizer_state_dict: dict | None = None
    scheduler_state_dict: dict | None = None
    metrics: dict | None = None
    hyperparams: dict | None = None


def save_checkpoint(path: str, ckpt: TrainingCheckpoint) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "stage": ckpt.stage,
        "epoch": ckpt.epoch,
        "global_step": ckpt.global_step,
        "adapter_state_dict": ckpt.adapter_state_dict,
        "metrics": ckpt.metrics or {},
        "hyperparams": ckpt.hyperparams or {},
    }
    if ckpt.gate_state_dict is not None:
        payload["gate_state_dict"] = ckpt.gate_state_dict
    if ckpt.optimizer_state_dict is not None:
        payload["optimizer_state_dict"] = ckpt.optimizer_state_dict
    if ckpt.scheduler_state_dict is not None:
        payload["scheduler_state_dict"] = ckpt.scheduler_state_dict

    torch.save(payload, path)


def load_adapter_state_dict(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("adapter_state_dict") or ckpt.get("model_state_dict") or ckpt.get("state_dict")
    if state is None:
        raise KeyError(f"No adapter state_dict found in checkpoint keys: {list(ckpt.keys())}")
    return state


def checkpoint_grad_accum_steps(resume_ckpt: dict) -> int | None:
    hyperparams = resume_ckpt.get("hyperparams") or {}
    saved = hyperparams.get("grad_accum_steps")
    return int(saved) if saved is not None else None


def resolve_resume_epoch_and_offset(
    *,
    resume_ckpt: dict,
    micro_steps_per_epoch: int,
    grad_accum_steps: int,
    checkpoint_dir: str,
    checkpoint_basename: str,
) -> tuple[int, int]:
    """
    Map a checkpoint to ``(start_epoch_0idx, micro_batch_offset)``.

    Checkpoints store 1-indexed ``epoch`` and ``global_step`` as optimizer updates.
    When ``grad_accum_steps`` is recorded, derive the dataloader offset within the
    saved epoch using the prior ``{basename}_epoch{N-1}.pt`` boundary when available.
    """
    global_step = int(resume_ckpt["global_step"])
    saved_accum = checkpoint_grad_accum_steps(resume_ckpt)
    resume_epoch_1idx = int(resume_ckpt["epoch"])
    start_epoch = max(0, resume_epoch_1idx - 1)

    if saved_accum is not None:
        epoch_start_step = 0
        if resume_epoch_1idx > 1:
            prev_epoch_path = os.path.join(
                checkpoint_dir,
                f"{checkpoint_basename}_epoch{resume_epoch_1idx - 1}.pt",
            )
            if os.path.isfile(prev_epoch_path):
                prev = torch.load(prev_epoch_path, map_location="cpu", weights_only=False)
                epoch_start_step = int(prev["global_step"])
            else:
                optimizer_steps_per_epoch = max(
                    1,
                    (micro_steps_per_epoch + grad_accum_steps - 1) // grad_accum_steps,
                )
                epoch_start_step = (resume_epoch_1idx - 1) * optimizer_steps_per_epoch
        steps_into_epoch = max(0, global_step - epoch_start_step)
        batch_offset = steps_into_epoch * saved_accum
        if batch_offset >= micro_steps_per_epoch:
            batch_offset = batch_offset % micro_steps_per_epoch
    else:
        batch_offset = global_step - start_epoch * micro_steps_per_epoch
        if batch_offset < 0:
            start_epoch = global_step // micro_steps_per_epoch
            batch_offset = global_step % micro_steps_per_epoch
        elif batch_offset >= micro_steps_per_epoch:
            batch_offset = batch_offset % micro_steps_per_epoch

    return start_epoch, batch_offset


def filter_adapter_state_dict(
    state: dict,
    *,
    num_queries: int,
    use_rate_controller: bool,
) -> dict:
    """
    Slice / drop checkpoint keys so they match the target :class:`StreamingAdapter` config.

    Stage-2 checkpoints may include ``rate_controller.*`` weights; ASR-only and stage-1
    trainers build adapters with ``use_rate_controller=False`` and must ignore those keys.
    """
    out = adapt_adapter_state_dict_num_queries(state, num_queries)
    if not use_rate_controller:
        out = {k: v for k, v in out.items() if not k.startswith("rate_controller.")}
    return out


def adapt_adapter_state_dict_num_queries(state: dict, num_queries: int) -> dict:
    """
    Slice adapter weights when loading a checkpoint trained with more queries than the model uses.

    Existing stage-1/2 checkpoints use ``num_queries=4``; inference can run with ``num_queries=2``
    by keeping the first *m* learnable query vectors (and matching rate-controller head rows).
    """
    out = dict(state)
    queries = out.get("queries")
    if queries is not None:
        ckpt_m = int(queries.shape[1])
        if ckpt_m > num_queries:
            out["queries"] = queries[:, :num_queries].contiguous()
        elif ckpt_m < num_queries:
            raise ValueError(
                f"Checkpoint has num_queries={ckpt_m} but model expects {num_queries}"
            )

    wkey = "rate_controller.gate_mlp.2.weight"
    bkey = "rate_controller.gate_mlp.2.bias"
    weight = out.get(wkey)
    if weight is not None and int(weight.shape[0]) > num_queries:
        out[wkey] = weight[:num_queries].contiguous()
    bias = out.get(bkey)
    if bias is not None and int(bias.shape[0]) > num_queries:
        out[bkey] = bias[:num_queries].contiguous()
    return out


def adapt_optimizer_state_dict_num_queries(
    optimizer_state: dict,
    *,
    adapter_state: dict,
    num_queries: int,
) -> dict:
    """
    Slice Adam momentum buffers when resuming a checkpoint trained with more queries.

    Mirrors :func:`adapt_adapter_state_dict_num_queries` on ``exp_avg`` / ``exp_avg_sq``.
    """
    queries = adapter_state.get("queries")
    if queries is None:
        return optimizer_state
    ckpt_m = int(queries.shape[1])
    if ckpt_m <= num_queries:
        return optimizer_state

    shape_slices: dict[tuple[int, ...], object] = {
        tuple(queries.shape): lambda t: t[:, :num_queries],
    }
    weight = adapter_state.get("rate_controller.gate_mlp.2.weight")
    if weight is not None:
        shape_slices[tuple(weight.shape)] = lambda t: t[:num_queries]
    bias = adapter_state.get("rate_controller.gate_mlp.2.bias")
    if bias is not None:
        shape_slices[tuple(bias.shape)] = lambda t: t[:num_queries]

    out = {"param_groups": optimizer_state["param_groups"], "state": {}}
    for key, param_state in optimizer_state["state"].items():
        new_state = dict(param_state)
        for buf_name in ("exp_avg", "exp_avg_sq"):
            buf = param_state.get(buf_name)
            if buf is None:
                continue
            slicer = shape_slices.get(tuple(buf.shape))
            if slicer is not None:
                new_state[buf_name] = slicer(buf).contiguous()
        out["state"][key] = new_state
    return out


def load_gate_state_dict_safe(
    gate: torch.nn.Module,
    ckpt: dict,
    *,
    warn: bool = True,
) -> None:
    """
    Load ``gate_state_dict`` when present; warn on architecture mismatch.

    Old mean-pool gate checkpoints are incompatible with :class:`TurnEndCommitGate`.
    """
    state = ckpt.get("gate_state_dict")
    if state is None:
        if warn:
            print("[WARN] Checkpoint has no gate_state_dict; gate stays randomly initialized.")
        return

    result = gate.load_state_dict(state, strict=False)
    if warn and (result.missing_keys or result.unexpected_keys):
        print(
            "[WARN] Gate checkpoint partial load — architecture may have changed "
            f"(missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)}). "
            "Re-train the gate or use a TurnEndCommitGate checkpoint."
        )
        if result.missing_keys:
            print(f"  missing: {result.missing_keys[:8]}{'...' if len(result.missing_keys) > 8 else ''}")
        if result.unexpected_keys:
            print(f"  unexpected: {result.unexpected_keys[:8]}{'...' if len(result.unexpected_keys) > 8 else ''}")


def infer_gate_silence_settings(gate_state_dict: dict) -> tuple[str, str]:
    """
    Infer ``(silence_mode, active_silence_path)`` from ``gate_state_dict`` keys.

    Checkpoints without ``hyperparams['gate_cfg']`` (e.g. older ``adapter_stage2.pt``)
    still carry learned heads as ``learned_silence_head.*`` / ``classifier_learned.*``.
    """
    keys = gate_state_dict.keys()
    has_learned = any(
        k.startswith("learned_silence_head.") or k.startswith("classifier_learned.")
        for k in keys
    )
    has_rule_classifier = any(
        k.startswith("classifier.") and not k.startswith("classifier_learned.")
        for k in keys
    )
    if has_learned and has_rule_classifier:
        return "both", "learned"
    if has_learned:
        return "learned", "learned"
    return "rule", "rule"


def resolve_gate_config_from_checkpoint(
    ckpt: dict,
    *,
    defaults: GateConfig,
) -> GateConfig:
    """Merge ``hyperparams['gate_cfg']`` onto env defaults; infer silence mode from weights if needed."""
    hyperparams = ckpt.get("hyperparams") or {}
    saved = hyperparams.get("gate_cfg")
    valid = {f.name for f in fields(GateConfig)}
    overrides: dict = (
        {k: v for k, v in saved.items() if k in valid} if isinstance(saved, dict) else {}
    )

    state = ckpt.get("gate_state_dict")
    if isinstance(state, dict) and state:
        inferred_mode, inferred_active = infer_gate_silence_settings(state)
        overrides.setdefault("silence_mode", inferred_mode)
        overrides.setdefault("active_silence_path", inferred_active)

    return replace(defaults, **overrides) if overrides else defaults


def hub_path_for_stage_epoch_upload(local_path: str, *, stage: int) -> str:
    """
    Hub path for one epoch checkpoint of ``stage`` only (repo root, no prefix).

    Only ``adapter_stage{N}_epoch{M}.pt`` is allowed. Resume files such as
    ``adapter_stage{N}.pt`` are never uploaded.
    """
    filename = os.path.basename(local_path)
    match = STAGE_EPOCH_FILENAME.match(filename)
    if not match:
        raise ValueError(
            f"Only adapter_stage{{N}}_epoch{{M}}.pt checkpoints are uploaded; got: {filename}"
        )
    file_stage = int(match.group(1))
    if file_stage != stage:
        raise ValueError(
            f"Checkpoint is stage {file_stage} but upload requested for stage {stage}: {filename}"
        )
    return filename


def upload_checkpoint_to_hub(
    local_path: str,
    *,
    repo_id: str,
    path_in_repo: str | None = None,
    revision: str = "main",
    private: bool = False,
    token: str | None = None,
) -> str:
    """
    Upload one checkpoint file to the Hugging Face Hub.

    Uses ``upload_file`` (additive): other repo files are left unchanged.
    """
    if not os.path.isfile(local_path):
        raise FileNotFoundError(local_path)

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError("huggingface_hub is required for checkpoint upload") from exc

    hub_path = path_in_repo or os.path.basename(local_path)
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=hub_path,
        repo_id=repo_id,
        revision=revision,
        commit_message=f"Upload {hub_path}",
    )
    return f"https://huggingface.co/{repo_id}/resolve/{revision}/{hub_path}"


def maybe_upload_stage_epoch_checkpoint(
    local_path: str,
    *,
    stage: int,
    repo_id: str,
    revision: str = "main",
    private: bool = False,
    token: str | None = None,
    enabled: bool = False,
) -> None:
    """Upload ``adapter_stage{stage}_epoch{N}.pt`` when enabled; never touches other stages' files."""
    if not enabled:
        return
    try:
        hub_path = hub_path_for_stage_epoch_upload(local_path, stage=stage)
        url = upload_checkpoint_to_hub(
            local_path,
            repo_id=repo_id,
            path_in_repo=hub_path,
            revision=revision,
            private=private,
            token=token,
        )
        print(f"Uploaded checkpoint to {url}")
    except ValueError:
        return
    except Exception as exc:
        print(f"[WARN] Hugging Face upload failed for {local_path}: {exc}")
