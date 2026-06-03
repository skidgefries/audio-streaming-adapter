"""
NLL retrieval evaluation for Stage 2 (ASR distillation) checkpoints.

Same metrics and procedure as ``eval_retrieval_nll.py`` (R@1, R@5, R@10 via mean
token NLL under frozen Qwen), but builds :class:`StreamingAdapter` with Stage 2
options (rate controller, etc.) so ``adapter_stage2.pt`` loads cleanly.

Loads Whisper + adapter first, encodes audio prefixes, then frees them before
loading the causal LM to reduce peak VRAM on a single GPU.
"""

from __future__ import annotations

import gc
import os
import sys
from contextlib import nullcontext

import torch
import torch.nn.functional as F

_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src_root = os.path.join(_pkg_root, "src")
sys.path.insert(0, _src_root)
sys.path.insert(0, _pkg_root)

from src.adapter.streaming_adapter import StreamingAdapter
from src.dataset import LibriSpeechPairs, load_mono_waveform_16k
from src.encoder.waveform_window_encoder import WhisperWindowFeatureExtractor
from training.utils.config import Stage2Config
from training.utils.devices import llm_input_device, resolve_device, visible_gpu_count
from training.utils.env import env_int, load_project_env
from training.utils.loaders import load_frozen_qwen_causal_lm

# ── Config ────────────────────────────────────────────────────────────────────
STAGE2 = Stage2Config.from_env()

WHISPER_DIM = 768
LLM_DIM = 4096
WHISPER_MODEL = "openai/whisper-small"
LLM_MODEL_ID = "Qwen/Qwen3-8B"

CHECKPOINT_PATH = os.path.join(_pkg_root, "checkpoints", "adapter_stage2.pt")
TEST_CLEAN_ROOT = os.path.join(
    _pkg_root, "datasets/librispeech_data/LibriSpeech/test-clean"
)
NUM_UTTERANCES = env_int("RETRIEVAL_NLL_NUM_UTTERANCES", 2620)
CANDIDATE_BATCH_SIZE = env_int("RETRIEVAL_NLL_BATCH_SIZE", 8)
MAX_TEXT_TOKENS = 512


def _init_eval_device() -> tuple[torch.device, torch.dtype, int]:
    """
    Resolve ``DEVICE`` and touch only that CUDA device.

    When ``DEVICE=cuda:1`` with ``CUDA_VISIBLE_DEVICES=0,1``, Whisper and Qwen
    both load on logical ``cuda:1`` (physical GPU 1); ``cuda:0`` is left alone.
    """
    load_project_env(_pkg_root)
    device = resolve_device()
    num_cuda = visible_gpu_count()
    torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    if device.type == "cuda":
        torch.cuda.init()
        torch.cuda.set_device(device)
        torch.zeros((), device=device)
        torch.cuda.empty_cache()
    return device, torch_dtype, num_cuda


def _assert_module_device(module: torch.nn.Module, device: torch.device, name: str) -> None:
    param_device = next(module.parameters()).device
    if param_device != device:
        raise RuntimeError(f"{name} expected on {device}, found on {param_device}")


def _maybe_autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _free_cuda(*objs: object, device: torch.device | None = None) -> None:
    for obj in objs:
        del obj
    gc.collect()
    if not torch.cuda.is_available():
        return
    if device is not None and device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    else:
        torch.cuda.empty_cache()


def load_audio_encoder_and_adapter(
    *,
    device: torch.device,
    torch_dtype: torch.dtype,
    checkpoint_path: str = CHECKPOINT_PATH,
):
    device_str = str(device)
    print("Loading Whisper...")
    audio = WhisperWindowFeatureExtractor(
        model_id=WHISPER_MODEL,
        device=device_str,
        torch_dtype=torch_dtype,
    )

    print("Loading Stage 2 adapter from checkpoint...")
    adapter = StreamingAdapter(
        d_encoder=WHISPER_DIM,
        d_llm=LLM_DIM,
        num_queries=4,
        num_layers=2,
        num_heads=4,
        d_ffn=2048,
        dropout=0.0,
        ema_alpha=0.8,
        learnable_ema=False,
        use_rate_controller=STAGE2.use_rate_controller,
        rate_threshold=0.5,
        target_rate=STAGE2.rate_target,
    ).to(device, dtype=torch.bfloat16)

    ckpt = torch.load(checkpoint_path, map_location=device)
    adapter.load_state_dict(ckpt["adapter_state_dict"])
    adapter.eval()
    _assert_module_device(audio.whisper, device, "Whisper")
    _assert_module_device(adapter, device, "Adapter")
    print(
        f"  Loaded {checkpoint_path}\n"
        f"  epoch={ckpt.get('epoch', '?')} step={ckpt.get('global_step', '?')} "
        f"rate_controller={STAGE2.use_rate_controller} target_rate={STAGE2.rate_target}\n"
        f"  Whisper + adapter on {device}\n"
    )
    return audio, adapter


def load_llm(*, device: torch.device, torch_dtype: torch.dtype):
    device_str = str(device)
    print("Loading Qwen causal LM...")
    qwen_models = load_frozen_qwen_causal_lm(
        model_id=LLM_MODEL_ID,
        device=device_str,
        torch_dtype=torch_dtype,
        device_map=None,
    )
    llm = qwen_models.causal_lm
    llm_device = llm_input_device(llm)
    if llm_device != device:
        raise RuntimeError(f"Qwen expected on {device}, found on {llm_device}")
    print(f"  Qwen causal LM on {llm_device}\n")
    return qwen_models.tokenizer, llm, qwen_models.embedder, llm_device


@torch.no_grad()
def compute_audio_prefix_tokens(audio_extractor, adapter, pairs, *, device: torch.device):
    audio_tokens_list = []
    kept_pairs = []

    for i, (audio_path, transcription) in enumerate(pairs):
        if i % 50 == 0:
            print(f"  Processing audio {i}/{len(pairs)}...")

        wave = load_mono_waveform_16k(audio_path)
        windows = audio_extractor.waveform_to_windows(wave)
        if len(windows) == 0:
            continue

        adapter.reset_streaming_state()
        chunks = []
        with _maybe_autocast(device):
            for w in windows:
                out = adapter.forward_window(w)
                chunks.append(out["tokens"])
        audio_tokens = torch.cat(chunks, dim=1)

        audio_tokens_list.append(audio_tokens.squeeze(0).cpu())
        kept_pairs.append((audio_path, transcription))

    return audio_tokens_list, kept_pairs


@torch.no_grad()
def tokenize_candidates(tokenizer, texts):
    tok = tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_TEXT_TOKENS,
    )
    return tok.input_ids, tok.attention_mask


@torch.no_grad()
def nll_matrix(
    audio_tokens_list,
    input_ids,
    attention_mask,
    llm,
    text_embedder,
    llm_device: torch.device,
    *,
    batch_size: int,
    autocast_device: torch.device,
):
    """
    Returns an [N, N] matrix where entry (i, j) is the mean per-token NLL of text j
    conditioned on audio i.
    """
    n = len(audio_tokens_list)
    assert input_ids.shape[0] == n
    assert attention_mask.shape[0] == n

    text_ids_shifted = input_ids[:, 1:].contiguous()
    text_mask_shifted = attention_mask[:, 1:].contiguous()

    pad_id = (
        llm.config.pad_token_id
        if getattr(llm.config, "pad_token_id", None) is not None
        else (text_embedder.weight.new_tensor([0], dtype=torch.long).item())
    )

    text_ids_shifted = text_ids_shifted.clone()
    text_ids_shifted[text_mask_shifted == 0] = int(pad_id)

    bos_token_id = (
        getattr(llm.config, "bos_token_id", None)
        if getattr(llm.config, "bos_token_id", None) is not None
        else getattr(llm.config, "eos_token_id", None)
    )
    if bos_token_id is None:
        bos_token_id = input_ids[0, 0].item()

    out_scores = torch.empty((n, n), dtype=torch.float32)

    for i in range(n):
        a = audio_tokens_list[i].to(llm_device)
        a_len = a.shape[0]
        a = a.unsqueeze(0)

        bos_ids = torch.tensor([[int(bos_token_id)]], device=llm_device, dtype=torch.long)
        bos_embed = text_embedder(bos_ids)

        for j0 in range(0, n, batch_size):
            j1 = min(n, j0 + batch_size)
            b = j1 - j0

            cand_ids = text_ids_shifted[j0:j1].to(llm_device)
            cand_mask = text_mask_shifted[j0:j1].to(llm_device)
            cand_embeds = text_embedder(cand_ids)

            prefix = torch.cat(
                [a.expand(b, -1, -1), bos_embed.expand(b, -1, -1)],
                dim=1,
            )
            inputs_embeds = torch.cat([prefix, cand_embeds], dim=1)

            labels_prefix = torch.full((b, a_len + 1), -100, device=llm_device, dtype=torch.long)
            labels_text = cand_ids.clone()
            labels_text[cand_mask == 0] = -100
            labels = torch.cat([labels_prefix, labels_text], dim=1)

            attn_prefix = torch.ones((b, a_len + 1), device=llm_device, dtype=torch.long)
            attn = torch.cat([attn_prefix, cand_mask], dim=1)

            with _maybe_autocast(autocast_device):
                out = llm(inputs_embeds=inputs_embeds, attention_mask=attn)
                logits = out.logits

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_tok = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
                ignore_index=-100,
            ).view(b, -1)
            valid = (shift_labels != -100).to(loss_tok.dtype)
            denom = valid.sum(dim=1).clamp_min(1.0)
            loss = (loss_tok * valid).sum(dim=1) / denom

            out_scores[i, j0:j1] = loss.detach().cpu()

            del inputs_embeds, logits, shift_logits, shift_labels, loss_tok, loss, out

        if (i + 1) % 25 == 0 or i == n - 1:
            print(f"  Scored {i + 1}/{n} audio queries...")

    return out_scores


def compute_recall_from_nll(nll_scores, ks=(1, 5, 10)):
    n = nll_scores.shape[0]
    ranked = nll_scores.argsort(dim=1, descending=False)

    results = {}
    for k in ks:
        top_k = ranked[:, :k]
        correct = torch.arange(n).unsqueeze(1)
        hits = (top_k == correct).any(dim=1).float()
        results[f"R@{k}"] = hits.mean().item() * 100
    return results


def main():
    device, torch_dtype, num_cuda = _init_eval_device()

    print(
        f"Device: {device} (visible CUDA devices: {num_cuda}, "
        f"single-GPU eval — Whisper then Qwen on {device})\n"
    )

    print(f"Loading test-clean from {TEST_CLEAN_ROOT}...")
    dataset = LibriSpeechPairs(TEST_CLEAN_ROOT)
    pairs = dataset.pairs[:NUM_UTTERANCES]
    print(
        f"Preparing {len(pairs)} utterances "
        f"(NLL batch={CANDIDATE_BATCH_SIZE}, max_text_tokens={MAX_TEXT_TOKENS})\n"
    )

    audio, adapter = load_audio_encoder_and_adapter(device=device, torch_dtype=torch_dtype)

    print("Computing audio prefix tokens...")
    audio_tokens_list, kept_pairs = compute_audio_prefix_tokens(
        audio, adapter, pairs, device=device
    )
    if not audio_tokens_list:
        raise RuntimeError("No audio windows produced any prefix tokens; nothing to evaluate.")

    texts = [t for _, t in kept_pairs]
    print(f"  Kept {len(texts)} utterances after filtering\n")

    print("Releasing Whisper and adapter before loading causal LM...")
    _free_cuda(audio, adapter, device=device)

    tokenizer, llm, text_embedder, llm_device = load_llm(device=device, torch_dtype=torch_dtype)

    print("Tokenizing candidate transcripts...")
    input_ids, attention_mask = tokenize_candidates(tokenizer, texts)

    print("Computing NLL matrix (this can be slow)...")
    nll_scores = nll_matrix(
        audio_tokens_list,
        input_ids,
        attention_mask,
        llm,
        text_embedder,
        llm_device,
        batch_size=CANDIDATE_BATCH_SIZE,
        autocast_device=device,
    )

    print("Computing retrieval metrics...")
    results = compute_recall_from_nll(nll_scores)

    print("\n" + "=" * 50)
    print("RETRIEVAL RESULTS — Stage 2 checkpoint (Audio → Text, NLL)")
    print("=" * 50)
    for k, v in results.items():
        print(f"  {k}: {v:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()
