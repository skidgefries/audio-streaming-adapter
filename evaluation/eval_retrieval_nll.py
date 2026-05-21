"""
Retrieval evaluation using NLL scoring (Audio → Text).

Instead of cosine similarity between pooled embeddings, this script ranks candidate
transcripts by *negative log-likelihood* under the frozen Qwen causal LM,
conditioned on the adapter-produced audio prefix tokens.

Lower NLL => better match.
"""

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
from src.encoder import WhisperWindowFeatureExtractor
from training.utils.loaders import default_device_and_dtype, load_frozen_qwen_causal_lm

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE, TORCH_DTYPE = default_device_and_dtype()

WHISPER_DIM = 768
LLM_DIM = 4096
WHISPER_MODEL = "openai/whisper-small"
LLM_MODEL_ID = "Qwen/Qwen3-8B"

CHECKPOINT_PATH = "checkpoints/adapter_adapter.pt"
TEST_CLEAN_ROOT = "datasets/librispeech_data/LibriSpeech/test-clean"
NUM_UTTERANCES = 200  # NLL retrieval is O(N^2); keep this modest by default

CANDIDATE_BATCH_SIZE = 16
MAX_TEXT_TOKENS = 256


def _maybe_autocast():
    if torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


# ── Load models ───────────────────────────────────────────────────────────────
def load_models():
    print("Loading Whisper...")
    audio = WhisperWindowFeatureExtractor(
        model_id=WHISPER_MODEL,
        device=DEVICE,
        torch_dtype=TORCH_DTYPE,
    )

    print("Loading Qwen causal LM...")
    qwen_models = load_frozen_qwen_causal_lm(
        model_id=LLM_MODEL_ID,
        device=DEVICE,
        torch_dtype=TORCH_DTYPE,
        device_map="auto",
    )
    tokenizer = qwen_models.tokenizer
    llm = qwen_models.causal_lm
    text_embedder = qwen_models.embedder

    print("Loading adapter from checkpoint...")
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
        use_rate_controller=False,
    ).to(DEVICE, dtype=torch.bfloat16)

    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    adapter.load_state_dict(ckpt["adapter_state_dict"])
    adapter.eval()
    print(f"  Loaded checkpoint from epoch {ckpt['epoch']}, step {ckpt['global_step']}\n")

    return audio, tokenizer, llm, text_embedder, adapter


@torch.no_grad()
def compute_audio_prefix_tokens(audio_extractor, adapter, pairs):
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
        with _maybe_autocast():
            for w in windows:
                out = adapter.forward_window(w)
                chunks.append(out["tokens"])
        audio_tokens = torch.cat(chunks, dim=1)  # (1, T, 4096)

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
def nll_matrix(audio_tokens_list, input_ids, attention_mask, llm, text_embedder, *, batch_size: int):
    """
    Returns an [N, N] matrix where entry (i, j) is the mean per-token NLL of text j
    conditioned on audio i.
    """
    n = len(audio_tokens_list)
    assert input_ids.shape[0] == n
    assert attention_mask.shape[0] == n

    # Shift like the Stage 2 trainer: predict tokens 1..end given prefix + token0
    text_ids_shifted = input_ids[:, 1:].contiguous()
    text_mask_shifted = attention_mask[:, 1:].contiguous()

    pad_id = (
        llm.config.pad_token_id
        if getattr(llm.config, "pad_token_id", None) is not None
        else (text_embedder.weight.new_tensor([0], dtype=torch.long).item())
    )

    # Ensure pad positions are a real in-vocab id, then mask with -100 in labels.
    text_ids_shifted = text_ids_shifted.clone()
    text_ids_shifted[text_mask_shifted == 0] = int(pad_id)

    bos_token_id = (
        getattr(llm.config, "bos_token_id", None)
        if getattr(llm.config, "bos_token_id", None) is not None
        else getattr(llm.config, "eos_token_id", None)
    )
    if bos_token_id is None:
        bos_token_id = input_ids.new_tensor([input_ids[0, 0].item()]).item()

    out_scores = torch.empty((n, n), dtype=torch.float32)

    for i in range(n):
        a = audio_tokens_list[i].to(DEVICE)  # (T_a, D)
        a_len = a.shape[0]
        a = a.unsqueeze(0)  # (1, T_a, D)

        bos_ids = torch.tensor([[int(bos_token_id)]], device=DEVICE, dtype=torch.long)
        bos_embed = text_embedder(bos_ids)  # (1, 1, D)

        for j0 in range(0, n, batch_size):
            j1 = min(n, j0 + batch_size)
            b = j1 - j0

            cand_ids = text_ids_shifted[j0:j1].to(DEVICE)  # (B, T_t)
            cand_mask = text_mask_shifted[j0:j1].to(DEVICE)  # (B, T_t)
            cand_embeds = text_embedder(cand_ids)  # (B, T_t, D)

            prefix = torch.cat(
                [
                    a.expand(b, -1, -1),
                    bos_embed.expand(b, -1, -1),
                ],
                dim=1,
            )  # (B, T_a+1, D)
            inputs_embeds = torch.cat([prefix, cand_embeds], dim=1)  # (B, S, D)

            labels_prefix = torch.full((b, a_len + 1), -100, device=DEVICE, dtype=torch.long)
            labels_text = cand_ids.clone()
            labels_text[cand_mask == 0] = -100
            labels = torch.cat([labels_prefix, labels_text], dim=1)  # (B, S_lbl)

            attn_prefix = torch.ones((b, a_len + 1), device=DEVICE, dtype=torch.long)
            attn = torch.cat([attn_prefix, cand_mask], dim=1)  # (B, S)

            with _maybe_autocast():
                out = llm(inputs_embeds=inputs_embeds, attention_mask=attn)
                logits = out.logits  # (B, S, V)

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
            loss = (loss_tok * valid).sum(dim=1) / denom  # (B,)

            out_scores[i, j0:j1] = loss.detach().cpu()

        if (i + 1) % 25 == 0 or i == n - 1:
            print(f"  Scored {i + 1}/{n} audio queries...")

    return out_scores


def compute_recall_from_nll(nll_scores, ks=(1, 5, 10)):
    n = nll_scores.shape[0]

    # Lower NLL is better => ascending sort
    ranked = nll_scores.argsort(dim=1, descending=False)

    results = {}
    for k in ks:
        top_k = ranked[:, :k]
        correct = torch.arange(n).unsqueeze(1)
        hits = (top_k == correct).any(dim=1).float()
        results[f"R@{k}"] = hits.mean().item() * 100
    return results


def main():
    torch.cuda.empty_cache()

    print(f"Loading test-clean from {TEST_CLEAN_ROOT}...")
    dataset = LibriSpeechPairs(TEST_CLEAN_ROOT)
    pairs = dataset.pairs[:NUM_UTTERANCES]
    print(f"Preparing {len(pairs)} utterances\n")

    audio, tokenizer, llm, text_embedder, adapter = load_models()

    print("Computing audio prefix tokens...")
    audio_tokens_list, kept_pairs = compute_audio_prefix_tokens(audio, adapter, pairs)
    if not audio_tokens_list:
        raise RuntimeError("No audio windows produced any prefix tokens; nothing to evaluate.")

    # Keep audio/text aligned by index after dropping empties.
    texts = [t for _, t in kept_pairs]
    print(f"  Kept {len(texts)} utterances after filtering\n")

    print("Tokenizing candidate transcripts...")
    input_ids, attention_mask = tokenize_candidates(tokenizer, texts)

    print("Computing NLL matrix (this can be slow)...")
    nll_scores = nll_matrix(
        audio_tokens_list,
        input_ids,
        attention_mask,
        llm,
        text_embedder,
        batch_size=CANDIDATE_BATCH_SIZE,
    )

    print("Computing retrieval metrics...")
    results = compute_recall_from_nll(nll_scores)

    print("\n" + "=" * 50)
    print("RETRIEVAL RESULTS (Audio → Text, NLL)")
    print("=" * 50)
    for k, v in results.items():
        print(f"  {k}: {v:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()

