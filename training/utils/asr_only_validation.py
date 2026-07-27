"""ASR validation: teacher-forced NLL, align/stability aux, greedy/beam decode, WER, BLEU-4."""

from __future__ import annotations

import json
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from llm.config import LlmGenerationParams, build_hf_generation_config
from src.adapter.streaming_adapter import StreamingAdapter
from src.adapter_llm_pipeline import WhisperAdapterLLMPipeline
from src.dataset import LibriSpeechPairs
from src.encoder import WhisperWindowFeatureExtractor
from training.utils.metrics import RunningMean, corpus_bleu4, normalize_asr_text, word_error_rate


@dataclass(frozen=True)
class UtteranceEncodeResult:
    tokens: torch.Tensor
    num_windows: int
    stability_loss: float = 0.0
    # Optional fields kept for older encode helpers; unused by this validator.
    sparse_loss: float = 0.0
    rate_loss: float = 0.0


def _utterance_id(audio_path: str) -> str:
    return Path(audio_path).stem


def _pad_audio_tokens(utterances: list[torch.Tensor]) -> torch.Tensor:
    max_len = max(t.shape[1] for t in utterances)
    padded = []
    for tokens in utterances:
        pad_len = max_len - tokens.shape[1]
        if pad_len > 0:
            padding = torch.zeros(
                tokens.shape[0],
                pad_len,
                tokens.shape[2],
                device=tokens.device,
                dtype=tokens.dtype,
            )
            tokens = torch.cat([tokens, padding], dim=1)
        padded.append(tokens)
    return torch.cat(padded, dim=0)


def _train_style_prefix_embeds_one(
    *,
    llm_model: torch.nn.Module,
    llm_tokenizer,
    audio_tokens: torch.Tensor,
    llm_device: torch.device,
    torch_dtype: torch.dtype,
    append_im_end: bool,
) -> torch.Tensor:
    """Build ``[audio]`` or ``[audio | im_end/BOS | no_think]`` embeds for one utterance."""
    audio = audio_tokens.to(device=llm_device, dtype=torch_dtype)
    if audio.ndim != 3 or audio.shape[0] != 1:
        raise ValueError(f"Expected audio_tokens shape (1, T, D), got {tuple(audio.shape)}")
    if not append_im_end:
        return audio
    sep_id = WhisperAdapterLLMPipeline.train_style_separator_token_id(llm_tokenizer)
    sep_ids = torch.tensor([[sep_id]], device=llm_device, dtype=torch.long)
    sep_embed = llm_model.get_input_embeddings()(sep_ids)
    no_think_embed = WhisperAdapterLLMPipeline.qwen_no_think_suffix_embeds(
        llm_model, llm_device, torch_dtype
    )
    return torch.cat([audio, sep_embed, no_think_embed], dim=1)


def _left_pad_embeds(embeds_list: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-pad variable-length ``(1, T, D)`` embeds to ``(B, T_max, D)`` + attention mask."""
    if not embeds_list:
        raise ValueError("embeds_list must be non-empty")
    max_t = max(int(e.shape[1]) for e in embeds_list)
    dim = int(embeds_list[0].shape[2])
    device = embeds_list[0].device
    dtype = embeds_list[0].dtype
    batch = len(embeds_list)
    out = torch.zeros(batch, max_t, dim, device=device, dtype=dtype)
    mask = torch.zeros(batch, max_t, dtype=torch.long, device=device)
    for i, emb in enumerate(embeds_list):
        t = int(emb.shape[1])
        out[i, max_t - t :] = emb[0]
        mask[i, max_t - t :] = 1
    return out, mask


@torch.no_grad()
def decode_asr_predictions_batch(
    *,
    llm_model: torch.nn.Module,
    llm_tokenizer,
    audio_tokens_list: list[torch.Tensor],
    llm_device: torch.device,
    torch_dtype: torch.dtype,
    generation: LlmGenerationParams,
    append_im_end: bool = True,
    trim_asr_tail: bool = False,
    maybe_autocast_fn: Callable[[torch.device], AbstractContextManager] | None = None,
) -> list[str]:
    """
    Batched greedy/beam decode from compressed audio tokens (train-style prefix).

    Each entry in ``audio_tokens_list`` is ``(1, T_i, D)``. Prefixes are left-padded so
    HuggingFace ``generate`` can run as one batch.
    """
    if not audio_tokens_list:
        return []
    if len(audio_tokens_list) == 1:
        return [
            decode_asr_prediction(
                llm_model=llm_model,
                llm_tokenizer=llm_tokenizer,
                audio_tokens=audio_tokens_list[0],
                llm_device=llm_device,
                torch_dtype=torch_dtype,
                generation=generation,
                append_im_end=append_im_end,
                trim_asr_tail=trim_asr_tail,
                maybe_autocast_fn=maybe_autocast_fn,
            )
        ]

    embeds_list = [
        _train_style_prefix_embeds_one(
            llm_model=llm_model,
            llm_tokenizer=llm_tokenizer,
            audio_tokens=tokens,
            llm_device=llm_device,
            torch_dtype=torch_dtype,
            append_im_end=append_im_end,
        )
        for tokens in audio_tokens_list
    ]
    input_embeds, attention_mask = _left_pad_embeds(embeds_list)
    gen_cfg = build_hf_generation_config(
        model=llm_model,
        tokenizer=llm_tokenizer,
        params=generation,
    )
    autocast = maybe_autocast_fn(llm_device) if maybe_autocast_fn else nullcontext()
    with autocast:
        out_ids = llm_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=gen_cfg,
        )
    texts: list[str] = []
    for row in out_ids:
        text = llm_tokenizer.decode(row, skip_special_tokens=True).strip()
        if trim_asr_tail:
            from adapter_llm_pipeline import _truncate_asr_chat_tail

            text = _truncate_asr_chat_tail(text)
        texts.append(text)
    return texts


@torch.no_grad()
def decode_asr_prediction(
    *,
    llm_model: torch.nn.Module,
    llm_tokenizer,
    audio_tokens: torch.Tensor,
    llm_device: torch.device,
    torch_dtype: torch.dtype,
    generation: LlmGenerationParams,
    append_im_end: bool = True,
    trim_asr_tail: bool = False,
    maybe_autocast_fn: Callable[[torch.device], AbstractContextManager] | None = None,
) -> str:
    """Greedy/beam decode from compressed audio tokens (train-style prefix)."""
    input_embeds = _train_style_prefix_embeds_one(
        llm_model=llm_model,
        llm_tokenizer=llm_tokenizer,
        audio_tokens=audio_tokens,
        llm_device=llm_device,
        torch_dtype=torch_dtype,
        append_im_end=append_im_end,
    )
    attention_mask = torch.ones(
        input_embeds.shape[0],
        input_embeds.shape[1],
        dtype=torch.long,
        device=input_embeds.device,
    )
    gen_cfg = build_hf_generation_config(
        model=llm_model,
        tokenizer=llm_tokenizer,
        params=generation,
    )
    autocast = maybe_autocast_fn(llm_device) if maybe_autocast_fn else nullcontext()
    with autocast:
        out_ids = llm_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=gen_cfg,
        )
    text = llm_tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
    if trim_asr_tail:
        from adapter_llm_pipeline import _truncate_asr_chat_tail

        text = _truncate_asr_chat_tail(text)
    return text


@torch.no_grad()
def validate_asr_only(
    *,
    adapter: StreamingAdapter,
    audio_extractor: WhisperWindowFeatureExtractor,
    llm_model: torch.nn.Module,
    llm_tokenizer,
    text_embedder: torch.nn.Module,
    encode_utterance_fn: Callable[..., UtteranceEncodeResult | None],
    build_inputs_for_asr_fn: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    asr_forward_loss_fn: Callable[..., torch.Tensor],
    compute_aux_loss_metrics_fn: Callable[..., dict[str, float]],
    val_root: str,
    train_device: torch.device,
    llm_device: torch.device,
    max_utterances: int | None,
    max_windows_per_utt: int | None,
    max_text_tokens: int,
    asr_micro_batch_size: int,
    generation: LlmGenerationParams,
    global_step: int,
    batch_size: int = 1,
    num_workers: int = 0,
    predictions_path: str | None = None,
    log_every: int = 50,
    maybe_autocast_fn: Callable[[torch.device], AbstractContextManager] | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """
    Run dev-clean validation for ASR + align + stability training.

    Losses match ``adapter_asr_align_trainer`` (no rate / gate / sparse).
    WER/BLEU use per-utterance greedy/beam decode.
    """
    adapter_was_training = adapter.training
    adapter.eval()

    val_full = LibriSpeechPairs(val_root)
    n_val = len(val_full) if max_utterances is None else min(max_utterances, len(val_full))
    val_dataset: Dataset = val_full if n_val == len(val_full) else Subset(val_full, range(n_val))
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    m_asr = RunningMean()
    m_align = RunningMean()
    m_stab = RunningMean()
    refs: list[str] = []
    hyps: list[str] = []
    items: list[dict[str, Any]] = []
    samples_done = 0

    print(
        f"\n  [val step {global_step}] dev-clean ({n_val} utterances, "
        f"{len(val_loader)} batches, batch_size={batch_size}): "
        f"losses + decode (max_new_tokens={generation.max_new_tokens}, "
        f"beams={generation.num_beams})...",
        flush=True,
    )

    for batch_idx, batch in enumerate(val_loader):
        audio_paths, transcriptions = batch
        batch_texts = list(transcriptions)

        audio_tokens_list: list[torch.Tensor] = []
        success_paths: list[str] = []
        success_texts: list[str] = []
        total_windows = 0
        total_stability_loss = 0.0

        for audio_path, text in zip(audio_paths, batch_texts, strict=True):
            encoded = encode_utterance_fn(
                adapter=adapter,
                audio_extractor=audio_extractor,
                audio_path=audio_path,
                train_device=train_device,
                max_windows_per_utt=max_windows_per_utt,
            )
            if encoded is None:
                continue
            audio_tokens_list.append(encoded.tokens)
            success_paths.append(audio_path)
            success_texts.append(text)
            total_windows += encoded.num_windows
            total_stability_loss += encoded.stability_loss

        if not audio_tokens_list:
            continue

        gt_tokens = llm_tokenizer(
            success_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_text_tokens,
        ).to(train_device)
        gt_embeds = text_embedder(gt_tokens.input_ids.to(llm_device)).to(train_device)
        audio_tokens = _pad_audio_tokens(audio_tokens_list)

        aux_metrics = compute_aux_loss_metrics_fn(
            audio_tokens=audio_tokens,
            gt_embeds=gt_embeds,
            total_stability_loss=total_stability_loss,
            total_windows=total_windows,
        )
        m_align.update(aux_metrics["align"])
        m_stab.update(aux_metrics["stability"])

        inputs_embeds, labels, attention_mask = build_inputs_for_asr_fn(
            audio_tokens=audio_tokens,
            gt_ids=gt_tokens.input_ids,
            gt_attention_mask=gt_tokens.attention_mask,
            llm_tokenizer=llm_tokenizer,
            text_embedder=text_embedder,
            llm_device=llm_device,
        )
        asr_loss = asr_forward_loss_fn(
            inputs_embeds=inputs_embeds,
            labels=labels,
            attention_mask=attention_mask,
            micro_batch_size=asr_micro_batch_size,
            device=llm_device,
        )
        m_asr.update(asr_loss.item())

        if batch_idx % 20 == 0 and batch_idx > 0:
            print(f"    val batch {batch_idx}/{len(val_loader)}", flush=True)

        # Per-utterance NLL (diagnostics) still sequential; WER decode is batched.
        utt_nlls: list[float] = []
        for text, tokens in zip(success_texts, audio_tokens_list, strict=True):
            gt_single = llm_tokenizer(
                [text],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_text_tokens,
            ).to(train_device)
            utt_inputs, utt_labels, utt_mask = build_inputs_for_asr_fn(
                audio_tokens=tokens,
                gt_ids=gt_single.input_ids,
                gt_attention_mask=gt_single.attention_mask,
                llm_tokenizer=llm_tokenizer,
                text_embedder=text_embedder,
                llm_device=llm_device,
            )
            utt_nll = asr_forward_loss_fn(
                inputs_embeds=utt_inputs,
                labels=utt_labels,
                attention_mask=utt_mask,
                micro_batch_size=1,
                device=llm_device,
            )
            utt_nlls.append(float(utt_nll.item()))

        predictions = decode_asr_predictions_batch(
            llm_model=llm_model,
            llm_tokenizer=llm_tokenizer,
            audio_tokens_list=audio_tokens_list,
            llm_device=llm_device,
            torch_dtype=audio_tokens_list[0].dtype,
            generation=generation,
            append_im_end=True,
            maybe_autocast_fn=maybe_autocast_fn,
        )

        for audio_path, text, prediction, utt_nll in zip(
            success_paths, success_texts, predictions, utt_nlls, strict=True
        ):
            reference = normalize_asr_text(text)
            prediction = normalize_asr_text(prediction)
            wer = word_error_rate(reference, prediction)

            items.append(
                {
                    "utterance_id": _utterance_id(audio_path),
                    "audio_path": audio_path,
                    "reference": reference,
                    "prediction": prediction,
                    "wer": wer,
                    "nll": utt_nll,
                    "align": aux_metrics["align"],
                    "stability": aux_metrics["stability"],
                }
            )
            refs.append(reference)
            hyps.append(prediction)
            samples_done += 1

            if log_every > 0 and (
                samples_done == 1 or samples_done % log_every == 0 or samples_done == n_val
            ):
                print(
                    f"  [{samples_done}/{n_val}] WER={wer:.3f} NLL={utt_nll:.4f} "
                    f"align={aux_metrics['align']:.4f} stab={aux_metrics['stability']:.4f} "
                    f"ref={reference[:48]}{'...' if len(reference) > 48 else ''}",
                    flush=True,
                )

    if adapter_was_training:
        adapter.train()

    if not items:
        raise RuntimeError(f"No validation samples found under {val_root}")

    avg_wer = sum(item["wer"] for item in items) / len(items)
    bleu4 = corpus_bleu4(refs, hyps)
    metrics = {
        "val/asr": m_asr.mean,
        "val/align": m_align.mean,
        "val/stability": m_stab.mean,
        "val/wer": avg_wer,
        "val/bleu4": bleu4,
        "val/num_samples": float(len(items)),
    }

    print(
        f"  [val step {global_step}] "
        f"ASR={m_asr.mean:.4f} align={m_align.mean:.4f} stab={m_stab.mean:.4f} | "
        f"WER={avg_wer:.4f} BLEU-4={bleu4:.4f} ({len(items)} samples)",
        flush=True,
    )

    if predictions_path and items:
        payload = {
            "global_step": global_step,
            "num_samples": len(items),
            "val_asr": m_asr.mean,
            "val_align": m_align.mean,
            "val_stability": m_stab.mean,
            "avg_wer": avg_wer,
            "bleu4": bleu4,
            "items": items,
        }
        out_path = Path(predictions_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        print(f"  [val step {global_step}] predictions -> {out_path}", flush=True)

    return metrics, items
