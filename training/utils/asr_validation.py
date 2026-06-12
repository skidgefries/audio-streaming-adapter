"""ASR validation: teacher-forced NLL, greedy/beam decode, WER, BLEU-4."""

from __future__ import annotations

import json
from contextlib import AbstractContextManager, nullcontext
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


def _utterance_id(audio_path: str) -> str:
    return Path(audio_path).stem


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
    maybe_autocast_fn: Callable[[torch.device], AbstractContextManager] | None = None,
) -> str:
    """Greedy/beam decode from compressed audio tokens (train-style prefix)."""
    llm_dev = llm_device
    audio = audio_tokens.to(device=llm_dev, dtype=torch_dtype)
    if append_im_end:
        sep_id = WhisperAdapterLLMPipeline.train_style_separator_token_id(llm_tokenizer)
        sep_ids = torch.tensor([[sep_id]], device=llm_dev, dtype=torch.long)
        sep_embed = llm_model.get_input_embeddings()(sep_ids)
        no_think_embed = WhisperAdapterLLMPipeline.qwen_no_think_suffix_embeds(
            llm_model, llm_dev, torch_dtype
        )
        input_embeds = torch.cat([audio, sep_embed, no_think_embed], dim=1)
    else:
        input_embeds = audio

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
    return llm_tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()


@torch.no_grad()
def validate_asr_only(
    *,
    adapter: StreamingAdapter,
    audio_extractor: WhisperWindowFeatureExtractor,
    llm_model: torch.nn.Module,
    llm_tokenizer,
    text_embedder: torch.nn.Module,
    encode_utterance_tokens_fn: Callable[..., torch.Tensor | None],
    build_inputs_for_asr_fn: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    asr_forward_loss_fn: Callable[..., torch.Tensor],
    val_root: str,
    train_device: torch.device,
    llm_device: torch.device,
    max_utterances: int | None,
    max_windows_per_utt: int | None,
    max_text_tokens: int,
    asr_micro_batch_size: int,
    generation: LlmGenerationParams,
    global_step: int,
    predictions_path: str | None = None,
    log_every: int = 50,
    maybe_autocast_fn: Callable[[torch.device], AbstractContextManager] | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """
    Run dev-clean validation: teacher-forced NLL, decode predictions, WER, BLEU-4.

    Processes one utterance at a time so references align with successful encodes.
    """
    adapter_was_training = adapter.training
    adapter.eval()

    val_full = LibriSpeechPairs(val_root)
    n_val = len(val_full) if max_utterances is None else min(max_utterances, len(val_full))
    val_dataset: Dataset = val_full if n_val == len(val_full) else Subset(val_full, range(n_val))
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    m_asr = RunningMean()
    refs: list[str] = []
    hyps: list[str] = []
    items: list[dict[str, Any]] = []

    print(
        f"\n  [val step {global_step}] dev-clean ({n_val} utterances): "
        f"NLL + decode (max_new_tokens={generation.max_new_tokens}, "
        f"beams={generation.num_beams})...",
        flush=True,
    )

    for idx, (audio_paths, transcriptions) in enumerate(val_loader):
        audio_path = audio_paths[0]
        reference = normalize_asr_text(transcriptions[0])

        tokens = encode_utterance_tokens_fn(
            adapter=adapter,
            audio_extractor=audio_extractor,
            audio_path=audio_path,
            train_device=train_device,
            max_windows_per_utt=max_windows_per_utt,
        )
        if tokens is None:
            continue

        gt_tokens = llm_tokenizer(
            [transcriptions[0]],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_text_tokens,
        ).to(train_device)
        inputs_embeds, labels, attention_mask = build_inputs_for_asr_fn(
            audio_tokens=tokens,
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

        prediction = decode_asr_prediction(
            llm_model=llm_model,
            llm_tokenizer=llm_tokenizer,
            audio_tokens=tokens,
            llm_device=llm_device,
            torch_dtype=tokens.dtype,
            generation=generation,
            append_im_end=True,
            maybe_autocast_fn=maybe_autocast_fn,
        )
        prediction = normalize_asr_text(prediction)
        wer = word_error_rate(reference, prediction)

        items.append(
            {
                "utterance_id": _utterance_id(audio_path),
                "audio_path": audio_path,
                "reference": reference,
                "prediction": prediction,
                "wer": wer,
                "nll": asr_loss.item(),
            }
        )
        refs.append(reference)
        hyps.append(prediction)

        if log_every > 0 and (
            idx == 0 or (idx + 1) % log_every == 0 or idx + 1 == n_val
        ):
            print(
                f"  [{len(items)}/{n_val}] WER={wer:.3f} NLL={asr_loss.item():.4f} "
                f"ref={reference[:48]}{'...' if len(reference) > 48 else ''}",
                flush=True,
            )

    if adapter_was_training:
        adapter.train()

    avg_wer = sum(item["wer"] for item in items) / max(len(items), 1)
    bleu4 = corpus_bleu4(refs, hyps)
    metrics = {
        "val/asr": m_asr.mean,
        "val/wer": avg_wer,
        "val/bleu4": bleu4,
        "val/num_samples": float(len(items)),
    }

    print(
        f"  [val step {global_step}] "
        f"ASR={m_asr.mean:.4f} WER={avg_wer:.4f} BLEU-4={bleu4:.4f} "
        f"({len(items)} samples)",
        flush=True,
    )

    if predictions_path and items:
        payload = {
            "global_step": global_step,
            "num_samples": len(items),
            "val_asr": m_asr.mean,
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
