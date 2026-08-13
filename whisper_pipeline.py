#!/usr/bin/env python3
"""
whisper_encoder_probe.py

Detailed layer-by-layer analysis of a HuggingFace `transformers` Whisper encoder.

Feeds one audio file through `openai/whisper-small` (or any whisper checkpoint)
and reports, in true execution order:

  - conv1 / conv2 frontend outputs (pre-GELU, since Whisper applies GELU
    functionally rather than as a module -- there's no hookable submodule for it)
  - inside every encoder block: self_attn_layer_norm -> q_proj/k_proj/v_proj/out_proj
    -> final_layer_norm -> fc1 -> activation_fn -> fc2
  - the final post-stack LayerNorm
  - coarse per-block hidden states (input/output boundary of each block)
  - per-block self-attention entropy (low entropy = peaky/focused attention,
    entropy near log(seq_len) = ~uniform attention, i.e. not really attending
    to specific frames)

Install (uv):
    uv add torch transformers librosa numpy

Usage:
    python whisper_encoder_probe.py --audio sample.wav
    python whisper_encoder_probe.py --audio sample.wav --model openai/whisper-small --save-dir ./activations
"""

import argparse
import json
import os
import sys
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from transformers import WhisperFeatureExtractor, WhisperModel

_pkg_root = os.path.dirname(os.path.abspath(__file__))
_src_root = os.path.join(_pkg_root, "src")
if _src_root not in sys.path:
    sys.path.insert(0, _src_root)


# --------------------------------------------------------------------------- #
# Hook manager
# --------------------------------------------------------------------------- #
class EncoderProbe:
    """Registers a forward hook on every LEAF submodule of a WhisperEncoder
    (i.e. modules with no children: Conv1d, Linear, LayerNorm, the GELU
    activation module) and records activation stats in true execution order.

    Container modules (the block itself, self_attn as a whole) are skipped
    since hooking them would double-count; you get their effective output
    via the last leaf inside them instead (e.g. out_proj == self_attn output,
    fc2 == MLP output before the residual add).
    """

    def __init__(self, encoder: torch.nn.Module):
        self.encoder = encoder
        self.records = []
        self._activations = OrderedDict()
        self._handles = []
        self._step = 0

    def _hook(self, name):
        def fn(module, inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(out):
                return
            with torch.no_grad():
                o = out.detach().float()
                self.records.append({
                    "step": self._step,
                    "name": name,
                    "type": module.__class__.__name__,
                    "shape": tuple(o.shape),
                    "mean": o.mean().item(),
                    "std": o.std().item(),
                    "min": o.min().item(),
                    "max": o.max().item(),
                    "l2_norm": o.norm().item(),
                    "pct_zero": (o == 0).float().mean().item() * 100,
                })
            self._activations[name] = out.detach().cpu()
            self._step += 1
        return fn

    def attach(self):
        for name, module in self.encoder.named_modules():
            if name == "":
                continue
            if len(list(module.children())) == 0:
                self._handles.append(module.register_forward_hook(self._hook(name)))
        # Also hook each block container directly (e.g. "layers.6"). This is the
        # only way to capture a block's RAW output -- for every block except the
        # last, that's identical to output_hidden_states[i+1], but for the LAST
        # block, output_hidden_states only ever gives you the value AFTER the
        # encoder's closing LayerNorm. This gives you the pre-norm version too.
        for i, block in enumerate(self.encoder.layers):
            self._handles.append(block.register_forward_hook(self._hook(f"layers.{i}")))

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def activations(self):
        return self._activations


# --------------------------------------------------------------------------- #
# Audio loading
# --------------------------------------------------------------------------- #
def load_audio(path, sr=16000):
    try:
        import librosa
        audio, _ = librosa.load(path, sr=sr, mono=True)
        return audio
    except ImportError:
        pass
    try:
        import soundfile as sf
        audio, file_sr = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if file_sr != sr:
            raise RuntimeError(
                f"File is {file_sr}Hz but librosa isn't installed to resample to {sr}Hz. "
                f"Run: uv add librosa"
            )
        return audio
    except ImportError:
        raise ImportError(
            "Need either librosa or soundfile to load audio. Run: uv add librosa soundfile"
        )


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _row(r, name_override=None):
    name = name_override if name_override is not None else r["name"]
    return (f"    {name:38} {r['type']:16} shape={str(r['shape']):20} "
            f"mean={r['mean']:>8.4f} std={r['std']:>8.4f} "
            f"norm={r['l2_norm']:>9.2f} zero%={r['pct_zero']:>5.1f}")


def group_records(records):
    frontend, layers, block_outputs, tail = [], {}, {}, []
    for r in records:
        name = r["name"]
        if name.startswith("layers."):
            parts = name.split(".")
            idx = int(parts[1])
            if len(parts) == 2:
                # block-level hook, e.g. "layers.6" -> the block's raw output
                block_outputs[idx] = r
            else:
                layers.setdefault(idx, []).append(r)
        elif name in ("conv1", "conv2"):
            frontend.append(r)
        else:
            tail.append(r)
    return frontend, layers, block_outputs, tail


def print_activation_report(probe: EncoderProbe):
    frontend, layers, block_outputs, tail = group_records(probe.records)

    print("\n" + "=" * 100)
    print("CONVOLUTIONAL FRONTEND  (outputs shown are PRE-GELU; GELU is applied functionally)")
    print("=" * 100)
    for r in frontend:
        print(_row(r))

    print("\n" + "=" * 100)
    print("TRANSFORMER ENCODER BLOCKS (pre-LN: norm -> attn -> +residual -> norm -> MLP -> +residual)")
    print("=" * 100)
    for idx in sorted(layers):
        print(f"\n  --- Block {idx} ---")
        for r in layers[idx]:
            short_name = r["name"].split(".", 2)[-1]  # strip "layers.N."
            print(_row(r, name_override=short_name))
        if idx in block_outputs:
            print(_row(block_outputs[idx], name_override="[BLOCK OUTPUT, raw, pre-final-norm]"))

    print("\n" + "=" * 100)
    print("FINAL NORMALIZATION")
    print("=" * 100)
    for r in tail:
        print(_row(r))


def print_block_boundaries(hidden_states):
    """hidden_states from output_hidden_states=True:
    [0]              = conv+pos output, input to block 0
    [i], 0<i<last    = output of block (i-1) == input to block i
    [-1]             = final output, AFTER the closing layer_norm
    """
    print("\n" + "=" * 100)
    print("PER-BLOCK HIDDEN STATE BOUNDARIES (coarse view)")
    print("=" * 100)
    n = len(hidden_states)
    for i, hs in enumerate(hidden_states):
        if i == 0:
            label = "conv+pos_embed output -> input to block 0"
        elif i == n - 1:
            label = f"output of block {i-1}, AFTER final layer_norm"
        else:
            label = f"output of block {i-1} -> input to block {i}"
        h = hs.float()
        print(f"  [{i:>2}] {label:48} shape={tuple(h.shape)} "
              f"mean={h.mean().item(): .4f} std={h.std().item(): .4f}")


def print_attention_entropy(attentions):
    print("\n" + "=" * 100)
    print("PER-BLOCK SELF-ATTENTION ENTROPY  (shape: batch, heads, tgt_frames, src_frames)")
    print("=" * 100)
    for i, attn in enumerate(attentions):
        p = attn.float().clamp_min(1e-9)
        entropy = -(p * p.log()).sum(-1).mean().item()
        max_entropy = float(np.log(attn.shape[-1]))
        print(f"  block {i:>2}: shape={tuple(attn.shape)}  "
              f"mean_entropy={entropy:.3f}  (uniform/max={max_entropy:.3f})  "
              f"max_weight={attn.max().item():.4f}")


# --------------------------------------------------------------------------- #
# Single-block embedding extraction (importable, for notebooks/other scripts)
# --------------------------------------------------------------------------- #
def load_model(model_name="openai/whisper-small", device=None):
    """Load the feature extractor + encoder once, so you can call
    get_block_embedding() repeatedly (e.g. over many audio files) without
    reloading weights every time."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    encoder = WhisperModel.from_pretrained(model_name, attn_implementation="eager").encoder
    encoder = encoder.to(device).eval()
    return feature_extractor, encoder


def get_block_embedding(audio_path, block, feature_extractor=None, encoder=None,
                         model="openai/whisper-small", device=None, apply_final_norm=False):
    """
    Run the Whisper encoder on one audio file and return ONE block's embedding
    as a single tensor.

        x = get_block_embedding("audio.wav", block=6)

    Args:
        audio_path: path to a wav/mp3/flac file.
        block: which block's output to return.
            "pre" or -1        -> conv+pos_embed output (input to block 0,
                                   before any transformer block runs at all)
            0 .. num_layers-1  -> the RAW output of that block (the exact
                                   residual-stream tensor fed into the next
                                   block -- this is what you want if you're
                                   trying to bypass the late-layer scale blow-up)
        feature_extractor, encoder: pass these in (from load_model()) to reuse
            a warm model across many calls. If omitted, loads fresh each call.
        model: HF hub id, only used if encoder/feature_extractor aren't passed in.
        apply_final_norm: only meaningful when block == num_layers - 1 (the
            last block). If True, returns the value AFTER the encoder's
            closing LayerNorm -- i.e. exactly what `last_hidden_state` /
            `encoder(...).last_hidden_state` normally gives you. Default False
            returns the raw pre-norm value instead.

    Returns:
        torch.Tensor of shape (1, num_frames, d_model), on the encoder's device.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if feature_extractor is None or encoder is None:
        feature_extractor, encoder = load_model(model, device)

    num_layers = encoder.config.encoder_layers
    valid = block in ("pre", -1) or (isinstance(block, int) and 0 <= block < num_layers)
    if not valid:
        raise ValueError(f"block must be 'pre'/-1 or an int in 0..{num_layers - 1}, got {block!r}")

    audio = load_audio(audio_path, sr=feature_extractor.sampling_rate)
    inputs = feature_extractor(audio, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    raw_block_outputs = {}

    def make_hook(idx):
        def hook(module, hook_inputs, output):
            raw_block_outputs[idx] = output[0] if isinstance(output, tuple) else output
        return hook

    handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(encoder.layers)]
    with torch.no_grad():
        outputs = encoder(input_features, output_hidden_states=True)
    for h in handles:
        h.remove()

    if block in ("pre", -1):
        return outputs.hidden_states[0]
    if block == num_layers - 1 and apply_final_norm:
        return outputs.last_hidden_state
    return raw_block_outputs[block]


def _parse_block_arg(s):
    if s.lower() == "pre":
        return "pre"
    return int(s)


# --------------------------------------------------------------------------- #
# Layer-selectable embeddings (training / analysis)
# --------------------------------------------------------------------------- #
EMBEDDING_MODES = ("single", "concat", "mean", "weighted_sum")


@dataclass
class LayerEmbedConfig:
    """Which Whisper encoder layers to feed the adapter, and how to combine them.

    Layer index ``i`` maps to HF ``hidden_states[i + 1]`` (output after block ``i``).
    When the last encoder block is selected and ``apply_final_norm`` is True, use
    ``last_hidden_state`` (matches stock stage2 / production encode).
    """

    layers: list[int] = field(default_factory=lambda: [11])
    mode: str = "single"  # single | concat | mean | weighted_sum
    apply_final_norm: bool = True

    def __post_init__(self):
        if isinstance(self.layers, tuple):
            self.layers = list(self.layers)
        if not self.layers:
            raise ValueError("LayerEmbedConfig.layers must be non-empty")
        mode = self.mode.strip().lower()
        if mode not in EMBEDDING_MODES:
            raise ValueError(
                f"mode must be one of {EMBEDDING_MODES}, got {self.mode!r}"
            )
        self.mode = mode
        if self.mode == "single" and len(self.layers) != 1:
            raise ValueError(
                f"mode='single' requires exactly one layer, got {self.layers}"
            )

    def validate_against_encoder(self, num_layers: int) -> None:
        for i in self.layers:
            if not isinstance(i, int) or i < 0 or i >= num_layers:
                raise ValueError(
                    f"layer index {i} out of range for encoder with "
                    f"{num_layers} layers (valid: 0..{num_layers - 1})"
                )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "LayerEmbedConfig":
        return cls(
            layers=list(d["layers"]),
            mode=d.get("mode", "single"),
            apply_final_norm=bool(d.get("apply_final_norm", True)),
        )


def embedding_dim(d_model: int, cfg: LayerEmbedConfig) -> int:
    """Effective adapter ``d_encoder`` for this layer config."""
    if cfg.mode == "concat":
        return int(d_model) * len(cfg.layers)
    return int(d_model)


class LearnableLayerAggregator(nn.Module):
    """Softmax-weighted sum over selected layer tensors (same ``D`` each)."""

    def __init__(self, num_layers: int):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.num_layers = int(num_layers)
        self.logits = nn.Parameter(torch.zeros(self.num_layers))

    def forward(self, tensors: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(tensors) != self.num_layers:
            raise ValueError(
                f"expected {self.num_layers} tensors, got {len(tensors)}"
            )
        weights = torch.softmax(self.logits, dim=0)
        stacked = torch.stack(list(tensors), dim=0)  # (N, B, T, D)
        w = weights.view(-1, *([1] * (stacked.ndim - 1)))
        return (stacked * w).sum(dim=0)

    def weight_dict(self) -> dict[str, float]:
        with torch.no_grad():
            w = torch.softmax(self.logits, dim=0).detach().cpu().tolist()
        return {f"layer_weight_{i}": float(v) for i, v in enumerate(w)}


def select_layer_tensors(
    hidden_states: Sequence[torch.Tensor],
    last_hidden_state: torch.Tensor,
    cfg: LayerEmbedConfig,
    *,
    num_layers: int,
) -> list[torch.Tensor]:
    """Pick tensors for ``cfg.layers`` from an encoder forward with hidden states."""
    cfg.validate_against_encoder(num_layers)
    out: list[torch.Tensor] = []
    for i in cfg.layers:
        if i == num_layers - 1 and cfg.apply_final_norm:
            out.append(last_hidden_state)
        else:
            # hidden_states[0] = conv+pos; hidden_states[i+1] = after block i
            out.append(hidden_states[i + 1])
    return out


def combine_layer_embeddings(
    tensors: Sequence[torch.Tensor],
    mode: str,
    aggregator: LearnableLayerAggregator | None = None,
) -> torch.Tensor:
    """Combine selected layer tensors into one ``(B, T, D_eff)`` embedding."""
    mode = mode.strip().lower()
    if not tensors:
        raise ValueError("tensors must be non-empty")
    if mode == "single":
        if len(tensors) != 1:
            raise ValueError(f"mode='single' needs 1 tensor, got {len(tensors)}")
        return tensors[0]
    if mode == "concat":
        return torch.cat(list(tensors), dim=-1)
    if mode == "mean":
        return torch.stack(list(tensors), dim=0).mean(dim=0)
    if mode == "weighted_sum":
        if aggregator is None:
            raise ValueError("mode='weighted_sum' requires a LearnableLayerAggregator")
        return aggregator(tensors)
    raise ValueError(f"unknown mode {mode!r}; expected one of {EMBEDDING_MODES}")


def encode_waveform_layers(
    waveform,
    *,
    feature_extractor,
    encoder: nn.Module,
    cfg: LayerEmbedConfig,
    aggregator: LearnableLayerAggregator | None = None,
    device: str | torch.device | None = None,
    torch_dtype: torch.dtype | None = None,
    sample_rate: int | None = None,
) -> torch.Tensor:
    """
    Mel-extract + Whisper encoder → select/combine layers → ``(1, T, D_eff)``.

    Encoder forward is under ``no_grad``; ``weighted_sum`` aggregation still
    receives gradients through ``aggregator``.
    """
    num_layers = int(encoder.config.encoder_layers)
    cfg.validate_against_encoder(num_layers)
    if cfg.mode == "weighted_sum" and aggregator is None:
        raise ValueError("aggregator is required when mode='weighted_sum'")

    if device is None:
        device = next(encoder.parameters()).device
    else:
        device = torch.device(device)
    if torch_dtype is None:
        torch_dtype = next(encoder.parameters()).dtype
    sr = int(sample_rate or getattr(feature_extractor, "sampling_rate", 16000))

    if isinstance(waveform, torch.Tensor):
        wave_np = waveform.detach().float().cpu().reshape(-1).numpy()
    else:
        wave_np = waveform

    mel = feature_extractor(wave_np, sampling_rate=sr, return_tensors="pt")
    input_features = mel.input_features.to(device=device, dtype=torch_dtype)

    with torch.no_grad():
        outputs = encoder(input_features, output_hidden_states=True)
        selected = select_layer_tensors(
            outputs.hidden_states,
            outputs.last_hidden_state,
            cfg,
            num_layers=num_layers,
        )
        # Detach encoder tensors so only aggregator (if any) is trainable.
        selected = [t.detach() for t in selected]

    combined = combine_layer_embeddings(selected, cfg.mode, aggregator=aggregator)
    if combined.dtype != torch_dtype or combined.device != device:
        combined = combined.to(device=device, dtype=torch_dtype)
    return combined


class LayerAwareWindowEncoder:
    """
    Overlapping raw-audio windows → per-chunk layer-selectable Whisper embedding.

    Drop-in for stage2's ``WhisperWindowFeatureExtractor.waveform_to_windows``
    contract: returns a list of ``(1, 1500, D_eff)`` tensors.
    """

    def __init__(
        self,
        *,
        model_id: str,
        device: str,
        torch_dtype: torch.dtype,
        layer_cfg: LayerEmbedConfig,
        aggregator: LearnableLayerAggregator | None = None,
        window_seconds: float = 0.8,
        stride_seconds: float = 0.4,
        sample_rate: int = 16000,
    ) -> None:
        from adapter.windowing import AudioWaveformWindowizer

        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        self.layer_cfg = layer_cfg
        self.aggregator = aggregator
        self.window_seconds = float(window_seconds)
        self.stride_seconds = float(stride_seconds)
        self.sample_rate = int(sample_rate)

        self.feature_extractor, self.encoder = load_model(model_id, device)
        self.encoder = self.encoder.to(device=device, dtype=torch_dtype).eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        num_layers = int(self.encoder.config.encoder_layers)
        self.layer_cfg.validate_against_encoder(num_layers)
        self.d_model = int(self.encoder.config.d_model)
        self.d_encoder = embedding_dim(self.d_model, self.layer_cfg)

        if self.layer_cfg.mode == "weighted_sum":
            if self.aggregator is None:
                self.aggregator = LearnableLayerAggregator(len(self.layer_cfg.layers))
            self.aggregator = self.aggregator.to(device=device, dtype=torch.float32)
        elif self.aggregator is not None:
            raise ValueError(
                f"aggregator is only used with mode='weighted_sum', got mode={self.layer_cfg.mode!r}"
            )

        self._audio_windowizer = AudioWaveformWindowizer(
            sample_rate=self.sample_rate,
            window_seconds=self.window_seconds,
            stride_seconds=self.stride_seconds,
        )

    def waveform_to_windows(self, waveform_16k_mono: torch.Tensor) -> list[torch.Tensor]:
        if isinstance(waveform_16k_mono, torch.Tensor):
            wave = waveform_16k_mono.detach().float().cpu()
        else:
            wave = torch.tensor(waveform_16k_mono, dtype=torch.float32)

        if wave.numel() == 0:
            return []

        dev = torch.device(self.device)
        audio_chunks = self._audio_windowizer(wave)
        enc_windows: list[torch.Tensor] = []
        for chunk in audio_chunks:
            enc = encode_waveform_layers(
                chunk,
                feature_extractor=self.feature_extractor,
                encoder=self.encoder,
                cfg=self.layer_cfg,
                aggregator=self.aggregator,
                device=self.device,
                torch_dtype=self.torch_dtype,
                sample_rate=self.sample_rate,
            ).to(device=dev, dtype=self.torch_dtype)
            enc_windows.append(enc.contiguous())
        return enc_windows


def parse_layers_arg(s: str) -> list[int]:
    """Parse ``'11'`` or ``'4,6,8,11'`` into a list of ints."""
    parts = [p.strip() for p in str(s).replace(" ", "").split(",") if p.strip()]
    if not parts:
        raise ValueError(f"empty layers string: {s!r}")
    return [int(p) for p in parts]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", required=True, help="Path to a wav/mp3/flac file")
    ap.add_argument("--model", default="openai/whisper-small", help="HF hub id or local path")
    ap.add_argument("--save-dir", default=None, help="If set, dump every activation as .npy + a stats.json here")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--block", type=_parse_block_arg, default=None,
                     help="Extract just one block's embedding into variable `x`. "
                          "Use an int 0..num_layers-1, or 'pre' for the conv+pos "
                          "output (input to block 0). Runs alongside the full report.")
    ap.add_argument("--apply-final-norm", action="store_true",
                     help="Only affects the last block: return the value AFTER the "
                          "encoder's closing LayerNorm instead of the raw pre-norm output.")
    args = ap.parse_args()

    print(f"Loading {args.model} on {args.device} ...")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model)
    full_model = WhisperModel.from_pretrained(args.model, attn_implementation="eager")
    encoder = full_model.encoder.to(args.device).eval()
    cfg = encoder.config

    print(f"  d_model={cfg.d_model}  encoder_layers={cfg.encoder_layers}  "
          f"heads={cfg.encoder_attention_heads}  ffn_dim={cfg.encoder_ffn_dim}  "
          f"n_mels={cfg.num_mel_bins}  max_source_positions={cfg.max_source_positions}")

    print(f"\nLoading audio from {args.audio} ...")
    audio = load_audio(args.audio, sr=feature_extractor.sampling_rate)
    print(f"  duration: {len(audio) / feature_extractor.sampling_rate:.2f}s "
          f"({len(audio)} samples @ {feature_extractor.sampling_rate}Hz)")
    print("  note: WhisperFeatureExtractor pads/trims every clip to 30s (3000 mel frames) by default")

    inputs = feature_extractor(audio, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt")
    input_features = inputs.input_features.to(args.device)
    print(f"  log-mel input_features shape: {tuple(input_features.shape)}  (batch, n_mels, frames)")

    probe = EncoderProbe(encoder)
    probe.attach()
    with torch.no_grad():
        outputs = encoder(input_features, output_attentions=True, output_hidden_states=True)
    probe.detach()

    print_activation_report(probe)
    print_block_boundaries(outputs.hidden_states)
    print_attention_entropy(outputs.attentions)

    print("\n" + "=" * 100)
    print(f"FINAL last_hidden_state: shape={tuple(outputs.last_hidden_state.shape)}")
    print("=" * 100)

    x = None
    if args.block is not None:
        if args.block in ("pre", -1):
            x = outputs.hidden_states[0]
            block_label = "pre (conv+pos_embed output, input to block 0)"
        else:
            if not (0 <= args.block < cfg.encoder_layers):
                raise ValueError(f"--block must be 'pre' or in 0..{cfg.encoder_layers - 1}, got {args.block}")
            if args.block == cfg.encoder_layers - 1 and args.apply_final_norm:
                x = outputs.last_hidden_state
                block_label = f"block {args.block} (AFTER final layer_norm)"
            else:
                x = probe.activations()[f"layers.{args.block}"]
                block_label = f"block {args.block} (raw, pre-final-norm)"

        print("\n" + "=" * 100)
        print(f"REQUESTED BLOCK EMBEDDING -> x = {block_label}")
        print("=" * 100)
        xf = x.float()
        print(f"  shape={tuple(x.shape)}  mean={xf.mean().item(): .4f}  std={xf.std().item(): .4f}  "
              f"min={xf.min().item(): .4f}  max={xf.max().item(): .4f}  norm={xf.norm().item():.2f}")

        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            fname = f"x_block_{args.block}{'_final_norm' if args.apply_final_norm else ''}.npy"
            np.save(os.path.join(args.save_dir, fname), x.detach().cpu().numpy())
            print(f"  saved to {os.path.join(args.save_dir, fname)}")

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        for name, tensor in probe.activations().items():
            np.save(os.path.join(args.save_dir, name.replace(".", "_") + ".npy"), tensor.numpy())
        np.save(os.path.join(args.save_dir, "final_last_hidden_state.npy"),
                outputs.last_hidden_state.detach().cpu().numpy())
        for i, hs in enumerate(outputs.hidden_states):
            np.save(os.path.join(args.save_dir, f"hidden_states_{i:02d}.npy"), hs.detach().cpu().numpy())
        for i, attn in enumerate(outputs.attentions):
            np.save(os.path.join(args.save_dir, f"attention_{i:02d}.npy"), attn.detach().cpu().numpy())
        with open(os.path.join(args.save_dir, "activation_stats.json"), "w") as f:
            json.dump(probe.records, f, indent=2)
        print(f"\nSaved {len(probe.activations())} activation tensors + hidden_states + attentions "
              f"+ activation_stats.json to {args.save_dir}/")

    print("\nDone.")


if __name__ == "__main__":
    main()