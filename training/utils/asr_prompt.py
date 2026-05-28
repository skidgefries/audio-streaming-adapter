"""Shared ASR instruction text for eval and (Stage 3+) prompt-conditioned training."""

DEFAULT_ASR_PROMPT = (
    "Transcribe the speech verbatim. Output only the transcript, "
    "with no extra commentary."
)

# Stage 2 trainer layout (teacher forcing): [audio_tokens | BOS | transcript_embeds]
# Eval: generate from audio embeddings only (see adapter_llm_pipeline train_style_asr).
TRAIN_STYLE_CONDITIONING = "audio_generate"

# Target layout after Stage 3 (prompt + audio): [--asr-prompt | audio_tokens] → generate
PROMPT_CONDITIONING = "prompt_audio"
