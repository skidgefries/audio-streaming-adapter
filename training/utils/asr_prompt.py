"""Shared ASR instruction text for eval and (Stage 3+) prompt-conditioned training."""

DEFAULT_ASR_PROMPT = (
    "Transcribe the speech verbatim. Output only the transcript, "
    "with no extra commentary. /no_think"
)

# Stage 2 trainer layout (teacher forcing): [audio_tokens | im_end/BOS | transcript_embeds]
# Eval (append_im_end=True): [audio_tokens | im_end/BOS] → generate
#   Qwen3 also appends the empty thinking block when those token ids are in-vocab.
#   Vicuna/Llama skip that suffix (ids 151667+ are out of range).
# Eval (append_im_end=False): [audio_tokens] → generate
TRAIN_STYLE_CONDITIONING = "audio_im_end_generate"
TRAIN_STYLE_NO_IM_END_CONDITIONING = "audio_generate"

# Target layout after Stage 3 (prompt + audio): [--asr-prompt | audio_tokens] → generate
PROMPT_CONDITIONING = "prompt_audio"
