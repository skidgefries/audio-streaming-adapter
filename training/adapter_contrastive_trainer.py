"""
Stage 1 Training: Audio-Text Alignment (Contrastive Learning)

Trains the streaming adapter using contrastive learning to align audio tokens
with text embeddings from the frozen LLM.

Loss: L = L_align + λ_stability · L_stability
"""

import os
import sys
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src_root = os.path.join(_pkg_root, "src")
sys.path.insert(0, _src_root)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, _pkg_root)

from src.adapter.streaming_adapter import StreamingAdapter
from src.dataset import LibriSpeechConfig, load_mono_waveform_16k, LibriSpeechPairsCustom, LibriSpeechPairs
from src.encoder import WhisperWindowFeatureExtractor
from training.utils.checkpointing import TrainingCheckpoint, save_checkpoint
from training.utils.config import CheckpointConfig, DataConfig, OptimConfig, Stage1Config, WandbConfig
from training.utils.logging import WandbLogger
from training.utils.losses import contrastive_infonce_loss
from training.utils.metrics import RunningMean
from training.utils.loaders import default_device_and_dtype, load_frozen_qwen_embeddings

# Configuration
DEVICE, TORCH_DTYPE = default_device_and_dtype()

# Model dimensions
WHISPER_DIM = 768   # whisper-small
LLM_DIM = 4096      # Qwen embedding dimension
WHISPER_MODEL = "openai/whisper-small"
LLM_MODEL_ID = "Qwen/Qwen3-8B"

# Training hyperparameters
STAGE = Stage1Config()
OPT = OptimConfig(lr=1e-4, weight_decay=0.01, grad_clip_norm=0.5, warmup_steps=1000)
# OPT = OptimConfig(lr=1e-3, weight_decay=0.01, grad_clip_norm=1.0, warmup_steps=0)
DATA = DataConfig(
    dataset_root=LibriSpeechConfig.default_train_clean_100_from_training_dir(os.path.dirname(__file__)).root,
    batch_size=16,
    num_workers=2,
    max_windows_per_utt=None,
)
CKPT = CheckpointConfig(dir="checkpoints", save_every_epochs=1)
WANDB = WandbConfig(enabled=True, project="audio-streaming-adapter", run_name="s1-centred")

SAVE_PATH = os.path.join(CKPT.dir, "adapter_stage1.pt")



# same dim1
def _pad_tokens(utterances: list[torch.Tensor]) -> torch.Tensor:
    max_len = max(t.shape[1] for t in utterances)
    padded = []
    for t in utterances:
        if t.shape[1] < max_len:
            pad = torch.zeros(1, max_len - t.shape[1], t.shape[2], device=t.device, dtype=t.dtype)
            t = torch.cat([t, pad], dim=1)
        padded.append(t)
    return torch.cat(padded, dim=0)


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    torch.cuda.empty_cache()
    print(f"Using device: {DEVICE}")
    audio = WhisperWindowFeatureExtractor(model_id=WHISPER_MODEL, device=DEVICE, torch_dtype=TORCH_DTYPE)

    qwen_models = load_frozen_qwen_embeddings(model_id=LLM_MODEL_ID, device=DEVICE, torch_dtype=TORCH_DTYPE, device_map="auto")
    llm_tokenizer = qwen_models.tokenizer
    text_embedder = qwen_models.embedder

    # Initialize streaming adapter (trainable)
    adapter = StreamingAdapter(
        d_encoder=WHISPER_DIM,
        d_llm=LLM_DIM,
        num_queries=4,
        num_layers=2,
        num_heads=4,
        d_ffn=2048,
        dropout=0.1,
        ema_alpha=0.8,
        learnable_ema=False,
        use_rate_controller=False,  # Fixed 4 tokens per window for stage 1
    ).to(DEVICE, dtype= torch.bfloat16)
    adapter.train()

    print(f"StreamingAdapter initialized:")
    print(f"  Encoder dim: {WHISPER_DIM}")
    print(f"  LLM dim: {LLM_DIM}")
    print(f"  Max tokens/window: 4\n")
    
    # Dataset and dataloader
    dataset = LibriSpeechPairs(DATA.dataset_root)
    dataloader = DataLoader(
        dataset, 
        batch_size=DATA.batch_size, 
        shuffle=True, 
        num_workers=DATA.num_workers
    )
 
#  check if the loss is working with a subset of the dataset   
    # dataset = LibriSpeechPairsCustom(
    #     dataset_root=DATA.dataset_root,
    #     file_ids=[
    #         "374-180299-0001",   #IN THE COURSE OF THE DAY I RECEIVED THIS note
    #         "374-180299-0002",   #BE AT PRUDENCE'S TO NIGHT AT EIGHT
    #         "7800-283478-0020",  #AS THE FOUR CHUMS WENT AWAY JERRY CHUCKLED
    #         "7800-283492-0012",  #OH NO IT ISN'T SO BAD AS THAT HE WAS ASSURED
    #         "7800-283493-0038",  #WELL SO LONG BOYS AND WE ALL WISH YOU SUCCESS
    #         "3240-131232-0001",  #BUT MAY AFTER A WHILE BE SHAKEN DOWN BY STORMS
    #         "1088-134315-0023",  #CLOSED THE DOOR CAREFULLY AND RETURNED TO THE HOUSE
    #         "1088-134315-0055",  #IN THAT CASE WAS A NEW STEEL KEY
    #     ]
    # )
    # dataloader = DataLoader(dataset, batch_size=DATA.batch_size, shuffle=False, num_workers=DATA.num_workers)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        adapter.parameters(), 
        lr=OPT.lr, 
        weight_decay=OPT.weight_decay
    )
    
    # scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=OPT.warmup_steps)
    
    total_steps = STAGE.epochs * len(dataloader)

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=OPT.warmup_steps,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps - OPT.warmup_steps,
        eta_min=1e-6,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[OPT.warmup_steps],
    )
    
    # Note: Losses computed in float32 for numerical stability, no mixed precision scaling needed

    # Create checkpoint directory
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    logger = WandbLogger(
        enabled=WANDB.enabled,
        project=WANDB.project,
        entity=WANDB.entity,
        run_name=WANDB.run_name,
        config={
            "stage": 1,
            "data": DATA.__dict__,
            "optim": OPT.__dict__,
            "stage_cfg": STAGE.__dict__,
        },
    )

    # ── Resume from checkpoint if available ───────────────────────────────────────
    start_epoch = 0
    if os.path.exists(SAVE_PATH):
            print(f"Resuming from checkpoint: {SAVE_PATH}")
            ckpt = torch.load(SAVE_PATH, map_location=DEVICE)
            adapter.load_state_dict(ckpt["adapter_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"]
            global_step = ckpt["global_step"]
            
            # Fast-forward scheduler past warmup
            for _ in range(global_step):
                scheduler.step()
                
            # Force correct LR
            for param_group in optimizer.param_groups:
                param_group['lr'] = OPT.lr
            # scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            
            print(f"  Resumed at epoch {start_epoch}, global_step {global_step}\n")
    else:
            global_step = 0
        
    
    print(f"Starting Stage 1 training: Audio-Text Alignment")
    print(f"  Epochs: {STAGE.epochs}")
    print(f"  Batch size: {DATA.batch_size}")
    print(f"  Learning rate: {OPT.lr}")
    print(f"  λ_stability: {STAGE.lambda_stability}")
    print(f"  Temperature: {STAGE.temperature}\n")

    for epoch in range(start_epoch, STAGE.epochs):
        m_total = RunningMean()
        m_align = RunningMean()
        m_stab = RunningMean()

        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{STAGE.epochs}")
        print(f"{'='*60}\n")

        for step, batch in enumerate(dataloader):
            audio_paths, transcriptions = batch
            batch_texts = list(transcriptions)

            # Tokenize ground truth transcriptions
            text_tokens = llm_tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(DEVICE)

            # Get text embeddings from frozen LLM (in float32 for stability)
            with torch.no_grad():
                label_ids = text_tokens.input_ids
                label_embeds = text_embedder(label_ids).float()  # Convert embeddings to float32

            # Check for NaN in embeddings
            if torch.isnan(label_embeds).any():
                print(f"[DEBUG] label_embeds contains NaN at step {step}!")
                print(f"  shape: {label_embeds.shape}")
                print(f"  min: {label_embeds.min()}, max: {label_embeds.max()}")
                raise ValueError("label_embeds contains NaN")
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                utterances = []
                stab_sum = torch.zeros((), device=DEVICE, dtype=torch.float32)
                for p in audio_paths:
                    wave = load_mono_waveform_16k(p)
                    windows = audio.waveform_to_windows(wave)
                    if DATA.max_windows_per_utt is not None:
                        windows = windows[: DATA.max_windows_per_utt]
                    adapter.reset_streaming_state()
                    chunks = []
                    for w in windows:
                        out = adapter.forward_window(w)
                        chunks.append(out["tokens"])
                        stab_sum = stab_sum + out["stability_loss"].float()
                    utterances.append(torch.cat(chunks, dim=1))

                audio_tokens = _pad_tokens(utterances)  # (B, T, D)

            # Check for NaN in audio_tokens
                if torch.isnan(audio_tokens).any():
                    print(f"[DEBUG] audio_tokens contains NaN at step {step}!")
                    print(f"  shape: {audio_tokens.shape}")
                    print(f"  min: {audio_tokens.min()}, max: {audio_tokens.max()}")
                    raise ValueError("audio_tokens contains NaN")

            # Compute losses
            # Contrastive loss (already internally converts to float32)
                align_loss, diag = contrastive_infonce_loss(
                    audio_tokens=audio_tokens,
                    text_embeddings=label_embeds,
                    temperature=STAGE.temperature,
                    return_diagnostics = True
                )
                stability_loss = stab_sum / float(len(audio_paths))

            if torch.isnan(stability_loss):
                print(f"[DEBUG] stability_loss is NaN at step {step}!")
                print(f"  stab_sum: {stab_sum}")
                print(f"  batch size: {len(audio_paths)}")
                raise ValueError("stability_loss contains NaN")

            # Total loss
            total_loss = align_loss + STAGE.lambda_stability * stability_loss

            if torch.isnan(total_loss):
                print(f"[DEBUG] total_loss is NaN at step {step}!")
                print(f"  align_loss: {align_loss}")
                print(f"  stability_loss: {stability_loss}")
                print(f"  lambda_stability: {STAGE.lambda_stability}")
                raise ValueError("total_loss contains NaN")

            optimizer.zero_grad()
            total_loss.backward()

            has_nan_grad = any(
                torch.isnan(param.grad).any()
                for param in adapter.parameters()
                if param.grad is not None
            )
            if has_nan_grad:
                print(f"[DEBUG] Gradients contain NaN at step {step}!")
                for name, param in adapter.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        print(f"  {name}: grad contains NaN")
                raise ValueError("Gradients contain NaN")

            for param in adapter.parameters():
                if param.grad is not None:
                    param.grad = torch.where(
                        torch.isnan(param.grad) | torch.isinf(param.grad),
                        torch.zeros_like(param.grad),
                        param.grad,
                    )

            torch.nn.utils.clip_grad_norm_(adapter.parameters(), OPT.grad_clip_norm)
            
            
            optimizer.step()   # ← actually applies the gradients
            scheduler.step()   # ← advances the LR warmup scheduler

            # Check gradient norm   
            total_norm = 0.0
            for param in adapter.parameters():
                if param.grad is not None:
                    total_norm += param.grad.data.norm(2).item() ** 2
            total_norm = total_norm**0.5
            if total_norm > 5.0:
                print(f"[WARN] Step {step}: Large gradient norm {total_norm:.2f}")

            # Update statistics (convert to float for consistent logging)
            m_total.update(total_loss.item())
            m_align.update(align_loss.item())
            m_stab.update(stability_loss.item())

            global_step += 1

            # Logging
            if step % 10 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"Step {step:4d}/{len(dataloader)} | "
                      f"Loss: {total_loss.item():.4f} | "
                      f"Align: {align_loss.item():.4f} | "
                      f"Stab: {stability_loss.item():.4f} | "
                      f"LR: {current_lr:.2e}")
                
                logger.log(
                    {
                        "train/loss": total_loss.item(),
                        "train/align": align_loss.item(),
                        "train/stability": stability_loss.item(),
                        "train/lr": current_lr,
                        "train/grad_norm": total_norm,
                        "diag/pos_sim": diag["pos_sim"],       
                        "diag/neg_sim": diag["neg_sim"],
                        "diag/pos_minus_neg": diag["pos_minus_neg"],
                        "diag/audio_std": diag["audio_std"],
                        "diag/text_std": diag["text_std"], 
                    },
                    step=global_step,
                )

            # Cleanup
            torch.cuda.empty_cache()

        # Save checkpoint
        print(f"\nEpoch {epoch + 1} complete:")
        print(f"  Avg Loss: {m_total.mean:.4f}")
        print(f"  Align Loss: {m_align.mean:.4f}")
        print(f"  Stability Loss: {m_stab.mean:.4f}\n")


        EPOCH_SAVE_PATH = os.path.join(CKPT.dir, f"adapter_stage1_epoch{epoch+1}.pt")
            
        save_checkpoint(
            SAVE_PATH,
            TrainingCheckpoint(
                stage=1,
                epoch=epoch + 1,
                global_step=global_step,
                adapter_state_dict=adapter.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=scheduler.state_dict(),
                metrics={"loss": m_total.mean, "align": m_align.mean, "stability": m_stab.mean},
            ),
        )
        
        save_checkpoint(
            EPOCH_SAVE_PATH,
            TrainingCheckpoint(
                stage=1,
                epoch=epoch + 1,
                global_step=global_step,
                adapter_state_dict=adapter.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=scheduler.state_dict(),
                metrics={"loss": m_total.mean, "align": m_align.mean, "stability": m_stab.mean},
            ),
        )
        print(f"Checkpoint saved to {SAVE_PATH}\n")

    print("Stage 1 training complete!")
    logger.finish()


if __name__ == "__main__":
    train()