"""
Whisper-to-LLM Projector Module

This module provides a simple linear projection network that maps Whisper encoder
embeddings to the embedding space of Large Language Models like Qwen.

The projector performs frame-by-frame dimensionality transformation without temporal
compression, serving as a baseline for comparison with the more advanced StreamingAdapter.

Architecture:
    Input (T, d_in) → Linear(d_in, hidden) → ReLU → Linear(hidden, d_out) → Output (T, d_out)

Default Configuration (Whisper-small to Qwen):
    - Input dimension: 768 (Whisper-small encoder output)
    - Hidden dimension: 2048
    - Output dimension: 4096 (Qwen3-8B embedding dimension)
    - Sequence length: Preserved (T=1500 frames)

Use Cases:
    - Baseline comparison with StreamingAdapter
    - Experiments requiring full temporal resolution
    - Debugging and testing audio-to-text pipelines
    - When temporal compression is not needed
"""

import torch
import torch.nn as nn


class WhisperToQwenProjector(nn.Module):
    """
    Simple linear projection network for Whisper-to-LLM dimension mapping.

    This projector performs frame-by-frame projection, preserving the full
    temporal sequence length. Unlike the StreamingAdapter, it does not provide
    any temporal compression, making it useful as a baseline or when full
    temporal resolution is required.

    Architecture:
        Linear(d_in, 2048) → ReLU activation → Linear(2048, d_out)

    Args:
        in_dim: Input dimension from Whisper encoder (default: 768 for whisper-small)
        out_dim: Output dimension matching LLM embedding (default: 4096 for Qwen3-8B)

    Example:
        >>> projector = WhisperToQwenProjector(in_dim=768, out_dim=4096)
        >>> encoder_output = torch.randn(1, 1500, 768)  # Whisper output
        >>> projected = projector(encoder_output)
        >>> print(projected.shape)  # torch.Size([1, 1500, 4096])
    """

    def __init__(self, in_dim: int = 768, out_dim: int = 4096):
        """
        Initialize the Whisper-to-Qwen projector.

        Args:
            in_dim: Dimension of Whisper encoder output (e.g., 768 for whisper-small)
            out_dim: Dimension of LLM embedding space (e.g., 4096 for Qwen3-8B)
        """
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        # Two-layer projection with ReLU activation
        self.proj = nn.Sequential(
            nn.Linear(in_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, out_dim),
        )

        # Initialize weights for better training stability
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize layer weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project Whisper encoder embeddings to LLM embedding space.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, in_dim)
               where sequence_length is typically 1500 frames for 30s of audio

        Returns:
            Projected tensor of shape (batch_size, sequence_length, out_dim)
            The sequence length is preserved through frame-by-frame projection

        Example:
            >>> projector = WhisperToQwenProjector()
            >>> x = torch.randn(2, 1500, 768)  # batch=2, frames=1500
            >>> y = projector(x)
            >>> print(y.shape)  # torch.Size([2, 1500, 4096])
        """
        return self.proj(x)

    def get_num_parameters(self) -> int:
        """
        Get the total number of trainable parameters.

        Returns:
            Total number of parameters
        """
        return sum(p.numel() for p in self.parameters())


# Test and demonstration code
if __name__ == "__main__":
    print("WhisperToQwenProjector - Demonstration\n")
    print("=" * 60)

    # Create model with default settings
    model = WhisperToQwenProjector()
    print(f"\nModel Configuration:")
    print(f"  Input dimension:  {model.in_dim}")
    print(f"  Output dimension: {model.out_dim}")
    print("  Hidden dimension: 2048")
    print(f"  Total parameters: {model.get_num_parameters():,}")

    # Test forward pass
    print(f"\nTesting forward pass:")
    batch_size = 2
    seq_len = 1500  # Typical Whisper output for 30s audio

    x = torch.randn(batch_size, seq_len, 768)
    print(f"  Input shape:    {x.shape}")

    y = model(x)
    print(f"  Output shape:   {y.shape}")

    # Verify output dimensions
    assert y.shape == (batch_size, seq_len, 4096), "Output shape mismatch!"

    # Test with different dimensions
    print(f"\nTesting custom dimensions:")
    custom_model = WhisperToQwenProjector(in_dim=1024, out_dim=4096)  # whisper-medium
    x_custom = torch.randn(1, 1500, 1024)
    y_custom = custom_model(x_custom)
    print(f"  Input shape:    {x_custom.shape}")
    print(f"  Output shape:   {y_custom.shape}")

    print("\n" + "=" * 60)
    print("All tests passed successfully!")
