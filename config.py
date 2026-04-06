"""Shared configuration for OmniVoice draft model distillation."""

from dataclasses import dataclass, field


@dataclass
class DraftConfig:
    """AR draft model config — predicts codebook-0 tokens left-to-right."""

    # Draft model architecture
    vocab_size: int = 1025          # codebook-0 vocab (0-1023 + mask placeholder)
    hidden_size: int = 512
    num_layers: int = 6
    num_heads: int = 8
    max_seq_len: int = 2048
    dropout: float = 0.1

    # Teacher dimensions (for conditioning projection)
    teacher_hidden_size: int = 1024
    num_codebooks: int = 8

    # Training
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    num_epochs: int = 5
    warmup_steps: int = 500
    grad_clip: float = 1.0
    kl_weight: float = 0.5          # weight for KL loss vs CE loss

    # Speculative decode
    draft_len: int = 10             # frames to draft ahead before verification
    acceptance_threshold: float = 0.0  # 0 = standard spec decode acceptance

    # Paths
    teacher_cache_dir: str = "./cache"
    checkpoint_dir: str = "./checkpoints"
    omnivoice_weights: str = ""     # path to OmniVoice weights


@dataclass
class DraftConfigLocal:
    """Tiny config for local Mac validation — proves the concept works."""

    vocab_size: int = 1025
    hidden_size: int = 256
    num_layers: int = 2
    num_heads: int = 4
    max_seq_len: int = 512
    dropout: float = 0.0

    teacher_hidden_size: int = 1024
    num_codebooks: int = 8

    batch_size: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    num_epochs: int = 3
    warmup_steps: int = 10
    grad_clip: float = 1.0
    kl_weight: float = 0.5

    draft_len: int = 5
    acceptance_threshold: float = 0.0

    teacher_cache_dir: str = "./cache_local"
    checkpoint_dir: str = "./checkpoints_local"
    omnivoice_weights: str = ""
