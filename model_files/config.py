from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 1024 # context length
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768