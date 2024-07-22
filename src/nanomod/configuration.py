from typing import Optional
from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore

@dataclass
class DataConfig:
    seq_len: int = field(default=256)
    num_tokens: Optional[int] = field(default=None)
    batch_size: int = field(default=64)
    num_workers: int = field(default=4)
    pin_memory: bool = field(default=True)

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    capacity_ratio: float = 0.5
    use_mod: bool = True
    mod_freq: int = 2
    use_darts: bool = False

@dataclass
class TrainConfig:
    lr: float = field(default=6e-4) 
    epochs: int = field(default=2)
    log_interval: int = field(default=100)
    eval_iterations: int = field(default=100)
    use_wandb: bool = field(default=False)
    use_fp16: bool = field(default=False)
    weight_decay: float = field(default=1e-1)
    beta1: float = field(default=0.9)
    beta2: float = field(default=0.95)
    warmup_iters: float = field(default=0.1)
    decay_iter: float = field(default=0.9)
    min_lr: float = field(default=6e-5)
    grad_clip: float = field(default=1.0)

@dataclass
class ExperimentConfig:
    model: GPTConfig = field(default_factory=GPTConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

def set_config_store() -> None:
    cs = ConfigStore.instance()
    cs.store(name="config", node=ExperimentConfig)