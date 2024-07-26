from typing import Optional, Tuple
from dataclasses import dataclass, field

import hydra
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
    epochs: int = field(default=2)
    log_interval: int = field(default=100)
    eval_iterations: int = field(default=100)
    use_wandb: bool = field(default=False)
    use_fp16: bool = field(default=False)
    watch_model: bool = field(default=False)
    watch_mode: str = field(default="all")

@dataclass
class TrainNormalConfig(TrainConfig):
    lr: float = field(default=6e-4) 
    grad_clip: float = field(default=1.0)
    weight_decay: float = field(default=1e-1)
    beta1: float = field(default=0.9)
    beta2: float = field(default=0.95)
    warmup_iters: float = field(default=0.1)
    decay_iter: float = field(default=0.9)
    min_lr: float = field(default=6e-5)

@dataclass
class TrainDnasConfig(TrainConfig):
    train_router_steps: float = field(default=0.4)
    lr_model: float = field(default=6e-4)
    lr_alphas: float = field(default=6e-4)
    grad_clip_model: Optional[float] = field(default=1.0)
    grad_clip_alphas: Optional[float] = field(default=1.0)

@dataclass
class DnasConfig:
    capacity_ratio_search_space: Tuple[float, ...] = field(default=(0.1, 0.4, 0.7, 1.0))
    share_router_weights: bool = field(default=False)
    gumbel_temperature: float = field(default=1.0)
    a: float = field(default=0.2)
    b: float = field(default=0.6)

@dataclass
class TrainExperimentConfig:
    model: GPTConfig = field(default_factory=GPTConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainNormalConfig = field(default_factory=TrainNormalConfig)

@dataclass
class SearchExperimentConfig:
    model: GPTConfig = field(default_factory=GPTConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainDnasConfig = field(default_factory=TrainDnasConfig)
    dnas: DnasConfig = field(default_factory=DnasConfig)


def set_config_store() -> None:
    cs = ConfigStore.instance()
    cs.store(name="config_train", node=TrainExperimentConfig)
    cs.store(name="config_search", node=SearchExperimentConfig)
    cs.store(group="train", name="normal", node=TrainNormalConfig)
    cs.store(group="train", name="dnas", node=TrainDnasConfig)

if __name__ == "__main__":
    # Just for testing purposes
    @hydra.main(config_path="../../config", config_name="config_search")
    def main_test(cfg: SearchExperimentConfig) -> None:
        print(cfg.train)
        print(cfg.dnas)

    set_config_store()
    main_test()