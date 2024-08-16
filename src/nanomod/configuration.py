from typing import Optional, Tuple
from dataclasses import dataclass, field

import hydra
from hydra.core.config_store import ConfigStore

@dataclass
class DataConfig:
    """
    Configuration for the data module.

    Attributes:
    - train_dataset: str:
        The dataset to use for training. Default is "openwebtext".
    - seq_len: int:
        The sequence length to use for training. Default is 256.
    - num_tokens: Optional[int]:
        The number of tokens to use for training. Default is None.
    - batch_size: int:
        The batch size to use for training. Default is 64.
    - num_workers: int:
        The number of workers to use for the DataLoader. Default is 4.
    - pin_memory: bool:
        Whether to pin memory in the DataLoader. Default is True.
    - seed: int:
        The seed to use for the random number generator. Default is 13.
    """
    train_dataset: str = field(default="openwebtext")
    seq_len: int = field(default=256)
    num_tokens: Optional[int] = field(default=None)
    batch_size: int = field(default=64)
    num_workers: int = field(default=4)
    pin_memory: bool = field(default=True)
    seed: int = field(default=13)

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
    use_darts: bool = False # IGNORED

@dataclass
class TrainConfig:
    """
    Configuration for the training module.

    Attributes:
    - epochs: int:
        The number of epochs to train for. Default is 2.
    - log_interval: int:
        The number of iterations between logging. Default is 100.
    - eval_iterations: int:
        The number of iterations on evaluation. Default is 100.
    - use_wandb: bool:
        Whether to use Weights & Biases for logging. Default is False.
    - use_fp16: bool:
        Whether to use mixed precision training prioratizing bf16 if available. Default is False.
    - watch_model: bool:
        Whether to watch the model with Weights & Biases. Default is False.
    - watch_mode: str:
        The mode to watch the model in. Default is "all".
    - max_steps: Optional[int]:
        The maximum number of steps to train for. Default is None.
    """
    epochs: int = field(default=2)
    log_interval: int = field(default=100)
    eval_iterations: int = field(default=100)
    use_wandb: bool = field(default=False)
    use_fp16: bool = field(default=False)
    watch_model: bool = field(default=False)
    watch_mode: str = field(default="all")
    max_steps: Optional[int] = field(default=None)

@dataclass
class TrainNormalConfig(TrainConfig):
    """
    Configuration for the normal training process.

    Attributes:
    - lr: float:
        The learning rate to use. Default is 6e-4.
    - grad_clip: float:
        The gradient clipping value to use. Default is 1.0.
    - weight_decay: float:
        The weight decay to use. Default is 1e-1.
    - beta1: float:
        The beta1 value to use. Default is 0.9.
    - beta2: float:
        The beta2 value to use. Default is 0.95.
    - warmup_iters: float:
        The warmup iterations to use as a percentage of total number of steps. Default is 0.1.
    - decay_iter: float:
        The decay iterations to use as a percentage of total number of steps. Default is 0.9.
    - min_lr: float:
        The minimum learning rate to use. Default is 6e-5
    """
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
    """
    Configuration for the DNAS training process.

    Attributes:
    - train_router_steps: float:
        The number of training steps to use for the router. Default is 0.4.
    - lr_model: float:
        The learning rate to use for the model. Default is 6e-4.
    - lr_alphas: float:
        The learning rate to use for the alphas. Default is 6e-3.
    - grad_clip_model: Optional[float]:
        The gradient clipping value to use for the model. Default is 1.0.
    - grad_clip_alphas: Optional[float]:
        The gradient clipping value to use for the alphas. Default is 1.0.
    - use_distillation: bool:
        Whether to use distillation. Default is False.
    - distillation_loss: str:
        The loss to use for distillation either "mse" or "kl". Default is "mse".
    """
    train_router_steps: float = field(default=0.4)
    lr_model: float = field(default=6e-4)
    lr_alphas: float = field(default=6e-3)
    grad_clip_model: Optional[float] = field(default=1.0)
    grad_clip_alphas: Optional[float] = field(default=1.0)
    use_distillation: bool = field(default=False)
    distillation_loss: str = field(default="mse")

@dataclass
class DnasConfig:
    """
    Configuration for the DNAS algorithm.

    Attributes:
    - capacity_ratio_search_space: Tuple[float, ...]:
        The search space for the capacity ratio. Default is (0.1, 0.4, 0.7, 1.0).
    - share_router_weights: bool:
        Whether to share the router weights. Default is False.
    - gumbel_temperature: float:
        The temperature to use for the Gumbel-Softmax. Default is 1.0.
    - a: float:
        The a value to use for the impact of compute loss on the final loss for DNAS. Default is 0.2.
    - b: float:
        The beta value to use for the scaling of DNAS compute loss. Default is 0.6.
    - compute_mode: str:
        The compute mode to use for the DNAS compute loss 
        either "none", "sqrt", "log", "mflop" or "normalized.Default is "none".
    - all_trainable: bool:
        Whether all parameters are trainable. Default is False.
    - fix_first_last: bool:
        Whether to fix the first and last layer. Default
    - hard_sampling: bool:
        Whether to use hard sampling i.e. argmax when sampling from DnasSearchModel. Default is False.
    """
    capacity_ratio_search_space: Tuple[float, ...] = field(default=(0.1, 0.4, 0.7, 1.0))
    share_router_weights: bool = field(default=False)
    gumbel_temperature: float = field(default=1.0)
    a: float = field(default=0.2)
    b: float = field(default=0.6)
    compute_mode: str = field(default="none")
    all_trainable: bool = field(default=False)
    fix_first_last: bool = field(default=False)
    hard_sampling: bool = field(default=False)

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
    cs.store(name="config_train_schema", node=TrainExperimentConfig)
    cs.store(name="config_search_schema", node=SearchExperimentConfig)
    cs.store(group="train", name="normal", node=TrainNormalConfig)
    cs.store(group="train", name="dnas", node=TrainDnasConfig)

if __name__ == "__main__":
    # Just for testing purposes
    @hydra.main(config_path="../../config", config_name="config_search", version_base=None)
    def main_test(cfg: SearchExperimentConfig) -> None:
        print(cfg.train)
        print(cfg.dnas)

    set_config_store()
    main_test()