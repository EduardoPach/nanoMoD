import time
import math
from typing import Tuple, Optional
from contextlib import nullcontext

import torch
import wandb
import pandas as pd
from hydra.core.config_store import ConfigStore

from nanomod.model import GPTConfig, DartsBlock
from nanomod.dataset import DataConfig

def get_best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def total_flops(cfg: GPTConfig) -> int:
    """
    Calculate the total number of FLOPs as per Chinchila Appendix F.
    """
    num_layers = cfg.n_layer
    num_heads = cfg.n_head
    seq_len = cfg.block_size
    hidden_size = cfg.n_embd
    vocab_size = cfg.vocab_size
    capacity_ratio = cfg.capacity_ratio
    use_mod = cfg.use_mod
    mod_freq = cfg.mod_freq
    mod_seq_len = int(capacity_ratio * seq_len)

    # Embedding FLOPs
    embedding_flop = lambda hidden_size, seq_len, vocab_size: 2 * seq_len * hidden_size * vocab_size

    # Attention FLOPs per Layer
    # 1. Query, Key, Value projection
    # 2. Dot product attention
    # 3. Softmax
    # 4. Softmax @ query reduction
    # 5. Output projection
    attn_flops_per_layer = lambda hidden_size, seq_len, num_heads: (
        (2 * 3 * seq_len * hidden_size * seq_len * num_heads) +
        (2 * seq_len * seq_len * seq_len * num_heads) +
        (3 * num_heads * seq_len * seq_len) +
        (2 * seq_len * seq_len * seq_len * num_heads) +
        (2 * seq_len * seq_len * num_heads * hidden_size)
    )
    
    # MLP FLOPs per Layer
    ffn_flops_per_layer = lambda hidden_size, seq_len: 2 * seq_len * (8 * hidden_size * hidden_size)

    # FLOPs per Block
    block_flops = lambda hidden_size, seq_len, num_heads: (
        attn_flops_per_layer(hidden_size, seq_len, num_heads) +
        ffn_flops_per_layer(hidden_size, seq_len)
    )

    # Output Logits
    output_flops = lambda hidden_size, seq_len, vocab_size: 2 * seq_len * hidden_size * vocab_size

    num_mod_layers = 0
    if use_mod:
        num_mod_layers = sum([int(i % mod_freq != 0) for i in range(num_layers)])
    num_normal_layers = num_layers - num_mod_layers

    total_flops = (
        embedding_flop(hidden_size, seq_len, vocab_size) +
        output_flops(hidden_size, seq_len, vocab_size) +
        (num_normal_layers * block_flops(hidden_size, seq_len, num_heads)) +
        (num_mod_layers * block_flops(hidden_size, mod_seq_len, num_heads))
    )

    return total_flops

@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    eval_iterations: int, ctx: torch.cuda.amp.autocast | nullcontext
) -> float:
    device = next(model.parameters()).device
    model.eval()
    total_loss = 0
    for i, (inputs, targets) in enumerate(dataloader):
        if i >= eval_iterations:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        with ctx:
            _, loss = model(inputs, targets)
        total_loss += loss.item()
    return total_loss / eval_iterations


def log_metrics(
    model: torch.nn.Module,
    ctx: torch.cuda.amp.autocast | nullcontext,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    eval_iterations: int,
    **kwargs
) -> None:
    model.eval()
    train_loss = estimate_loss(model, train_loader, eval_iterations, ctx)
    val_loss = estimate_loss(model, val_loader, eval_iterations, ctx)
    model.train()

    log_dict = {
        "train/loss": train_loss,
        "val/loss": val_loss
    }
    log_dict.update(kwargs)

    wandb.log(log_dict)

# learning rate decay scheduler (cosine with warmup)
def set_learning_rate(
    optimizer: torch.optim.Optimizer, 
    step: int, 
    warmup_iters: int, 
    lr_decay_iters: int, 
    max_learning_rate: float, 
    min_lr: float
) -> float:
    # 1) linear warmup for warmup_iters steps
    if step < warmup_iters:
        return max_learning_rate * step / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if step > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (step - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    new_lr =  min_lr + coeff * (max_learning_rate - min_lr)

    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

    return new_lr


def train_step(
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    dataloader: torch.utils.data.DataLoader, 
    device: torch.device, 
    train_ctx: torch.cuda.amp.autocast | nullcontext,
    scaler: torch.cuda.amp.GradScaler,
    grad_clip: Optional[float] = None
) -> Tuple[float, float, float]:
    model.train()
    inputs, targets = next(iter(dataloader))
    inputs, targets = inputs.to(device), targets.to(device)
    optimizer.zero_grad(set_to_none=True)
    is_cuda = device.type == "cuda"

    # Ugly, but we need to measure latency and throughput
    if is_cuda:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    else:
        start = time.time()

    with train_ctx:
        logits, loss = model(inputs, targets)

    if is_cuda:
        end.record()
        torch.cuda.synchronize()
        latency = start.elapsed_time(end) / 1000
    else:
        latency = time.time() - start 
    
    throughput = inputs.size(0) * inputs.size(1) / latency # tokens per second

    scaler.scale(loss).backward()
    if grad_clip is not None:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    scaler.step(optimizer)
    scaler.update()

    return latency, throughput, loss.item()

def log_table(**kwargs) -> None:
    if kwargs is None:
        return
    df = pd.DataFrame([kwargs])
    wandb.log({"table": wandb.Table(dataframe=df)})

def log_capacity_ratio_profile(model: torch.nn.Module) -> None:
    block_idxs = []
    capacity_ratios = []
    for block_idx, block in enumerate(model.transformer.h):
        block_idxs.append(block_idx)
        if not hasattr(block, "capacity_ratio"):
            capacity_ratios.append(1)
            continue
        capacity_ratio = block.capacity_ratio
        capacity_ratio = capacity_ratio.item() if torch.is_tensor(capacity_ratio) else capacity_ratio
        capacity_ratios.append(capacity_ratio)
    
    df = pd.DataFrame({"Block Index": block_idxs, "Capacity Ratio": capacity_ratios})
    wandb.log({"capacity_ratio_profile": wandb.Table(dataframe=df)})

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

def get_compute_loss(model: torch.nn.Module):
    """Compute the avg normalized FLOPs for the model to induce a loss on compute."""
    attn_flops_per_layer = lambda hidden_size, seq_len, num_heads: (
        (2 * 3 * seq_len * hidden_size * seq_len * num_heads) +
        (2 * seq_len * seq_len * seq_len * num_heads) +
        (3 * num_heads * seq_len * seq_len) +
        (2 * seq_len * seq_len * seq_len * num_heads) +
        (2 * seq_len * seq_len * num_heads * hidden_size)
    )
    ffn_flops_per_layer = lambda hidden_size, seq_len: 2 * seq_len * (8 * hidden_size * hidden_size)
    block_flops = lambda hidden_size, seq_len, num_heads: (
        attn_flops_per_layer(hidden_size, seq_len, num_heads) +
        ffn_flops_per_layer(hidden_size, seq_len)
    )

    blocks = model.transformer.h
    max_flops = block_flops(blocks[0].hidden_size, blocks[0].seq_len, blocks[0].num_heads)

    loss = 0
    for block in blocks:
        weights = torch.nn.functional.softmax(block.alphas, dim=0)
        if not isinstance(block, DartsBlock):
            raise ValueError(f"Block {block} is not an instance of DartsBlock")
        
        for b, a in zip(block.blocks, weights):
            normalized_flops = block_flops(block.hidden_size, b.capacity, block.num_heads) / max_flops
            loss += normalized_flops * a
    
    loss = loss / len(blocks)
        
    return loss