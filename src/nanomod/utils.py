import os
import time
import math
from typing import Tuple, Optional, Dict
from contextlib import nullcontext

import torch
import wandb
import pandas as pd

from nanomod.configuration import TrainExperimentConfig, GPTConfig

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

def log_model(
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer,
    model_config: GPTConfig,
    output_dir: str = "checkpoint_dir", 
    **kwargs
) -> None:
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_config': dict(model_config),
    }

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ckpt_path = os.path.join(output_dir, "ckpt.pt")
    torch.save(checkpoint, ckpt_path)
    artifact = wandb.Artifact(f"model-ckpt", type='model', metadata=kwargs)
    artifact.add_file(ckpt_path)
    wandb.run.log_artifact(artifact)

@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    eval_iterations: int, 
    ctx: torch.cuda.amp.autocast | nullcontext
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

class LearningRateScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_iters: int, lr_decay_iters: int, max_learning_rate: float, min_lr: float) -> None:
        self.optimizer = optimizer
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = lr_decay_iters
        self.max_learning_rate = max_learning_rate
        self.min_lr = min_lr
    
    def update(self, step: int) -> float:
        return set_learning_rate(
            self.optimizer, 
            step, 
            self.warmup_iters, 
            self.lr_decay_iters, 
            self.max_learning_rate, 
            self.min_lr
        )

class TemperatureExponentialDecay:
    def __init__(self, max_temperature: float, min_temperature: float, max_steps: int) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        if max_value <= min_value:
            raise ValueError("max_value must be greater than min_value")
        
        self.max_temperature = max_temperature
        self.min_temperature = min_temperature
        self.max_steps = max_steps
        
        # Calculate the decay factor
        self.decay_factor = (min_temperature / max_temperature) ** (1 / (max_steps - 1))
    
    def update(self, step: int) -> float:
        """
        Get the temperature for a given step.
        
        Parameters:
            step (int): The step for which to get the temperature.
        
        Returns:
            float: The temperature at the given step.
        """
        if step < 0 or step >= self.max_steps:
            raise ValueError("step must be within the range [0, max_steps)")
        
        temperature = self.max_temperature * (self.decay_factor ** step)
        return temperature


def get_train_context_and_scaler(
    cfg: TrainExperimentConfig, 
    device: torch.device
) -> Tuple[torch.cuda.amp.autocast, torch.cuda.amp.GradScaler]:
    dtype = torch.float32 if not cfg.train.use_fp16 else (torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16)
    return torch.cuda.amp.autocast(enabled=(cfg.train.use_fp16 and device.type == "cuda"), dtype=dtype), torch.cuda.amp.GradScaler(enabled=cfg.train.use_fp16)

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

def get_flop_per_block(hidden_size: int, seq_len: int, num_heads: int, capacity_ratio: float) -> float:
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

    return block_flops(hidden_size, int(capacity_ratio * seq_len), num_heads)

def load_checkpoint(checkpoint: str = "model-ckpt:latest") -> Tuple[Dict[str, torch.Tensor], GPTConfig]:
    artifact = wandb.use_artifact(f'eduardopacheco/nanoMoD/{checkpoint}', type='model')
    artifact_dir = artifact.download()
    checkpoint = torch.load(os.path.join(artifact_dir, "ckpt.pt"), map_location="cpu")
    config = GPTConfig(**checkpoint['model_config'])
    state_dict = checkpoint['model']

    return state_dict, config

def log_table(**kwargs) -> None:
    if kwargs is None:
        return
    df = pd.DataFrame([kwargs])
    wandb.log({"table": wandb.Table(dataframe=df)})