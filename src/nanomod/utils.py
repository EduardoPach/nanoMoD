import time
from typing import Tuple
from contextlib import nullcontext

import torch
import wandb

from nanomod.model import GPTConfig

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
    step: int,
    tokens_per_step: int,
    flops_per_forward: int,
    latency: float,
    throughput: float,
) -> None:
    model.eval()
    num_params = model.get_num_params()
    train_loss = estimate_loss(model, train_loader, eval_iterations, ctx)
    val_loss = estimate_loss(model, val_loader, eval_iterations, ctx)
    model.train()

    wandb.log(
        {
            "train/loss": train_loss,
            "val/loss": val_loss,
            "step": step,
            "tokens_seen": tokens_per_step * (step + 1),
            "total_flops": flops_per_forward * (step + 1),
            "latency": latency,
            "tokens_per_sec": throughput,
            "num_params": num_params,
        }
    )

def train_step(
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    dataloader: torch.utils.data.DataLoader, 
    device: torch.device, 
    train_ctx: torch.cuda.amp.autocast | nullcontext,
    scaler: torch.cuda.amp.GradScaler
) -> Tuple[float, float, float]:
    model.train()
    inputs, targets = next(iter(dataloader))
    inputs, targets = inputs.to(device), targets.to(device)
    optimizer.zero_grad()
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
    scaler.step(optimizer)
    scaler.update()

    return latency, throughput, loss.item()