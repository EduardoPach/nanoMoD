from typing import Tuple, List
from dataclasses import dataclass, field
from contextlib import nullcontext

import wandb
import torch
import hydra
from tqdm import tqdm

from nanomod.model import GPT
from nanomod.dataset import get_dataloaders
from nanomod import utils


def get_train_context_and_scaler(cfg: ExperimentConfig, device: torch.device) -> Tuple[torch.cuda.amp.autocast, torch.cuda.amp.GradScaler]:
    dtype = torch.float32 if not cfg.train.use_fp16 else (torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16)
    return torch.cuda.amp.autocast(enabled=(cfg.train.use_fp16 and device.type == "cuda"), dtype=dtype), torch.cuda.amp.GradScaler(enabled=cfg.train.use_fp16)


def train_step(
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    dataloader: torch.utils.data.DataLoader, 
    device: torch.device, 
    train_ctx: torch.cuda.amp.autocast | nullcontext,
    scaler: torch.cuda.amp.GradScaler,
    grad_clip: Optional[float] = None,
    add_compute_loss: bool = False
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

    if add_compute_loss:
        loss_compute = utils.get_compute_loss(model)
        loss = loss + loss_compute

    scaler.scale(loss).backward()
    if grad_clip is not None:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    scaler.step(optimizer)
    scaler.update()

    optimizer.zero_grad(set_to_none=True)

    return latency, throughput, loss.item()

@hydra.main(config_path="config", config_name="config")
def train(cfg: utils.ExperimentConfig) -> None:
    assert cfg.model.block_size == cfg.data.seq_len, f"Model block size {cfg.model.block_size} must match data sequence length {cfg.data.seq_len}"

    device = utils.get_best_device()
    model = GPT(cfg.model)
    model.to(device)
    model.train()

    train_ctx, scaler = get_train_context_and_scaler(cfg, device)
    train_loader, val_loader = get_dataloaders(cfg.data)

    optimizer_weights = model.configure_optimizers(cfg.train.weight_decay, cfg.train.lr, (cfg.train.beta1, cfg.train.beta2), device.type)
    optimizer_alphas = torch.optim.Adam(
        [v for k, v in model.named_parameters() if "alpha" in k], 
        lr=cfg.train.alpha_lr, 
        betas=(cfg.train.beta1, cfg.train.beta2)
    )

    num_steps = len(train_loader) * cfg.train.epochs
    flops_per_forward = utils.total_flops(cfg.model) * cfg.data.batch_size
    tokens_per_step = cfg.data.batch_size * cfg.data.seq_len
    wandb_mode = "online" if cfg.train.use_wandb else "disabled"

    pbar = tqdm(range(num_steps), total=num_steps, desc=f"Searching MoD Profile: Step 0 - Loss: NaN")
    accum_latency = 0
    accum_throughput = 0

    with wandb.init(project="nanoMoD", config=dict(cfg), job_type="dnas", mode=wandb_mode) as run:
        for step in pbar:
            # Set learning rate for Weights step
            lr = utils.set_learning_rate(
                optimizer=optimizer_weights,
                step=step,
                warmup_iters=(num_steps * cfg.train.warmup_iters),
                lr_decay_iters=int(num_steps * cfg.train.decay_iter),
                max_learning_rate=cfg.train.lr,
                min_lr=cfg.train.min_lr
            )

            # Step for Alphas (i.e. arch search)
            _, _ = utils.train_step(
                model=model,
                optimizer=optimizer_alphas,
                dataloader=val_loader,
                device=device,
                train_ctx=train_ctx,
                scaler=scaler,
                add_compute_loss=cfg.train.add_compute_loss,
                grad_clip=None
            )

            # Step for model weights
            _, _, loss = utils.train_step(
                model=model,
                optimizer=optimizer_weights,
                dataloader=train_loader,
                device=device,
                train_ctx=train_ctx,
                scaler=scaler,
                grad_clip=cfg.train.grad_clip
            )

            pbar.set_description(f"Searching MoD Profile: Step {step} - Loss: {loss:.4f}")

            if ((step+1) % cfg.train.log_interval) == 0:
                utils.log_metrics(
                    model=model,
                    ctx=train_ctx,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    eval_iterations=cfg.train.eval_iterations,
                    step=step,
                    tokens_seen=(tokens_per_step * (step + 1)),
                    total_flops=(flops_per_forward * (step + 1)),
                    latency=avg_latency,
                    tokens_per_sec=avg_throughput,
                    lr=lr
                )
        if cfg.model.learnable_capacity_ratio:
            utils.log_capacity_ratio_profile(model)

if __name__ == "__main__":
    utils.set_config_store()
    train()