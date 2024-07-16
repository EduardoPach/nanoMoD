from typing import Tuple, List
from dataclasses import dataclass, field
from contextlib import nullcontext

import wandb
import torch
import hydra
from tqdm import tqdm
from hydra.core.config_store import ConfigStore

from nanomod.model import GPT, GPTConfig
from nanomod.dataset import get_dataloaders, DataConfig
from nanomod import utils


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


def get_train_context_and_scaler(cfg: ExperimentConfig, device: torch.device) -> Tuple[torch.cuda.amp.autocast, torch.cuda.amp.GradScaler]:
    dtype = torch.float32 if not cfg.train.use_fp16 else (torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16)
    return torch.cuda.amp.autocast(enabled=(cfg.train.use_fp16 and device.type == "cuda"), dtype=dtype), torch.cuda.amp.GradScaler(enabled=cfg.train.use_fp16)
    

@hydra.main(config_path="config", config_name="config")
def train(cfg: ExperimentConfig) -> None:
    assert cfg.model.block_size == cfg.data.seq_len, f"Model block size {cfg.model.block_size} must match data sequence length {cfg.data.seq_len}"

    device = utils.get_best_device()
    model = GPT(cfg.model)
    model.to(device)
    model.train()
    train_ctx, scaler = get_train_context_and_scaler(cfg, device)
    optimizer = model.configure_optimizers(cfg.train.weight_decay, cfg.train.lr, (cfg.train.beta1, cfg.train.beta2), device.type)
    train_loader, val_loader = get_dataloaders(cfg.data)

    num_steps = len(train_loader) * cfg.train.epochs
    flops_per_forward = utils.total_flops(cfg.model) * cfg.data.batch_size
    tokens_per_step = cfg.data.batch_size * cfg.data.seq_len
    wandb_mode = "online" if cfg.train.use_wandb else "disabled"

    pbar = tqdm(range(num_steps), total=num_steps, desc=f"GPT Training: Step 0 - Loss: NaN")
    accum_latency = 0
    accum_throughput = 0

    with wandb.init(project="nanoMoD", config=dict(cfg), job_type="train", mode=wandb_mode) as run:
        utils.log_table(
            num_params=model.get_num_params(),
            flops_per_forward=flops_per_forward, 
            tokens_per_step=tokens_per_step, 
            use_mod=cfg.model.use_mod,
            capacity_ratio=cfg.model.capacity_ratio,
            num_layers=cfg.model.n_layer,
            num_heads=cfg.model.n_head,
            hidden_size=cfg.model.n_embd,
            seq_len=cfg.model.block_size
        )
        for step in pbar:
            lr = utils.set_learning_rate(
                optimizer=optimizer,
                step=step,
                warmup_iters=(num_steps * cfg.train.warmup_iters),
                lr_decay_iters=int(num_steps * cfg.train.decay_iter),
                max_learning_rate=cfg.train.lr,
                min_lr=cfg.train.min_lr
            )

            latency, tokens_per_sec, loss = utils.train_step(
                model=model,
                optimizer=optimizer,
                dataloader=train_loader,
                device=device,
                train_ctx=train_ctx,
                scaler=scaler,
                grad_clip=cfg.train.grad_clip
            )

            # Warming up the latency and throughput metrics
            if step + 1 >= 10:
                accum_latency += latency
                accum_throughput += tokens_per_sec

            avg_latency = accum_latency / (step + 1)
            avg_throughput = accum_throughput / (step + 1)

            pbar.set_description(f"GPT Training: Step {step} - Loss: {loss:.4f} - Latency: {avg_latency:.2f} - Throughput: {avg_throughput:.2f}")

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


if __name__ == "__main__":
    set_config_store()
    train()