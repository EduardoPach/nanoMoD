from typing import Tuple, Optional
from contextlib import nullcontext

import hydra
import wandb
import torch
from tqdm import tqdm
from torch.optim import AdamW

from nanomod import utils
from nanomod.model import DnasSearchModel, GPT
from nanomod.dataset import get_dataloaders
from nanomod.configuration import SearchExperimentConfig, set_config_store


def train_model_step(
    model: DnasSearchModel, 
    optimizer: torch.optim.Optimizer, 
    train_loader: torch.utils.data.DataLoader,
    ctx: torch.cuda.amp.autocast | nullcontext,
    scaler: torch.cuda.amp.GradScaler,
    grad_clip: Optional[float] = None
) -> float:
    model.train()
    device = next(model.parameters()).device
    inputs, targets = next(iter(train_loader))
    inputs, targets = inputs.to(device), targets.to(device)
    optimizer.zero_grad(set_to_none=True)
    with ctx:
        _, loss = model(inputs, targets)
    
    scaler.scale(loss).backward()
    if grad_clip is not None:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    scaler.step(optimizer)
    scaler.update()

    return loss.item()


def train_alphas_step(
    model: DnasSearchModel, 
    a: float,
    b: float,
    optimizer: torch.optim.Optimizer, 
    train_loader: torch.utils.data.DataLoader,
    ctx: torch.cuda.amp.autocast | nullcontext,
    scaler: torch.cuda.amp.GradScaler,
    grad_clip: Optional[float] = None,
) -> Tuple[float, float]:
    model.train()
    device = next(model.parameters()).device
    inputs, targets = next(iter(train_loader))
    inputs, targets = inputs.to(device), targets.to(device)
    optimizer.zero_grad(set_to_none=True)

    with ctx:
        _, loss, loss_compute = model(inputs, targets, return_loss_compute=True)
    
    total_loss = loss + a * loss_compute ** b

    scaler.scale(total_loss).backward()
    if grad_clip is not None:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    scaler.step(optimizer)
    scaler.update()

    return loss.item(), loss_compute.item()

@hydra.main(config_path="config", config_name="config_search")
def main(cfg: SearchExperimentConfig) -> None:
    device = utils.get_best_device()
    wandb_mode = "online" if cfg.train.use_wandb else "disabled"
    ctx, scaler_model = utils.get_train_context_and_scaler(cfg, device)
    _, scaler_alphas = utils.get_train_context_and_scaler(cfg, device)

    train_loader, val_loader = get_dataloaders(cfg.data)

    num_steps = len(train_loader) * cfg.train.epochs
    num_steps_model = int(num_steps * cfg.train.train_router_steps)
    tokens_per_step = cfg.data.batch_size * cfg.data.seq_len

    pbar = tqdm(range(num_steps), total=num_steps, desc=f"Searching Model: Step 0 - Training Model Weights", unit="step")

    with wandb.init(project="nanoMoD", config=dict(cfg), job_type="search", mode=wandb_mode):
        state_dict, model_config = utils.load_checkpoint()
        base_model = GPT(model_config)
        base_model.load_state_dict(state_dict)
        model = DnasSearchModel(base_model, cfg.dnas)
        model.to(device)
        model.train()

        optimizer_router = AdamW(model.get_router_weights(), lr=cfg.train.lr_model)
        optimizer_alphas = AdamW(model.get_alphas(), lr=cfg.train.lr_alphas)
        for step in pbar:
            if step < num_steps_model:
                loss = train_model_step(
                    model=model, 
                    optimizer=optimizer_router, 
                    train_loader=train_loader,
                    ctx=ctx,
                    scaler=scaler_model,
                    grad_clip=cfg.train.grad_clip_model
                )
                pbar.set_description(f"Searching Model: Step {step} - Training Model Weights - Loss: {loss:.4f}")
            else:
                loss, loss_compute = train_alphas_step(
                    model=model,
                    a=cfg.dnas.a,
                    b=cfg.dnas.b,
                    optimizer=optimizer_alphas, 
                    train_loader=train_loader,
                    ctx=ctx,
                    scaler=scaler_alphas,
                    grad_clip=cfg.train.grad_clip_alphas
                )
                pbar.set_description(f"Searching Model: Step {step} - Training Alphas - Loss: {loss:.4f} - Loss Compute: {loss_compute:.4f}")

            if step % cfg.train.log_interval == 0:
                utils.log_metrics(
                    model=model,
                    ctx=ctx,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    eval_iterations=cfg.train.eval_iterations,
                    step=step,
                    tokens_seen=(tokens_per_step * (step + 1)),
                    **model.capacity_profile
                )

        # Log one more time at the end
        utils.log_metrics(
            model=model,
            ctx=ctx,
            train_loader=train_loader,
            val_loader=val_loader,
            eval_iterations=cfg.train.eval_iterations,
            step=step,
            tokens_seen=(tokens_per_step * (step + 1)),
            **model.capacity_profile
        )

if __name__ == "__main__":
    set_config_store()
    main()