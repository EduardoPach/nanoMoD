import math
from logging import getLogger
from typing import Tuple, Optional
from contextlib import nullcontext
from collections import defaultdict

import hydra
import wandb
import torch
import pandas as pd
from tqdm import tqdm
from torch.optim import AdamW

from nanomod import utils
from nanomod.model import DnasSearchModel, GPT
from nanomod.dataset import get_dataloaders
from nanomod.configuration import SearchExperimentConfig, set_config_store

logger = getLogger(__name__)


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
    temperature: Optional[float] = None
) -> Tuple[float, float, float]:
    model.train()
    device = next(model.parameters()).device
    inputs, targets = next(iter(train_loader))
    inputs, targets = inputs.to(device), targets.to(device)
    optimizer.zero_grad(set_to_none=True)

    if temperature is not None:
        model.set_temperature(temperature)

    with ctx:
        _, loss, loss_compute = model(inputs, targets, return_loss_compute=True)
    
    total_loss = loss + a * loss_compute ** b

    scaler.scale(total_loss).backward()
    if grad_clip is not None:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    scaler.step(optimizer)
    scaler.update()

    return total_loss.item(), loss.item(), loss_compute.item()

@hydra.main(config_path="config", config_name="config_search")
def search(cfg: SearchExperimentConfig) -> None:
    device = utils.get_best_device()
    wandb_mode = "online" if cfg.train.use_wandb else "disabled"
    ctx, scaler_model = utils.get_train_context_and_scaler(cfg, device)
    _, scaler_alphas = utils.get_train_context_and_scaler(cfg, device)

    train_loader, val_loader = get_dataloaders(cfg.data)

    num_steps = len(train_loader) * cfg.train.epochs
    num_steps_model_only = int(num_steps * cfg.train.train_router_steps)
    num_steps_alphas = num_steps - num_steps_model_only

    alphas_historical = defaultdict(list)
    pbar = tqdm(range(num_steps), total=num_steps, desc=f"Searching Model: Step 0 - Training Model Weights", unit="step")

    with wandb.init(project="nanoMoD", config=dict(cfg), job_type="search", mode=wandb_mode):
        state_dict, model_config = utils.load_checkpoint(use_wandb=cfg.train.use_wandb)
        base_model = GPT(model_config)
        base_model.load_state_dict(state_dict)
        model = DnasSearchModel(base_model, cfg.dnas)
        if not cfg.dnas.all_trainable:
            model.freeze()
        model.to(device)
        model.train()

        if cfg.train.watch_model:
            wandb.watch(model, log=cfg.train.watch_mode, log_freq=cfg.train.log_interval)

        optimizer_router = AdamW(model.get_router_weights(), lr=cfg.train.lr_model)
        optimizer_alphas = AdamW(model.get_alphas(), lr=cfg.train.lr_alphas)

        scheduler_router = utils.LearningRateScheduler(
            optimizer=optimizer_router, 
            warmup_iters=int(num_steps_model_only * 0.2), 
            lr_decay_iters=int(num_steps_model_only + num_steps_alphas * 0.5), 
            max_learning_rate=cfg.train.lr_model, 
            min_lr=cfg.train.lr_model / 10, 
        )

        scheduler_alphas = utils.LearningRateScheduler(
            optimizer=optimizer_alphas, 
            warmup_iters=int(num_steps_alphas * 0.1), 
            lr_decay_iters=int(num_steps_alphas * 0.9),
            max_learning_rate=cfg.train.lr_alphas, 
            min_lr=cfg.train.lr_alphas / 10
        )

        temperature_scheduler = utils.TemperatureExponentialDecay(
            max_temperature=cfg.dnas.gumbel_temperature, 
            min_temperature=cfg.dnas.gumbel_temperature / 10, 
            max_steps=num_steps_alphas
        )

        alphas_values = []
        for step in pbar:
            optimizer_alphas.zero_grad(set_to_none=True)
            optimizer_router.zero_grad(set_to_none=True)

            lr_model = scheduler_router.update(step+1)

            loss_model_step = train_model_step(
                model=model, 
                optimizer=optimizer_router, 
                train_loader=train_loader,
                ctx=ctx,
                scaler=scaler_model,
                grad_clip=cfg.train.grad_clip_model
            )
    
        
            if step > (num_steps_model_only - 1):
                lr_alphas = scheduler_alphas.update(step - num_steps_model_only + 1)
                temperature = temperature_scheduler.update(step - num_steps_model_only)

                total_loss, loss, loss_compute = train_alphas_step(
                    model=model,
                    a=cfg.dnas.a,
                    b=cfg.dnas.b,
                    optimizer=optimizer_alphas, 
                    train_loader=train_loader,
                    ctx=ctx,
                    scaler=scaler_alphas,
                    grad_clip=cfg.train.grad_clip_alphas,
                    temperature=temperature
                )
                pbar.set_description(f"Searching Model: Step {step} - Training Alphas - Loss: {loss:.4f} - Loss Compute: {loss_compute:.4f} - Total Loss: {total_loss:.4f}")
                wandb.log({
                    "search/loss_model_step": loss_model_step, 
                    "search/total_loss": total_loss, 
                    "search/loss_ce": loss, 
                    "search/loss_compute": loss_compute, 
                    "search/step": step,
                    "search/lr_alphas": lr_alphas,
                    "search/lr_model": lr_model,
                    "search/temperature": temperature
                })
            
            else:
                wandb.log({"search/loss_model_step": loss_model_step, "search/step": step, "search/lr_model": lr_model})
                pbar.set_description(f"Searching Model: Step {step} - Training Model Weights - Loss: {loss_model_step:.4f}")


            if step % cfg.train.log_interval == 0:
                for block_idx, block in enumerate(model.get_blocks()):
                    alphas_values = block.alphas.detach().cpu()

                    weight_values = alphas_values.softmax(dim=-1).tolist()
                    alphas_values = alphas_values.tolist()
                    capacity_ratios = cfg.dnas.capacity_ratio_search_space
                    idx = [block_idx] * len(capacity_ratios)
                    step_list = [step] * len(capacity_ratios)

                    alphas_historical["block_idx"].extend(idx)
                    alphas_historical["capacity_ratio"].extend(capacity_ratios)
                    alphas_historical["alphas"].extend(alphas_values)
                    alphas_historical["weights"].extend(weight_values)
                    alphas_historical["step"].extend(step_list)
                
                picked_alphas = {}
                for block_idx, block in enumerate(model.get_blocks()):
                    picked_idx = block.alphas.detach().argmax(dim=-1).item()
                    picked_alphas[f"search/layer_{block_idx}"] = cfg.dnas.capacity_ratio_search_space[picked_idx]
                
                wandb.log(picked_alphas)

        # Log one more time at the end
        table = wandb.Table(dataframe=pd.DataFrame(alphas_historical))
        logger.info(f"Sampling model...")
        sample_model = model.sample_architecture()
        logger.info(f"Evaluating model...")
        sample_model_val_loss = utils.estimate_loss(sample_model, val_loader, cfg.train.eval_iterations, ctx)
        logger.info(f"Sample model validation loss: {sample_model_val_loss:.4f} -> logging to wandb...")
        wandb.log({"search/alphas_historical": table, "search/val_loss": sample_model_val_loss})

def main() -> None:
    set_config_store()
    search()

if __name__ == "__main__":
    main()