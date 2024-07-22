import wandb
import hydra
from tqdm import tqdm

from nanomod import utils
from nanomod.model import GPT
from nanomod.dataset import get_dataloaders
from nanomod.configuration import ExperimentConfig, set_config_store

@hydra.main(config_path="config", config_name="config")
def train(cfg: ExperimentConfig) -> None:
    assert cfg.model.block_size == cfg.data.seq_len, f"Model block size {cfg.model.block_size} must match data sequence length {cfg.data.seq_len}"

    device = utils.get_best_device()
    model = GPT(cfg.model)
    model.to(device)
    model.train()
    train_ctx, scaler = utils.get_train_context_and_scaler(cfg, device)
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

        utils.log_model(
            model, 
            optimizer,
            cfg.model,
            tokens_seen=(tokens_per_step * num_steps), 
            total_flops=(flops_per_forward * num_steps)
        )

if __name__ == "__main__":
    set_config_store()
    train()