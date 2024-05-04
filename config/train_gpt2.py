wandb_log = True
wandb_project = 'mod-gpt-shakespeare'
wandb_run_name='gpt2-124M'

batch_size = 12
block_size = 1024
gradient_accumulation_steps = 1

# eval stuff
eval_interval = 500
eval_iters = 200
log_interval = 10