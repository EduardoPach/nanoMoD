import os
from typing import Optional, Tuple
from dataclasses import dataclass, field

import torch
import numpy as np

from nanomod.configuration import DataConfig

class OpenWebText(torch.utils.data.Dataset):
    def __init__(self, split: str, seq_len: int, num_tokens: Optional[int] = None, seed: int = 13) -> None:
        root = "/teamspace/studios/this_studio/nanoMoD/data/openwebtext/"
        self.split = split
        self.seq_len = seq_len
        self.num_tokens = num_tokens
        self.data = np.memmap(os.path.join(root, f"{split}.bin"), dtype=np.uint16, mode='r')
        self.seed = seed
        self.indices = self._set_indices()

    def _set_indices(self):
        np.random.seed(self.seed)
        if self.num_tokens is None:
            return np.arange(len(self.data) - self.seq_len)
        else:
            return np.random.randint(0, len(self.data) - self.seq_len, self.num_tokens // self.seq_len)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = self.indices[idx]
        x = self.data[idx:idx+self.seq_len]
        y = self.data[idx+1:idx+self.seq_len+1]
        return torch.from_numpy(x.astype(np.int64)), torch.from_numpy(y.astype(np.int64))



class RandomTokenDataset(torch.utils.data.Dataset):
    def __init__(self, seq_len: int, vocab_size: int, num_tokens: Optional[int] = None, seed: int = 13, dynamic: bool = False):
        self.vocab_size = vocab_size
        self.num_tokens = int(num_tokens) if num_tokens is not None else int(10e6)  # Convert to integer
        self.seq_len = seq_len
        self.num_sequences = self.num_tokens // self.seq_len
        self.seed = seed
        self.dynamic = dynamic
        self.dynamic_counter = 0

        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        
        if not dynamic:
            self.rng = torch.Generator()
            self.rng.manual_seed(seed)

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.dynamic:
            # Static behavior: always the same sequence for the same index
            self.rng.manual_seed(self.seed + idx)
        else:
            # Dynamic but reproducible behavior
            dynamic_seed = self.seed + idx + self.dynamic_counter
            self.generator.manual_seed(dynamic_seed)
            self.dynamic_counter += 1
        
        # Generate a sequence of length seq_len + 1
        full_sequence = torch.randint(0, self.vocab_size, (self.seq_len + 1,), generator=self.generator)
        
        # Split the sequence into input and target
        input_sequence = full_sequence[:-1]
        target_sequence = full_sequence[1:]
        
        return input_sequence, target_sequence

    def reset_dynamic_counter(self):
        """Reset the dynamic counter, e.g., at the start of each epoch"""
        self.dynamic_counter = 0

def get_dataloaders(cfg: DataConfig, vocab_size: Optional[int] = None) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Get the train and validation data loaders. Regardless of the `train_dataset` configuration, the validation dataset
    is always OpenWebText.

    Args:
    - cfg (DataConfig): The data configuration.
    - vocab_size (int): The size of the vocabulary.
    """
    if cfg.train_dataset == "openwebtext":
        if vocab_size is not None:
            print("Warning: vocab_size is ignored when using OpenWebText.")
        train_dataset = OpenWebText("train", cfg.seq_len, cfg.num_tokens, cfg.seed)
    elif cfg.train_dataset == "random":
        train_dataset = RandomTokenDataset(cfg.seq_len, vocab_size, None, cfg.seed)
    else:
        raise ValueError(f"Unknown dataset: {cfg.train_dataset}")
    val_dataset = OpenWebText("val", cfg.seq_len, cfg.num_tokens, cfg.seed)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size, 
        num_workers=cfg.num_workers, 
        pin_memory=cfg.pin_memory,
        shuffle=True, 
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=cfg.batch_size, 
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        shuffle=False, 
    )
    return train_loader, val_loader