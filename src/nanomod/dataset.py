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

def get_dataloaders(cfg: DataConfig) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    train_dataset = OpenWebText("train", cfg.seq_len, cfg.num_tokens)
    val_dataset = OpenWebText("val", cfg.seq_len, cfg.num_tokens)
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