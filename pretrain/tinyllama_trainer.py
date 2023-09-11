import math
import sys
import time
from pathlib import Path
from typing import Any, Optional
import os

import lightning as L
import numpy as np
import torch

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import FSDPStrategy
from torch.utils.data import DataLoader, IterableDataset
from dotenv import load_dotenv

load_dotenv()

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt import Config
from lit_gpt.model import GPT, Block
from lit_gpt.speed_monitor import SpeedMonitorCallback, estimate_flops, measure_flops
from lit_gpt.utils import (
    chunked_cross_entropy,
    get_default_supported_precision,
)

model_name = "SmolKat_310M"  #
num_devices = 1

configs = {
    "SmolKat_10M": {
        "micro_batch_size": 64,
        "target_batch_size": 256,
    },
    "SmolKat_120M": {
        "micro_batch_size": 8,
        "target_batch_size": 256,
    },
    "SmolKat_310M": {
        "micro_batch_size": 4,
        "target_batch_size": 256,
    },
}


configs[model_name]["gradient_accumulation_steps"] = int(
    configs[model_name]["target_batch_size"]
    / configs[model_name]["micro_batch_size"]
    * 1
    / num_devices
)

print(
    f"Running with Accumulation Steps = {configs[model_name]['gradient_accumulation_steps']}"
)

name = "concoction"
out_dir = Path("out") / name
data_dir = Path("data") / name
save_interval = 500
eval_interval = 1000
eval_iters = 100
log_interval = 1
dataset_logging_interval = 10_000
# Hyperparameters
micro_batch_size = configs[model_name]["micro_batch_size"]
gradient_accumulation_steps = configs[model_name]["gradient_accumulation_steps"]
## Batch Size = micro_batch_size * gradient_accumulation_steps
assert gradient_accumulation_steps > 0
max_iters = 600000  # num_epochs * (epoch_size // micro_batch_size) // devices
warmup_iters = 3000
lr_decay_iters = max_iters

learning_rate = 6e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
min_lr = 6e-5

hparams = {
    k: v
    for k, v in locals().items()
    if isinstance(v, (int, float, str)) and not k.startswith("_")
}


# class Dataset(IterableDataset):
#     def __init__(self, data_file: Path, block_size: int, print_every: int = 1e10):
#         super().__init__()
#         self.data_file = data_file
#         self.block_size = block_size
#         self.print_every = print_every

#         # Create a shuffled list of indices
#         self.data = np.memmap(self.data_file, dtype=np.uint16, mode="r")
#         self.visited_indices = torch.randperm(len(self.data) - self.block_size).tolist()
#         self.counter = 0  # to count the number of indices we've visited

#     def __iter__(self):
#         step = 0
#         while True:
#             i = self.visited_indices[self.counter % len(self.visited_indices)]
#             self.counter += 1

#             # Once we've visited all indices twice, reshuffle
#             if self.counter == 2 * len(self.visited_indices):
#                 self.counter = 0
#                 self.visited_indices = torch.randperm(
#                     len(self.data) - self.block_size
#                 ).tolist()

#             x = torch.from_numpy((self.data[i : i + self.block_size]).astype(np.int64))
#             y = torch.from_numpy(
#                 (self.data[i + 1 : i + 1 + self.block_size]).astype(np.int64)
#             )

#             # Print every N steps
#             step += 1
#             if step % self.print_every == 0:
#                 print(
#                     f"Dataset {step}: Working on index {i}, {self.block_size*step} tokens visited"
#                 )

#             yield x, y


class Dataset(IterableDataset):
    def __init__(self, data_file: Path, block_size: int):
        super().__init__()
        self.data_file = data_file
        self.block_size = block_size

    def __iter__(self):
        data = np.memmap(self.data_file, dtype=np.uint16, mode="r")
        while True:
            i = torch.randint(len(data) - self.block_size, (1,)).item()
            x = torch.from_numpy((data[i : i + self.block_size]).astype(np.int64))
            y = torch.from_numpy(
                (data[i + 1 : i + 1 + self.block_size]).astype(np.int64)
            )
            yield x, y


class LightningGPTModule(L.LightningModule):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.module: Optional[torch.nn.Module] = None
        self.measured_flops: Optional[int] = None

    def configure_model(self) -> None:
        self.module = GPT(self.config)
        if os.path.exists("out/concoction/last.ckpt"):
            checkpoint = torch.load(
                "out/concoction/last.ckpt"
            )  # , map_location="cuda")
            self.module.load_state_dict(checkpoint["state_dict"])
        else:
            self.module.apply(self.module._init_weights)
        self.module = torch.compile(self.module)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.module.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(beta1, beta2),
            foreach=False,
        )

    def on_fit_start(self) -> None:
        trainer = self.trainer
        with torch.device("meta"):
            meta_model = GPT(self.module.config)
            # "estimated" is not as precise as "measured". Estimated is optimistic but widely used in the wild.
            # When comparing MFU or FLOP numbers with other projects that use estimated FLOPs,
            # consider setting `self.measured_flops = estimated_flops` instead
            estimated_flops = estimate_flops(meta_model) * micro_batch_size
            self.print(
                f"Estimated TFLOPs: {estimated_flops * trainer.world_size / 1e12:.2f}"
            )
            x = torch.randint(0, 1, (micro_batch_size, meta_model.config.block_size))
            self.measured_flops = measure_flops(meta_model, x)
            self.print(
                f"Measured TFLOPs: {self.measured_flops * trainer.world_size / 1e12:.2f}"
            )

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        if not decay_lr:
            return
        # determine and set the learning rate for this iteration
        lr = get_lr(self.trainer.fit_loop.total_batch_idx)
        for optimizer in self.trainer.strategy.optimizers:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        input_ids, targets = batch
        logits = self.module(input_ids)
        loss = chunked_cross_entropy(logits, targets, chunk_size=0)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        input_ids, targets = batch
        logits = self.module(input_ids)
        loss = chunked_cross_entropy(logits, targets, chunk_size=0)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)


def main(devices: int = 1, precision: Optional[str] = None) -> None:
    precision = precision or get_default_supported_precision(training=True)

    if devices > 1:
        strategy = FSDPStrategy(
            auto_wrap_policy={Block},
            activation_checkpointing_policy={Block},
            # the argument is not available in the Trainer strategy, but it's the default anyways
            # state_dict_type="full",
            limit_all_gathers=True,
            cpu_offload=False,
        )
    else:
        strategy = "auto"

    speed_monitor = SpeedMonitorCallback(
        length_fn=lambda batch: batch[0].size(1),
        batch_size=micro_batch_size,
        window_size=50,
        time_unit="seconds",
    )
    model_checkpoint = ModelCheckpoint(
        dirpath=out_dir, every_n_train_steps=save_interval, save_last=True, verbose=True
    )
    wandb_logger = WandbLogger(log_model=True)

    trainer = L.Trainer(
        devices=devices,
        strategy=strategy,
        precision=precision,
        logger=wandb_logger,
        callbacks=[speed_monitor, model_checkpoint],
        max_steps=max_iters,
        max_epochs=1,
        limit_val_batches=eval_iters,
        accumulate_grad_batches=gradient_accumulation_steps,
        log_every_n_steps=log_interval,
        val_check_interval=eval_interval,
    )

    L.seed_everything(
        1337, workers=True
    )  # same seed for every process to init model (FSDP)

    trainer.print(hparams)

    if trainer.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    config = Config.from_name(model_name)
    trainer.print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()
    model = LightningGPTModule(config)
    trainer.print(
        f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds."
    )

    train_data = Dataset(str(data_dir / "train.bin"), config.block_size)
    val_data = Dataset(str(data_dir / "val.bin"), config.block_size)
    train_dataloader = DataLoader(
        train_data, batch_size=micro_batch_size, num_workers=8
    )
    val_dataloader = DataLoader(val_data, batch_size=micro_batch_size, num_workers=8)

    t0 = time.perf_counter()
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path="last")
    trainer.print(f"Training time: {(time.perf_counter()-t0):.2f}s")
    if trainer.strategy.root_device.type == "cuda":
        trainer.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(main)
