"""
Training entrypoint for the Global Planetary Engine (SFNO).

Usage:
    python -m training.train_global_engine --config-name sfno_base

Real training needs a real Zarr-backed IterableDataset producing
(x_t, cond, x_{t+1}, ..., x_{t+k}) rollout windows — `ERA5RolloutDataset`
below is written against that contract but reads through
`app/data/zarr_pipeline.ZarrDataLake`, so it works the moment a real
`zarr_store_uri` is configured. See docs/TRAINING.md for the full,
step-by-step "how do I actually run this" guide (compute, data access,
timeline, cost).
"""
from __future__ import annotations

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, IterableDataset

from app.data.zarr_pipeline import ZarrDataLake
from app.models.sfno import SFNO
from training.losses import GlobalEngineLoss


class ERA5RolloutDataset(IterableDataset):
    """Yields (x0, cond, targets) windows of `rollout_steps` consecutive
    6-hourly analysis states. Each worker opens its own lazy Zarr handle
    (safe for Dask-backed multi-worker DataLoader)."""

    def __init__(self, store_uri: str, years: tuple[int, int], rollout_steps: int):
        self.store_uri = store_uri
        self.years = years
        self.rollout_steps = rollout_steps

    def __iter__(self):
        lake = ZarrDataLake(self.store_uri)
        ds = lake.open().sel(time=slice(f"{self.years[0]}-01-01", f"{self.years[1]}-12-31"))
        n_times = ds.sizes["time"]
        for t in range(n_times - self.rollout_steps - 1):
            window = ds.isel(time=slice(t, t + self.rollout_steps + 1))
            x0 = torch.tensor(window.isel(time=0).to_array().values, dtype=torch.float32)
            targets = torch.stack(
                [torch.tensor(window.isel(time=i).to_array().values, dtype=torch.float32)
                 for i in range(1, self.rollout_steps + 1)]
            )
            cond = torch.zeros(8, dtype=torch.float32)  # GHG/ENSO/MJO scalars, see zarr_pipeline
            yield x0, cond, targets


class GlobalEngineLightningModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.model = SFNO(**cfg.model)
        self.loss_fn = GlobalEngineLoss(
            lambda1=cfg.loss.lambda1_rmse,
            lambda2=cfg.loss.lambda2_spectral,
            lambda3=cfg.loss.lambda3_perceptual,
            lambda4=cfg.loss.lambda4_physics,
        )
        self.rollout_steps = 1  # overwritten per curriculum stage by the training loop

    def training_step(self, batch, batch_idx):
        x0, cond, targets = batch
        state = x0
        total_loss = 0.0
        for k in range(self.rollout_steps):
            state = self.model(state, cond)
            losses = self.loss_fn(state, targets[:, k])
            total_loss = total_loss + losses["total"]
            self.log_dict({f"train/{k_}": v for k_, v in losses.items()}, prog_bar=(k == 0))
        return total_loss / self.rollout_steps

    def configure_optimizers(self):
        opt_cfg = self.hparams.optim
        optim = torch.optim.AdamW(
            self.model.parameters(), lr=opt_cfg.lr, weight_decay=opt_cfg.weight_decay, betas=tuple(opt_cfg.betas)
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=self.hparams.trainer.max_epochs)
        return {"optimizer": optim, "lr_scheduler": sched}


@hydra.main(version_base=None, config_path="configs", config_name="sfno_base")
def main(cfg: DictConfig) -> None:
    module = GlobalEngineLightningModule(cfg)

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.precision,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        val_check_interval=cfg.trainer.val_check_interval,
    )

    for stage in cfg.curriculum.stages:
        module.rollout_steps = stage.rollout_steps
        train_ds = ERA5RolloutDataset(cfg.data.zarr_store_uri, tuple(cfg.data.train_years), stage.rollout_steps)
        train_loader = DataLoader(train_ds, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers)
        trainer.fit_loop.max_epochs = trainer.current_epoch + stage.epochs
        trainer.fit(module, train_loader)

    torch.save(module.model.state_dict(), "checkpoints/sfno_global_v1.pt")


if __name__ == "__main__":
    main()
