"""
Training entrypoint for the Satellite AOI Downscaling diffusion head.

Usage:
    python -m training.train_downscaler --config-name diffusion_downscaler

Standard DDPM epsilon-prediction training: sample a random timestep, add
noise to the hi-res target field, predict the noise conditioned on the
terrain raster + coarse-forecast tokens, minimize MSE against the true
noise. See docs/TRAINING.md for the data-pairing strategy, and
scripts/build_aoi_pairs_manifest.py to build the manifest this script reads.

Requires requirements-train.txt (hydra/pytorch_lightning/omegaconf) — the
dataset itself (training/data.py) only needs the base requirements.txt.
"""
from __future__ import annotations

import hydra
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from app.models.diffusion_unet import DiffusionDownscaler
from training.data import AOIPairDataset, cosine_noise_schedule


class DownscalerLightningModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.model = DiffusionDownscaler(**cfg.model)
        self.register_buffer("alphas_cumprod", cosine_noise_schedule(cfg.diffusion.train_timesteps))

    def training_step(self, batch, batch_idx):
        coarse, terrain, tokens, target = batch
        b = target.shape[0]
        t = torch.randint(0, self.hparams.diffusion.train_timesteps, (b,), device=target.device)
        noise = torch.randn_like(target)

        a_bar = self.alphas_cumprod[t].view(b, 1, 1, 1)
        x_t = a_bar.sqrt() * target + (1 - a_bar).sqrt() * noise

        eps_pred = self.model(x_t, terrain, tokens, t)
        loss = F.mse_loss(eps_pred, noise)
        self.log("train/mse", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt_cfg = self.hparams.optim
        optim = torch.optim.AdamW(self.model.parameters(), lr=opt_cfg.lr, weight_decay=opt_cfg.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=self.hparams.trainer.max_epochs)
        return {"optimizer": optim, "lr_scheduler": sched}


@hydra.main(version_base=None, config_path="configs", config_name="diffusion_downscaler")
def main(cfg: DictConfig) -> None:
    module = DownscalerLightningModule(cfg)
    train_ds = AOIPairDataset(manifest_path=cfg.data.manifest_path, patch_size=cfg.data.aoi_patch_size)
    train_loader = DataLoader(train_ds, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers)

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.precision,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
    )
    trainer.fit(module, train_loader)
    torch.save(module.model.state_dict(), "checkpoints/diffusion_downscaler_v1.pt")


if __name__ == "__main__":
    main()
