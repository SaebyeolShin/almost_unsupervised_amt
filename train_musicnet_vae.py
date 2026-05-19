import os
import random
from collections import defaultdict
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import F1Score
import torch.nn as nn
from pathlib import Path
from ema_pytorch import EMA
import hydra
from omegaconf import DictConfig, OmegaConf

from midi_autoencoder.lucidrains_ae import UNetStyleVAE

# ── Fixed test IDs — always excluded from training ────────────────────────────
FIXED_TEST_IDS = {
    '1759', '1819', '2106', '2191', '2298',
    '2303', '2382', '2416', '2556', '2628',
}


# ── Dataset ───────────────────────────────────────────────────────────────────

class MusicNetEMRollDataset(Dataset):
    """
    Loads MusicNet EM MIDI chunks of shape (C, T, 88).
    Returns (C, T, 88) tensor in [-1, 1].
    """
    def __init__(self, chunk_paths: list, num_instruments: int):
        self.paths = [Path(p) for p in chunk_paths]
        self.C = num_instruments

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        arr = np.load(self.paths[idx], allow_pickle=False)  # (C, T, 88)
        if arr.ndim == 2:
            arr = arr[np.newaxis]                            # (1, T, 88) edge case
        arr = arr[:self.C].astype(np.float32)
        arr = (arr > 0).astype(np.float32) * 2.0 - 1.0     # {0,1} → [-1,1]
        return torch.from_numpy(arr)


def build_split(preprocessed_root: Path, num_instruments: int,
                val_fraction: float, seed: int):
    """
    Returns (train_paths, val_paths) — song-level split,
    excluding FIXED_TEST_IDS.
    """
    midi_dir = preprocessed_root / "midi"
    all_chunks = sorted(midi_dir.glob("*_chunk*.npy"))

    # group by song ID
    song_chunks: dict = defaultdict(list)
    for p in all_chunks:
        sid = p.name.rsplit("_chunk", 1)[0]
        if sid not in FIXED_TEST_IDS:
            song_chunks[sid].append(p)

    song_ids = sorted(song_chunks.keys())
    rng = random.Random(seed)
    rng.shuffle(song_ids)

    n_val = max(1, int(len(song_ids) * val_fraction))
    val_ids = set(song_ids[:n_val])
    train_ids = set(song_ids[n_val:])

    train_paths = [p for sid in train_ids for p in song_chunks[sid]]
    val_paths   = [p for sid in val_ids   for p in song_chunks[sid]]

    print(f"[build_split] train songs={len(train_ids)} ({len(train_paths)} chunks)  "
          f"val songs={len(val_ids)} ({len(val_paths)} chunks)")
    return train_paths, val_paths


# ── Lightning Module ──────────────────────────────────────────────────────────

class MusicNetVAELightning(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        C = cfg.model.num_instruments

        self.model = UNetStyleVAE(
            in_channels=C,
            out_channels=C,
            ch=cfg.model.ch,
            ch_mult=tuple(cfg.model.ch_mult),
            num_res_blocks=cfg.model.num_res_blocks,
            z_channels=cfg.model.z_channels,
            kl_weight=cfg.model.kl_weight,
            binary_mode=cfg.model.binary_mode,
            attn_heads=cfg.model.attn_heads,
            attn_dim_head=cfg.model.attn_dim_head,
            loss_type=cfg.model.loss_type,
            focal_gamma=cfg.model.focal_gamma,
            focal_alpha=cfg.model.focal_alpha,
        )

        self.ema_model = EMA(
            self.model,
            beta=cfg.training.ema_decay,
            update_after_step=500,
            update_every=10,
        )

        self.inst_names = ['Piano','Harpsi','Violin','Viola','Cello',
                           'Pizz','FrHorn','Oboe','Bassoon','Clarinet','Flute']

        self.val_f1     = F1Score(task="binary")
        self.val_f1_ema = F1Score(task="binary")
        self.val_f1_per_inst     = nn.ModuleList([F1Score(task="binary") for _ in range(C)])
        self.val_f1_per_inst_ema = nn.ModuleList([F1Score(task="binary") for _ in range(C)])

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.ema_model.update()

    def training_step(self, batch, batch_idx):
        x = batch                                             # (B, C, T, 88)
        dec, recon_loss, kl_loss = self.model(x, return_loss=True)
        loss = recon_loss + self.model.kl_weight * kl_loss
        self.log("train/loss",       loss,       prog_bar=True)
        self.log("train/recon_loss", recon_loss)
        self.log("train/kl_loss",    kl_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch                                             # (B, C, T, 88)
        target = (x > 0).long()

        # online
        dec, recon_loss, kl_loss = self.model(x, return_loss=True)
        loss = recon_loss + self.model.kl_weight * kl_loss
        self.log("val/loss",       loss,       prog_bar=True, sync_dist=True)
        self.log("val/recon_loss", recon_loss, sync_dist=True)
        self.log("val/kl_loss",    kl_loss,    sync_dist=True)
        pred = (dec > 0).long()
        self.val_f1(pred, target)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        for c, (metric, name) in enumerate(zip(self.val_f1_per_inst, self.inst_names)):
            metric(pred[:, c], target[:, c])
            self.log(f"val/f1_{name}", metric, on_step=False, on_epoch=True)

        # EMA
        self.ema_model.eval()
        dec_ema, recon_loss_ema, kl_loss_ema = self.ema_model(x, return_loss=True)
        loss_ema = recon_loss_ema + self.model.kl_weight * kl_loss_ema
        self.log("val/ema_loss",    loss_ema,    prog_bar=True, sync_dist=True)
        pred_ema = (dec_ema > 0).long()
        self.val_f1_ema(pred_ema, target)
        self.log("val/ema_f1", self.val_f1_ema, on_step=False, on_epoch=True, prog_bar=True)
        for c, (metric, name) in enumerate(zip(self.val_f1_per_inst_ema, self.inst_names)):
            metric(pred_ema[:, c], target[:, c])
            self.log(f"val/ema_f1_{name}", metric, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.training.learning_rate,
        )
        scheduler = CosineAnnealingLR(
            opt,
            T_max=self.cfg.training.max_epochs,
            eta_min=self.cfg.training.learning_rate * 0.01,
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


# ── Main ──────────────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="configs", config_name="vae_musicnet")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(cfg.seed)

    if not cfg.get("run_name"):
        raise ValueError("run_name is required!")

    staged_root = Path(cfg.data.preprocessed_root)

    out_dir = os.path.join(os.path.abspath(cfg.logging.save_dir), cfg.run_name)
    if os.path.exists(out_dir):
        raise FileExistsError(f"Run dir {out_dir} already exists!")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    ckpt_dir = os.path.join(out_dir, "checkpoints")
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(ckpt_dir, "config.yaml"))

    # Data
    train_paths, val_paths = build_split(
        staged_root,
        num_instruments=cfg.data.num_instruments,
        val_fraction=cfg.data.val_fraction,
        seed=cfg.seed,
    )

    C = cfg.data.num_instruments
    train_ds = MusicNetEMRollDataset(train_paths, C)
    val_ds   = MusicNetEMRollDataset(val_paths,   C)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=True,
        persistent_workers=(cfg.data.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=False,
        persistent_workers=(cfg.data.num_workers > 0),
    )

    model = MusicNetVAELightning(cfg)

    wandb_logger = WandbLogger(
        name=cfg.run_name,
        project=cfg.logging.project,
        entity=cfg.logging.entity,
        offline=cfg.logging.offline,
        save_dir=out_dir,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            monitor="val/ema_loss",
            save_top_k=cfg.training.save_top_k,
            mode="min",
            filename="musicnet-vae-{epoch:02d}-{step:07d}",
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        precision=cfg.training.precision,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=cfg.training.log_every_n_steps,
        val_check_interval=cfg.training.val_check_interval,
        gradient_clip_val=cfg.training.gradient_clip_val,
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
