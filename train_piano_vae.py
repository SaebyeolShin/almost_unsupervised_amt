import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import F1Score, Precision, Recall, Accuracy
from transformers import get_cosine_schedule_with_warmup
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from PIL import Image
from ema_pytorch import EMA

from midi_autoencoder.lucidrains_ae import UNetStyleVAE


# ── Dataset ─────────────────────────────────────────────────────────────────
class RollDataset(Dataset):
    def __init__(self, paths):
        self.paths = [Path(p) for p in paths]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        arr = np.load(self.paths[index], allow_pickle=False)
        if arr.ndim == 2:
            arr = arr[None]                          # (256, 88) → (1, 256, 88)
        arr = (arr > 0).astype(np.float32) * 2 - 1  # {0,1} → {-1,1}
        return torch.from_numpy(arr)

# ── Lightning Module ────────────────────────────────────────────────────────
class MidiVAELightning(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.model = UNetStyleVAE(
            in_channels=cfg.model.in_channels,
            out_channels=cfg.model.out_channels,
            ch=cfg.model.ch,
            ch_mult=tuple(cfg.model.ch_mult),
            ch_mult_1d=tuple(cfg.model.ch_mult_1d) if getattr(cfg.model, "ch_mult_1d", None) else None,
            num_res_blocks=cfg.model.num_res_blocks,
            z_channels=cfg.model.z_channels,
            kl_weight=cfg.model.kl_weight,
            binary_mode=cfg.model.binary_mode,
            attn_heads=getattr(cfg.model, "attn_heads", 4),
            attn_dim_head=getattr(cfg.model, "attn_dim_head", 32),
            loss_type=getattr(cfg.model, "loss_type", "bce"),
            focal_gamma=getattr(cfg.model, "focal_gamma", 2.0),
            focal_alpha=getattr(cfg.model, "focal_alpha", 0.25)
        )

        # EMA Setup
        self.ema_decay = cfg.training.ema_decay
        self.ema_model = EMA(
            self.model,
            beta=self.ema_decay,
            update_after_step=500,
            update_every=10
        )
        
        # Metrics - Online
        self.val_f1 = F1Score(task="binary")
        self.val_precision = Precision(task="binary")
        self.val_recall = Recall(task="binary")
        self.val_acc = Accuracy(task="binary")

        # Metrics - EMA
        self.val_f1_ema = F1Score(task="binary")
        self.val_precision_ema = Precision(task="binary")
        self.val_recall_ema = Recall(task="binary")
        self.val_acc_ema = Accuracy(task="binary")
        
    def forward(self, x):
        return self.model(x)
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.ema_model.update()

    def on_train_epoch_start(self):
        self.ema_model.eval()

    def training_step(self, batch, batch_idx):
        x = batch
        dec, recon_loss, kl_loss = self.model(x, return_loss=True)
        loss = recon_loss + self.model.kl_weight * kl_loss
        
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/recon_loss", recon_loss, prog_bar=True)
        self.log("train/kl_loss", kl_loss, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch
        
        dec, recon_loss, kl_loss = self.model(x, return_loss=True)
        loss = recon_loss + self.model.kl_weight * kl_loss

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/recon_loss", recon_loss)
        self.log("val/kl_loss", kl_loss)

        target = (x > 0).long()
        preds  = (dec > 0).long()

        self.val_f1(preds, target)
        self.val_precision(preds, target)
        self.val_recall(preds, target)
        self.val_acc(preds, target)

        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/precision", self.val_precision, on_step=False, on_epoch=True)
        self.log("val/recall", self.val_recall, on_step=False, on_epoch=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True)

        self.ema_model.eval()
        dec_ema, recon_loss_ema, kl_loss_ema = self.ema_model(x, return_loss=True)
        loss_ema = recon_loss_ema + self.model.kl_weight * kl_loss_ema

        self.log("val/ema_loss", loss_ema, prog_bar=True)
        self.log("val/ema_recon_loss", recon_loss_ema)
        self.log("val/ema_kl_loss", kl_loss_ema)

        preds_ema = (dec_ema > 0).long()

        self.val_f1_ema(preds_ema, target)
        self.val_precision_ema(preds_ema, target)
        self.val_recall_ema(preds_ema, target)
        self.val_acc_ema(preds_ema, target)

        self.log("val/ema_f1", self.val_f1_ema, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/ema_precision", self.val_precision_ema, on_step=False, on_epoch=True)
        self.log("val/ema_recall", self.val_recall_ema, on_step=False, on_epoch=True)
        self.log("val/ema_acc", self.val_acc_ema, on_step=False, on_epoch=True)
        
        if batch_idx == 0:
            # Log images for both
            self.log_images(x, dec, suffix="")
            self.log_images(x, dec_ema, suffix="_ema")
            
            # Get z for visualization
            mean, logvar = self.model.encode(x)
            z = self.model.reparameterize(mean, logvar)
            self.log_latent_pca(z, suffix="")

            # For EMA, access the underlying model via .ema_model
            mean_ema, logvar_ema = self.ema_model.ema_model.encode(x)
            z_ema = self.ema_model.ema_model.reparameterize(mean_ema, logvar_ema)
            self.log_latent_pca(z_ema, suffix="_ema")
            
        return loss
        
    def log_images(self, real, fake, suffix=""):
        images_list = []
        for i in range(real.size(0)):
            r = (real[i].float().cpu().clamp(-1, 1) + 1) / 2  # [-1,1] → [0,1]
            f = (fake[i].float().cpu().clamp(-1, 1) + 1) / 2
            comp = np.concatenate([r.numpy()[0], f.numpy()[0]], axis=1)
            images_list.append(comp)
        if isinstance(self.logger, WandbLogger):
            self.logger.log_image(
                key=f"val/reconstruction{suffix}",
                images=images_list,
                caption=[f"Real (left) | Recon (right) — sample {i}" for i in range(len(images_list))],
            )

    def log_latent_pca(self, z, suffix=""):
        from sklearn.decomposition import PCA
        
        # z: (B, C, H, W)
        if z.ndim == 3:
            # (B, C, T) -> (B, C, T, 1)
            z = z.unsqueeze(-1)
            
        B, C, H, W = z.shape
        
        # Move to numpy, (B, H, W, C)
        z_np = z.detach().float().cpu().permute(0, 2, 3, 1).numpy()
        
        # Flatten to (N_pixels_total, C) where N_pixels_total = B * H * W
        flat_z = z_np.reshape(-1, C)
        
        # PCA to 3 components
        n_components = min(3, C)
        pca = PCA(n_components=n_components)
        # Fit on the entire batch for consistent coloring
        rgb_flat = pca.fit_transform(flat_z)
        
        # Normalize to [0, 1]
        rgb_min = rgb_flat.min(axis=0)
        rgb_max = rgb_flat.max(axis=0)
        rgb_flat = (rgb_flat - rgb_min) / (rgb_max - rgb_min + 1e-6)
        
        # Pad if needed
        if n_components < 3:
            padding = np.zeros((rgb_flat.shape[0], 3 - n_components))
            rgb_flat = np.concatenate([rgb_flat, padding], axis=1)
            
        # Reshape back to (B, H, W, 3)
        rgb_imgs_np = rgb_flat.reshape(B, H, W, 3)
        
        images_list = []
        for i in range(B):
            img = rgb_imgs_np[i] # (H, W, 3)
            
            # Resize for better visibility using PIL
            img_pil = Image.fromarray((img * 255).astype(np.uint8))
            img_pil = img_pil.resize((W * 4, H * 4), resample=Image.NEAREST)
            images_list.append(img_pil)
        
        # Log
        if isinstance(self.logger, WandbLogger):
            self.logger.log_image(
                key=f"val/latent_pca{suffix}", 
                images=images_list, 
                caption=[f"Sample {i}: Latent PCA" for i in range(len(images_list))]
            )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.cfg.training.learning_rate
        )
        
        # Cosine Decay with Linear Warmup
        if self.trainer.max_epochs:
            total_steps = self.trainer.estimated_stepping_batches
            warmup_steps = 1000
            
            
            
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=warmup_steps, 
                num_training_steps=total_steps
            )
            
            scheduler_config = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
            
            return [optimizer], [scheduler_config]
            
        return optimizer

# ── Main ────────────────────────────────────────────────────────────────────
@hydra.main(version_base=None, config_path="configs", config_name="vae_piano")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    pl.seed_everything(cfg.seed)
    
    if not cfg.get("run_name"):
        raise ValueError("run_name is required! Please pass it as run_name=...")

    if not os.path.exists(cfg.data.roll_dir):
        raise FileNotFoundError(f"Roll directory not found: {cfg.data.roll_dir}")

    out_dir = os.path.join(os.path.abspath(cfg.logging.save_dir), cfg.run_name)
    if os.path.exists(out_dir):
        raise FileExistsError(f"Run directory {out_dir} already exists! Please use a different run_name.")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    ckpt_dir = os.path.join(out_dir, "checkpoints")
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(ckpt_dir, "config.yaml"))

    data_root = Path(cfg.data.roll_dir)

    train_dir = data_root / "train" / "midi"
    val_dir   = data_root / "validation" / "midi"

    if not train_dir.exists():
        raise FileNotFoundError(f"[main] train_dir not found: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"[main] val_dir not found: {val_dir}")

    train_files = sorted(train_dir.glob("*.npy"))
    val_files   = sorted(val_dir.glob("*.npy"))

    if len(train_files) == 0:
        raise FileNotFoundError(f"[main] No .npy files found in: {train_dir}")
    if len(val_files) == 0:
        raise FileNotFoundError(f"[main] No .npy files found in: {val_dir}")

    print(f"[main] Train size: {len(train_files)}")
    print(f"[main] Val size:   {len(val_files)}")

    if cfg.debug:
        print("[main] Debug mode: setting num_workers=0")
        cfg.data.num_workers = 0

    train_ds = RollDataset(train_files)
    val_ds   = RollDataset(val_files)

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
    
    model = MidiVAELightning(cfg)

    wandb_logger = WandbLogger(
        name=cfg.run_name,
        project=cfg.logging.project,
        entity=cfg.logging.entity,
        offline=cfg.logging.offline,
        save_dir=out_dir,
        config=OmegaConf.to_container(cfg, resolve=True)
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor="val/loss",
        save_top_k=cfg.training.save_top_k,
        mode="min",
        filename="midi-vae-{epoch:02d}-{step:07d}"
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=cfg.training.log_every_n_steps,
        val_check_interval=cfg.training.val_check_interval,
        check_val_every_n_epoch=None,
        gradient_clip_val=cfg.training.gradient_clip_val,
        precision=cfg.training.precision
    )
    
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()