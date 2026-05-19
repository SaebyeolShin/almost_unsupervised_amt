from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import os
from torch.utils.data import Dataset, DataLoader, Sampler
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from PIL import Image
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import get_cosine_schedule_with_warmup
from ema_pytorch import EMA
from models.generator import CqtToLatentGenerator, LatentToCqtGenerator

from utils import (
    load_vae_from_checkpoint, load_cqt_npy, list_split_files,
    piano_metrics_from_rolls, ImagePool,
    FPS, NOTE_MIN, NOTE_MAX,
)
from models.discriminator import MultiScaleDiscriminator, GANLoss
from pytorch_lightning.utilities.combined_loader import CombinedLoader
import random

def split_train_val(train_cqt, train_midi, val_fraction: float, seed: int):
    """
    Returns:
      train_cqt2, train_midi2, val_cqt, val_midi, train_idx, val_idx
    """
    assert len(train_cqt) == len(train_midi)
    n = len(train_cqt)
    if n < 2:
        raise RuntimeError(f"Not enough training pairs to split: n={n}")

    # ensure at least one sample in each split
    k = int(round(n * float(val_fraction)))
    k = max(1, min(k, n - 1))

    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    val_idx = np.sort(perm[:k])
    train_idx = np.sort(perm[k:])

    train_cqt2 = [train_cqt[i] for i in train_idx]
    train_midi2 = [train_midi[i] for i in train_idx]
    val_cqt = [train_cqt[i] for i in val_idx]
    val_midi = [train_midi[i] for i in val_idx]

    print(f"[split_train_val] n={n} -> train={len(train_cqt2)}, val={len(val_cqt)} (val_fraction={val_fraction})")
    return train_cqt2, train_midi2, val_cqt, val_midi, train_idx, val_idx

def resolve_splits(root, seed: int, val_fraction: float = 0.05):
    """
    Returns:
      (train_cqt, train_midi), (val_cqt, val_midi), (test_cqt, test_midi), split_info
    split_info:
      - train_idx / val_idx for optional index export
      - test_from: "test" | "validation" | None
    """
    train = list_split_files(root, "train", strict=True)
    val = list_split_files(root, "validation", strict=False)
    test = list_split_files(root, "test", strict=False)

    train_cqt, train_midi = train
    val_cqt, val_midi = val
    test_cqt, test_midi = test

    split_info = {"train_idx": None, "val_idx": None, "test_from": None}

    # case A) train/validation only -> use validation as test, carve new val from train
    if (val_cqt is not None) and (test_cqt is None):
        print("[resolve_splits] No test split -> using validation as TEST.")
        test_cqt, test_midi = val_cqt, val_midi
        split_info["test_from"] = "validation"

        print("[resolve_splits] Creating new VALIDATION from TRAIN.")
        train_cqt, train_midi, val_cqt, val_midi, tr_idx, va_idx = split_train_val(
            train_cqt, train_midi, val_fraction=val_fraction, seed=seed
        )
        split_info["train_idx"] = tr_idx
        split_info["val_idx"] = va_idx

    # case B) train/test only -> carve val from train
    elif (val_cqt is None) and (test_cqt is not None):
        print("[resolve_splits] No validation split -> creating VALIDATION from TRAIN.")
        train_cqt, train_midi, val_cqt, val_midi, tr_idx, va_idx = split_train_val(
            train_cqt, train_midi, val_fraction=val_fraction, seed=seed
        )
        split_info["train_idx"] = tr_idx
        split_info["val_idx"] = va_idx
        split_info["test_from"] = "test"

    # case C) train only -> cannot proceed without test
    elif (val_cqt is None) and (test_cqt is None):
        raise RuntimeError(
            "[resolve_splits] Only train split found. Policy requires TEST. "
            "Provide validation/ (will be used as test) or test/ folder."
        )

    # case D) all three splits present -> use as-is
    else:
        print("[resolve_splits] Found train/validation/test splits -> using as-is.")
        split_info["test_from"] = "test"

    return (train_cqt, train_midi), (val_cqt, val_midi), (test_cqt, test_midi), split_info

def split_paired_val(cqt_paths, midi_paths, n_val_songs: int, seed: int):
    """
    Split paired chunks into train/val by song ID.
    Song ID is the stem prefix before '_chunk' (e.g. '1234' from '1234_chunk000001.npy').
    Returns (train_cqt, train_midi, val_cqt, val_midi).
    """
    assert len(cqt_paths) == len(midi_paths)
    song_ids = sorted({Path(p).name.rsplit("_chunk", 1)[0] for p in cqt_paths})
    rng = np.random.RandomState(seed)
    val_songs = set(rng.choice(song_ids, size=min(n_val_songs, len(song_ids)), replace=False))

    train_cqt, train_midi, val_cqt, val_midi = [], [], [], []
    for cqt, midi in zip(cqt_paths, midi_paths):
        sid = Path(cqt).name.rsplit("_chunk", 1)[0]
        if sid in val_songs:
            val_cqt.append(cqt);   val_midi.append(midi)
        else:
            train_cqt.append(cqt); train_midi.append(midi)

    print(f"[split_paired_val] val_songs={len(val_songs)}  "
          f"train_chunks={len(train_cqt)}  val_chunks={len(val_cqt)}")
    return train_cqt, train_midi, val_cqt, val_midi

# Instrument mapping (matches preprocess_musicnet.py / preprocess_musicnet_em.py)
UNIQ_PROGRAMS = [1, 7, 41, 42, 43, 44, 61, 69, 71, 72, 74]
K_INST        = len(UNIQ_PROGRAMS)           # 11
PIANO_CHANNEL = UNIQ_PROGRAMS.index(1)       # 0  (program 1 = Acoustic Grand Piano)

# ======================================================
# Dataset: Unaligned CQT <-> Roll
# ======================================================
class UnalignedCqtRollDataset(Dataset):
    """
    cqt : CQT npy (dB scale, [-80, 0])
    midi : roll npy (binary {0,1})

    - cqt(CQT): dB → [0,1] → [-1,1]
    - midi(roll): {0,1} → [-1,1]
    """
    def __init__(self, cqt_paths, midi_paths, db_min: float = -80.0, db_max: float = 0.0):
        super().__init__()
        self.cqt_paths = [Path(p) for p in cqt_paths]  # CQT
        self.midi_paths = [Path(p) for p in midi_paths]  # roll

        self.cqt_size = len(self.cqt_paths)
        self.midi_size = len(self.midi_paths)

        self.db_min = float(db_min)
        self.db_max = float(db_max)

        print(f"[Dataset] CQT:  {self.cqt_size} files")
        print(f"[Dataset] MIDI: {self.midi_size} files")
        if self.cqt_size == 0 or self.midi_size == 0:
            raise RuntimeError("no .npy file in one of CQT / MIDI.")

    def __len__(self):
        return max(self.cqt_size, self.midi_size)

    def _load_cqt(self, path: Path):
        return load_cqt_npy(path, self.db_min, self.db_max)

    def _load_roll(self, path: Path):
        arr = np.load(path, allow_pickle=False, mmap_mode=None)
        if arr.ndim == 2:
            # Piano-only format (e.g. MAESTRO): expand to (K, T, 88) at channel 0 (piano)
            T, P = arr.shape
            full = np.zeros((K_INST, T, P), dtype=np.float32)
            full[PIANO_CHANNEL] = arr
            arr = full
        elif arr.ndim != 3:
            raise ValueError(f"Unexpected roll shape {arr.shape} from {path}")

        arr = arr.astype("float32", copy=False)
        bin_arr = (arr > 0).astype("float32")
        x11 = bin_arr * 2.0 - 1.0
        return torch.from_numpy(np.ascontiguousarray(x11))

    def __getitem__(self, index):
        cqt_path = self.cqt_paths[index % self.cqt_size]
        midi_index = np.random.randint(0, self.midi_size)
        midi_path = self.midi_paths[midi_index]

        cqt = self._load_cqt(cqt_path)
        midi = self._load_roll(midi_path)

        return {
            "cqt": cqt,
            "midi": midi,
            "cqt_path": str(cqt_path),
            "midi_path": str(midi_path),
        }
        
class PairedCqtRollDataset(Dataset):
    """
    Dataset for paired CQT and MIDI files (by index).
    Used for validation to ensure we evaluate transcription metrics on correct pairs.
    """
    def __init__(self, cqt_paths, midi_paths, db_min=-80.0, db_max=0.0):
        super().__init__()
        # Assumes cqt_paths[i] corresponds to midi_paths[i]
        self.cqt_paths = [Path(p) for p in cqt_paths]
        self.midi_paths = [Path(p) for p in midi_paths]
        assert len(self.cqt_paths) == len(self.midi_paths), "Paths must be same length"
        
        
        self.db_min = float(db_min)
        self.db_max = float(db_max)
        self.active_mask = []  # list of (C,) bool
        for p in self.midi_paths:
            arr = np.load(p, allow_pickle=False, mmap_mode="r")   # (C,T,88)
            act = (arr > 0).any(axis=(1,2))                      # (C,)
            self.active_mask.append(act)
            
    def get_active_mask(self, idx: int) -> np.ndarray:
        return self.active_mask[idx]

    def __len__(self):
        return len(self.cqt_paths)

    def _load_cqt(self, path: Path):
        return load_cqt_npy(path, self.db_min, self.db_max)

    def _load_midi(self, path: Path):
        arr = np.load(path, allow_pickle=False, mmap_mode=None)
        if arr.ndim == 2:
            T, P = arr.shape
            full = np.zeros((K_INST, T, P), dtype=np.float32)
            full[PIANO_CHANNEL] = arr
            arr = full
        elif arr.ndim != 3:
            raise ValueError(f"Unexpected roll shape {arr.shape} from {path}")
        arr = arr.astype("float32", copy=False)
        bin_arr = (arr > 0).astype("float32")
        x11 = bin_arr * 2.0 - 1.0
        return torch.from_numpy(x11)

    def __getitem__(self, idx):
        cqt_path = self.cqt_paths[idx]
        midi_path = self.midi_paths[idx]

        cqt = self._load_cqt(cqt_path)
        midi = self._load_midi(midi_path)

        return {
            "cqt": cqt,
            "midi": midi,
            "cqt_path": str(cqt_path),
            "midi_path": str(midi_path),
        }

class InstBalancedBatchSampler(Sampler):
    def __init__(self, dataset: PairedCqtRollDataset, batch_size: int, seed: int = 0, drop_last: bool = True):
        self.ds = dataset
        self.bs = int(batch_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)

        # build per-instrument pools of active sample indices
        C = len(self.ds.active_mask[0])
        pools = [[] for _ in range(C)]
        for i in range(len(self.ds)):
            act = self.ds.get_active_mask(i)
            for c in range(C):
                if act[c]:
                    pools[c].append(i)

        self.pools = pools
        self.C = C
        self.epoch = 0

        # fallback: if an instrument has no active samples, use the full index pool
        self.all_idx = list(range(len(self.ds)))
        for c in range(C):
            if len(self.pools[c]) == 0:
                self.pools[c] = self.all_idx
    
    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

        rng = random.Random(self.seed + self.epoch)

        # prepare a shuffled iterator per instrument
        iters = []
        for c in range(self.C):
            lst = self.pools[c][:]
            rng.shuffle(lst)
            iters.append(lst)

        ptr = [0] * self.C
        n = len(self.ds)
        n_drop = (n // self.bs) * self.bs if self.drop_last else n
        yielded = 0
        batch_idx = 0

        while True:
            batch = []

            if self.C > self.bs:
                inst_order = list(range(self.C))
                rng.shuffle(inst_order)
                inst_order = inst_order[:self.bs]
            else:
                inst_order = range(self.C)

            # one sample per selected instrument
            for c in inst_order:
                if len(batch) >= self.bs:
                    break
                lst = iters[c]
                if ptr[c] >= len(lst):
                    rng.shuffle(lst)
                    ptr[c] = 0
                batch.append(lst[ptr[c]])
                ptr[c] += 1

            # fill remaining slots randomly
            while len(batch) < self.bs:
                batch.append(rng.choice(self.all_idx))

            if batch_idx % world_size == rank:
                yield batch
            batch_idx += 1
            yielded += len(batch)
            if yielded >= n_drop:
                break


    def __len__(self):
        import torch.distributed as dist
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        total = len(self.ds) // self.bs if self.drop_last else (len(self.ds) + self.bs - 1) // self.bs
        return max(1, total // world_size)


class CqtRollCycleLightning(pl.LightningModule):
    def __init__(self, cfg, image_dir, num_instruments: int):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.automatic_optimization = False
        self.num_instruments = int(num_instruments)
        
        # GAN warmup tracking
        self.gan_warmup_steps = cfg.training.get("gan_warmup_steps", 0)

        # Load VAE
        if not cfg.model.vae_ckpt:
            raise ValueError("vae_ckpt must be provided in config")
        self.vae = load_vae_from_checkpoint(cfg.model.vae_ckpt, use_ema=cfg.model.use_ema)

        self.z_channels = int(self.vae.z_channels)

        # Generators
        c2m_cfg = cfg.model.gen_cqt_to_midi
        self.gen_cqt_to_midi = CqtToLatentGenerator(
            ngf=c2m_cfg.ngf,
            z_channels=self.z_channels,
            ch_mult=list(c2m_cfg.ch_mult),
            num_res_blocks=c2m_cfg.num_res_blocks,
            attn_heads=c2m_cfg.attn_heads,
            attn_dim_head=c2m_cfg.attn_dim_head,
            attn_start_level=c2m_cfg.get("attn_start_level", 2)
        )

        m2c_cfg = cfg.model.gen_midi_to_cqt
        self.gen_midi_to_cqt = LatentToCqtGenerator(
            ngf=m2c_cfg.ngf,
            z_channels=self.z_channels,
            ch_mult=list(m2c_cfg.ch_mult),
            num_res_blocks=m2c_cfg.num_res_blocks,
            attn_heads=m2c_cfg.attn_heads,
            attn_dim_head=m2c_cfg.attn_dim_head,
            attn_start_level=m2c_cfg.get("attn_start_level", 2)
        )

        # Discriminators
        # MIDI (Latent) Discriminator
        self.disc_midi = MultiScaleDiscriminator(
            input_nc=self.z_channels,
            ndf=cfg.model.disc_midi.ndf,
            n_layers=cfg.model.disc_midi.n_layers,
            scales=cfg.model.disc_midi.scales,
            kernel_size=tuple(cfg.model.disc_midi.kernel_size),
            stride=tuple(cfg.model.disc_midi.stride),
            padding=tuple(cfg.model.disc_midi.padding),
            use_residual=cfg.model.disc_midi.get("use_residual", False)
        )
        
        # CQT Discriminator
        self.disc_cqt = MultiScaleDiscriminator(
            input_nc=1, 
            ndf=cfg.model.disc_cqt.ndf, 
            n_layers=cfg.model.disc_cqt.n_layers,
            scales=cfg.model.disc_cqt.scales,
            kernel_size=tuple(cfg.model.disc_cqt.kernel_size),
            stride=tuple(cfg.model.disc_cqt.stride),
            padding=tuple(cfg.model.disc_cqt.padding),
            use_residual=cfg.model.disc_cqt.get("use_residual", False)
        )

        # Losses
        self.criterionGAN = GANLoss(use_fm_loss=cfg.model.use_fm_loss)
        self.criterionCycle = nn.L1Loss()

        # Image pools (for D)
        self.pool_cqt = ImagePool(pool_size=cfg.model.pool_size)
        self.pool_midi = ImagePool(pool_size=cfg.model.pool_size)
        
        # EMA
        self.use_ema = cfg.model.get("use_ema", False)
        if self.use_ema:
            print(f"[EMA] Initializing Generator EMA (beta={cfg.model.ema_beta})")
            self.ema_c2m = EMA(
                self.gen_cqt_to_midi,
                beta=cfg.model.ema_beta,
                update_after_step=cfg.model.ema_update_after_step,
                update_every=cfg.model.ema_update_every,
            )
            self.ema_m2c = EMA(
                self.gen_midi_to_cqt,
                beta=cfg.model.ema_beta,
                update_after_step=cfg.model.ema_update_after_step,
                update_every=cfg.model.ema_update_every,
            )

        # for visualization
        self.image_dir = Path(image_dir)
        self.save_val_vis = cfg.training.get("save_val_vis", True)
        self.save_test_vis = cfg.training.get("save_test_vis", True)
        self.save_ema_vis = cfg.training.get("save_ema_vis", True)

    def forward(self, real_cqt, real_midi_latent, use_gen_ema=False):
        if use_gen_ema and self.use_ema:
            gen_c2m = self.ema_c2m
            gen_m2c = self.ema_m2c
        else:
            gen_c2m = self.gen_cqt_to_midi
            gen_m2c = self.gen_midi_to_cqt

        fake_midi_z = gen_c2m(real_cqt)
        rec_cqt     = gen_m2c(fake_midi_z)
        fake_cqt    = gen_m2c(real_midi_latent)
        rec_midi_z  = gen_c2m(fake_cqt)

        return fake_midi_z, rec_cqt, fake_cqt, rec_midi_z

    def configure_optimizers(self):
        opt_G = torch.optim.AdamW(
            list(self.gen_cqt_to_midi.parameters()) + list(self.gen_midi_to_cqt.parameters()),
            lr=self.cfg.training.learning_rate,
            betas=(self.cfg.training.beta1, 0.99),
        )
        opt_D = torch.optim.AdamW(
            list(self.disc_midi.parameters()) + list(self.disc_cqt.parameters()),
            lr=self.cfg.training.learning_rate,
            betas=(self.cfg.training.beta1, 0.99),
        )

        total_steps = self.cfg.training.num_train_steps + int(self.cfg.training.get("extra_steps", 0))
        warmup_steps = self.cfg.training.get("warmup_steps", 1000)

        sched_G = get_cosine_schedule_with_warmup(
            opt_G,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        sched_D = get_cosine_schedule_with_warmup(
            opt_D,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        return (
            [opt_G, opt_D],
            [
                {"scheduler": sched_G, "interval": "step", "frequency": 1},
                {"scheduler": sched_D, "interval": "step", "frequency": 1},
            ],
        )

    def _compute_grad_norm(self, model):
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

    def _get_gan_weight(self):
        """
        Returns the GAN loss weight for Generator's adversarial loss.
        Linearly anneals from min_weight to 1 during warmup period.
        Note: This only affects G's adversarial loss; D trains at full strength from step 0.
        """
        if self.gan_warmup_steps <= 0:
            return 1.0

        min_weight = self.cfg.training.get("gan_warmup_min_weight", 0.1)
        progress = min(self.global_step / self.gan_warmup_steps, 1.0)
        # Linearly interpolate from min_weight to 1.0
        return min_weight + (1.0 - min_weight) * progress
    
    def _encode_midi_latent(self, midi_roll_bctp: torch.Tensor) -> torch.Tensor:
        """
        midi_roll_bctp: (B, C, T, 88) in [-1, 1]
        returns z: (B, z_channels, H, W)  — joint multi-instrument latent
        """
        if midi_roll_bctp.ndim != 4:
            raise ValueError(f"Expected (B,C,T,88), got {midi_roll_bctp.shape}")
        B, C, T, P = midi_roll_bctp.shape
        if C != self.num_instruments:
            raise ValueError(f"num_instruments mismatch: data C={C}, model={self.num_instruments}")

        with torch.no_grad():
            mean, logvar = self.vae.encode(midi_roll_bctp)   # (B, z_channels, H, W)
            z = self.vae.reparameterize(mean, logvar).detach()
        return z


    def _decode_midi_latent_to_roll(self, z_bchw: torch.Tensor) -> torch.Tensor:
        """(B, z_channels, H, W) → (B, C, T, 88) in [-1, 1]"""
        logits = self.vae.decode(z_bchw)
        return torch.sigmoid(logits) * 2.0 - 1.0
    
    def training_step(self, batch, batch_idx):
        # CombinedLoader yields {"unpaired": {...}, "paired": {...}}
        b_u = batch["unpaired"]
        b_p = batch["paired"]

        real_cqt = b_u["cqt"]
        real_midi_roll = b_u["midi"]
        real_midi_latent = self._encode_midi_latent(real_midi_roll)

        sup_cqt = b_p["cqt"]
        sup_midi_roll = b_p["midi"]
        gt01 = (sup_midi_roll > 0).float()          # (B,C,T,88)
        inst_on = gt01.sum(dim=(0,2,3))             # (C,)
        self.log_dict({f"train/sup_inst_on_{c:02d}": inst_on[c] for c in range(inst_on.numel())},
                    on_step=True, prog_bar=False, sync_dist=True)
        sup_midi_latent = self._encode_midi_latent(sup_midi_roll)
        
        assert real_midi_roll.ndim == 4, f"train midi shape wrong: {real_midi_roll.shape}"
        assert sup_midi_roll.ndim == 4, f"train sup midi shape wrong: {sup_midi_roll.shape}"

        lambda_cqt = self.cfg.model.lambda_cqt
        lambda_midi = self.cfg.model.lambda_midi
        
        acc_steps = self.cfg.training.accumulate_grad_batches
        d_steps_per_g_step = self.cfg.training.get("d_steps_per_g_step", 1)

        opt_G, opt_D = self.optimizers()
        scheds = self.lr_schedulers()
        if isinstance(scheds, (list, tuple)):
            sched_G, sched_D = scheds
        else:
            sched_G = scheds
            sched_D = None

        # Get GAN loss weight (annealed during warmup)
        gan_weight = self._get_gan_weight()

        # Determine if we should update G this step
        # G runs every d_steps_per_g_step batches
        update_G = (batch_idx % d_steps_per_g_step == 0)
        g_batch_count = batch_idx // d_steps_per_g_step

        # ------------------
        # 1) G update (only every d_steps_per_g_step)
        # ------------------
        
        if update_G:
            # Disable D gradients during G update to prevent leakage
            self.disc_midi.requires_grad_(False)
            self.disc_cqt.requires_grad_(False)

            fake_midi_z = self.gen_cqt_to_midi(real_cqt)
            rec_cqt     = self.gen_midi_to_cqt(fake_midi_z)
            fake_cqt    = self.gen_midi_to_cqt(real_midi_latent)
            rec_midi_z  = self.gen_cqt_to_midi(fake_cqt)

            # Adversarial loss (G side)
            fmaps_fake_midi = self.disc_midi(fake_midi_z)
            loss_adv_midi = self.criterionGAN.adversarial_loss(fmaps_fake_midi)

            # Cycle-aware feature matching (real vs cycle-reconstructed)
            fmaps_real_midi = self.disc_midi(real_midi_latent)
            fmaps_rec_midi  = self.disc_midi(rec_midi_z)
            loss_fm_midi = self.criterionGAN.feature_matching_loss(fmaps_real_midi, fmaps_rec_midi)
            loss_G_midi  = loss_adv_midi + loss_fm_midi * self.cfg.model.fm_ratio

            fmaps_fake_cqt = self.disc_cqt(fake_cqt)
            loss_adv_cqt = self.criterionGAN.adversarial_loss(fmaps_fake_cqt)

            fmaps_real_cqt = self.disc_cqt(real_cqt)
            fmaps_rec_cqt  = self.disc_cqt(rec_cqt)
            loss_fm_cqt = self.criterionGAN.feature_matching_loss(fmaps_real_cqt, fmaps_rec_cqt)
            loss_G_cqt  = loss_adv_cqt + loss_fm_cqt * self.cfg.model.fm_ratio

            # Cycle-consistency loss
            loss_cycle_cqt  = self.criterionCycle(rec_cqt, real_cqt) * lambda_cqt
            loss_cycle_midi = self.criterionCycle(rec_midi_z, real_midi_latent) * lambda_midi

            # Supervised loss (paired batch)
            lambda_sup_midi = float(self.cfg.training.get("lambda_sup_midi", 1.0))
            lambda_sup_cqt  = float(self.cfg.training.get("lambda_sup_cqt", 1.0))

            sup_pred_cqt    = self.gen_midi_to_cqt(sup_midi_latent)
            sup_fake_midi_z = self.gen_cqt_to_midi(sup_cqt)
            loss_sup_midi   = self.criterionCycle(sup_fake_midi_z, sup_midi_latent) * lambda_sup_midi
            loss_sup_cqt    = self.criterionCycle(sup_pred_cqt, sup_cqt) * lambda_sup_cqt

            self.log("train/loss_sup_midi", loss_sup_midi, prog_bar=False, on_step=True, sync_dist=True)
            self.log("train/loss_sup_cqt", loss_sup_cqt, on_step=True, prog_bar=False, sync_dist=True)

            # Total G loss with annealed GAN weight
            loss_G = gan_weight * (loss_G_midi + loss_G_cqt) + loss_cycle_cqt + loss_cycle_midi + loss_sup_midi + loss_sup_cqt

            # backward & step 
            self.manual_backward(loss_G / acc_steps)

            # Re-enable D gradients
            self.disc_midi.requires_grad_(True)
            self.disc_cqt.requires_grad_(True)
            
            if (g_batch_count + 1) % acc_steps == 0:
                # Log G gradient norm
                grad_norm_G = self._compute_grad_norm(self.gen_cqt_to_midi) + self._compute_grad_norm(self.gen_midi_to_cqt)
                self.log("train/grad_norm_G", grad_norm_G, prog_bar=False, on_step=True, sync_dist=True)
                
                # Clip G gradients
                self.clip_gradients(opt_G, gradient_clip_val=self.cfg.training.max_grad_norm, gradient_clip_algorithm="norm")
                
                opt_G.step()
                opt_G.zero_grad()
                if sched_G is not None:
                    sched_G.step()

            if (self.global_step + 1) % self.cfg.training.vis_freq == 0:
                self._save_cycle_images(
                    real_cqt, fake_midi_z, rec_cqt, real_midi_roll, fake_cqt, rec_midi_z,
                    tag="train"
                )
            # Log G losses
            self.log("train/loss_G", loss_G, prog_bar=True, on_step=True, sync_dist=True)
            self.log("train/loss_G_midi", loss_G_midi, prog_bar=False, on_step=True, sync_dist=True)
            self.log("train/loss_adv_midi", loss_adv_midi, prog_bar=False, on_step=True, sync_dist=True)
            self.log("train/loss_fm_midi", loss_fm_midi, prog_bar=False, on_step=True, sync_dist=True)
            
            self.log("train/loss_G_cqt", loss_G_cqt, prog_bar=False, on_step=True, sync_dist=True)
            self.log("train/loss_adv_cqt", loss_adv_cqt, prog_bar=False, on_step=True, sync_dist=True)
            self.log("train/loss_fm_cqt", loss_fm_cqt, prog_bar=False, on_step=True, sync_dist=True)

            self.log("train/loss_cycle_cqt", loss_cycle_cqt, prog_bar=False, on_step=True, sync_dist=True)
            self.log("train/loss_cycle_midi", loss_cycle_midi, prog_bar=False, on_step=True, sync_dist=True)
            self.log("train/gan_weight", gan_weight, prog_bar=False, on_step=True, sync_dist=True)

            
            # Update EMA
            if self.use_ema:
                self.ema_c2m.update()
                self.ema_m2c.update()
        else:
            # D-only step: generate fakes without tracking G gradients
            with torch.no_grad():
                fake_midi_z = self.gen_cqt_to_midi(real_cqt)
                fake_cqt    = self.gen_midi_to_cqt(real_midi_latent)

            loss_G = torch.tensor(0.0, device=real_cqt.device)

        # ------------------
        # 2) D update (full strength, no warmup scaling)
        # ------------------
        # Detach fakes for D update
        fake_midi_det = fake_midi_z.detach()
        fake_cqt_det = fake_cqt.detach()

        # Use ImagePool
        fake_midi_pool = self.pool_midi.query(fake_midi_det)
        fake_cqt_pool = self.pool_cqt.query(fake_cqt_det)

        # D_midi loss
        fmaps_real_midi = self.disc_midi(real_midi_latent)
        fmaps_fake_midi = self.disc_midi(fake_midi_pool)
        loss_D_midi, loss_D_midi_real, loss_D_midi_fake, metrics_midi = self.criterionGAN.discriminator_loss(fmaps_real_midi, fmaps_fake_midi)

        # D_cqt loss
        fmaps_real_cqt = self.disc_cqt(real_cqt)
        fmaps_fake_cqt = self.disc_cqt(fake_cqt_pool)
        loss_D_cqt, loss_D_cqt_real, loss_D_cqt_fake, metrics_cqt = self.criterionGAN.discriminator_loss(fmaps_real_cqt, fmaps_fake_cqt)

        # D trains at full strength (no gan_weight scaling)
        # This ensures D learns the real distribution early during G's warmup
        loss_D = loss_D_midi + loss_D_cqt

        self.manual_backward(loss_D / acc_steps)

        if (batch_idx + 1) % acc_steps == 0:
            # Log D gradient norm
            grad_norm_D = self._compute_grad_norm(self.disc_midi) + self._compute_grad_norm(self.disc_cqt)
            self.log("train/grad_norm_D", grad_norm_D, prog_bar=False, on_step=True, sync_dist=True)

            # Clip D gradients
            self.clip_gradients(opt_D, gradient_clip_val=self.cfg.training.max_grad_norm, gradient_clip_algorithm="norm")

            opt_D.step()
            opt_D.zero_grad()
            if sched_D is not None:
                sched_D.step()

        # Log D losses
        self.log("train/loss_D", loss_D, prog_bar=True, on_step=True, sync_dist=True)
        self.log("train/loss_D_midi", loss_D_midi, prog_bar=False, on_step=True, sync_dist=True)
        self.log("train/loss_D_cqt", loss_D_cqt, prog_bar=False, on_step=True, sync_dist=True)
        
        # Log D metrics
        self.log("train/acc_real_midi", metrics_midi["acc_real"], prog_bar=False, on_step=True, sync_dist=True)
        self.log("train/acc_fake_midi", metrics_midi["acc_fake"], prog_bar=False, on_step=True, sync_dist=True)
        self.log("train/score_real_midi", metrics_midi["score_real"], prog_bar=False, on_step=True, sync_dist=True)
        self.log("train/score_fake_midi", metrics_midi["score_fake"], prog_bar=False, on_step=True, sync_dist=True)

        self.log("train/acc_real_cqt", metrics_cqt["acc_real"], prog_bar=False, on_step=True, sync_dist=True)
        self.log("train/acc_fake_cqt", metrics_cqt["acc_fake"], prog_bar=False, on_step=True, sync_dist=True)
        self.log("train/score_real_cqt", metrics_cqt["score_real"], prog_bar=False, on_step=True, sync_dist=True)
        self.log("train/score_fake_cqt", metrics_cqt["score_fake"], prog_bar=False, on_step=True, sync_dist=True)
        
        train_loss = loss_G + loss_D
        self.log("train/train_loss", train_loss, prog_bar=True, on_step=True, sync_dist=True)
        return {"train_loss": train_loss.detach()}
    
    def on_train_epoch_start(self):
        if hasattr(self, "paired_batch_sampler") and self.paired_batch_sampler is not None:
            self.paired_batch_sampler.set_epoch(self.current_epoch)
        if hasattr(self, "unpaired_sampler") and self.unpaired_sampler is not None:
            self.unpaired_sampler.set_epoch(self.current_epoch)

    def _compute_metrics(self, fake_midi_z, real_midi_roll):
        fake_midi_roll = self._decode_midi_latent_to_roll(fake_midi_z)

        fake_cpu = fake_midi_roll.detach().float().cpu().numpy()
        real_cpu = real_midi_roll.detach().float().cpu().numpy()
        B, C, T, P = fake_cpu.shape

        pred01 = (fake_cpu > 0).astype(np.uint8)
        gt01   = (real_cpu > 0).astype(np.uint8)

        # Frame F1: union roll across all instruments
        gt_m   = gt01.max(axis=1)
        pred_m = pred01.max(axis=1)
        frame_f1_list = [piano_metrics_from_rolls(gt_m[i], pred_m[i])["Frame F1"] for i in range(B)]

        # Multi-Instrument Frame F1: per-channel frame F1, averaged across channels and batch
        mi_f1_list = [
            piano_metrics_from_rolls(gt01[i, c], pred01[i, c])["Frame F1"]
            for i in range(B) for c in range(C)
        ]

        return {
            "frame_f1":      float(np.mean(frame_f1_list)) if frame_f1_list else 0.0,
            "multi_inst_f1": float(np.mean(mi_f1_list))   if mi_f1_list   else 0.0,
        }


    def validation_step(self, batch, batch_idx):
        real_cqt = batch["cqt"]
        real_midi_roll = batch["midi"]
        real_midi_latent = self._encode_midi_latent(real_midi_roll)
        assert real_midi_roll.ndim == 4, f"train midi shape wrong: {real_midi_roll.shape}"

        # Online
        fake_midi_z, rec_cqt, fake_cqt, rec_midi_z = self.forward(
            real_cqt, real_midi_latent, use_gen_ema=False
        )

        loss_cycle_cqt  = self.criterionCycle(rec_cqt, real_cqt)
        loss_cycle_midi = self.criterionCycle(rec_midi_z, real_midi_latent)
        val_loss = loss_cycle_cqt + loss_cycle_midi

        self.log("val/val_cycle_cqt",  loss_cycle_cqt,  on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/val_cycle_midi", loss_cycle_midi, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/val_loss",       val_loss,        on_epoch=True, prog_bar=True, sync_dist=True)

        metrics = self._compute_metrics(fake_midi_z, real_midi_roll)
        self.log("val/frame_f1",      metrics["frame_f1"],      on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/multi_inst_f1", metrics["multi_inst_f1"], on_epoch=True, prog_bar=True, sync_dist=True)

        # EMA model
        if self.use_ema:
            fake_midi_z_ema, rec_cqt_ema, _, rec_midi_z_ema = self.forward(
                real_cqt, real_midi_latent, use_gen_ema=True
            )
            loss_cycle_cqt_ema  = self.criterionCycle(rec_cqt_ema, real_cqt)
            loss_cycle_midi_ema = self.criterionCycle(rec_midi_z_ema, real_midi_latent)
            val_loss_ema = loss_cycle_cqt_ema + loss_cycle_midi_ema

            self.log("val/val_cycle_cqt_ema",  loss_cycle_cqt_ema,  on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("val/val_cycle_midi_ema", loss_cycle_midi_ema, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("val/val_loss_ema",       val_loss_ema,        on_epoch=True, prog_bar=True, sync_dist=True)

            metrics_ema = self._compute_metrics(fake_midi_z_ema, real_midi_roll)
            self.log("val/frame_f1_ema",      metrics_ema["frame_f1"],      on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("val/multi_inst_f1_ema", metrics_ema["multi_inst_f1"], on_epoch=True, prog_bar=True, sync_dist=True)

        if self.save_val_vis and batch_idx == 0:
            self._save_cycle_images(
                real_cqt, fake_midi_z, rec_cqt, real_midi_roll, fake_cqt, rec_midi_z,
                tag="val"
            )
            if self.use_ema and self.save_ema_vis:
                with torch.no_grad():
                    fake_midi_z_ema, rec_cqt_ema, fake_cqt_ema, rec_midi_z_ema = self.forward(
                        real_cqt, real_midi_latent, use_gen_ema=True
                    )
                self._save_cycle_images(
                    real_cqt, fake_midi_z_ema, rec_cqt_ema, real_midi_roll, fake_cqt_ema, rec_midi_z_ema,
                    tag="val_ema"
                )

        return val_loss

    
    
    def test_step(self, batch, batch_idx):
        real_cqt = batch["cqt"]
        real_midi_roll = batch["midi"]

        assert real_midi_roll.ndim == 4, f"train midi shape wrong: {real_midi_roll.shape}"

        real_midi_latent = self._encode_midi_latent(real_midi_roll)
        fake_midi_z, rec_cqt, fake_cqt, rec_midi_z = self.forward(real_cqt, real_midi_latent, use_gen_ema=False)

        loss_cycle_cqt = self.criterionCycle(rec_cqt, real_cqt)
        loss_cycle_midi = self.criterionCycle(rec_midi_z, real_midi_latent)
        test_loss = loss_cycle_cqt + loss_cycle_midi

        self.log("test/test_cycle_cqt", loss_cycle_cqt, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test/test_cycle_midi", loss_cycle_midi, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test/test_loss", test_loss, on_epoch=True, prog_bar=True, sync_dist=True)

        metrics = self._compute_metrics(fake_midi_z, real_midi_roll)
        self.log("test/frame_f1",      metrics["frame_f1"],      on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test/multi_inst_f1", metrics["multi_inst_f1"], on_epoch=True, prog_bar=True, sync_dist=True)

        if self.use_ema:
            fake_midi_z_ema, rec_cqt_ema, fake_cqt_ema, rec_midi_z_ema = self.forward(
                real_cqt, real_midi_latent, use_gen_ema=True
            )
            loss_cycle_cqt_ema  = self.criterionCycle(rec_cqt_ema, real_cqt)
            loss_cycle_midi_ema = self.criterionCycle(rec_midi_z_ema, real_midi_latent)
            test_loss_ema = loss_cycle_cqt_ema + loss_cycle_midi_ema

            self.log("test/test_cycle_cqt_ema",  loss_cycle_cqt_ema,  on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("test/test_cycle_midi_ema", loss_cycle_midi_ema, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("test/test_loss_ema",       test_loss_ema,       on_epoch=True, prog_bar=True, sync_dist=True)

            metrics_ema = self._compute_metrics(fake_midi_z_ema, real_midi_roll)
            self.log("test/frame_f1_ema",      metrics_ema["frame_f1"],      on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("test/multi_inst_f1_ema", metrics_ema["multi_inst_f1"], on_epoch=True, prog_bar=True, sync_dist=True)

        if self.save_test_vis and batch_idx == 0:
            self._save_cycle_images(
                real_cqt, fake_midi_z, rec_cqt, real_midi_roll, fake_cqt, rec_midi_z,
                tag="test"
            )

            if self.use_ema and self.save_ema_vis:
                self._save_cycle_images(
                    real_cqt, fake_midi_z_ema, rec_cqt_ema, real_midi_roll, fake_cqt_ema, rec_midi_z_ema,
                    tag="test_ema"
                )
        return test_loss

    def _save_cycle_images(
        self,
        real_cqt, fake_midi, rec_cqt, real_midi_roll, fake_cqt, rec_midi,
        max_samples=1, tag="train"
    ):
        if self.trainer is not None and not self.trainer.is_global_zero:
            return

        self.image_dir.mkdir(parents=True, exist_ok=True)

        bsz = min(max_samples, real_cqt.size(0))

        # Decode latents for visualization
        fake_midi_roll = self._decode_midi_latent_to_roll(fake_midi[:bsz])
        rec_midi_roll  = self._decode_midi_latent_to_roll(rec_midi[:bsz])

        def _to_np01(x):
            x = x[:bsz].detach().to(torch.float32).cpu().numpy()
            x = (x + 1.0) / 2.0
            return np.clip(x, 0.0, 1.0)

        real_cqt_np = _to_np01(real_cqt)     # (B,1,256,352)
        rec_cqt_np  = _to_np01(rec_cqt)
        fake_cqt_np = _to_np01(fake_cqt)

        real_midi_np = _to_np01(real_midi_roll)    # (B,C,256,88)
        fake_midi_np = _to_np01(fake_midi_roll)    # (B,C,256,88)
        rec_midi_np  = _to_np01(rec_midi_roll)     # (B,C,256,88)

        # ---- helpers: make single-channel image (H,W) as np in [0,1] then invert for display
        def _roll_aggregate(roll_bctp):  # (B,C,T,P)
            if roll_bctp.ndim == 4:
                return roll_bctp.max(axis=1, keepdims=True)  # (B,1,T,P)
            elif roll_bctp.ndim == 3:
                return roll_bctp[:, None, ...]
            else:
                raise ValueError(f"roll ndim unexpected: {roll_bctp.shape}")

        def _save_grid(out_path: Path, panels_1ch):
            # panels_1ch: list of (name, arr) where arr is (B,1,H,W) in [0,1]
            rows = []
            for i in range(bsz):
                row_imgs = []
                for _, arr in panels_1ch:
                    img = 1.0 - arr[i, 0]  # invert
                    row_imgs.append(img)
                row = np.concatenate(row_imgs, axis=1)
                rows.append(row)
            grid = np.concatenate(rows, axis=0)
            img_255 = (grid * 255.0).astype(np.uint8)
            Image.fromarray(img_255, mode="L").save(out_path)

        step = int(self.global_step)

        # =========================
        # A) Aggregate (all instruments merged) 
        # =========================
        real_midi_ag = _roll_aggregate(real_midi_np)  # (B,1,256,88)
        fake_midi_ag = _roll_aggregate(fake_midi_np)
        rec_midi_ag  = _roll_aggregate(rec_midi_np)

        panels_ag = [
            ("real_cqt",  real_cqt_np),
            ("fake_midi", fake_midi_ag),
            ("rec_cqt",   rec_cqt_np),
            ("real_midi", real_midi_ag),
            ("fake_cqt",  fake_cqt_np),
            ("rec_midi",  rec_midi_ag),
        ]

        out_path_ag = self.image_dir / f"{tag}_AGG_epoch{self.current_epoch:03d}_step{step:07d}.png"
        _save_grid(out_path_ag, panels_ag)
        print(f"[viz] saved AGG grid -> {out_path_ag}")

        # =========================
        # B) Instrument folder (inst00~)
        # =========================
        C = real_midi_np.shape[1]
        for c in range(C):
            inst_dir = self.image_dir / f"inst{c:02d}"
            inst_dir.mkdir(parents=True, exist_ok=True)

            # (B,1,T,P)
            real_midi_c = real_midi_np[:, c:c+1]
            fake_midi_c = fake_midi_np[:, c:c+1]
            rec_midi_c  = rec_midi_np[:, c:c+1]

            panels_c = [
                ("real_cqt",  real_cqt_np),
                ("fake_midi", fake_midi_c),
                ("rec_cqt",   rec_cqt_np),
                ("real_midi", real_midi_c),
                ("fake_cqt",  fake_cqt_np),
                ("rec_midi",  rec_midi_c),
            ]
            out_path_c = inst_dir / f"{tag}_epoch{self.current_epoch:03d}_step{step:07d}.png"
            _save_grid(out_path_c, panels_c)

        print(f"[viz] saved per-instrument grids under {self.image_dir}")


# ======================================================
# Data helpers for musicnet-EM split structure
# ======================================================

def load_paired_paths(split_root: Path, split: str):
    """Returns (cqt_paths, midi_paths) matched by filename for a paired split."""
    cqt_dir  = split_root / split / "audio"
    midi_dir = split_root / split / "midi"
    if not cqt_dir.exists() or not midi_dir.exists():
        raise RuntimeError(f"Split '{split}' not found under {split_root}")
    cqt_files  = {f.name: f for f in cqt_dir.glob("*.npy")}
    midi_files = {f.name: f for f in midi_dir.glob("*.npy")}
    common = sorted(set(cqt_files) & set(midi_files))
    if not common:
        raise RuntimeError(f"No paired .npy files for split='{split}'")
    print(f"[load_paired_paths] {split}: {len(common)} paired chunks")
    return [cqt_files[n] for n in common], [midi_files[n] for n in common]


def load_unpaired_midi_paths(split_root: Path):
    """Returns MIDI paths from unpaired_midi/midi/ (musicnet-EM only)."""
    midi_dir = split_root / "unpaired_midi" / "midi"
    paths = sorted(midi_dir.glob("*.npy")) if midi_dir.exists() else []
    print(f"[load_unpaired_midi] EM unpaired: {len(paths)} chunks")
    return paths


def load_unpaired_cqt_paths(unpaired_audio_dir):
    """Returns CQT paths from unpaired_audio/ (GM). Raises if not found."""
    if not unpaired_audio_dir:
        raise ValueError("data.unpaired_audio_dir must be set (GM preprocessed audio dir)")
    d = Path(unpaired_audio_dir)
    if not d.exists():
        raise FileNotFoundError(f"unpaired_audio_dir not found: {d}")
    paths = sorted(d.glob("*.npy"))
    if not paths:
        raise RuntimeError(f"No .npy files in unpaired_audio_dir: {d}")
    print(f"[load_unpaired_cqt] GM audio: {len(paths)} chunks from {d}")
    return paths


# ======================================================
# main / CLI
# ======================================================

@hydra.main(version_base=None, config_path="configs", config_name="cycle_musicnet")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(cfg.seed)

    if not cfg.get("run_name"):
        raise ValueError("run_name is required!")

    split_root_s = Path(cfg.data.split_root)

    out_dir = os.path.join(os.path.abspath(cfg.logging.save_dir), cfg.run_name)
    if os.path.exists(out_dir) and not cfg.training.get("resume_ckpt", None):
        raise FileExistsError(f"Run dir {out_dir} already exists!")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    ckpt_dir = os.path.join(out_dir, "checkpoints")
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(ckpt_dir, "config.yaml"))

    # ── Load splits ────────────────────────────────────────────────────────────
    test_cqt,  test_midi  = load_paired_paths(split_root_s, "test")
    val_cqt,   val_midi   = load_paired_paths(split_root_s, "val")
    train_cqt, train_midi = load_paired_paths(split_root_s, "paired")

    unpaired_midi = load_unpaired_midi_paths(split_root_s)
    _unpaired_audio_dir = cfg.data.get("unpaired_audio_dir", None)
    unpaired_cqt  = load_unpaired_cqt_paths(_unpaired_audio_dir)

    # Subsample CQT to maintain a fixed midi:cqt ratio (default 1:9)
    cqt_midi_ratio = int(cfg.data.get("unpaired_cqt_midi_ratio", 9))
    target_cqt = len(unpaired_midi) * cqt_midi_ratio
    if len(unpaired_cqt) > target_cqt:
        rng = random.Random(cfg.seed)
        unpaired_cqt = rng.sample(unpaired_cqt, target_cqt)
        print(f"[main] subsampled unpaired_cqt to {len(unpaired_cqt)} (ratio 1:{cqt_midi_ratio})")

    print(f"[main] test={len(test_cqt)}  paired_train={len(train_cqt)}  "
          f"val={len(val_cqt)}  unpaired_midi={len(unpaired_midi)}  "
          f"unpaired_cqt={len(unpaired_cqt)}")

    # ── Datasets ───────────────────────────────────────────────────────────────
    test_ds     = PairedCqtRollDataset(test_cqt, test_midi)
    val_ds      = PairedCqtRollDataset(val_cqt, val_midi)
    paired_ds   = PairedCqtRollDataset(train_cqt, train_midi)
    unpaired_ds = UnalignedCqtRollDataset(unpaired_cqt, unpaired_midi)

    # ── Samplers / loaders ─────────────────────────────────────────────────────
    _world_size = int(os.environ.get("WORLD_SIZE", 1))
    _rank = int(os.environ.get("RANK", 0))
    if _world_size > 1:
        from torch.utils.data.distributed import DistributedSampler
        unpaired_sampler = DistributedSampler(
            unpaired_ds, num_replicas=_world_size, rank=_rank, shuffle=True, drop_last=True
        )
    else:
        unpaired_sampler = None
    unpaired_loader = DataLoader(
        unpaired_ds,
        batch_size=cfg.data.train_batch_size,
        sampler=unpaired_sampler,
        shuffle=(unpaired_sampler is None),
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=(cfg.data.num_workers > 0),
        drop_last=True,
    )

    paired_sampler = InstBalancedBatchSampler(
        paired_ds,
        batch_size=cfg.data.train_batch_size,
        seed=cfg.seed,
        drop_last=True,
    )
    paired_loader = DataLoader(
        paired_ds,
        batch_sampler=paired_sampler,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=(cfg.data.num_workers > 0),
    )

    train_loader = CombinedLoader(
        {"unpaired": unpaired_loader, "paired": paired_loader},
        mode="max_size_cycle",
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.data.valid_batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.data.valid_batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=False,
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    arr = np.load(Path(train_midi[0]), allow_pickle=False, mmap_mode="r")
    if arr.ndim != 3:
        raise ValueError(f"Expected (C,T,P) midi roll, got {arr.shape}")
    num_inst = int(arr.shape[0])

    model = CqtRollCycleLightning(cfg, image_dir=os.path.join(out_dir, "images"), num_instruments=num_inst)
    model.paired_batch_sampler = paired_sampler
    model.unpaired_sampler = unpaired_sampler

    # ── Callbacks / Logger ─────────────────────────────────────────────────────
    latest_ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir, filename="latest",
        save_top_k=1, monitor=None,
        every_n_train_steps=cfg.training.steps_per_checkpoint,
        save_last=False,
    )
    best_ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir, filename="best-loss",
        save_top_k=1, monitor="val/val_loss_ema", mode="min",
    )
    best_frame_f1_cb = ModelCheckpoint(
        dirpath=ckpt_dir, filename="best-frame-f1",
        save_top_k=1, monitor="val/frame_f1_ema", mode="max",
    )
    best_mi_f1_cb = ModelCheckpoint(
        dirpath=ckpt_dir, filename="best-mi-f1",
        save_top_k=1, monitor="val/multi_inst_f1_ema", mode="max",
    )
    lr_cb = LearningRateMonitor(logging_interval="step")
    pbar  = TQDMProgressBar()

    logger = None
    if not cfg.logging.offline:
        logger = WandbLogger(
            project=cfg.logging.project,
            save_dir=out_dir,
            name=cfg.run_name,
            entity=cfg.logging.entity,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # ── Trainer ────────────────────────────────────────────────────────────────
    resume_ckpt = cfg.training.get("resume_ckpt", None)
    if resume_ckpt:
        resume_ckpt = Path(resume_ckpt)
        if not resume_ckpt.exists():
            raise FileNotFoundError(f"resume_ckpt not found: {resume_ckpt}")
        cfg_path = resume_ckpt.parent / "config.yaml"
        if cfg_path.exists():
            old_cfg = OmegaConf.load(cfg_path)
            for k in ["training.resume_ckpt", "training.num_train_steps", "run_name", "logging"]:
                if OmegaConf.select(cfg, k) is not None:
                    OmegaConf.update(old_cfg, k, OmegaConf.select(cfg, k), merge=False)
            cfg = old_cfg

    trainer = pl.Trainer(
        max_steps=cfg.training.num_train_steps,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        num_nodes=cfg.training.num_nodes,
        strategy=DDPStrategy(find_unused_parameters=True, static_graph=False)
                 if (cfg.training.devices > 1 or cfg.training.num_nodes > 1) else "auto",
        callbacks=[latest_ckpt_cb, best_ckpt_cb, best_frame_f1_cb, best_mi_f1_cb, lr_cb, pbar],
        logger=logger,
        precision=cfg.training.precision,
        log_every_n_steps=10,
        enable_progress_bar=True,
        use_distributed_sampler=False,
    )

    trainer.fit(model, train_loader, val_loader,
                ckpt_path=str(resume_ckpt) if resume_ckpt else None)
    trainer.test(model, dataloaders=test_loader)


if __name__ == "__main__":
    main()