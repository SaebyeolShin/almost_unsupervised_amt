from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from midi_autoencoder.lucidrains_ae import UNetStyleVAE

# ── Constants ─────────────────────────────────────────────────────────────────
FPS      = 50
NOTE_MIN = 21
NOTE_MAX = 108

# ── VAE loading ───────────────────────────────────────────────────────────────
def load_vae_from_checkpoint(ckpt_path: str, use_ema: bool = False, map_location="cpu"):
    """Load a frozen UNetStyleVAE from a PyTorch Lightning checkpoint."""
    print(f"[load_vae] loading from {ckpt_path} (use_ema={use_ema})")
    checkpoint = torch.load(ckpt_path, map_location=map_location, weights_only=False)

    hparams = checkpoint.get("hyper_parameters", {})
    if "cfg" in hparams:
        m_cfg = hparams["cfg"].model
        C = m_cfg.get("num_instruments") or m_cfg.get("in_channels", 1)
        model_kwargs = {
            "in_channels":    C,
            "out_channels":   C,
            "ch":             m_cfg.ch,
            "ch_mult":        tuple(m_cfg.ch_mult),
            "ch_mult_1d":     tuple(m_cfg.ch_mult_1d) if getattr(m_cfg, "ch_mult_1d", None) else None,
            "num_res_blocks": m_cfg.num_res_blocks,
            "z_channels":     m_cfg.z_channels,
            "kl_weight":      m_cfg.kl_weight,
            "binary_mode":    m_cfg.binary_mode,
            "attn_heads":     getattr(m_cfg, "attn_heads", 4),
            "attn_dim_head":  getattr(m_cfg, "attn_dim_head", 32),
            "loss_type":      getattr(m_cfg, "loss_type", "bce"),
            "focal_gamma":    getattr(m_cfg, "focal_gamma", 2.0),
            "focal_alpha":    getattr(m_cfg, "focal_alpha", 0.25),
        }
    else:
        model_kwargs = hparams

    vae = UNetStyleVAE(**model_kwargs)
    state_dict = checkpoint["state_dict"]
    new_state_dict = {}

    if use_ema:
        ema_prefix = "ema_model.ema_model."
        has_ema = any(k.startswith(ema_prefix) for k in state_dict)
        if has_ema:
            print("[load_vae] Found EMA weights, loading them.")
            for k, v in state_dict.items():
                if k.startswith(ema_prefix):
                    new_state_dict[k[len(ema_prefix):]] = v
        else:
            print("[load_vae] WARNING: use_ema=True but no EMA weights found. Falling back to online model.")
            for k, v in state_dict.items():
                if k.startswith("model."):
                    new_state_dict[k[6:]] = v
                elif not k.startswith("ema_model."):
                    new_state_dict[k] = v
    else:
        for k, v in state_dict.items():
            if k.startswith("model."):
                new_state_dict[k[6:]] = v
            elif not k.startswith("ema_model."):
                new_state_dict[k] = v

    vae.load_state_dict(new_state_dict)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    print("[load_vae] success")
    return vae

# ── Data helpers ──────────────────────────────────────────────────────────────
def load_cqt_npy(path: Path, db_min: float = -80.0, db_max: float = 0.0) -> torch.Tensor:
    """Load a CQT .npy chunk, normalize dB → [-1, 1]. Returns (1, H, W)."""
    arr = np.load(path, allow_pickle=False, mmap_mode=None)
    if arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim == 3 and arr.shape[0] != 1:
        raise ValueError(f"Unexpected CQT shape: {path}, shape={arr.shape}")
    arr = arr.astype("float32", copy=True)
    x01 = np.clip((arr - db_min) / (db_max - db_min), 0.0, 1.0)
    return torch.from_numpy(x01 * 2.0 - 1.0)


def list_split_files(root, split: str = "train", strict: bool = True):
    """
    Scan root/split/audio/*.npy and root/split/midi/*.npy for matched pairs.
    Returns (cqt_paths, roll_paths), or (None, None) when strict=False and split is absent.
    """
    root = Path(root)
    cqt_dir  = root / split / "audio"
    roll_dir = root / split / "midi"

    if not cqt_dir.exists() or not roll_dir.exists():
        if strict:
            raise RuntimeError(f"Split '{split}' not found under {root}")
        print(f"[list_split_files] split='{split}' not found, skipping.")
        return None, None

    cqt_files  = {f.name: f for f in cqt_dir.glob("*.npy")}
    roll_files = {f.name: f for f in roll_dir.glob("*.npy")}
    common = sorted(set(cqt_files) & set(roll_files))

    if not common:
        if strict:
            raise RuntimeError(f"No paired .npy files for split='{split}'")
        print(f"[list_split_files] split='{split}' has no paired files, skipping.")
        return None, None

    print(f"[list_split_files] split={split}, paired files={len(common)}")
    return [cqt_files[n] for n in common], [roll_files[n] for n in common]

# ── MIDI metrics ──────────────────────────────────────────────────────────────
def piano_metrics_from_rolls(gt_roll_bin: np.ndarray, pred_roll_bin: np.ndarray) -> dict:
    """Frame-level precision/recall/F1 between two binary (T, 88) piano rolls."""
    gt_f   = gt_roll_bin.reshape(-1).astype(np.float32)
    pred_f = pred_roll_bin.reshape(-1).astype(np.float32)

    tp = np.sum(pred_f * gt_f)
    fp = np.sum(pred_f * (1.0 - gt_f))
    fn = np.sum((1.0 - pred_f) * gt_f)

    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)

    return {
        "Frame precision": float(prec),
        "Frame recall":    float(rec),
        "Frame F1":        float(f1),
    }

# ── Piano datasets ───────────────────────────────────────────────────────────
class UnalignedCqtRollDataset(Dataset):
    """Unpaired CQT + piano roll dataset. CQT and MIDI are sampled independently."""
    def __init__(self, cqt_paths, midi_paths, db_min: float = -80.0, db_max: float = 0.0):
        super().__init__()
        self.cqt_paths  = [Path(p) for p in cqt_paths]
        self.midi_paths = [Path(p) for p in midi_paths]
        self.cqt_size   = len(self.cqt_paths)
        self.midi_size  = len(self.midi_paths)
        self.db_min = float(db_min)
        self.db_max = float(db_max)
        print(f"[Dataset] CQT:  {self.cqt_size} files")
        print(f"[Dataset] MIDI: {self.midi_size} files")
        if self.cqt_size == 0 or self.midi_size == 0:
            raise RuntimeError("no .npy file in one of CQT / MIDI.")

    def __len__(self):
        return max(self.cqt_size, self.midi_size)

    def _load_roll(self, path: Path) -> torch.Tensor:
        arr = np.load(path, allow_pickle=False, mmap_mode=None)
        if arr.ndim == 2:
            arr = arr[None, ...]
        elif arr.ndim == 3 and arr.shape[0] != 1:
            raise ValueError(f"Unexpected roll shape: {path}, shape={arr.shape}")
        arr = arr.astype("float32", copy=True)
        return torch.from_numpy((arr > 0).astype("float32") * 2.0 - 1.0)

    def __getitem__(self, index):
        cqt_path  = self.cqt_paths[index % self.cqt_size]
        midi_path = self.midi_paths[np.random.randint(0, self.midi_size)]
        return {
            "cqt":       load_cqt_npy(cqt_path, self.db_min, self.db_max),
            "midi":      self._load_roll(midi_path),
            "cqt_path":  str(cqt_path),
            "midi_path": str(midi_path),
        }


class PairedCqtRollDataset(Dataset):
    """Paired CQT + piano roll dataset (index-aligned)."""
    def __init__(self, cqt_paths, midi_paths, db_min: float = -80.0, db_max: float = 0.0):
        super().__init__()
        self.cqt_paths  = [Path(p) for p in cqt_paths]
        self.midi_paths = [Path(p) for p in midi_paths]
        assert len(self.cqt_paths) == len(self.midi_paths), "Paths must be same length"
        self.db_min = float(db_min)
        self.db_max = float(db_max)

    def __len__(self):
        return len(self.cqt_paths)

    def _load_roll(self, path: Path) -> torch.Tensor:
        arr = np.load(path, allow_pickle=False, mmap_mode=None)
        if arr.ndim == 2:
            arr = arr[None, ...]
        elif arr.ndim == 3 and arr.shape[0] != 1:
            raise ValueError(f"Unexpected roll shape: {path}, shape={arr.shape}")
        arr = arr.astype("float32", copy=True)
        return torch.from_numpy((arr > 0).astype("float32") * 2.0 - 1.0)

    def __getitem__(self, idx):
        cqt_path  = self.cqt_paths[idx]
        midi_path = self.midi_paths[idx]
        return {
            "cqt":       load_cqt_npy(cqt_path, self.db_min, self.db_max),
            "midi":      self._load_roll(midi_path),
            "cqt_path":  str(cqt_path),
            "midi_path": str(midi_path),
        }


# ── Image pool (discriminator replay buffer) ──────────────────────────────────
class ImagePool:
    def __init__(self, pool_size: int = 128):
        self.pool_size = pool_size
        self.images: list = []

    def query(self, images: torch.Tensor) -> torch.Tensor:
        if self.pool_size == 0:
            return images
        return_images = []
        for i in range(images.size(0)):
            img = images[i].unsqueeze(0)
            if len(self.images) < self.pool_size:
                self.images.append(img)
                return_images.append(img)
            elif np.random.uniform() > 0.5:
                idx = np.random.randint(0, self.pool_size)
                tmp = self.images[idx].clone()
                self.images[idx] = img
                return_images.append(tmp)
            else:
                return_images.append(img)
        return torch.cat(return_images, dim=0)
