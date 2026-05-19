import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from train_musicnet import CqtRollCycleLightning, PairedCqtRollDataset
from utils import list_split_files, piano_metrics_from_rolls, FPS


def multi_inst_frame_f1(gt01_ctp: np.ndarray, pred01_ctp: np.ndarray) -> float:
    """(C, T, 88) binary arrays -> per-channel frame F1 averaged across channels."""
    f1s = [piano_metrics_from_rolls(gt01_ctp[c], pred01_ctp[c])["Frame F1"]
           for c in range(gt01_ctp.shape[0])]
    return float(np.mean(f1s))


# ── Visualisation ─────────────────────────────────────────────────────────────

def _np01(t, bsz):
    x = t[:bsz].detach().float().cpu().numpy()
    return np.clip((x + 1.0) / 2.0, 0.0, 1.0)


def _make_panels_rgb(real_cqt_np, gt_midi_np, pred_midi_np,
                     rec_cqt_np, fake_cqt_np, rec_midi_np):
    from matplotlib import cm
    cqt_cmap = cm.get_cmap("magma")

    def _cqt(a):
        return (cqt_cmap(1.0 - a)[..., :3] * 255).astype(np.uint8)

    def _midi(a):
        rgb = np.ones((*a.shape, 3), dtype=np.uint8) * 255
        rgb[a > 0.5] = [11, 30, 91]
        return rgb

    return [_cqt(real_cqt_np), _midi(pred_midi_np),
            _cqt(rec_cqt_np),  _midi(gt_midi_np),
            _cqt(fake_cqt_np), _midi(rec_midi_np)]


def _save_grid(path, rows_of_panels):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    rows = [np.concatenate(panels, axis=1) for panels in rows_of_panels]
    Image.fromarray(np.concatenate(rows, axis=0)).save(path)
    print(f"[viz] {path}")


def save_viz(out_png, real_cqt, real_midi_roll, pred_roll, cycle_tensors, max_samples=2):
    bsz = min(max_samples, real_cqt.size(0))

    real_cqt_np  = _np01(real_cqt,                       bsz)
    gt_midi_np   = _np01(real_midi_roll,                  bsz)
    pred_midi_np = _np01(pred_roll,                       bsz)
    rec_cqt_np   = _np01(cycle_tensors["rec_cqt"],       bsz)
    fake_cqt_np  = _np01(cycle_tensors["fake_cqt"],      bsz)
    rec_midi_np  = _np01(cycle_tensors["rec_midi_roll"], bsz)

    C = gt_midi_np.shape[1]

    # aggregate (union across channels)
    rows_agg = [
        _make_panels_rgb(
            real_cqt_np[i, 0],
            gt_midi_np[i].max(axis=0),
            pred_midi_np[i].max(axis=0),
            rec_cqt_np[i, 0],
            fake_cqt_np[i, 0],
            rec_midi_np[i].max(axis=0),
        )
        for i in range(bsz)
    ]
    _save_grid(out_png, rows_agg)

    # per-instrument
    base = Path(out_png)
    for c in range(C):
        rows_c = [
            _make_panels_rgb(
                real_cqt_np[i, 0],
                gt_midi_np[i, c],
                pred_midi_np[i, c],
                rec_cqt_np[i, 0],
                fake_cqt_np[i, 0],
                rec_midi_np[i, c],
            )
            for i in range(bsz)
        ]
        _save_grid(str(base.parent / f"inst{c:02d}" / base.name), rows_c)


# ── Eval loop ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_eval(args):
    device = args.device

    model = CqtRollCycleLightning.load_from_checkpoint(args.ckpt, map_location=device)
    model.eval()
    model.to(device)

    cqt_paths, midi_paths = list_split_files(args.data_root, args.split)
    print(f"[eval] {args.split}: {len(cqt_paths)} chunks")

    ds = PairedCqtRollDataset(cqt_paths, midi_paths)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True, drop_last=False)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = out_dir / "viz"

    frame_f1_list, mi_f1_list = [], []
    saved = 0

    for batch in loader:
        real_cqt       = batch["cqt"].to(device)
        real_midi_roll = batch["midi"].to(device)

        real_midi_latent = model._encode_midi_latent(real_midi_roll)
        fake_midi_z, rec_cqt, fake_cqt, rec_midi_z = model.forward(
            real_cqt, real_midi_latent, use_gen_ema=True
        )
        pred_roll     = model._decode_midi_latent_to_roll(fake_midi_z)
        rec_midi_roll = model._decode_midi_latent_to_roll(rec_midi_z)

        pred01 = (pred_roll.detach().float().cpu().numpy() > 0).astype(np.uint8)
        gt01   = (real_midi_roll.detach().float().cpu().numpy() > 0).astype(np.uint8)

        for i in range(pred01.shape[0]):
            frame_f1_list.append(
                piano_metrics_from_rolls(gt01[i].max(axis=0), pred01[i].max(axis=0))["Frame F1"]
            )
            mi_f1_list.append(multi_inst_frame_f1(gt01[i], pred01[i]))

        if saved < args.max_viz:
            cycle_tensors = dict(rec_cqt=rec_cqt, fake_cqt=fake_cqt, rec_midi_roll=rec_midi_roll)
            save_viz(
                str(viz_dir / f"batch{saved:04d}_agg.png"),
                real_cqt, real_midi_roll, pred_roll, cycle_tensors,
                max_samples=min(2, real_cqt.size(0)),
            )
            saved += 1

    results = {
        "ckpt":                str(args.ckpt),
        "data_root":           str(args.data_root),
        "split":               args.split,
        "num_samples":         len(frame_f1_list),
        "frame_f1":            float(np.mean(frame_f1_list)) if frame_f1_list else 0.0,
        "multi_inst_frame_f1": float(np.mean(mi_f1_list))   if mi_f1_list   else 0.0,
    }

    out_json = out_dir / f"{args.split}_metrics.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== EVAL RESULTS ===")
    print(f"  num_samples         : {results['num_samples']}")
    print(f"  frame_f1            : {results['frame_f1']:.4f}")
    print(f"  multi_inst_frame_f1 : {results['multi_inst_frame_f1']:.4f}")
    print(f"\nSaved: {out_json}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",        required=True,
                   help="Path to checkpoint, e.g. [YOUR_LOG_DIR]/checkpoints/best-frame-f1.ckpt")
    p.add_argument("--data_root",   required=True,
                   help="Split root with test/, val/, paired/ subfolders")
    p.add_argument("--split",       default="test", choices=["paired", "val", "test"])
    p.add_argument("--out_dir",     default="eval_results/musicnet")
    p.add_argument("--device",      default="cuda")
    p.add_argument("--batch_size",  type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_viz",     type=int, default=4,
                   help="Number of batches to save visualizations for")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eval(args)