import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont
import librosa
import soundfile as sf

from train_piano import CqtRollCycleLightning
from utils import list_split_files, PairedCqtRollDataset, piano_metrics_from_rolls

SR_DEFAULT = 44100
FPS_DEFAULT = 50
NOTE_MIN, NOTE_MAX = 21, 108

def tensor_to_2d(x: torch.Tensor) -> np.ndarray:
    """(1,H,W) or (H,W) torch -> (H,W) np"""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
    arr = x.numpy()
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D or (1,2D), got {arr.shape}")
    return arr

def cqt_to_audio(
    cqt_11: torch.Tensor,
    out_wav: str,
    db_min: float,
    db_max: float,
    sr: int = SR_DEFAULT,
    fps: int = FPS_DEFAULT,
    bins_per_octave: int = 48,
    fmin_note: str = "A0",
):
    """
    cqt_11: (256,352) or (1,256,352) in [-1,1]
    Writes wav using librosa.griffinlim_cqt on magnitude.
    """
    x11 = tensor_to_2d(cqt_11)  # (T, F) = (256,352)

    # [-1,1] -> [0,1] -> [db_min, db_max]
    x01 = (x11 + 1.0) / 2.0
    x01 = np.clip(x01, 0.0, 1.0)
    db = x01 * (db_max - db_min) + db_min  # (T,F)

    # librosa expects (n_bins, t)
    mag = librosa.db_to_amplitude(db.T)  # (F,T)

    hop_length = int(round(sr / fps))
    y = librosa.griffinlim_cqt(
        mag,
        sr=sr,
        hop_length=hop_length,
        fmin=librosa.note_to_hz(fmin_note),
        bins_per_octave=bins_per_octave,
    )

    peak = np.max(np.abs(y))
    if peak > 1e-6:
        y = y / peak * 0.95

    Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_wav, y.astype(np.float32), sr)
    print(f"[saved CQT audio] {out_wav}")

def midi_to_audio(
    roll_11: torch.Tensor,
    out_wav: str,
    sr: int = SR_DEFAULT,
    fps: int = FPS_DEFAULT,
):
    """
    roll_11: (256,88) or (1,256,88) in [-1,1] (or logits). We binarize by >0.
    Very simple additive-sine synth per frame.
    """
    roll_2d = tensor_to_2d(roll_11)  # (T,88)
    roll_bin = (roll_2d > 0).astype(np.float32)

    T, N = roll_bin.shape
    num_notes = NOTE_MAX - NOTE_MIN + 1
    assert N == num_notes, f"Expected {num_notes} notes, got {N}"

    frame_dur = 1.0 / fps
    frame_len = int(round(sr * frame_dur))
    total_len = T * frame_len

    audio = np.zeros(total_len, dtype=np.float32)
    midi_nums = np.arange(NOTE_MIN, NOTE_MAX + 1)
    freqs = librosa.midi_to_hz(midi_nums)  # (88,)

    for t in range(T):
        active = np.where(roll_bin[t] > 0)[0]
        if len(active) == 0:
            continue

        start = t * frame_len
        end = start + frame_len

        t_global = np.linspace(
            t * frame_dur,
            (t + 1) * frame_dur,
            frame_len,
            endpoint=False
        ).astype(np.float32)

        frame_wave = np.zeros_like(t_global, dtype=np.float32)
        for idx_note in active:
            f = freqs[idx_note]
            frame_wave += np.sin(2.0 * np.pi * f * t_global).astype(np.float32)

        frame_wave /= max(1, len(active))
        audio[start:end] += frame_wave

    peak = np.max(np.abs(audio))
    if peak > 1e-6:
        audio = audio / peak * 0.95

    Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_wav, audio, sr)
    print(f"[saved ROLL audio] {out_wav}")


def downsample_cqt(cqt_tf: np.ndarray) -> np.ndarray:
    """
    cqt_tf: (T=256, F=352) float
    return: (T=256, F=88) by mean-pooling over freq (352 -> 88 by /4)
    """
    T, F = cqt_tf.shape
    if (T, F) != (256, 352):
        raise ValueError(f"Expected CQT (256,352) but got {cqt_tf.shape}")
    return cqt_tf.reshape(256, 88, 4).mean(axis=2)

BLACK_PCS = {1, 3, 6, 8, 10}  # C#, D#, F#, G#, A#

def midi_to_octave_label_for_c(midi: int) -> str:
    octave = midi // 12 - 1
    return f"C{octave}"

def piano_required_left_width(scale: int, w_roll: int, font: ImageFont.ImageFont,
                              keyboard_frac: float = 0.10,
                              tick_len_base: int = 4,
                              gap_base: int = 2,
                              margin_base: int = 6) -> int:
    ls = scale / 3.0

    white_w = int(round(w_roll * keyboard_frac))
    white_w = max(int(round(12 * ls)), min(white_w, int(round(26 * ls))))

    tick_len = max(3, int(round(tick_len_base * ls)))
    label_gap = max(3, int(round(4 * ls)))
    kb_gap_to_roll = max(2, int(round(gap_base * ls)))
    margin = max(6, int(round(margin_base * ls)))

    tmp = Image.new("RGB", (10, 10))
    d = ImageDraw.Draw(tmp)
    label_w = int(d.textlength("C8", font=font)) if hasattr(d, "textlength") else font.getsize("C8")[0]
    return label_w + label_gap + white_w + tick_len + kb_gap_to_roll + margin

def draw_piano_with_c_labels(
    canvas: Image.Image,
    roll_x0: int,            # roll (MIDI block) left x
    y0: int,
    w_roll: int,
    h_roll: int,
    scale: int,
    note_min: int = 21,
    n_keys: int = 88,
    flip: bool = True,
    keyboard_frac: float = 0.10,
    c_font_mul: float = 1.35,       # C label font size (increase to 1.5~1.8 for larger labels)
    tick_outside: bool = True,      # tick is drawn outside the keyboard
):
    draw = ImageDraw.Draw(canvas)
    ls = scale / 3.0

    # ---- widths (short + thin) ----
    white_w = int(round(w_roll * keyboard_frac))
    white_w = max(int(round(12 * ls)), min(white_w, int(round(26 * ls))))

    black_w = max(1, int(round(white_w * 0.45)))
    black_h_ratio = 0.55

    kb_gap_to_roll = max(2, int(round(2 * ls)))    # gap between roll and keyboard
    tick_len = max(3, int(round(4 * ls)))          # tick length
    label_gap = max(3, int(round(4 * ls)))

    line_w = 1
    outline = (40, 40, 40)

    row_h = h_roll / float(n_keys)

    # ---- fonts ----
    try:
        base_font_size = max(9, int(round(9 * ls)))
        c_font_size    = max(base_font_size + 2, int(round(base_font_size * c_font_mul)))
        font   = ImageFont.truetype("DejaVuSans.ttf", base_font_size)
        font_c = ImageFont.truetype("DejaVuSans.ttf", c_font_size)
    except:
        font = ImageFont.load_default()
        font_c = ImageFont.load_default()

    # ---- x positions ----
    # keyboard sits just left of roll
    kb_x1 = roll_x0 - kb_gap_to_roll
    kb_x0 = kb_x1 - white_w

    # ticks are outside keyboard (never overlap)
    if tick_outside:
        tick_x1 = kb_x0
        tick_x0 = kb_x0 - tick_len
    else:
        tick_x1 = kb_x0 + 1
        tick_x0 = tick_x1 - tick_len

    label_x1 = tick_x0 - label_gap  # label is left of tick

    def y_bounds_for_row(i: int):
        i_draw = (n_keys - 1 - i) if flip else i
        y_top = int(round(y0 + i_draw * row_h))
        y_bot = int(round(y0 + (i_draw + 1) * row_h))
        return y_top, y_bot

    # 1) WHITE keys
    for i in range(n_keys):
        midi = note_min + i
        if (midi % 12) in BLACK_PCS:
            continue
        y_top, y_bot = y_bounds_for_row(i)
        draw.rectangle([kb_x0, y_top, kb_x1, y_bot - 1],
                       fill=(250, 250, 250), outline=outline, width=line_w)

    # 2) BLACK keys (attached to kb_x0 exactly)
    black_x0 = kb_x0
    black_x1 = black_x0 + black_w
    for i in range(n_keys):
        midi = note_min + i
        if (midi % 12) not in BLACK_PCS:
            continue
        y_top, y_bot_full = y_bounds_for_row(i)
        y_bot = y_top + max(1, int(round((y_bot_full - y_top) * black_h_ratio)))
        draw.rectangle([black_x0, y_top, black_x1, y_bot],
                       fill=(50, 50, 50), outline=outline, width=line_w)

    # keyboard border
    draw.rectangle([kb_x0, y0, kb_x1, y0 + h_roll - 1], outline=outline, width=line_w)

    # 3) C ticks + labels (C1..C8)
    for i in range(n_keys + 1):
        midi = note_min + i
        if midi % 12 != 0:
            continue
        i_draw = (n_keys - i) if flip else i
        y = int(round(y0 + i_draw * row_h))

        # tick (outside)
        draw.line([(tick_x0, y), (tick_x1, y)], fill=(80, 80, 80), width=1)

        # label
        if 24 <= midi <= 108:
            label = midi_to_octave_label_for_c(midi)
            tw = int(draw.textlength(label, font=font_c)) if hasattr(draw, "textlength") else font_c.getsize(label)[0]
            draw.text((label_x1 - tw, y - max(6, int(round(6 * ls)))), label,
                      fill=(0, 0, 0), font=font_c)

    return canvas

def draw_black_border(canvas: Image.Image, x0: int, y0: int, w: int, h: int, width: int = 1):
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([x0, y0, x0 + w - 1, y0 + h - 1], outline=(30, 30, 30), width=width)
    return canvas

def draw_inset_legend(canvas: Image.Image, x0: int, y0: int, scale: int = 6):
    """
    OPAQUE legend box inside MIDI block (top-left).
    - single row: [TP][FN][FP]
    - no "Legend" text
    - opaque background (no grid bleed-through)
    - box size fits content (no extra right margin)
    """
    draw = ImageDraw.Draw(canvas)
    ls = scale / 3.0

    # Colors (TP blue / FP warm red-orange / FN green)
    TP = (0, 114, 178)
    FN = (0, 158, 115)
    FP = (213, 94, 0)

    try:
        font_size = max(14, int(round(14 * ls)))  # bigger text
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except:
        font = ImageFont.load_default()

    # slightly bigger swatches/spacing
    pad_x  = max(8, int(round(9 * ls)))
    pad_y  = max(6, int(round(7 * ls)))
    swatch = max(14, int(round(16 * ls)))
    gap    = max(10, int(round(12 * ls)))  # between items
    tgap   = max(7, int(round(8 * ls)))    # between swatch and text

    items = [("TP", TP), ("FN", FN), ("FP", FP)]

    def text_w(s: str) -> int:
        if hasattr(draw, "textbbox"):
            bb = draw.textbbox((0, 0), s, font=font)
            return bb[2] - bb[0]
        if hasattr(draw, "textlength"):
            return int(draw.textlength(s, font=font))
        return font.getsize(s)[0]

    def text_h(s: str) -> int:
        if hasattr(draw, "textbbox"):
            bb = draw.textbbox((0, 0), s, font=font)
            return bb[3] - bb[1]
        return font.getsize(s)[1]

    th = max(text_h("TP"), text_h("FN"), text_h("FP"))
    row_h = max(swatch, th)

    # ---- compute exact content width (fix extra right margin) ----
    content_w = 0
    for i, (lab, _) in enumerate(items):
        content_w += swatch + tgap + text_w(lab)
        if i != len(items) - 1:
            content_w += gap

    box_w = content_w + 2 * pad_x
    box_h = row_h + 2 * pad_y

    # ---- draw opaque box + border ----
    draw.rectangle([x0, y0, x0 + box_w, y0 + box_h], fill=(255, 255, 255))
    draw.rectangle([x0, y0, x0 + box_w, y0 + box_h], outline=(0, 0, 0), width=max(2, int(round(2 * ls))))

    # ---- draw items ----
    cx = x0 + pad_x
    cy = y0 + pad_y + (row_h - swatch) // 2
    ty = y0 + pad_y + (row_h - th) // 2

    for i, (lab, col) in enumerate(items):
        # swatch
        draw.rectangle([cx, cy, cx + swatch, cy + swatch], fill=col, outline=(0, 0, 0), width=max(2, int(round(2 * ls))))
        cx += swatch + tgap

        # label
        draw.text((cx, ty), lab, fill=(0, 0, 0), font=font)
        cx += text_w(lab)

        if i != len(items) - 1:
            cx += gap

    return canvas

def draw_88key_grid_rgba(
    canvas: Image.Image,
    x0: int,
    y0: int,
    w: int,
    h: int,
    n_rows: int = 88,
    rgba=(220, 220, 220, 70),
    c_rgba=(150, 150, 150, 160),
    width: int = 1,
    note_min: int = 21,
):
    if canvas.mode != "RGBA":
        base = canvas.convert("RGBA")
    else:
        base = canvas

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)

    row_h = h / float(n_rows)
    for r in range(n_rows + 1):
        y = int(round(y0 + r * row_h))
        midi = note_min + r
        is_c = (midi % 12 == 0)
        color = c_rgba if is_c else rgba
        odraw.line([(x0, y), (x0 + w - 1, y)], fill=color, width=width)

    return Image.alpha_composite(base, overlay)

def save_one_sample_tpfnfp_png(
    out_png: str,
    cqt_tf: np.ndarray,      # (256,352) in [-1,1]
    gt_tf: np.ndarray,       # (256,88)  in [-1,1]
    pred_tf: np.ndarray,     # (256,88)  in [-1,1]
    scale: int = 6,
    keyboard_frac: float = 0.10,
):
    # ---- CQT ----
    cqt01 = (cqt_tf + 1.0) / 2.0
    cqt01 = 1.0 - np.clip(cqt01, 0.0, 1.0)
    cqt01 = downsample_cqt(cqt01)  # (256,88)
    cqt_img = (cqt01.T * 255).astype(np.uint8)  # (88,256)

    # ---- TP/FN/FP ----
    gt_bin = (gt_tf > 0)
    pr_bin = (pred_tf > 0)

    tp = (gt_bin & pr_bin).T
    fn = (gt_bin & (~pr_bin)).T
    fp = ((~gt_bin) & pr_bin).T

    H, W = tp.shape  # (88,256)

    rgb = np.ones((H, W, 3), dtype=np.uint8) * 255
    rgb[tp] = np.array([0, 114, 178], dtype=np.uint8)     # TP
    rgb[fn] = np.array([0, 158, 115], dtype=np.uint8)     # FN
    rgb[fp] = np.array([213, 94, 0], dtype=np.uint8)     # FP

    # ---- layout (preserve original aspect ratio) ----
    pad = 6
    legend_h = 18
    rowlabel_h = 12
    gap = 2
    header_h = pad + legend_h + gap + rowlabel_h + pad

    midlabel_h = 12
    sep = pad + midlabel_h + pad

    total_w = W + 2 * pad
    total_h = header_h + H + sep + H + pad

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 10)
    except:
        font = ImageFont.load_default()

    # --- Legend row ---
    x = pad
    y = pad
    x += 50

    # --- Row label above CQT ---
    draw.text((pad, pad + legend_h + gap), "Input CQT", fill=(0, 0, 0), font=font)

    # paste CQT
    cqt_pil = Image.fromarray(cqt_img, mode="L").convert("RGB")
    y_cqt = header_h
    canvas.paste(cqt_pil, (pad, y_cqt))

    # --- mid text ---
    y_mid = y_cqt + H + pad
    draw.text((pad, y_mid), "GT vs Pred MIDI (TP/FN/FP)", fill=(0, 0, 0), font=font)

    # paste TF
    tf_pil = Image.fromarray(rgb, mode="RGB")
    y_tf = y_cqt + H + sep
    canvas.paste(tf_pil, (pad, y_tf))

    # paste CQT
    cqt_pil = Image.fromarray(cqt_img, mode="L").convert("RGB")
    y_cqt = header_h
    canvas.paste(cqt_pil, (pad, y_cqt))

    # mid text
    y_mid = y_cqt + H + pad
    draw.text((pad, y_mid), "GT vs Pred MIDI (TP/FN/FP)", fill=(0, 0, 0), font=font)

    # paste TF
    tf_pil = Image.fromarray(rgb, mode="RGB")
    y_tf = y_cqt + H + sep
    canvas.paste(tf_pil, (pad, y_tf))

    # ---- resize (single scale) ----
    s = int(scale) if scale is not None else 1
    if s != 1:
        canvas = canvas.resize((canvas.width * s, canvas.height * s), resample=Image.NEAREST)

    # ---- left margin for piano ----
    try:
        font_for_margin = ImageFont.truetype("DejaVuSans.ttf", max(12, int(round(12 * (s / 3.0)))))
    except:
        font_for_margin = ImageFont.load_default()

    W_s = W * s
    H_s = H * s
    extra_left = piano_required_left_width(s, W_s, font_for_margin, keyboard_frac=keyboard_frac)

    new_canvas = Image.new("RGB", (canvas.width + extra_left, canvas.height), (255, 255, 255))
    new_canvas.paste(canvas, (extra_left, 0))
    canvas = new_canvas

    pad_s = pad * s + extra_left
    y_cqt_s = y_cqt * s
    y_tf_s  = y_tf  * s
    
    inset_margin = max(6, int(round(6 * (s / 3.0))))
    

    # ---- draw piano next to MIDI block (not CQT) ----
    canvas = draw_piano_with_c_labels(
        canvas,
        roll_x0=pad_s,
        y0=y_tf_s,
        w_roll=W_s,
        h_roll=H_s,
        scale=s,
        flip=True,
        keyboard_frac=keyboard_frac,
        c_font_mul=1.35,     # increase to 1.5~1.8 for larger C1~C8 labels
        tick_outside=True,   # tick does not overlap the keyboard
    )

    # ---- grids + borders (both panels) ----
    canvas = draw_88key_grid_rgba(canvas, x0=pad_s, y0=y_cqt_s, w=W_s, h=H_s)
    canvas = draw_88key_grid_rgba(canvas, x0=pad_s, y0=y_tf_s,  w=W_s, h=H_s)
    canvas = canvas.convert("RGB")
    canvas = draw_black_border(canvas, pad_s, y_cqt_s, W_s, H_s, width=1)
    canvas = draw_black_border(canvas, pad_s, y_tf_s,  W_s, H_s, width=1)
    
    ls = s / 3.0
    legend_x = pad_s + max(6, int(round(6 * ls)))   # inside top-left of MIDI block
    legend_y = y_tf_s + max(6, int(round(6 * ls)))
    canvas = draw_inset_legend(canvas, legend_x, legend_y, scale=s)

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)

@torch.no_grad()
def run_eval(args):
    device = args.device

    model = CqtRollCycleLightning.load_from_checkpoint(args.ckpt, map_location=device)
    model.eval()
    model.to(device)

    cqt_paths, midi_paths = list_split_files(args.data_root, split=args.split)
    ds = PairedCqtRollDataset(cqt_paths, midi_paths)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )

    out_dir = Path(args.out_dir)
    viz_dir = out_dir / "viz"
    audio_dir = out_dir / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    frame_list = []
    saved = 0

    for bidx, batch in enumerate(loader):
        real_cqt = batch["cqt"].to(device)        # (B,1,256,352) [-1,1]
        gt_roll  = batch["midi"].to(device)       # (B,1,256,88)  [-1,1]

        real_midi_latent = model.vae.encode(gt_roll)[0].detach()

        fake_midi_z, rec_cqt, fake_cqt, rec_midi_z = model.forward(real_cqt, real_midi_latent, use_gen_ema=True)

        pred_roll = model.vae.decode(fake_midi_z)

        # ---- metrics per-sample ----
        pred_np = pred_roll.detach().float().cpu().numpy()
        gt_np   = gt_roll.detach().float().cpu().numpy()

        B = pred_np.shape[0]
        for i in range(B):
            p = pred_np[i, 0]
            g = gt_np[i, 0]
            pred_bin = (p > 0).astype(int)
            gt_bin   = (g > 0).astype(int)
            m = piano_metrics_from_rolls(gt_bin, pred_bin)

            frame_list.append(m["Frame F1"])

        # ---- viz + audio ----
        if saved < args.max_viz:
            pred_cpu = pred_roll.detach().float().cpu()  # (B,1,256,88)
            gt_cpu   = gt_roll.detach().float().cpu()
            cqt_cpu  = real_cqt.detach().float().cpu()   # (B,1,256,352)

            B = pred_cpu.shape[0]
            for i in range(B):
                if saved >= args.max_viz:
                    break

                stem = f"{args.split}_sample_{saved:06d}"

                # PNG
                out_png = viz_dir / f"{stem}.png"
                save_one_sample_tpfnfp_png(
                    out_png=str(out_png),
                    cqt_tf=cqt_cpu[i, 0].numpy(),   # (256,352)
                    gt_tf=gt_cpu[i, 0].numpy(),     # (256,88)
                    pred_tf=pred_cpu[i, 0].numpy(), # (256,88)
                )
                
                out_cycle = viz_dir / f"cyclegrid_b{stem}.png"
                save_cycle_grid_color_png(
                    out_png=str(out_cycle),
                    real_cqt=real_cqt,
                    fake_midi_latent=fake_midi_z,
                    rec_cqt=rec_cqt,
                    real_midi_roll=gt_roll,
                    fake_cqt=fake_cqt,
                    rec_midi_latent=rec_midi_z,
                    model=model,
                    max_samples=min(4, real_cqt.size(0)),
                )

                # WAVs
                # input CQT
                cqt_to_audio(
                    cqt_cpu[i, 0],
                    out_wav=str(audio_dir / f"{stem}_inputcqt.wav"),
                    db_min=args.db_min,
                    db_max=args.db_max,
                    sr=args.sr,
                    fps=args.fps,
                )
                # gt roll
                midi_to_audio(
                    gt_cpu[i, 0],
                    out_wav=str(audio_dir / f"{stem}_gtroll.wav"),
                    sr=args.sr,
                    fps=args.fps,
                )
                # pred roll
                midi_to_audio(
                    pred_cpu[i, 0],
                    out_wav=str(audio_dir / f"{stem}_predroll.wav"),
                    sr=args.sr,
                    fps=args.fps,
                )

                saved += 1

    results = {
        "ckpt":        str(args.ckpt),
        "data_root":   str(args.data_root),
        "split":       args.split,
        "num_samples": len(frame_list),
        "frame_f1":    float(np.mean(frame_list)) if frame_list else 0.0,
    }

    out_json = out_dir / f"{args.split}_metrics.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print("\n=== EVAL RESULTS ===")
    print(f"  num_samples : {results['num_samples']}")
    print(f"  frame_f1    : {results['frame_f1']:.4f}")
    print(f"\nSaved: {out_json}")

def save_cycle_grid_color_png(
    out_png: str,
    real_cqt: torch.Tensor,        # (B,1,256,352) in [-1,1]
    fake_midi_latent: torch.Tensor,# (B,*,*,*) latent or (B,1,256,88) if discrete
    rec_cqt: torch.Tensor,         # (B,1,256,352) in [-1,1]
    real_midi_roll: torch.Tensor,  # (B,1,256,88)  in [-1,1]
    fake_cqt: torch.Tensor,        # (B,1,256,352) in [-1,1]
    rec_midi_latent: torch.Tensor, # latent or roll
    model,                         # CqtRollCycleLightning
    max_samples: int = 4,
):
    """
    Save a 6-panel cycle visualization as a single RGB grid:
      real_cqt | fake_midi | rec_cqt | real_midi | fake_cqt | rec_midi
    CQT panels: magma colormap
    MIDI panels: WHITE background + DARK NAVY notes (thresholded)
    """
    from matplotlib import cm

    bsz = min(max_samples, real_cqt.size(0))

    # --- decode MIDI latents for visualization ---
    with torch.no_grad():
        fake_midi_roll = model.vae.decode(fake_midi_latent[:bsz])
        rec_midi_roll  = model.vae.decode(rec_midi_latent[:bsz])

    def _to_np01(x: torch.Tensor) -> np.ndarray:
        x = x[:bsz].detach().to(torch.float32).cpu().numpy()
        x = (x + 1.0) / 2.0   # [-1,1] -> [0,1]
        x = np.clip(x, 0.0, 1.0)
        return x

    # to [0,1]
    real_cqt_np   = _to_np01(real_cqt)
    fake_midi_np  = _to_np01(fake_midi_roll)
    rec_cqt_np    = _to_np01(rec_cqt)
    real_midi_np  = _to_np01(real_midi_roll)
    fake_cqt_np   = _to_np01(fake_cqt)
    rec_midi_np   = _to_np01(rec_midi_roll)

    panels = [
        ("real_cqt",  real_cqt_np),
        ("fake_midi", fake_midi_np),
        ("rec_cqt",   rec_cqt_np),
        ("real_midi", real_midi_np),
        ("fake_cqt",  fake_cqt_np),
        ("rec_midi",  rec_midi_np),
    ]

    # CQT colormap (purple-ish)
    cqt_cmap = cm.get_cmap("magma")

    def _apply_cqt_cmap(img01_2d: np.ndarray) -> np.ndarray:
        rgba = cqt_cmap(img01_2d)  # (H,W,4) float
        rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
        return rgb

    def _render_midi_white_bg(img01_2d: np.ndarray, thr: float = 0.5) -> np.ndarray:
        """
        White background, dark navy notes.
        thr=0.5 corresponds to original roll > 0 (since [-1,1]->[0,1]).
        """
        H, W = img01_2d.shape
        rgb = np.ones((H, W, 3), dtype=np.uint8) * 255  # white bg
        active = img01_2d > thr
        rgb[active] = np.array([11, 30, 91], dtype=np.uint8)  # #0B1E5B
        return rgb

    rows_rgb = []
    for i in range(bsz):
        row_imgs = []
        for name, arr in panels:
            img01 = arr[i, 0]  # (H,W) in [0,1]

            if "cqt" in name:
                # invert CQT so high energy appears bright
                img_for_cqt = 1.0 - img01
                rgb = _apply_cqt_cmap(img_for_cqt)
            else:
                # do not invert MIDI; keep white background
                rgb = _render_midi_white_bg(img01, thr=0.5)

            row_imgs.append(rgb)

        row_rgb = np.concatenate(row_imgs, axis=1)
        rows_rgb.append(row_rgb)

    grid_rgb = np.concatenate(rows_rgb, axis=0)
    im = Image.fromarray(grid_rgb, mode="RGB")

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    im.save(out_png)
    print(f"[saved cycle color grid] {out_png}")

    
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True,
                   help="Path to model checkpoint, e.g. [YOUR_LOG_DIR]/checkpoints/best-f1.ckpt")
    p.add_argument("--data_root", type=str, default="[YOUR_DATA_DIR]",
                   help="Root of processed MAESTRO dataset (must contain test/audio and test/midi subdirs)")
    p.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"])
    p.add_argument("--out_dir", type=str, default="eval_output")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_viz", type=int, default=8)
    # audio params
    p.add_argument("--sr", type=int, default=SR_DEFAULT)
    p.add_argument("--fps", type=int, default=FPS_DEFAULT)
    p.add_argument("--db_min", type=float, default=-80.0)
    p.add_argument("--db_max", type=float, default=0.0)

    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_eval(args)