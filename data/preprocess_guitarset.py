import argparse
import hashlib
import json
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import librosa
import jams
from tqdm import tqdm

# -------------------------
# Constants / defaults
# -------------------------
DATASET_DIRNAME = "GuitarSet"

NOTE_MIN, NOTE_MAX = 21, 108
NUM_NOTES = NOTE_MAX - NOTE_MIN + 1

FPS_DEFAULT = 50
CHUNK_FRAMES_DEFAULT = 256

HOP_LENGTH_44100 = 882  # 44100 / 50
HOP_LENGTH_48000 = 960  # 48000 / 50

TRACKID_RE = re.compile(
    r"^(?P<pid>\d+)_((?P<style>[A-Za-z]+)(?P<prog>\d+))-(?P<tempo>\d+)-(?P<key>[A-G][b#]?)_.*$"
)

STYLE_MAP = {
    "Jazz": "Jazz",
    "BN": "Bossa Nova",
    "Rock": "Rock",
    "SS": "Singer-Songwriter",
    "Funk": "Funk",
}

REMOTES = {
    "annotations": {
        "filename": "annotation.zip",
        "url": "https://zenodo.org/record/3371780/files/annotation.zip?download=1",
        "checksum": "b39b78e63d3446f2e54ddb7a54df9b10",
        "destination_dir": "annotation",
    },
    "audio_mic": {
        "filename": "audio_mono-mic.zip",
        "url": "https://zenodo.org/record/3371780/files/audio_mono-mic.zip?download=1",
        "checksum": "275966d6610ac34999b58426beb119c3",
        "destination_dir": "audio_mono-mic",
    },
    "audio_mix": {
        "filename": "audio_mono-pickup_mix.zip",
        "url": "https://zenodo.org/record/3371780/files/audio_mono-pickup_mix.zip?download=1",
        "checksum": "aecce79f425a44e2055e46f680e10f6a",
        "destination_dir": "audio_mono-pickup_mix",
    },
    "audio_hex_original": {
        "filename": "audio_hex-pickup_original.zip",
        "url": "https://zenodo.org/record/3371780/files/audio_hex-pickup_original.zip?download=1",
        "checksum": "f9911bf217cb40e9e68edf3726ef86cc",
        "destination_dir": "audio_hex-pickup_original",
    },
    "audio_hex_debleeded": {
        "filename": "audio_hex-pickup_debleeded.zip",
        "url": "https://zenodo.org/record/3371780/files/audio_hex-pickup_debleeded.zip?download=1",
        "checksum": "c31d97279464c9a67e640cb9061fb0c6",
        "destination_dir": "audio_hex-pickup_debleeded",
    },
}

# -------------------------
# Helpers: hashing / download
# -------------------------
def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_url(url: str, dest: Path):
    """Simple downloader using urllib (no extra deps)."""
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = r.headers.get("Content-Length")
        total = int(total) if total is not None else None

        pbar = tqdm(total=total, unit="B", unit_scale=True, desc=f"download {dest.name}")
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            pbar.update(len(chunk))
        pbar.close()

    tmp.replace(dest)


def extract_zip(zip_path: Path, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)


def ensure_guitarset_downloaded(
    data_home: str,
    partial: Optional[List[str]] = None,
    force_overwrite: bool = False,
    cleanup: bool = True,
) -> str:
    """
    Download and extract GuitarSet into <data_home>/GuitarSet/...
    partial: subset of keys in REMOTES (e.g., ["annotations","audio_mic"])
    """
    data_home = Path(data_home).expanduser().resolve()
    root = data_home / DATASET_DIRNAME
    root.mkdir(parents=True, exist_ok=True)

    want = partial if partial is not None else list(REMOTES.keys())
    for key in want:
        if key not in REMOTES:
            raise ValueError(f"Unknown remote key: {key}")

    zips_dir = root / "_zips"
    zips_dir.mkdir(parents=True, exist_ok=True)

    for key in want:
        meta = REMOTES[key]
        zip_name = meta["filename"]
        url = meta["url"]
        checksum = meta["checksum"]
        dest_subdir = root / meta["destination_dir"]
        zip_path = zips_dir / zip_name

        # If extracted exists and not forcing overwrite, skip
        if dest_subdir.exists() and any(dest_subdir.iterdir()) and not force_overwrite:
            print(f"[skip] {key}: already extracted at {dest_subdir}")
            continue

        # If forcing overwrite, wipe the destination subdir
        if force_overwrite and dest_subdir.exists():
            shutil.rmtree(dest_subdir, ignore_errors=True)

        # Download zip if missing or overwrite
        if force_overwrite and zip_path.exists():
            zip_path.unlink()

        if not zip_path.exists():
            print(f"[download] {key} -> {zip_path}")
            download_url(url, zip_path)

        # Verify checksum
        got = _md5(zip_path)
        if got != checksum:
            raise RuntimeError(
                f"Checksum mismatch for {zip_path} ({key}).\n"
                f"  expected md5={checksum}\n"
                f"  got      md5={got}"
            )

        # Extract into root
        print(f"[extract] {zip_path.name} -> {dest_subdir}")
        extract_zip(zip_path, dest_subdir)

        if cleanup:
            try:
                zip_path.unlink()
            except Exception:
                pass

    print(f"[done] dataset root = {root}")
    return str(root)


# -------------------------
# Indexing / parsing track id
# -------------------------
@dataclass
class TrackInfo:
    track_id: str
    style_code: str
    style_name: str
    progression: int
    tempo: int
    key: str


def parse_track_id(track_id: str) -> TrackInfo:
    m = TRACKID_RE.match(track_id)
    if not m:
        raise ValueError(f"Unrecognized track_id format: {track_id}")

    style_code = m.group("style")
    prog = int(m.group("prog"))
    tempo = int(m.group("tempo"))
    key = m.group("key")
    style_name = STYLE_MAP.get(style_code, style_code)

    return TrackInfo(
        track_id=track_id,
        style_code=style_code,
        style_name=style_name,
        progression=prog,
        tempo=tempo,
        key=key,
    )


def build_index(dataset_root: str, audio_kind: str) -> Dict[str, Tuple[str, str]]:
    """
    Return dict: track_id -> (audio_path, jams_path)
    audio_kind: mic|mix|hex|hex_cln
    """
    root = Path(dataset_root)

    jams_dir = root / "annotation"
    if not jams_dir.exists():
        raise FileNotFoundError(f"Missing annotation dir: {jams_dir}")

    jams_files = sorted(jams_dir.rglob("*.jams"))
    if not jams_files:
        raise FileNotFoundError(f"No .jams files found under {jams_dir}")

    if audio_kind == "mic":
        audio_dir = root / "audio_mono-mic"
        suffix = "_mic.wav"
    elif audio_kind == "mix":
        audio_dir = root / "audio_mono-pickup_mix"
        suffix = "_mix.wav"
    elif audio_kind == "hex":
        audio_dir = root / "audio_hex-pickup_original"
        suffix = "_hex.wav"
    elif audio_kind == "hex_cln":
        audio_dir = root / "audio_hex-pickup_debleeded"
        suffix = "_hex_cln.wav"
    else:
        raise ValueError(f"audio_kind must be one of mic|mix|hex|hex_cln, got {audio_kind}")

    if not audio_dir.exists():
        raise FileNotFoundError(f"Missing audio dir for {audio_kind}: {audio_dir}")

    out: Dict[str, Tuple[str, str]] = {}
    missing_audio = 0
    for jp in jams_files:
        tid = jp.stem
        ap = audio_dir / f"{tid}{suffix}"
        if not ap.exists():
            missing_audio += 1
            continue
        out[tid] = (str(ap), str(jp))

    if not out:
        raise RuntimeError(f"No (audio,jams) pairs found for audio_kind={audio_kind}")

    if missing_audio > 0:
        print(f"[warn] missing audio files for {missing_audio} tracks (audio_kind={audio_kind})")

    return out


def make_style_progression_split(track_ids: List[str]) -> Dict[str, str]:
    """
    mt3 paper split:
      For each style: progression 1&2 -> train, progression 3 -> validation
    """
    split: Dict[str, str] = {}
    bad = 0
    for tid in track_ids:
        try:
            info = parse_track_id(tid)
        except Exception:
            bad += 1
            continue

        if info.progression in (1, 2):
            split[tid] = "train"
        elif info.progression == 3:
            split[tid] = "validation"
        else:
            split[tid] = "train"

    if bad:
        print(f"[warn] could not parse {bad} track_ids, dropped from split")
    return split


# -------------------------
# Feature extraction
# -------------------------
def pad_audio_to_hop(y: np.ndarray, hop_length: int) -> np.ndarray:
    r = len(y) % hop_length
    if r != 0:
        y = np.pad(y, (0, hop_length - r), mode="constant")
    return y


def audio_to_cqt_db(
    audio_path: str,
    fps: int,
    bins_per_octave: int,
    n_bins: int,
    segment_threshold_frames: int = 15360,
) -> Tuple[np.ndarray, int]:
    sr = librosa.get_samplerate(audio_path)

    if sr == 44100:
        hop_length = HOP_LENGTH_44100
    elif sr == 48000:
        hop_length = HOP_LENGTH_48000
    else:
        hop_length = int(sr / fps)
        print(f"[WARN] Unusual sample rate {sr} Hz for {audio_path}")

    y, _ = librosa.load(audio_path, sr=sr, mono=True, dtype=np.float32, res_type="kaiser_fast")
    y = pad_audio_to_hop(y, hop_length)
    total_frames = len(y) // hop_length

    if total_frames <= segment_threshold_frames:
        C = librosa.cqt(
            y, sr=sr, hop_length=hop_length,
            fmin=librosa.note_to_hz("A0"),
            n_bins=n_bins, bins_per_octave=bins_per_octave,
            pad_mode="reflect",
        )
        mag = np.abs(C).astype(np.float32)
        mag = np.nan_to_num(mag, nan=0.0, posinf=0.0, neginf=0.0)
        X = librosa.amplitude_to_db(mag, ref=np.max).T.astype(np.float32)
    else:
        padding = 60000
        cqt_list = []
        for start_frame in range(0, total_frames, segment_threshold_frames):
            end_frame = min(start_frame + segment_threshold_frames, total_frames)
            start_sample = max(0, start_frame * hop_length - padding)
            end_sample = min(len(y), end_frame * hop_length + padding)
            y_seg = y[start_sample:end_sample]

            C_seg = librosa.cqt(
                y_seg, sr=sr, hop_length=hop_length,
                fmin=librosa.note_to_hz("A0"),
                n_bins=n_bins, bins_per_octave=bins_per_octave,
                pad_mode="reflect",
            )
            actual_left_pad = start_frame * hop_length - start_sample
            frame_offset = actual_left_pad // hop_length
            frames_in_seg = end_frame - start_frame
            C_seg = C_seg[:, frame_offset : frame_offset + frames_in_seg]
            cqt_list.append(C_seg)

        C = np.concatenate(cqt_list, axis=1)
        mag = np.abs(C).astype(np.float32)
        mag = np.nan_to_num(mag, nan=0.0, posinf=0.0, neginf=0.0)
        X = librosa.amplitude_to_db(mag, ref=np.max).T.astype(np.float32)

    return X, sr


def jams_to_roll_binary(
    jams_path: str,
    fps: int,
    note_min: int,
    note_max: int,
) -> np.ndarray:
    """
    Build pianoroll (T, 88) from GuitarSet JAMS.
    - unions across 6 'note_midi' annotations (one per string)
    - output is binary {0,1} (no velocity info available)
    """
    jam = jams.load(jams_path)
    annos = jam.search(namespace="note_midi")
    if len(annos) == 0:
        return np.zeros((1, note_max - note_min + 1), dtype=np.float32)

    # total duration = max end time across all annotations
    t_end = 0.0
    for anno in annos:
        intervals, _ = anno.to_interval_values()
        if len(intervals) > 0:
            t_end = max(t_end, float(np.max(intervals[:, 1])))

    T = int(np.ceil(t_end * fps)) + 1
    P = note_max - note_min + 1
    roll = np.zeros((T, P), dtype=np.float32)

    for anno in annos:
        intervals, values = anno.to_interval_values()
        for (t0, t1), v in zip(intervals, values):
            # robust pitch extraction
            try:
                if isinstance(v, dict):
                    p = float(v.get("value", v.get("midi", 0.0)))
                else:
                    p = float(v)
                p = int(round(p))
            except Exception:
                continue

            if p < note_min or p > note_max:
                continue

            s = int(np.floor(float(t0) * fps))
            e = int(np.ceil(float(t1) * fps))
            if e <= s:
                e = s + 1
            s = max(0, s)
            e = min(T, e)

            roll[s:e, p - note_min] = 127.0

    return roll


def split_chunks(X: np.ndarray, chunk_frames: int) -> List[np.ndarray]:
    n = X.shape[0] // chunk_frames
    return [X[i * chunk_frames : (i + 1) * chunk_frames] for i in range(n)]


# -------------------------
# Main preprocess
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_home", type=str, default = "./guitarset")
    ap.add_argument("--partial_download", nargs="*", default=None, help="Subset of: annotations audio_mic audio_mix audio_hex_original audio_hex_debleeded (default: all)")
    ap.add_argument("--force_overwrite", action="store_true", help="Re-download and overwrite extracted dirs")
    ap.add_argument("--cleanup", action="store_true", help="Delete zip files after extraction (default: True)")
    ap.set_defaults(cleanup=True)
    ap.add_argument("--audio_kind", type=str, default="mic", choices=["mic", "mix", "hex", "hex_cln"], help="Which audio to use for CQT")
    ap.add_argument("--bins_per_octave", type=int, default=48, help="CQT bins per octave (use 48 to match your piano model CQT resolution)")
    ap.add_argument("--n_bins", type=int, default=352, help="Total CQT bins (for 48 bpo, use 352)")
    ap.add_argument("--compute_global_norm", action="store_true", help="Compute global mean/std over saved CQT chunks (in raw dB scale, matching MAESTRO)")
    ap.add_argument("--max_tracks", type=int, default=-1, help="For debugging: limit number of tracks")

    args = ap.parse_args()

    # 1) download
    dataset_root = ensure_guitarset_downloaded(
        data_home=args.data_home,
        partial=args.partial_download,
        force_overwrite=args.force_overwrite,
        cleanup=args.cleanup,
    )

    # 2) index pairs
    pairs = build_index(dataset_root, audio_kind=args.audio_kind)
    track_ids = sorted(pairs.keys())

    # 3) split
    split_map = make_style_progression_split(track_ids)

    # 4) prepare output dirs
    out_root = Path(args.data_home)
    for sp in ["train", "validation"]:
        (out_root / sp / "audio").mkdir(parents=True, exist_ok=True)
        (out_root / sp / "midi").mkdir(parents=True, exist_ok=True)


    # 5) preprocess loop
    s1 = 0.0
    s2 = 0.0
    n = 0

    dropped_frames_total = 0
    total_frames_total = 0
    saved_chunks = {"train": 0, "validation": 0}
    sr_44100_count = 0
    sr_48000_count = 0
    sr_other_count = 0

    skipped: List[str] = []

    iterable = track_ids[: args.max_tracks] if args.max_tracks > 0 else track_ids

    for tid in tqdm(iterable, desc="Preprocess GuitarSet", unit="track"):
        sp = split_map.get(tid)
        if sp not in ("train", "validation"):
            continue

        audio_path, jams_path = pairs[tid]

        try:
            # audio -> CQT in RAW dB (no clipping!)
            X_cqt_db, _sr = audio_to_cqt_db(
                audio_path=audio_path,
                fps=FPS_DEFAULT,
                bins_per_octave=args.bins_per_octave,
                n_bins=args.n_bins,
            )  # (T_cqt, n_bins) in RAW dB
            
            if _sr == 44100:
                sr_44100_count += 1
            elif _sr == 48000:
                sr_48000_count += 1
            else:
                sr_other_count += 1

            # jams -> roll binary {0,1}
            X_roll = jams_to_roll_binary(
                jams_path=jams_path,
                fps=FPS_DEFAULT,
                note_min=NOTE_MIN,
                note_max=NOTE_MAX,
            )  # (T_roll, 88) in {0,1}

            # align roll to CQT length (CQT is "ground truth" time axis from audio)
            T_cqt = X_cqt_db.shape[0]
            T_roll = X_roll.shape[0]
            if T_roll < T_cqt:
                X_roll = np.pad(X_roll, ((0, T_cqt - T_roll), (0, 0)), mode="constant")
            elif T_roll > T_cqt:
                X_roll = X_roll[:T_cqt]
            T = T_cqt

            # chunking stats
            T_kept = (T // CHUNK_FRAMES_DEFAULT) * CHUNK_FRAMES_DEFAULT
            total_frames_total += T
            dropped_frames_total += (T - T_kept)

            if T_kept == 0:
                skipped.append(f"{tid}: too short after alignment (T={T})")
                continue

            X_cqt_db = X_cqt_db[:T_kept]
            X_roll = X_roll[:T_kept]

            cqt_chunks = split_chunks(X_cqt_db, CHUNK_FRAMES_DEFAULT)
            roll_chunks = split_chunks(X_roll, CHUNK_FRAMES_DEFAULT)
            if len(cqt_chunks) != len(roll_chunks):
                raise RuntimeError(f"Chunk mismatch: cqt={len(cqt_chunks)} roll={len(roll_chunks)}")

            out_cqt_dir = out_root / sp / "audio"
            out_roll_dir = out_root / sp / "midi"
            
            for i, (cc, rr) in enumerate(zip(cqt_chunks, roll_chunks)):
                if cc.shape != (CHUNK_FRAMES_DEFAULT, args.n_bins):
                    continue
                if rr.shape != (CHUNK_FRAMES_DEFAULT, NUM_NOTES):
                    continue

                stem = f"{tid}_chunk{i:04d}"
                np.save(out_cqt_dir / f"{stem}.npy", cc.astype(np.float32))
                np.save(out_roll_dir / f"{stem}.npy", rr.astype(np.float32))
                saved_chunks[sp] += 1

                if args.compute_global_norm:
                    tmp = cc.astype(np.float64, copy=False)
                    s1 += float(tmp.sum())
                    s2 += float((tmp * tmp).sum())
                    n += tmp.size

        except Exception as e:
            skipped.append(f"{tid}: {e}")

    # summary
    print("\n" + "=" * 70)
    print("GuitarSet preprocess summary")
    print("=" * 70)
    print(f"dataset_root: {dataset_root}")
    print(f"audio_kind: {args.audio_kind}")
    print(f"fps: {FPS_DEFAULT}, chunk_frames: {CHUNK_FRAMES_DEFAULT}")
    print(f"CQT: bins_per_octave={args.bins_per_octave}, n_bins={args.n_bins}, saved_as=RAW_dB")
    print(f"Roll: pitches=[{NOTE_MIN},{NOTE_MAX}] saved_as={{0,127}}")
    print(f"saved train chunks: {saved_chunks['train']}")
    print(f"saved validation chunks: {saved_chunks['validation']}")

    print()
    print(f"Sample rate statistics:")
    print(f"  44.1 kHz files: {sr_44100_count}")
    print(f"  48.0 kHz files: {sr_48000_count}")
    if sr_other_count > 0:
        print(f"  Other sample rates: {sr_other_count}")

    if total_frames_total > 0 and dropped_frames_total > 0:
        pct = 100.0 * dropped_frames_total / float(total_frames_total)
        print(f"\ndropped frames: {dropped_frames_total}/{total_frames_total} ({pct:.2f}%)")

    if skipped:
        print(f"\n[warn] skipped {len(skipped)} tracks")
        for msg in skipped[:10]:
            print("  -", msg)
        if len(skipped) > 10:
            print(f"  ... and {len(skipped)-10} more")

    split_json = {
        "train": [tid for tid in track_ids if split_map.get(tid) == "train"],
        "validation": [tid for tid in track_ids if split_map.get(tid) == "validation"],
    }
    split_path = out_root / "guitarset_split_style_prog.json"
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_json, f, indent=2)
    print(f"\n[saved] split -> {split_path}")

    if args.compute_global_norm and n > 0:
        mean = float(s1 / n)
        var = max(1e-12, float(s2 / n - mean * mean))
        std = float(np.sqrt(var))
        norm_path = out_root / "guitarset_cqt_global_norm.json"
        with open(norm_path, "w", encoding="utf-8") as f:
            json.dump({"mean": mean, "std": std, "num_samples": int(n)}, f, indent=2)
        print(f"[saved] global norm -> {norm_path}")
        print(f"  mean={mean:.6f}, std={std:.6f}")

    print("=" * 70)


if __name__ == "__main__":
    main()