#!/usr/bin/env python3
"""
Filter and preprocess Gardner Museum audio into CQT chunks.

Step 1 — Filter: exclude GM files that overlap with MusicNet-EM recordings using
  - Primary filter  : composer + catalog token exact-match from filename
  - Secondary filter: duration similarity (±2s)

Step 2 — Preprocess: convert kept files to CQT chunks matching the
  musicnet_em_preprocessed/audio/ format:
    audio/{stem}_chunk{i:06d}.npy  →  (256, 352) float32 dB, bpo=48

Supports resume: skips files whose first chunk already exists.
"""

import argparse, csv, json, re
from collections import defaultdict
from pathlib import Path

import numpy as np
import librosa
from tqdm import tqdm

# ── CQT constants (must match preprocess_musicnet_em.py) ──────────────────────
FPS           = 50
CHUNK_FRAMES  = 256
BPO           = 48
N_BINS        = 352
HOP_44100     = 882
HOP_48000     = 960
SEG_THRESHOLD = 15360
SEG_PADDING   = 60000

DUR_ABS_SEC   = 2.0    # ±2 s threshold for duration-based filtering


# ── Filter helpers ─────────────────────────────────────────────────────────────
def _filename_tokens(stem: str):
    return set(re.split(r'[_\-\s]+', stem.lower()))

def _composer_token(composer: str) -> str:
    return composer.strip().lower().split()[0]

def _load_em_songs(em_root: Path):
    meta_csv = em_root / 'musicnet_metadata.csv'
    em_midi  = em_root / 'musicnet_em'
    em_ids   = {f.stem for f in em_midi.glob('*.mid')}
    songs = []
    with open(meta_csv) as f:
        for row in csv.DictReader(f):
            if row['id'] in em_ids:
                songs.append({
                    'id':       row['id'],
                    'composer': row['composer'].strip(),
                    'catalog':  row['catalog_name'].strip().lower(),
                    'seconds':  float(row['seconds']),
                })
    return songs

def _load_gm_durations(gm_dir: Path):
    durations = {}
    for f in tqdm(sorted(gm_dir.glob('*.mp3')), desc='Reading GM durations', unit='file'):
        try:
            durations[f.name] = librosa.get_duration(path=str(f))
        except Exception as e:
            print(f'  [WARN] {f.name}: {e}')
    return durations

def _run_primary(em_songs, gm_files):
    hits = defaultdict(list)
    for gm_f in gm_files:
        tokens = _filename_tokens(gm_f.stem)
        for song in em_songs:
            comp = _composer_token(song['composer'])
            cat  = song['catalog']
            if not cat:
                continue
            if comp in tokens and cat in tokens:
                hits[gm_f.name].append(song['id'])
    return dict(hits)

def _run_secondary(em_songs, gm_durations):
    cat_total = defaultdict(float)
    cat_ids   = defaultdict(list)
    for s in em_songs:
        cat_total[s['catalog']] += s['seconds']
        cat_ids[s['catalog']].append(s['id'])

    def close(a, b):
        return abs(a - b) <= DUR_ABS_SEC

    hits = defaultdict(set)
    for gm_name, gm_dur in gm_durations.items():
        for s in em_songs:
            if close(gm_dur, s['seconds']):
                hits[gm_name].add(s['id'])
        for cat, total in cat_total.items():
            if close(gm_dur, total):
                for eid in cat_ids[cat]:
                    hits[gm_name].add(eid)
    return {k: sorted(v) for k, v in hits.items()}

def filter_gm(em_root: Path, gm_dir: Path):
    """Returns sorted list of kept GM filenames."""
    print('Loading MusicNet-EM metadata...')
    em_songs = _load_em_songs(em_root)
    print(f'  EM songs: {len(em_songs)}')

    gm_files = sorted(gm_dir.glob('*.mp3'))
    print(f'  Gardner Museum files: {len(gm_files)}')

    gm_durations = _load_gm_durations(gm_dir)

    print('\nRunning primary filter (composer + catalog token)...')
    primary   = _run_primary(em_songs, gm_files)
    print('Running secondary filter (duration similarity)...')
    secondary = _run_secondary(em_songs, gm_durations)

    all_gm        = {f.name for f in gm_files}
    flagged_names = set(primary) | set(secondary)
    kept          = sorted(all_gm - flagged_names)

    print(f'\n{"="*60}')
    print(f'  Both filters      : {len(set(primary) & set(secondary)):4d} GM files  (confirmed duplicate)')
    print(f'  Primary only      : {len(set(primary) - set(secondary)):4d} GM files  (same piece, different performance)')
    print(f'  Secondary only    : {len(set(secondary) - set(primary)):4d} GM files  (duration match ±2s only)')
    print(f'  TOTAL EXCLUDED    : {len(flagged_names):4d} GM files')
    print(f'  Kept for training : {len(kept):4d} GM files')
    print(f'{"="*60}')

    return kept


# ── CQT preprocessing ──────────────────────────────────────────────────────────
def _wav_to_cqt(path: str) -> np.ndarray:
    """Returns (T, 352) float32 dB."""
    sr  = librosa.get_samplerate(path)
    hop = HOP_44100 if sr == 44100 else (HOP_48000 if sr == 48000 else int(sr / FPS))
    y, _ = librosa.load(path, sr=sr, mono=True, dtype=np.float32, res_type='kaiser_fast')
    r = len(y) % hop
    if r:
        y = np.pad(y, (0, hop - r), mode='constant')
    total_frames = len(y) // hop

    def _cqt(seg):
        C   = librosa.cqt(seg, sr=sr, hop_length=hop,
                          fmin=librosa.note_to_hz('A0'),
                          n_bins=N_BINS, bins_per_octave=BPO,
                          pad_mode='reflect')
        mag = np.abs(C).astype(np.float32)
        mag = np.nan_to_num(mag, nan=0.0, posinf=0.0, neginf=0.0)
        return librosa.amplitude_to_db(mag, ref=np.max).astype(np.float32)

    if total_frames <= SEG_THRESHOLD:
        return _cqt(y).T

    parts = []
    for sf in range(0, total_frames, SEG_THRESHOLD):
        ef  = min(sf + SEG_THRESHOLD, total_frames)
        ss  = max(0, sf * hop - SEG_PADDING)
        es  = min(len(y), ef * hop + SEG_PADDING)
        C   = _cqt(y[ss:es])
        off = (sf * hop - ss) // hop
        parts.append(C[:, off:off + (ef - sf)])
    return np.concatenate(parts, axis=1).T

def _split_chunks(X: np.ndarray):
    n = X.shape[0] // CHUNK_FRAMES
    return [X[i * CHUNK_FRAMES:(i + 1) * CHUNK_FRAMES] for i in range(n)]


# ── Main ───────────────────────────────────────────────────────────────────────
def main(args):
    gm_dir   = Path(args.gm_dir)
    out_root = Path(args.out_root)
    out_audio = out_root / 'audio'
    out_audio.mkdir(parents=True, exist_ok=True)

    # Step 1: filter
    kept = filter_gm(em_root=Path(args.em_root), gm_dir=gm_dir)

    # Step 2: preprocess
    print(f'\nPreprocessing {len(kept)} kept files → {out_audio}')
    saved, skipped, resumed = 0, [], 0

    for fname in tqdm(kept, unit='file'):
        src  = gm_dir / fname
        stem = Path(fname).stem

        if (out_audio / f'{stem}_chunk000000.npy').exists():
            resumed += 1
            continue

        if not src.exists():
            skipped.append(f'{fname}: file not found')
            continue

        try:
            cqt    = _wav_to_cqt(str(src))
            chunks = _split_chunks(cqt)
            for i, c in enumerate(chunks):
                np.save(out_audio / f'{stem}_chunk{i:06d}.npy', c)
            saved += len(chunks)
        except Exception as e:
            skipped.append(f'{fname}: {e}')

    meta = {
        'format': '(256,352) float32 dB bpo=48',
        'fps': FPS, 'chunk_frames': CHUNK_FRAMES,
        'kept_files': len(kept),
        'skipped': skipped,
    }
    (out_root / 'meta.json').write_text(json.dumps(meta, indent=2))

    print(f'\n{"="*55}')
    print(f'Saved chunks : {saved}')
    print(f'Resumed      : {resumed} files (already done)')
    print(f'Skipped      : {len(skipped)} files')
    for s in skipped:
        print(f'  [SKIP] {s}')
    print(f'{"="*55}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Filter Gardner Museum audio against MusicNet-EM and preprocess into CQT chunks.'
    )
    ap.add_argument('--em_root', type=str, required=True,
                    help='Root of MusicNet-EM dataset (must contain musicnet_em/ and musicnet_metadata.csv)')
    ap.add_argument('--gm_dir', type=str, required=True,
                    help='Directory containing Gardner Museum audio files (.mp3)')
    ap.add_argument('--out_root', type=str, default='./gm_preprocessed',
                    help='Output directory for CQT chunks')
    args = ap.parse_args()
    main(args)
