#!/usr/bin/env python3
"""
Preprocess ALL musicnet-EM songs into a shared directory.

Output format:
  midi  : (256, 88)  float32, 0-127
  audio : (256, 352) float32, dB  (bpo=48)

Default output root: ./musicnet_em_preprocessed/
  audio/{id}_chunk{i:06d}.npy
  midi/{id}_chunk{i:06d}.npy

Supports resume: skips songs whose chunks already exist.
"""

import argparse, json
from pathlib import Path
from typing import List

import numpy as np
import pretty_midi
import librosa
from tqdm import tqdm

# ── Constants ──────────────────────────────────────────────────────────────────
NOTE_MIN, NOTE_MAX = 21, 108
N_NOTES       = NOTE_MAX - NOTE_MIN + 1
FPS           = 50
CHUNK_FRAMES  = 256
BPO           = 48
N_BINS        = 352
HOP_44100     = 882
HOP_48000     = 960
SEG_THRESHOLD = 15360
SEG_PADDING   = 60000

# ── Instrument mapping ────────────────────────────────────────────────────────
# EM MIDI files use 0-indexed pretty_midi programs (Piano=0, Violin=40, ...).
# The original MusicNet annotations use 1-indexed programs, so preprocess_musicnet.py
# used [1,7,41,...]. Here we use the actual 0-indexed programs found in the EM files.
UNIQ_PROGRAMS = [0, 6, 40, 41, 42, 45, 60, 68, 70, 71, 73]
# inst00=Piano, inst01=Harpsichord, inst02=Violin, inst03=Viola, inst04=Cello,
# inst05=Pizzicato, inst06=FrenchHorn, inst07=Oboe, inst08=Bassoon,
# inst09=Clarinet, inst10=Flute
INST2IDX      = {p: i for i, p in enumerate(UNIQ_PROGRAMS)}
K             = len(UNIQ_PROGRAMS)  # 11

# ── Source paths (set via --data_root) ────────────────────────────────────────
EM_ROOT     = None  # set in main() from --data_root
MIDI_DIR    = None
TRAIN_AUDIO = None
TEST_AUDIO  = None


# ── Helpers ────────────────────────────────────────────────────────────────────
def find_audio(song_id: str) -> Path:
    for d in (TEST_AUDIO, TRAIN_AUDIO):
        p = d / f'{song_id}.wav'
        if p.exists():
            return p
    raise FileNotFoundError(f'No audio for song {song_id}')


def pad_to_hop(y: np.ndarray, hop: int) -> np.ndarray:
    r = len(y) % hop
    return np.pad(y, (0, hop - r), mode='constant') if r else y


def wav_to_cqt(wav_path: str) -> np.ndarray:
    """Returns (T, 352) float32 dB."""
    sr  = librosa.get_samplerate(wav_path)
    hop = HOP_44100 if sr == 44100 else (HOP_48000 if sr == 48000 else int(sr / FPS))
    y, _ = librosa.load(wav_path, sr=sr, mono=True, dtype=np.float32, res_type='kaiser_fast')
    y = pad_to_hop(y, hop)
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


def midi_to_pianoroll(midi_path: str, T: int) -> np.ndarray:
    """Returns (K, T, 88) float32, 0-127. Per-instrument channels."""
    pm   = pretty_midi.PrettyMIDI(midi_path)
    roll = np.zeros((K, T, N_NOTES), dtype=np.float32)
    for inst in pm.instruments:
        if inst.program not in INST2IDX:
            continue
        k       = INST2IDX[inst.program]
        ir      = inst.get_piano_roll(fs=FPS)[NOTE_MIN:NOTE_MAX + 1]  # (88, T_inst)
        T_use   = min(T, ir.shape[1])
        roll[k, :T_use, :] = ir[:, :T_use].T
    return roll


def chunk(X: np.ndarray) -> List[np.ndarray]:
    n = X.shape[0] // CHUNK_FRAMES
    return [X[i * CHUNK_FRAMES:(i + 1) * CHUNK_FRAMES] for i in range(n)]


def chunk_roll(roll: np.ndarray) -> List[np.ndarray]:
    """roll: (K, T, P) -> list of (K, CHUNK_FRAMES, P)"""
    n = roll.shape[1] // CHUNK_FRAMES
    return [roll[:, i * CHUNK_FRAMES:(i + 1) * CHUNK_FRAMES, :] for i in range(n)]


# ── Main ───────────────────────────────────────────────────────────────────────
def main(args):
    global EM_ROOT, MIDI_DIR, TRAIN_AUDIO, TEST_AUDIO
    EM_ROOT     = Path(args.data_root)
    MIDI_DIR    = EM_ROOT / 'musicnet_em'
    TRAIN_AUDIO = EM_ROOT / 'train_data'
    TEST_AUDIO  = EM_ROOT / 'test_data'

    out       = Path(args.out_root)
    out_audio = out / 'audio'
    out_midi  = out / 'midi'
    out_audio.mkdir(parents=True, exist_ok=True)
    out_midi.mkdir(parents=True, exist_ok=True)

    all_ids = sorted(f.stem for f in MIDI_DIR.glob('*.mid'))
    print(f'Total EM songs: {len(all_ids)}')
    print(f'Output root   : {out}')

    saved_chunks = 0
    skipped_songs = []
    resumed = 0

    for sid in tqdm(all_ids, unit='song'):
        # Resume: skip if first chunk already exists
        if (out_midi / f'{sid}_chunk000000.npy').exists():
            resumed += 1
            continue

        try:
            cqt  = wav_to_cqt(str(find_audio(sid)))
            T    = len(cqt)
            roll = midi_to_pianoroll(str(MIDI_DIR / f'{sid}.mid'), T)

            roll_chunks = chunk_roll(roll)
            cqt_chunks  = chunk(cqt)

            for i, (rc, cc) in enumerate(zip(roll_chunks, cqt_chunks)):
                stem = f'{sid}_chunk{i:06d}'
                np.save(out_midi  / f'{stem}.npy', rc)
                np.save(out_audio / f'{stem}.npy', cc)
            saved_chunks += len(roll_chunks)

        except Exception as e:
            skipped_songs.append(f'{sid}: {e}')

    # Save metadata
    meta = {
        'format': {'midi': '(256,88) float32 0-127', 'audio': '(256,352) float32 dB bpo=48'},
        'fps': FPS, 'chunk_frames': CHUNK_FRAMES,
        'total_songs': len(all_ids),
        'skipped': skipped_songs,
    }
    (out / 'meta.json').write_text(json.dumps(meta, indent=2))

    print(f'\n{"="*55}')
    print(f'Saved chunks : {saved_chunks}')
    print(f'Resumed      : {resumed} songs (already done)')
    print(f'Skipped      : {len(skipped_songs)} songs')
    for s in skipped_songs:
        print(f'  [SKIP] {s}')
    print(f'{"="*55}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', type=str, required=True,
                    help='Root of MusicNet EM dataset (must contain musicnet_em/, train_data/, test_data/)')
    ap.add_argument('--out_root', type=str, default='./musicnet_em_preprocessed',
                    help='Output directory for preprocessed chunks')
    args = ap.parse_args()
    main(args)
