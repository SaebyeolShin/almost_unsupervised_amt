#!/usr/bin/env python3
"""
Create a named experiment split from preprocessed musicnet-EM chunks.

Reads from:
  {preprocessed_root}/audio/{id}_chunk*.npy
  {preprocessed_root}/midi/{id}_chunk*.npy

Creates symlinks under:
  {preprocessed_root}/{split_name}/
    test/audio/          <- original MusicNet 10 fixed test songs only
    test/midi/
    val/audio/           <- n_val_songs held out from paired (song-level, fixed by seed)
    val/midi/
    paired/audio/        <- top n_paired songs minus val songs (train only)
    paired/midi/
    unpaired_midi/midi/  <- all remaining musicnet-EM songs (excludes fixed test + paired + val)
    split_manifest.json

Usage:
  python musicnet_em_make_split.py --split_name main
  python musicnet_em_make_split.py --split_name main --n_paired 65 --n_val_songs 5 --seed 42
"""

import argparse, json, os, random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ── Fixed test IDs (original MusicNet benchmark) ──────────────────────────────
FIXED_TEST_IDS = {
    '1759', '1819', '2106', '2191', '2298',
    '2303', '2382', '2416', '2556', '2628',
}

PREPROCESSED_ROOT = Path('./musicnet_em_preprocessed')


def score_songs_by_instrument_coverage(root: Path, song_ids: List[str]) -> List[Tuple[str, int, int, List[int]]]:
    """
    For each song, count per-instrument active chunks.
    Returns list of (song_id, n_instruments, total_active_chunks, per_inst_counts)
    sorted by (n_instruments desc, total_active_chunks desc).
    """
    midi_dir = root / 'midi'
    scored = []
    for sid in song_ids:
        chunks = sorted(midi_dir.glob(f'{sid}_chunk*.npy'))
        if not chunks:
            scored.append((sid, 0, 0, [0] * 11))
            continue
        inst_counts = np.zeros(11, dtype=int)
        for cp in chunks:
            arr = np.load(cp, allow_pickle=False, mmap_mode='r')  # (C,T,88)
            active = (arr > 0).any(axis=(1, 2))                   # (C,)
            inst_counts += active.astype(int)
        n_inst = int((inst_counts > 0).sum())
        total  = int(inst_counts.sum())
        scored.append((sid, n_inst, total, inst_counts.tolist()))
    scored.sort(key=lambda x: (-x[1], -x[2]))
    return scored


def assign_splits(root: Path, all_ids: List[str],
                  n_paired: int, n_val_songs: int, seed: int) -> Dict[str, List[str]]:
    rng   = random.Random(seed)
    fixed = [i for i in all_ids if i in FIXED_TEST_IDS]
    pool  = [i for i in all_ids if i not in FIXED_TEST_IDS]

    # Sort pool by instrument coverage; top n_paired become paired+val candidates,
    # rest become unpaired_midi.
    print('  Scoring songs by instrument coverage...')
    scored = score_songs_by_instrument_coverage(root, pool)

    paired_pool = [sid for sid, *_ in scored[:n_paired]]
    unpaired    = [sid for sid, *_ in scored[n_paired:]]

    # Song-level val split from paired_pool (fixed by seed)
    val_songs   = sorted(rng.sample(paired_pool, min(n_val_songs, len(paired_pool))))
    val_set     = set(val_songs)
    paired      = [sid for sid in paired_pool if sid not in val_set]

    # Print coverage summary per split
    INST_NAMES = ['Piano','Harpsi','Violin','Viola','Cello',
                  'Pizz','FrHorn','Oboe','Bassoon','Clarinet','Flute']

    paired_scored   = [s for s in scored[:n_paired] if s[0] not in val_set]
    val_scored      = [s for s in scored[:n_paired] if s[0] in val_set]
    unpaired_scored = scored[n_paired:]

    for split_name, split_scored in [('paired (train)', paired_scored),
                                     ('val',            val_scored),
                                     ('unpaired_midi',  unpaired_scored)]:
        n_inst_dist: Dict[int, int] = {}
        per_inst = np.zeros(11, dtype=int)
        for _, n_inst, _, counts in split_scored:
            n_inst_dist[n_inst] = n_inst_dist.get(n_inst, 0) + 1
            per_inst += np.array(counts, dtype=int)
        print(f'\n  [{split_name}] {len(split_scored)} songs')
        for k in sorted(n_inst_dist, reverse=True):
            print(f'    {k} instruments: {n_inst_dist[k]} songs')
        print(f'  Per-instrument active chunks:')
        for c, cnt in enumerate(per_inst):
            print(f'    inst{c:02d} ({INST_NAMES[c]:8s}): {cnt}')

    return {
        'test':          sorted(fixed),
        'val':           sorted(val_songs),
        'paired':        sorted(paired),
        'unpaired_midi': sorted(unpaired),
    }


def get_chunks_for_song(root: Path, modality: str, song_id: str) -> List[Path]:
    """Return sorted list of chunk paths for a given song and modality."""
    return sorted((root / modality).glob(f'{song_id}_chunk*.npy'))


def make_symlinks(chunk_paths: List[Path], link_dir: Path):
    """Create relative symlinks in link_dir pointing to chunk_paths."""
    link_dir.mkdir(parents=True, exist_ok=True)
    for src in chunk_paths:
        dst = link_dir / src.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        # relative path from link_dir to src
        rel = os.path.relpath(src, link_dir)
        dst.symlink_to(rel)


def main(args):
    root      = Path(args.preprocessed_root)
    split_dir = root / args.split_name

    # Discover all song IDs from midi directory
    all_ids = sorted({p.name.rsplit('_chunk', 1)[0]
                      for p in (root / 'midi').glob('*_chunk*.npy')})
    print(f'Preprocessed songs found: {len(all_ids)}')

    # Assign splits
    splits = assign_splits(root, all_ids,
                           n_paired=args.n_paired,
                           n_val_songs=args.n_val_songs,
                           seed=args.seed)
    for name, ids in splits.items():
        print(f'  {name:15s}: {len(ids)} songs')

    # Build symlinks
    link_map = [
        ('test',          'audio', split_dir / 'test'          / 'audio'),
        ('test',          'midi',  split_dir / 'test'          / 'midi'),
        ('val',           'audio', split_dir / 'val'           / 'audio'),
        ('val',           'midi',  split_dir / 'val'           / 'midi'),
        ('paired',        'audio', split_dir / 'paired'        / 'audio'),
        ('paired',        'midi',  split_dir / 'paired'        / 'midi'),
        ('unpaired_midi', 'midi',  split_dir / 'unpaired_midi' / 'midi'),
    ]

    chunk_counts = {}
    for split_name, modality, link_dir in link_map:
        song_ids = splits[split_name]
        # unpaired_midi has no audio
        if split_name == 'unpaired_midi' and modality == 'audio':
            continue

        chunks = []
        for sid in song_ids:
            chunks.extend(get_chunks_for_song(root, modality, sid))

        make_symlinks(chunks, link_dir)
        key = f'{split_name}/{modality}'

        chunk_counts[key] = len(chunks)
        print(f'  {key:30s}: {len(chunks)} symlinks → {link_dir}')

    # Save manifest
    manifest = {
        'split_name':     args.split_name,
        'seed':           args.seed,
        'n_paired':       args.n_paired,
        'n_val_songs':    args.n_val_songs,
        'fixed_test_ids': sorted(FIXED_TEST_IDS),
        'splits':         splits,
        'chunk_counts':   chunk_counts,
    }
    manifest_path = split_dir / 'split_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f'\nManifest saved → {manifest_path}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--split_name',        type=str, default='main')
    ap.add_argument('--preprocessed_root', type=str,
                    default=str(PREPROCESSED_ROOT))
    ap.add_argument('--n_paired',          type=int, default=65)
    ap.add_argument('--n_val_songs',       type=int, default=5)
    ap.add_argument('--seed',              type=int, default=42)
    args = ap.parse_args()
    main(args)
