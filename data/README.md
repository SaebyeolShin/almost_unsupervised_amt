# Data Preprocessing

This directory contains all preprocessing scripts. Run them in order for each dataset.

---

## Piano (MAESTRO)

### 1. Download MAESTRO

Download MAESTRO v3.0.0 from https://magenta.tensorflow.org/datasets/maestro and extract to a local directory.

### 2. Preprocess

```bash
python data/preprocess_maestro.py \
    --input_dir /path/to/maestro-v3.0.0 \
    --out_root ./maestro_preprocessed
```

Output: `./maestro_preprocessed/{train,validation,test}/{audio,midi}/*.npy`

- `audio/`: CQT spectrograms, shape `(256, 352)`, float32 dB (bpo=48)
- `midi/`: Piano rolls, shape `(256, 88)`, float32 0–127

### 3. (Optional) GuitarSet domain mixing

Download GuitarSet from https://guitarset.weebly.com/ and extract the audio files.

```bash
python data/preprocess_guitarset.py \
    --input_dir /path/to/guitarset/audio_mono-mic \
    --out_root ./guitarset_preprocessed
```

Output: `./guitarset_preprocessed/train/audio/*.npy`

Pass `data.guitar_root=./guitarset_preprocessed` to `train_piano.py` to enable CQT domain mixing.

---

## Multi-Instrument (MusicNet-EM)

### 1. Download MusicNet

Download MusicNet from https://zenodo.org/record/5120004 (the EM variant with aligned MIDI).
Extract to a local directory containing `musicnet_em/`, `train_data/`, `test_data/`, and `musicnet_metadata.csv`.

### 2. Preprocess MusicNet-EM

```bash
python data/preprocess_musicnet_em.py \
    --data_root /path/to/musicnet \
    --out_root ./musicnet_em_preprocessed
```

Output: `./musicnet_em_preprocessed/{audio,midi}/*.npy`

- `audio/`: CQT spectrograms, shape `(256, 352)`, float32 dB
- `midi/`: Multi-instrument rolls, shape `(11, 256, 88)`, float32 0–127
  - 11 channels: Piano, Harpsichord, Violin, Viola, Cello, Pizzicato, French Horn, Oboe, Bassoon, Clarinet, Flute

### 3. Create train/val/test split

```bash
python data/musicnet_em_make_split.py \
    --preprocessed_root ./musicnet_em_preprocessed \
    --split_name main
```

Output: `./musicnet_em_preprocessed/main/{paired,val,test}/{audio,midi}/*.npy` (symlinks)

10 songs are held out as a fixed test set; the rest are split into paired train / val.
Pass `data.split_root=./musicnet_em_preprocessed/main` to `train_musicnet.py`.

### 4. Download and preprocess Gardner Museum audio (unpaired CQT)

**Step 1 — Download** the Isabella Stewart Gardner Museum concert recordings using the provided URL list:

```bash
python data/gardener_url_download.py \
    --urls data/urls.txt \
    --out /path/to/gardner_museum_audio
```

**Step 2 — Preprocess** the downloaded audio into CQT chunks:

```bash
python data/preprocess_gardner_museum.py \
    --em_root /path/to/musicnet \
    --gm_dir /path/to/gardner_museum_audio \
    --out_root ./gm_preprocessed
```

This script automatically:
1. Filters out Gardner Museum recordings that overlap with MusicNet-EM (by filename and duration)
2. Preprocesses the remaining recordings into CQT chunks

Output: `./gm_preprocessed/audio/*.npy`

Pass `data.unpaired_audio_dir=./gm_preprocessed/audio` to `train_musicnet.py`.

---

## Verifying alignment

To visually verify that CQT and MIDI chunks are temporally aligned:

```bash
python data/verify_chunk_alignment.py \
    --wav_dir  ./musicnet_em_preprocessed/audio \
    --midi_dir ./musicnet_em_preprocessed/midi \
    --num_files 4 \
    --output_dir ./alignment_check
```