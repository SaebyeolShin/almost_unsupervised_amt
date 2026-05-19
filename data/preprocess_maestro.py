import os, csv, argparse, json
from pathlib import Path
import numpy as np
import pretty_midi, librosa
from tqdm import tqdm  

NOTE_MIN, NOTE_MAX = 21, 108
NUM_NOTES = NOTE_MAX - NOTE_MIN + 1
FPS = 50
CHUNK_FRAMES = 256

# Fixed hop lengths for different sample rates
HOP_LENGTH_44100 = 882  # 44100 / 50 = 882
HOP_LENGTH_48000 = 960  # 48000 / 50 = 960

def split_chunks(X, frames=CHUNK_FRAMES):
    n = X.shape[0] // frames
    return [X[i*frames:(i+1)*frames] for i in range(n)]

def count_existing_chunks(base_name, out_dir):
    """Count how many chunks already exist for a given base_name"""
    if not out_dir.exists():
        return 0
    chunk_pattern = f"{base_name}_chunk*.npy"
    existing_chunks = list(out_dir.glob(chunk_pattern))
    return len(existing_chunks)

def get_expected_chunk_count(file_path, mode, fps=FPS, chunk_frames=CHUNK_FRAMES):
    """Calculate expected number of chunks for a file (matching actual preprocessing logic)"""
    if mode == 'midi':
        pm = pretty_midi.PrettyMIDI(str(file_path))
        midi_roll = pm.get_piano_roll(fs=fps)
        midi_frames = midi_roll.shape[1]
        
        # For MIDI-only mode, just use MIDI frames
        total_frames = midi_frames
    else:  # wav or both mode
        actual_sr = librosa.get_samplerate(str(file_path))
        if actual_sr == 44100:
            hop_length = HOP_LENGTH_44100
        elif actual_sr == 48000:
            hop_length = HOP_LENGTH_48000
        else:
            hop_length = int(actual_sr / fps)
        
        # Load audio length
        y, _ = librosa.load(str(file_path), sr=actual_sr, mono=True, dtype=np.float32, res_type="kaiser_fast")
        
        # Apply padding (MT3-style) to match what wav_to_cqt does
        remainder = len(y) % hop_length
        if remainder != 0:
            pad_amount = hop_length - remainder
            padded_length = len(y) + pad_amount
        else:
            padded_length = len(y)
        
        total_frames = padded_length // hop_length
    
    return total_frames // chunk_frames

def pad_audio_to_hop_length(audio, hop_length):
    """Pad audio to be divisible by hop_length (MT3-style)."""
    remainder = len(audio) % hop_length
    if remainder != 0:
        pad_amount = hop_length - remainder
        audio = np.pad(audio, [0, pad_amount], mode='constant')
    return audio

def midi_to_pianoroll(midi_path, fps=FPS):
    pm = pretty_midi.PrettyMIDI(midi_path)
    roll = pm.get_piano_roll(fs=fps)[NOTE_MIN:NOTE_MAX+1]    # (88, T)
    return roll.T.astype(np.float32)                       # (T, 88)

def wav_to_cqt(wav_path, bins_per_octave=12, auto_segment=True, segment_threshold=15360, debug=False):
    """
    Convert audio to CQT using file's native sample rate.
    Pads audio to be divisible by hop_length (MT3-style).
    Automatically uses segmentation for long files to save memory.
    
    Args:
        wav_path: Path to audio file
        bins_per_octave: CQT bins per octave
        auto_segment: If True, automatically use segmentation for files > segment_threshold frames
        segment_threshold: Threshold in frames (default: 15360 = ~5 min at 50 FPS)
    """
    # Get actual sample rate of the file using librosa
    try:
        # Just peek at the file metadata without loading audio
        actual_sr = librosa.get_samplerate(wav_path)
    except Exception as e:
        raise RuntimeError(f"Failed to get sample rate from {wav_path}: {e}") from e
    
    # Determine appropriate hop_length for this sample rate
    if actual_sr == 44100:
        hop_length = HOP_LENGTH_44100  # 882
    elif actual_sr == 48000:
        hop_length = HOP_LENGTH_48000  # 960
    else:
        # Fallback for unusual sample rates
        hop_length = int(actual_sr / FPS)
    
    # Load at native sample rate (no resampling)
    y, _ = librosa.load(wav_path, sr=actual_sr, mono=True, dtype=np.float32, res_type="kaiser_fast")
    
    # Pad audio to be divisible by hop_length (MT3-style)
    y = pad_audio_to_hop_length(y, hop_length)
    
    # Calculate total frames
    total_frames = len(y) // hop_length
    
    if debug:
        print(f"[DEBUG] Audio length: {len(y)} samples, total_frames: {total_frames}, segment_threshold: {segment_threshold}")
    
    # Decide whether to use segmentation
    use_segmentation = auto_segment and total_frames > segment_threshold
    
    if debug:
        print(f"[DEBUG] use_segmentation: {use_segmentation} (auto_segment={auto_segment}, total_frames > segment_threshold: {total_frames > segment_threshold})")
    
    # Calculate number of bins based on bins_per_octave
    # 88 piano keys span 7.25 octaves (A0 to C8)
    # So n_bins = bins_per_octave * 7.25, but we round to match piano range
    if bins_per_octave == 12:
        n_bins = 88  # Standard: 12 bins/octave * 7.33 octaves ≈ 88
    else:
        # For higher resolutions: scale proportionally
        # 88 keys with 12 bins/octave → 88 * (bins_per_octave / 12)
        n_bins = int(88 * bins_per_octave / 12)
    
    if not use_segmentation:
        # Process entire audio at once (for short files)
        C = librosa.cqt(
            y, sr=actual_sr, hop_length=hop_length,
            fmin=librosa.note_to_hz('A0'),
            n_bins=n_bins, bins_per_octave=bins_per_octave,
            pad_mode="reflect"
        )
        mag = np.abs(C).astype(np.float32)
        mag = np.nan_to_num(mag, nan=0.0, posinf=0.0, neginf=0.0)
        return librosa.amplitude_to_db(mag, ref=np.max).T.astype(np.float32)  # (T, n_bins)
    
    else:
        # Process in segments to save memory (for long files)
        segment_samples = segment_threshold * hop_length
        padding = 60000  # Extra samples for filter support
        
        cqt_list = []
        
        if debug:
            print(f"[DEBUG] Segmentation: total_frames={total_frames}, will process {(total_frames + segment_threshold - 1) // segment_threshold} segments")
        
        for start_frame in range(0, total_frames, segment_threshold):
            end_frame = min(start_frame + segment_threshold, total_frames)
            
            if debug:
                print(f"[DEBUG] Segment: start_frame={start_frame}, end_frame={end_frame}, frames_in_seg={end_frame-start_frame}")
            
            # Calculate sample indices with padding
            start_sample = max(0, start_frame * hop_length - padding)
            end_sample = min(len(y), end_frame * hop_length + padding)
            
            # Extract segment
            y_segment = y[start_sample:end_sample]
            
            if debug:
                print(f"[DEBUG]   y_segment: start_sample={start_sample}, end_sample={end_sample}, length={len(y_segment)}")
            
            # Compute CQT for segment
            C_segment = librosa.cqt(
                y_segment, sr=actual_sr, hop_length=hop_length,
                fmin=librosa.note_to_hz('A0'),
                n_bins=n_bins, bins_per_octave=bins_per_octave,
                pad_mode="reflect"
            )
            
            if debug:
                print(f"[DEBUG]   C_segment after CQT: shape={C_segment.shape}")
            
            # Calculate frame offset due to padding
            # The CQT of y_segment starts at sample 0 of y_segment
            # We need to figure out which CQT frames correspond to our target frames
            
            # How many samples of left padding did we actually include?
            actual_left_padding = start_frame * hop_length - start_sample
            
            # The CQT will produce frames starting from sample 0 of y_segment
            # So frame 0 of C_segment corresponds to sample start_sample in original audio
            # Which corresponds to frame (start_sample // hop_length) in original audio
            # We want frames corresponding to [start_frame, end_frame) in original audio
            
            # Number of frames the CQT should produce (approximately)
            expected_cqt_frames = len(y_segment) // hop_length
            
            # The offset into C_segment where our target frames start
            # This is how many frames into C_segment we need to skip
            frame_offset = actual_left_padding // hop_length
            
            if debug:
                print(f"[DEBUG]   actual_left_padding={actual_left_padding}, frame_offset={frame_offset}, C_segment.shape[1]={C_segment.shape[1]}")
                print(f"[DEBUG]   will extract [{frame_offset}:{frame_offset + (end_frame - start_frame)}]")
            
            # Extract frames for this segment (accounting for padding)
            frames_in_segment = end_frame - start_frame
            C_segment = C_segment[:, frame_offset:frame_offset + frames_in_segment]
            
            if debug:
                print(f"[DEBUG] After extraction: C_segment shape = {C_segment.shape}, extracted {frames_in_segment} frames")
            
            cqt_list.append(C_segment)
        
        # Concatenate all segments
        C = np.concatenate(cqt_list, axis=1)
        
        mag = np.abs(C).astype(np.float32)
        mag = np.nan_to_num(mag, nan=0.0, posinf=0.0, neginf=0.0)
        return librosa.amplitude_to_db(mag, ref=np.max).T.astype(np.float32)  # (T, 88 or 352)

def process_aligned_pair(midi_path, wav_path, bins_per_octave=12, debug=False):
    """
    Process MIDI and audio pair with pre-alignment.
    Determines target length first, then pads/trims both to match before CQT computation.
    
    Args:
        midi_path: Path to MIDI file
        wav_path: Path to audio file
        bins_per_octave: CQT resolution
        debug: Enable debug output
        
    Returns:
        (X_midi, X_wav): Aligned piano roll and CQT features, both with shape (T, ...)
    """
    # Get sample rate to determine hop_length
    actual_sr = librosa.get_samplerate(wav_path)
    if actual_sr == 44100:
        hop_length = HOP_LENGTH_44100
    elif actual_sr == 48000:
        hop_length = HOP_LENGTH_48000
    else:
        hop_length = int(actual_sr / FPS)
    
    # Load MIDI to get its length
    pm = pretty_midi.PrettyMIDI(midi_path)
    midi_roll = pm.get_piano_roll(fs=FPS)[NOTE_MIN:NOTE_MAX+1]
    X_midi = midi_roll.T.astype(np.float32)  # (T_midi, 88), 0~127
    midi_frames = X_midi.shape[0]
    
    # Load audio to get its length
    y, _ = librosa.load(wav_path, sr=actual_sr, mono=True, dtype=np.float32, res_type="kaiser_fast")
    
    # Audio determines the target length (ceiling division to ensure we cover all audio)
    audio_frames = (len(y) + hop_length - 1) // hop_length
    target_frames = audio_frames
    
    if debug:
        print(f"[DEBUG] Pre-alignment: MIDI={midi_frames} frames, Audio={len(y)} samples = {audio_frames} frames, Target={target_frames} frames")
    
    # Pad audio to exactly target_frames * hop_length samples
    target_samples = target_frames * hop_length
    if len(y) < target_samples:
        pad_amount = target_samples - len(y)
        y = np.pad(y, [0, pad_amount], mode='constant')
        if debug:
            print(f"[DEBUG] Padded audio by {pad_amount} samples to reach {target_samples} samples")
    
    # Match MIDI to audio length (pad or trim as needed)
    if len(X_midi) < target_frames:
        pad_amount = target_frames - len(X_midi)
        X_midi = np.pad(X_midi, [(0, pad_amount), (0, 0)], mode='constant')
        if debug:
            print(f"[DEBUG] Padded MIDI by {pad_amount} frames to reach {target_frames} frames")
    elif len(X_midi) > target_frames:
        trim_amount = len(X_midi) - target_frames
        X_midi = X_midi[:target_frames]
        if debug:
            print(f"[DEBUG] Trimmed MIDI by {trim_amount} frames to reach {target_frames} frames")
    
    if debug:
        print(f"[DEBUG] After padding: MIDI={len(X_midi)} frames, Audio={len(y)} samples = {len(y)//hop_length} frames")
    
    # Now compute CQT on the pre-aligned audio
    # Calculate n_bins
    if bins_per_octave == 12:
        n_bins = 88
    else:
        n_bins = int(88 * bins_per_octave / 12)
    
    # Decide if we need segmentation
    use_segmentation = target_frames > 15360
    
    if not use_segmentation:
        # Process entire audio at once
        C = librosa.cqt(
            y, sr=actual_sr, hop_length=hop_length,
            fmin=librosa.note_to_hz('A0'),
            n_bins=n_bins, bins_per_octave=bins_per_octave,
            pad_mode="reflect"
        )
        mag = np.abs(C).astype(np.float32)
        mag = np.nan_to_num(mag, nan=0.0, posinf=0.0, neginf=0.0)
        X_wav = librosa.amplitude_to_db(mag, ref=np.max).T.astype(np.float32)
    else:
        # Use segmented processing
        segment_threshold = 15360
        padding = 60000
        cqt_list = []
        
        for start_frame in range(0, target_frames, segment_threshold):
            end_frame = min(start_frame + segment_threshold, target_frames)
            
            # Calculate sample indices with padding
            start_sample = max(0, start_frame * hop_length - padding)
            end_sample = min(len(y), end_frame * hop_length + padding)
            
            # Extract segment
            y_segment = y[start_sample:end_sample]
            
            # Compute CQT for segment
            C_segment = librosa.cqt(
                y_segment, sr=actual_sr, hop_length=hop_length,
                fmin=librosa.note_to_hz('A0'),
                n_bins=n_bins, bins_per_octave=bins_per_octave,
                pad_mode="reflect"
            )
            
            # Calculate frame offset
            actual_left_padding = start_frame * hop_length - start_sample
            frame_offset = actual_left_padding // hop_length
            
            # Extract frames for this segment
            frames_in_segment = end_frame - start_frame
            C_segment = C_segment[:, frame_offset:frame_offset + frames_in_segment]
            
            cqt_list.append(C_segment)
        
        # Concatenate all segments
        C = np.concatenate(cqt_list, axis=1)
        mag = np.abs(C).astype(np.float32)
        mag = np.nan_to_num(mag, nan=0.0, posinf=0.0, neginf=0.0)
        X_wav = librosa.amplitude_to_db(mag, ref=np.max).T.astype(np.float32)
    
    # Final trim to ensure exact match (CQT might produce slightly different length)
    min_len = min(len(X_midi), len(X_wav))
    X_midi = X_midi[:min_len]
    X_wav = X_wav[:min_len]
    
    return X_midi, X_wav

def main(args):
    root = Path(args.input_dir)
    out_root = Path(args.out_root)

    for split in ['train', 'validation', 'test']:
        (out_root / split / 'midi').mkdir(parents=True, exist_ok=True)
        (out_root / split / 'audio').mkdir(parents=True, exist_ok=True)

    # Support both MAESTRO v2 and v3 CSV filenames
    for csv_name in ('maestro-v3.0.0.csv', 'maestro-v2.0.0.csv'):
        csv_path = root / csv_name
        if csv_path.exists():
            break
    else:
        raise FileNotFoundError(f"No MAESTRO CSV found in {root}")
    
    # Statistics for global normalization
    s1, s2, n = 0.0, 0.0, 0
    midi_chunk_count = {'train': 0, 'validation': 0, 'test': 0}
    wav_chunk_count = {'train': 0, 'validation': 0, 'test': 0}
    paired_count = {'train': 0, 'validation': 0, 'test': 0}
    skipped_files = []
    
    # Track sample rate statistics
    sr_44100_count = 0
    sr_48000_count = 0
    sr_other_count = 0

    print(f"[INFO] Processing MAESTRO dataset from {csv_path}")
    print(f"[INFO] Mode: {args.mode}")
    print(f"[INFO] Using native sample rates with fixed hop_lengths:")
    print(f"        44.1 kHz → hop_length = {HOP_LENGTH_44100}")
    print(f"        48.0 kHz → hop_length = {HOP_LENGTH_48000}")
    print(f"[INFO] Output will be organized by train/validation/test splits")
    
    if args.subfolder:
        print(f"[INFO] Processing only subfolder: {args.subfolder}")
    else:
        print(f"[INFO] Processing ALL subfolders")
    
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    
    # Filter by subfolder if specified
    if args.subfolder:
        original_count = len(rows)
        rows = [row for row in rows if row['midi_filename'].startswith(args.subfolder + '/') or 
                                        row['midi_filename'].startswith(args.subfolder + '\\')]
        print(f"[INFO] Filtered from {original_count} to {len(rows)} files in subfolder '{args.subfolder}'")
        
        if len(rows) == 0:
            print(f"[ERROR] No files found in subfolder '{args.subfolder}'")
            print(f"[INFO] Available subfolders in first few rows:")
            with open(csv_path) as f:
                sample_rows = list(csv.DictReader(f))[:10]
            subfolders = set()
            for row in sample_rows:
                subfolder = Path(row['midi_filename']).parts[0]
                subfolders.add(subfolder)
            for sf in sorted(subfolders):
                print(f"  - {sf}")
            return
    
    for row in tqdm(rows, desc="Processing pairs", unit="file"):
        midi_rel = row['midi_filename']
        wav_rel = row['audio_filename']
        split = row['split']  # Get split info from CSV
        midi_path = root / midi_rel
        wav_path = root / wav_rel
        
        base_name = Path(midi_rel).stem.replace('.mid', '')
        
        out_midi = out_root / split / 'midi'
        out_wav  = out_root / split / 'audio'
        
        try:
            # Check if file is already processed (skip if both MIDI and WAV chunks match expected count)
            skip_file = False
            if args.mode == 'both':
                # For 'both' mode, check if both MIDI and WAV chunks exist and match
                # Use WAV to determine expected count since preprocessing aligns MIDI to WAV
                if midi_path.exists() and wav_path.exists():
                    try:
                        expected_chunks = get_expected_chunk_count(wav_path, 'wav')
                        existing_midi = count_existing_chunks(base_name, out_midi)
                        existing_wav = count_existing_chunks(base_name, out_wav)
                        
                        if existing_midi == expected_chunks and existing_wav == expected_chunks and expected_chunks > 0:
                            skip_file = True
                            if args.debug:
                                tqdm.write(f"[SKIP] {base_name}: Already processed ({expected_chunks} chunks)")
                    except Exception:
                        pass  # If we can't check, process the file
            elif args.mode == 'midi':
                if midi_path.exists():
                    try:
                        expected_chunks = get_expected_chunk_count(midi_path, 'midi')
                        existing_chunks = count_existing_chunks(base_name, out_midi)
                        if existing_chunks == expected_chunks and expected_chunks > 0:
                            skip_file = True
                            if args.debug:
                                tqdm.write(f"[SKIP] {base_name}: Already processed ({expected_chunks} chunks)")
                    except Exception:
                        pass
            elif args.mode == 'wav':
                if wav_path.exists():
                    try:
                        expected_chunks = get_expected_chunk_count(wav_path, 'wav')
                        existing_chunks = count_existing_chunks(base_name, out_wav)
                        if existing_chunks == expected_chunks and expected_chunks > 0:
                            skip_file = True
                            if args.debug:
                                tqdm.write(f"[SKIP] {base_name}: Already processed ({expected_chunks} chunks)")
                    except Exception:
                        pass
            
            if skip_file:
                continue
            
            # Check actual sample rate
            if wav_path.exists():
                actual_sr = librosa.get_samplerate(str(wav_path))
                if actual_sr == 44100:
                    sr_44100_count += 1
                elif actual_sr == 48000:
                    sr_48000_count += 1
                else:
                    sr_other_count += 1
                    tqdm.write(f"[WARN] Unusual sample rate {actual_sr} Hz for {base_name}")
            
            if args.mode == 'midi':
                if not midi_path.exists():
                    skipped_files.append(f"MIDI not found: {midi_path}")
                    continue
                    
                X_midi = midi_to_pianoroll(str(midi_path))
                midi_chunks = split_chunks(X_midi, CHUNK_FRAMES)
                
                for i, chunk in enumerate(midi_chunks):
                    if chunk.shape[0] != CHUNK_FRAMES:
                        continue
                    np.save(out_midi / f"{base_name}_chunk{i:04d}.npy", chunk)
                    midi_chunk_count[split] += 1
                    
            elif args.mode == 'wav':
                if not wav_path.exists():
                    skipped_files.append(f"WAV not found: {wav_path}")
                    continue

                X_wav = wav_to_cqt(str(wav_path), args.bins_per_octave, debug=args.debug)
                wav_chunks = split_chunks(X_wav, CHUNK_FRAMES)
                
                for i, chunk in enumerate(wav_chunks):
                    if chunk.shape[0] != CHUNK_FRAMES:  # Only check time dimension
                        continue
                    np.save(out_wav / f"{base_name}_chunk{i:04d}.npy", chunk.astype(np.float32))
                    wav_chunk_count[split] += 1
                    
                    if args.compute_global_norm:
                        tmp = chunk.astype(np.float64, copy=False)
                        s1 += tmp.sum()
                        s2 += (tmp * tmp).sum()
                        n += tmp.size
                        
            elif args.mode == 'both':
                # Check both files exist
                if not midi_path.exists():
                    skipped_files.append(f"MIDI not found: {midi_path}")
                    continue
                if not wav_path.exists():
                    skipped_files.append(f"WAV not found: {wav_path}")
                    continue
                
                # Process with pre-alignment: determine target length first
                X_midi, X_wav = process_aligned_pair(
                    str(midi_path), str(wav_path), 
                    args.bins_per_octave, debug=args.debug
                )
                
                if args.debug:
                    print(f"[DEBUG] {base_name}: After aligned processing - MIDI len={len(X_midi)}, WAV len={len(X_wav)}")
                
                # They should already be the same length
                assert len(X_midi) == len(X_wav), \
                    f"Length mismatch after aligned processing: MIDI={len(X_midi)}, WAV={len(X_wav)}"
                
                # Chunk both with same split points
                midi_chunks = split_chunks(X_midi, CHUNK_FRAMES)
                wav_chunks = split_chunks(X_wav, CHUNK_FRAMES)
                
                # They should have same length now
                assert len(midi_chunks) == len(wav_chunks), \
                    f"Chunk count mismatch: {len(midi_chunks)} vs {len(wav_chunks)}"
                
                # Save paired chunks with matching indices
                for i, (midi_chunk, wav_chunk) in enumerate(zip(midi_chunks, wav_chunks)):
                    # Double-check shapes - MIDI is always (256, 88), WAV varies by bins_per_octave
                    if midi_chunk.shape != (CHUNK_FRAMES, 88):
                        continue
                    if wav_chunk.shape[0] != CHUNK_FRAMES:  # Only check time dimension
                        continue
                    
                    chunk_id = f"{base_name}_chunk{i:04d}"
                    np.save(out_midi / f"{chunk_id}.npy", midi_chunk)
                    np.save(out_wav / f"{chunk_id}.npy", wav_chunk.astype(np.float32))
                    
                    paired_count[split] += 1
                    midi_chunk_count[split] += 1
                    wav_chunk_count[split] += 1
                    
                    if args.compute_global_norm:
                        tmp = wav_chunk.astype(np.float64, copy=False)
                        s1 += tmp.sum()
                        s2 += (tmp * tmp).sum()
                        n += tmp.size
                        
        except Exception as e:
            skipped_files.append(f"Error processing {base_name}: {e}")
            tqdm.write(f"[ERROR] {base_name}: {e}")
    
    # Print summary
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Sample rate statistics:")
    print(f"  44.1 kHz files: {sr_44100_count}")
    print(f"  48.0 kHz files: {sr_48000_count}")
    if sr_other_count > 0:
        print(f"  Other sample rates: {sr_other_count}")
    print()
    
    if args.mode == 'midi':
        for split in ['train', 'validation', 'test']:
            print(f"[MIDI {split:11s}] Total chunks saved: {midi_chunk_count[split]}")
    elif args.mode == 'wav':
        for split in ['train', 'validation', 'test']:
            print(f"[WAV {split:11s}] Total chunks saved: {wav_chunk_count[split]}")
    elif args.mode == 'both':
        for split in ['train', 'validation', 'test']:
            print(f"[PAIRED {split:11s}] Total paired chunks: {paired_count[split]}")
            print(f"[MIDI {split:11s}] Total chunks: {midi_chunk_count[split]}")
            print(f"[WAV {split:11s}] Total chunks: {wav_chunk_count[split]}")
    
    if skipped_files:
        print(f"\n[WARN] Skipped {len(skipped_files)} files")
        if len(skipped_files) <= 10:
            for msg in skipped_files:
                print(f"  - {msg}")
        else:
            for msg in skipped_files[:5]:
                print(f"  - {msg}")
            print(f"  ... and {len(skipped_files)-5} more")
    
    # Global normalization
    if args.compute_global_norm and n > 0:
        mean = float(s1 / n)
        var = max(1e-12, s2 / n - mean**2)
        std = float(np.sqrt(var))
        gnorm_path = out_root / "cqt_global_norm.json"
        with open(gnorm_path, "w") as f:
            json.dump({"mean": mean, "std": std, "num_samples": n}, f, indent=2)
        print(f"\n[NORM] Global statistics saved to: {gnorm_path}")
        print(f"  mean = {mean:.4f}, std = {std:.4f}")
    
    print("="*60)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, required=True,
                    help="MAESTRO root directory (must contain maestro-v3.0.0.csv and audio files)")
    ap.add_argument("--out_root", type=str, default="./maestro_preprocessed",
                    help="Output root; creates {split}/audio/ and {split}/midi/ subdirs")
    ap.add_argument("--mode", type=str, choices=["midi","wav","both"], default="both")
    ap.add_argument("--compute_global_norm", action="store_true")
    ap.add_argument("--bins_per_octave", type=int, default=12,
                    help="Number of bins per octave for CQT")
    ap.add_argument("--subfolder", type=str, default=None,
                    help="Process only files in this subfolder (e.g., '2004', '2006', '2008', etc.)")
    ap.add_argument("--debug", action="store_true",
                    help="Enable debug output for troubleshooting")
    args = ap.parse_args()
    main(args)