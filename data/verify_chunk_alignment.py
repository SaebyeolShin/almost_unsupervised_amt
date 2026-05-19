"""
Verify alignment of MIDI and CQT chunks by visualizing selected chunk pairs.
Example usage:
python verify_chunk_alignment.py --output_dir ./check_misalignment --file_pattern $file_name"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import argparse

def get_base_name(chunk_filename):
    """Extract base name from chunk filename (remove _chunkXXXX.npy)"""
    # e.g., "aaa_chunk0042.npy" -> "aaa"
    return '_'.join(chunk_filename.replace('.npy', '').split('_')[:-1])

def get_chunk_number(chunk_filename):
    """Extract chunk number from filename"""
    # e.g., "aaa_chunk0042.npy" -> 42
    chunk_part = chunk_filename.replace('.npy', '').split('_')[-1]
    return int(chunk_part.replace('chunk', ''))

def group_chunks_by_file(chunk_dir):
    """Group chunk files by their base name"""
    chunk_files = sorted(Path(chunk_dir).glob("*.npy"))
    
    file_groups = defaultdict(list)
    for chunk_file in chunk_files:
        base_name = get_base_name(chunk_file.name)
        chunk_num = get_chunk_number(chunk_file.name)
        file_groups[base_name].append((chunk_num, chunk_file))
    
    # Sort chunks within each group
    for base_name in file_groups:
        file_groups[base_name].sort(key=lambda x: x[0])
    
    return file_groups

def select_chunks_to_visualize(chunks, num_start=2, num_middle=2, num_end=2):
    """Select chunks from beginning, middle, and end"""
    total_chunks = len(chunks)
    
    if total_chunks < num_start + num_middle + num_end:
        # Not enough chunks, just return all
        return chunks
    
    # Beginning chunks
    start_chunks = chunks[:num_start]
    
    # End chunks
    end_chunks = chunks[-num_end:]
    
    # Middle chunks
    middle_idx = total_chunks // 2
    middle_start = max(num_start, middle_idx - num_middle // 2)
    middle_end = min(total_chunks - num_end, middle_start + num_middle)
    middle_chunks = chunks[middle_start:middle_end]
    
    return start_chunks, middle_chunks, end_chunks

def plot_chunk_pair(midi_chunk, wav_chunk, chunk_num, base_name, section, output_path=None):
    """Plot a single MIDI-WAV chunk pair"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    
    # Plot MIDI (piano roll)
    im0 = axes[0].imshow(midi_chunk.T, aspect='auto', origin='lower', 
                         cmap='gray_r', interpolation='nearest')
    axes[0].set_title(f'{base_name} - Chunk {chunk_num:04d} ({section}) - MIDI Piano Roll', 
                     fontsize=11, fontweight='bold')
    axes[0].set_ylabel('MIDI Note (21-108)', fontsize=9)
    axes[0].set_yticks([0, 21, 43, 65, 87])
    axes[0].set_yticklabels(['A0', 'A2', 'A4', 'A6', 'C8'])
    plt.colorbar(im0, ax=axes[0], label='Active')
    
    # Plot CQT (full resolution, no downsampling)
    im1 = axes[1].imshow(wav_chunk.T, aspect='auto', origin='lower', 
                         cmap='viridis', interpolation='nearest')
    axes[1].set_title(f'{base_name} - Chunk {chunk_num:04d} ({section}) - CQT ({wav_chunk.shape[1]} bins)', 
                     fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Frequency Bin', fontsize=9)
    axes[1].set_xlabel('Time Frames', fontsize=9)
    plt.colorbar(im1, ax=axes[1], label='dB')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def verify_alignment(midi_dir, wav_dir, output_dir=None, num_files=None, 
                    num_start=2, num_middle=2, num_end=2, file_pattern=None):
    """Main function to verify chunk alignment"""
    
    midi_dir = Path(midi_dir)
    wav_dir = Path(wav_dir)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Group chunks by base file name
    print("Grouping MIDI chunks...")
    midi_groups = group_chunks_by_file(midi_dir)
    print(f"Found {len(midi_groups)} unique files in MIDI directory")
    
    print("Grouping WAV chunks...")
    wav_groups = group_chunks_by_file(wav_dir)
    print(f"Found {len(wav_groups)} unique files in WAV directory")
    
    # Find common files
    common_files = set(midi_groups.keys()) & set(wav_groups.keys())
    print(f"Found {len(common_files)} files with both MIDI and WAV chunks")
    
    # Filter by pattern if specified
    if file_pattern:
        common_files = {f for f in common_files if file_pattern.lower() in f.lower()}
        print(f"After filtering by pattern '{file_pattern}': {len(common_files)} files")
    
    if not common_files:
        print("ERROR: No common files found! Check your directory paths or pattern.")
        return
    
    # Only in MIDI
    midi_only = set(midi_groups.keys()) - set(wav_groups.keys())
    if midi_only:
        print(f"WARNING: {len(midi_only)} files only have MIDI chunks")
        if len(midi_only) <= 5:
            for f in list(midi_only)[:5]:
                print(f"  - {f}")
    
    # Only in WAV
    wav_only = set(wav_groups.keys()) - set(midi_groups.keys())
    if wav_only:
        print(f"WARNING: {len(wav_only)} files only have WAV chunks")
        if len(wav_only) <= 5:
            for f in list(wav_only)[:5]:
                print(f"  - {f}")
    
    print("\n" + "="*80)
    
    # Limit number of files to process
    files_to_process = sorted(common_files)
    if num_files:
        files_to_process = files_to_process[:num_files]
        print(f"Processing first {num_files} files...")
    else:
        print(f"Processing all {len(files_to_process)} files...")
    
    # Process each file
    misaligned_files = []
    
    for base_name in files_to_process:
        midi_chunks = midi_groups[base_name]
        wav_chunks = wav_groups[base_name]
        
        print(f"\n{base_name}:")
        print(f"  MIDI chunks: {len(midi_chunks)}")
        print(f"  WAV chunks: {len(wav_chunks)}")
        
        # Check if chunk counts match
        if len(midi_chunks) != len(wav_chunks):
            print(f"  ⚠️  WARNING: Chunk count mismatch! MIDI={len(midi_chunks)}, WAV={len(wav_chunks)}")
            print(f"      Will visualize common chunks only")
            
            # Find common chunk numbers
            midi_nums_set = set(c[0] for c in midi_chunks)
            wav_nums_set = set(c[0] for c in wav_chunks)
            common_nums = sorted(midi_nums_set & wav_nums_set)
            
            if not common_nums:
                print(f"      No common chunk numbers found, skipping...")
                misaligned_files.append(base_name)
                continue
            
            print(f"      Found {len(common_nums)} common chunks")
            
            # Filter to only common chunks
            midi_chunks = [(num, path) for num, path in midi_chunks if num in common_nums]
            wav_chunks = [(num, path) for num, path in wav_chunks if num in common_nums]
        
        # Check if chunk numbers match
        midi_nums = [c[0] for c in midi_chunks]
        wav_nums = [c[0] for c in wav_chunks]
        if midi_nums != wav_nums:
            print(f"  ⚠️  WARNING: Chunk numbers don't match even after filtering!")
            misaligned_files.append(base_name)
            continue
        
        # Select chunks to visualize
        start_chunks, middle_chunks, end_chunks = select_chunks_to_visualize(
            midi_chunks, num_start, num_middle, num_end
        )
        
        print(f"  Visualizing: {len(start_chunks)} start + {len(middle_chunks)} middle + {len(end_chunks)} end")
        
        # Visualize selected chunks
        for section_name, chunks in [("START", start_chunks), 
                                     ("MIDDLE", middle_chunks), 
                                     ("END", end_chunks)]:
            for chunk_num, midi_path in chunks:
                # Find corresponding WAV chunk - try without leading zeros first
                wav_path = wav_dir / f"{base_name}_chunk{chunk_num}.npy"
                
                # If not found, try with leading zeros (old format)
                if not wav_path.exists():
                    wav_path = wav_dir / f"{base_name}_chunk{chunk_num:04d}.npy"
                
                if not wav_path.exists():
                    print(f"    ⚠️  Missing WAV chunk {chunk_num}")
                    continue
                
                # Load chunks
                try:
                    midi_chunk = np.load(midi_path)
                    wav_chunk = np.load(wav_path)
                except Exception as e:
                    print(f"    ⚠️  Error loading chunk {chunk_num}: {e}")
                    continue
                
                # Check shapes
                # Detect shape automatically from first chunk
                expected_midi_shape = midi_chunk.shape[0]
                expected_wav_bins = wav_chunk.shape[1]
                
                if midi_chunk.shape[1] != 88:
                    print(f"    ⚠️  Unexpected MIDI shape at chunk {chunk_num}: {midi_chunk.shape}")
                    misaligned_files.append(base_name)
                    continue
                
                if wav_chunk.shape[0] != midi_chunk.shape[0]:
                    print(f"    ⚠️  Shape mismatch at chunk {chunk_num}: MIDI={midi_chunk.shape}, WAV={wav_chunk.shape}")
                    misaligned_files.append(base_name)
                    continue
                
                # Plot with full resolution (no downsampling)
                if output_dir:
                    output_path = output_dir / f"{base_name}_{section_name}_chunk{chunk_num:04d}.png"
                    plot_chunk_pair(midi_chunk, wav_chunk, chunk_num, base_name, 
                                  section_name, output_path)
                    print(f"    ✓ Saved: {output_path.name}")
                else:
                    plot_chunk_pair(midi_chunk, wav_chunk, chunk_num, base_name, section_name)
        
        print(f"  ✓ {base_name} aligned correctly")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Files processed: {len(files_to_process)}")
    print(f"Files with issues: {len(misaligned_files)}")
    
    if misaligned_files:
        print("\nFiles with alignment issues:")
        for f in misaligned_files[:10]:
            print(f"  - {f}")
        if len(misaligned_files) > 10:
            print(f"  ... and {len(misaligned_files) - 10} more")
    else:
        print("\n✓ All files aligned correctly!")
    
    if output_dir:
        print(f"\nVisualizations saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Verify alignment of MIDI and CQT chunks'
    )
    parser.add_argument('--midi_dir', type=str,
                       help='Directory containing MIDI chunks (.npy files)',
                       default='./debug_midi')
    parser.add_argument('--wav_dir', type=str,
                       help='Directory containing WAV/CQT chunks (.npy files)',
                       default='./debug_wav')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save visualization plots (if None, display interactively)')
    parser.add_argument('--num_files', type=int, default=None,
                       help='Number of files to process (None = all)')
    parser.add_argument('--num_start', type=int, default=2,
                       help='Number of chunks to visualize from start')
    parser.add_argument('--num_middle', type=int, default=2,
                       help='Number of chunks to visualize from middle')
    parser.add_argument('--num_end', type=int, default=2,
                       help='Number of chunks to visualize from end')
    parser.add_argument('--file_pattern', type=str, default=None,
                       help='Filter files by pattern (e.g., "Schubert" or "2018" or "PIANO081")')
    
    args = parser.parse_args()
    
    verify_alignment(
        args.midi_dir,
        args.wav_dir,
        args.output_dir,
        args.num_files,
        args.num_start,
        args.num_middle,
        args.num_end,
        args.file_pattern
    )