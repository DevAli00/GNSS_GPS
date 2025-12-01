import numpy as np
import matplotlib.pyplot as plt
import struct
import os

def inspect_gps_wav(filename, duration_seconds=2):
    """
    Read and inspect GPS IQ data from WAV file (memory-efficient for large files)
    Handles both standard WAV and raw binary formats
    
    Parameters:
    -----------
    filename : str
        Path to the .wav file
    duration_seconds : int
        Number of seconds to load (to avoid memory issues)
    """
    print("=" * 50)
    print("PHASE 0: GPS Data Inspection")
    print("=" * 50)
    
    file_size = os.path.getsize(filename)
    print(f"\n📁 File Info:")
    print(f"  File: {filename}")
    print(f"  Size: {file_size / (1024**2):.1f} MB")
    
    # Try to read WAV header manually
    try:
        with open(filename, 'rb') as f:
            # Read RIFF header
            riff = f.read(4)
            if riff != b'RIFF':
                print(f"\n⚠️  Warning: Not a standard RIFF WAV file (header: {riff})")
                print("  Attempting to read as raw binary IQ data...")
                return read_raw_binary(filename, duration_seconds)
            
            file_size_header = struct.unpack('<I', f.read(4))[0]
            wave_header = f.read(4)
            
            if wave_header != b'WAVE':
                print(f"\n⚠️  Warning: Not a WAVE file (header: {wave_header})")
                return read_raw_binary(filename, duration_seconds)
            
            # Find fmt chunk
            fmt_found = False
            while True:
                chunk_id = f.read(4)
                if len(chunk_id) < 4:
                    break
                    
                chunk_size = struct.unpack('<I', f.read(4))[0]
                
                if chunk_id == b'fmt ':
                    fmt_found = True
                    # Read fmt chunk
                    audio_format = struct.unpack('<H', f.read(2))[0]
                    n_channels = struct.unpack('<H', f.read(2))[0]
                    fs = struct.unpack('<I', f.read(4))[0]
                    byte_rate = struct.unpack('<I', f.read(4))[0]
                    block_align = struct.unpack('<H', f.read(2))[0]
                    bits_per_sample = struct.unpack('<H', f.read(2))[0]
                    
                    print(f"\n📊 WAV Format Properties:")
                    print(f"  Audio Format: {audio_format} (1=PCM)")
                    print(f"  Channels: {n_channels}")
                    print(f"  Sample Rate (fs): {fs:,} Hz")
                    print(f"  Byte Rate: {byte_rate:,}")
                    print(f"  Block Align: {block_align}")
                    print(f"  Bits/Sample: {bits_per_sample}")
                    
                    # Skip any remaining fmt data
                    if chunk_size > 16:
                        f.read(chunk_size - 16)
                    
                elif chunk_id == b'data':
                    if not fmt_found:
                        raise ValueError("data chunk before fmt chunk!")
                    
                    data_size = chunk_size
                    n_frames = data_size // (n_channels * bits_per_sample // 8)
                    
                    print(f"\n📊 Data Properties:")
                    print(f"  Data Size: {data_size / (1024**2):.1f} MB")
                    print(f"  Total Frames: {n_frames:,}")
                    print(f"  Total Duration: {n_frames/fs:.2f} seconds")
                    
                    # Calculate how many frames to read
                    chunk_frames = int(duration_seconds * fs)
                    bytes_to_read = chunk_frames * n_channels * bits_per_sample // 8
                    
                    print(f"\n✂️  Loading first {duration_seconds} seconds ({chunk_frames:,} frames)")
                    print(f"  Bytes to read: {bytes_to_read / (1024**2):.1f} MB")
                    
                    # Read data chunk
                    raw_data = f.read(bytes_to_read)
                    
                    # Determine dtype
                    if bits_per_sample == 8:
                        dtype = np.uint8
                    elif bits_per_sample == 16:
                        dtype = np.int16
                    elif bits_per_sample == 32:
                        dtype = np.int32
                    else:
                        raise ValueError(f"Unsupported bits per sample: {bits_per_sample}")
                    
                    # Parse the data
                    return process_iq_data(raw_data, dtype, n_channels, fs, bits_per_sample)
                else:
                    # Skip unknown chunk
                    f.read(chunk_size)
            
            raise ValueError("No data chunk found in WAV file")
            
    except Exception as e:
        print(f"\n⚠️  WAV parsing failed: {e}")
        print("  Attempting to read as raw binary IQ data...")
        return read_raw_binary(filename, duration_seconds)


def read_raw_binary(filename, duration_seconds=2, assumed_fs=2048000, dtype=np.int16):
    """
    Read file as raw binary IQ data (fallback method)
    Assumes stereo int16 format at 2.048 MHz
    """
    print(f"\n📊 Raw Binary Mode:")
    print(f"  Assumed Sample Rate: {assumed_fs:,} Hz")
    print(f"  Assumed Format: Stereo int16 (IQ)")
    
    chunk_frames = int(duration_seconds * assumed_fs)
    bytes_to_read = chunk_frames * 2 * 2  # 2 channels * 2 bytes per sample
    
    print(f"\n✂️  Loading first {duration_seconds} seconds ({chunk_frames:,} frames)")
    
    with open(filename, 'rb') as f:
        raw_data = f.read(bytes_to_read)
    
    return process_iq_data(raw_data, dtype, 2, assumed_fs, 16)


def process_iq_data(raw_data, dtype, n_channels, fs, bits_per_sample):
    """Process raw IQ data into complex numpy array and visualize"""
    
    # Convert to numpy array
    chunk = np.frombuffer(raw_data, dtype=dtype)
    
    if n_channels == 2:
        chunk = chunk.reshape(-1, 2)
    
    print(f"\n  Data Shape: {chunk.shape}")
    print(f"  Data Type: {chunk.dtype}")
    print(f"  Memory Used: {chunk.nbytes / (1024**2):.1f} MB")
    
    # Print raw data statistics BEFORE normalization
    print(f"\n📊 Raw Data Statistics (Before Normalization):")
    if n_channels == 2:
        print(f"  I-channel: min={chunk[:, 0].min():6d}, max={chunk[:, 0].max():6d}, mean={chunk[:, 0].mean():.2f}")
        print(f"  Q-channel: min={chunk[:, 1].min():6d}, max={chunk[:, 1].max():6d}, mean={chunk[:, 1].mean():.2f}")
    else:
        print(f"  Data: min={chunk.min():6d}, max={chunk.max():6d}, mean={chunk.mean():.2f}")
    
    # Handle stereo IQ data
    if n_channels == 2:
        print(f"\n  Stereo file detected: IQ data")
        
        # Convert to float WITHOUT normalization first
        i_channel = chunk[:, 0].astype(np.float32)
        q_channel = chunk[:, 1].astype(np.float32)
        
        # Print raw float statistics
        print(f"\n  Raw float I-channel RMS: {np.sqrt(np.mean(i_channel**2)):.2f}")
        print(f"  Raw float Q-channel RMS: {np.sqrt(np.mean(q_channel**2)):.2f}")
        
        # Create complex signal (not normalized yet)
        iq_data = i_channel + 1j * q_channel
        
    else:
        print("  Mono file - treating as real data")
        iq_data = chunk.astype(np.float32).astype(complex)
    
    # Calculate RMS BEFORE any processing
    rms_before = np.sqrt(np.mean(np.abs(iq_data)**2))
    print(f"\n  Complex signal RMS (before normalization): {rms_before:.2f}")
    
    # Skip transient - use 0.5 seconds instead of fixed 5000 samples
    transient_skip = int(0.5 * fs)  # 0.5 seconds
    if len(iq_data) > transient_skip:
        print(f"\n🧼  Cleaning: Skipping first {transient_skip} transient samples ({transient_skip/fs:.2f}s)...")
        iq_data = iq_data[transient_skip:]
    else:
        print(f"\n⚠️  Warning: Data chunk is too small to remove {transient_skip} transient samples.")
        print(f"   Skipping first 5000 samples instead...")
        if len(iq_data) > 5000:
            iq_data = iq_data[5000:]
    
    # Remove DC bias
    print(f"🧼  Cleaning: Removing DC bias...")
    iq_data = iq_data - np.mean(iq_data)
    
    # AGC Normalization: Normalize by RMS to target level
    rms_after_dc = np.sqrt(np.mean(np.abs(iq_data)**2))
    target_rms = 0.15  # Target RMS level (empirically chosen for GPS)
    
    print(f"\n⚡ AGC Normalization:")
    print(f"  Current RMS: {rms_after_dc:.4f}")
    print(f"  Target RMS:  {target_rms:.4f}")
    
    if rms_after_dc > 0:
        agc_gain = target_rms / rms_after_dc
        iq_data = iq_data * agc_gain
        print(f"  AGC Gain: {agc_gain:.4f}x")
    else:
        print(f"  ⚠️  Warning: RMS is zero, skipping AGC")
    
    # Final statistics
    final_rms = np.sqrt(np.mean(np.abs(iq_data)**2))
    print(f"  Final RMS: {final_rms:.4f}")
    
    print(f"\n  IQ Data Shape: {iq_data.shape}")
    print(f"  IQ Data Type: {iq_data.dtype}")
    
    # Compute spectrum
    print("\n🔍 Computing FFT spectrum...")
    spectrum = np.abs(np.fft.fft(iq_data))**2
    f_axis = np.fft.fftfreq(len(iq_data), 1/fs)
    
    # Shift for plotting
    f_shifted = np.fft.fftshift(f_axis)
    spectrum_shifted = np.fft.fftshift(spectrum)
    
    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: Time domain (I and Q)
    time = np.arange(len(iq_data)) / fs
    plot_samples = min(1000, len(iq_data))
    axes[0].plot(time[:plot_samples] * 1e3, np.real(iq_data[:plot_samples]), label='I (Real)', alpha=0.7)
    axes[0].plot(time[:plot_samples] * 1e3, np.imag(iq_data[:plot_samples]), label='Q (Imag)', alpha=0.7)
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title(f'IQ Data - Time Domain (First {plot_samples} samples)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Power spectrum (full view)
    axes[1].plot(f_shifted / 1e6, 10 * np.log10(spectrum_shifted + 1e-10))
    axes[1].set_xlabel('Frequency (MHz)')
    axes[1].set_ylabel('Power (dB)')
    axes[1].set_title('Power Spectrum - Full View')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Zoomed spectrum (GPS L1 band is around ±2 MHz from center)
    mask = np.abs(f_shifted) < 5e6  # ±5 MHz
    axes[2].plot(f_shifted[mask] / 1e6, 10 * np.log10(spectrum_shifted[mask] + 1e-10))
    axes[2].set_xlabel('Frequency (MHz)')
    axes[2].set_ylabel('Power (dB)')
    axes[2].set_title('Power Spectrum - Zoomed (±5 MHz)')
    axes[2].grid(True, alpha=0.3)
    axes[2].axvline(0, color='r', linestyle='--', alpha=0.5, label='Center Frequency')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('phase0_inspection.png', dpi=150, bbox_inches='tight')
    print("\n✅ Plot saved as 'phase0_inspection.png'")
    plt.show()
    
    # Basic statistics
    print("\n📈 Signal Statistics:")
    print(f"  Mean Power: {np.mean(np.abs(iq_data)**2):.2e}")
    print(f"  Peak Power: {np.max(np.abs(iq_data)**2):.2e}")
    print(f"  RMS: {np.sqrt(np.mean(np.abs(iq_data)**2)):.4f}")
    print(f"  SNR estimate: {10*np.log10(np.max(spectrum_shifted)/np.mean(spectrum_shifted)):.1f} dB")
    
    # Find peak frequency (IF estimation)
    peak_freq_idx = np.argmax(spectrum_shifted)
    peak_freq = f_shifted[peak_freq_idx]
    print(f"  Peak Frequency (Est. IF): {peak_freq/1e6:.3f} MHz")
    
    return iq_data, fs, peak_freq


if __name__ == "__main__":
    wav_file = "/Users/alielboury/Documents/Learning/S9/Proj_GNSS/gps/groupe_20M.wav"  
    
    try:
        iq_data, fs = inspect_gps_wav(wav_file, duration_seconds=2)
        print("\n" + "=" * 50)
        print("✨ Phase 0 Complete!")
        print("=" * 50)
        print("\nNext Steps:")
        print("  1. Check the spectrum plot for GPS signal power")
        print("  2. Verify sample rate (should be ~2 MHz or higher)")
        print("  3. Move to Phase 1: C/A Code Generation")
        
    except FileNotFoundError:
        print(f"\n❌ Error: File '{wav_file}' not found!")
        print("Please check if the file exists in the current directory.")
        print(f"Current directory: {os.getcwd()}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()