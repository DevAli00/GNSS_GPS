import numpy as np
import setup
import acquisition
import os
from scipy import signal

def preprocess_signal(iq_data, fs, if_freq=None):
    """
    Advanced signal preprocessing for real GPS data.
    
    This function applies critical preprocessing steps that are essential
    for successful acquisition with real-world GPS signals.
    """
    print("\n" + "=" * 50)
    print("SIGNAL PREPROCESSING")
    print("=" * 50)
    
    # 1. Remove DC offset (critical for real data)
    print("\n1️⃣  Removing DC offset...")
    dc_offset = np.mean(iq_data)
    iq_data = iq_data - dc_offset
    print(f"   DC offset removed: {np.abs(dc_offset):.4f}")
    
    # 2. Estimate and remove IF if not provided
    if if_freq is None:
        print("\n2️⃣  Estimating Intermediate Frequency (IF)...")
        # Use PSD to find IF more accurately
        nperseg = min(8192, len(iq_data) // 4)
        freqs, psd = signal.welch(iq_data, fs, nperseg=nperseg, return_onesided=False)
        
        # Shift to center
        freqs = np.fft.fftshift(freqs)
        psd = np.fft.fftshift(psd)
        
        # Find peak (excluding DC region)
        dc_mask = np.abs(freqs) > 10000  # Exclude ±10 kHz around DC
        psd_masked = psd.copy()
        psd_masked[~dc_mask] = 0
        
        if np.max(psd_masked) > 0:
            peak_idx = np.argmax(psd_masked)
            if_freq = freqs[peak_idx]
        else:
            if_freq = 0
        
        print(f"   Estimated IF: {if_freq/1e6:.4f} MHz")
    
    # 3. Mix to baseband if IF is significant
    if np.abs(if_freq) > 5000:  # If IF > 5 kHz
        print(f"\n3️⃣  Mixing signal to baseband (IF: {if_freq/1e3:.1f} kHz)...")
        t = np.arange(len(iq_data)) / fs
        iq_data = iq_data * np.exp(-1j * 2 * np.pi * if_freq * t)
        # Remove residual DC
        iq_data = iq_data - np.mean(iq_data)
        print("   ✅ Baseband conversion complete")
    else:
        print(f"\n3️⃣  Signal already at baseband (IF: {if_freq/1e3:.1f} kHz)")
    
    # 4. Apply bandpass filter to isolate GPS L1 C/A band
    print("\n4️⃣  Applying GPS L1 C/A bandpass filter...")
    # GPS L1 C/A signal bandwidth is approximately ±1.023 MHz
    # We use a slightly wider filter (±1.5 MHz) to account for Doppler
    nyq = fs / 2
    
    # Design bandpass filter centered at 0 (baseband)
    # GPS C/A bandwidth: main lobe is ±1.023 MHz, we use ±2 MHz
    filter_bandwidth = min(2.0e6, nyq * 0.9)  # 2 MHz or 90% of Nyquist
    
    if filter_bandwidth < 1.5e6:
        print(f"   ⚠️  Warning: Sample rate too low for optimal filtering")
        print(f"      Bandwidth limited to {filter_bandwidth/1e6:.2f} MHz")
    
    # Low-pass filter (equivalent to bandpass at baseband)
    filter_order = 5
    Wn = filter_bandwidth / nyq
    b, a = signal.butter(filter_order, Wn, btype='low')
    iq_data = signal.filtfilt(b, a, iq_data)
    print(f"   Filter bandwidth: ±{filter_bandwidth/1e6:.2f} MHz")
    
    # 5. AGC - Normalize power
    print("\n5️⃣  Applying AGC (Automatic Gain Control)...")
    rms = np.sqrt(np.mean(np.abs(iq_data)**2))
    if rms > 0:
        iq_data = iq_data / rms
        print(f"   Signal normalized (original RMS: {rms:.4f})")
    
    # 6. Optional decimation if sample rate is very high
    # Keep sample rate at least 4x chip rate (4.092 MHz) for good correlation
    min_sample_rate = 4.092e6
    max_decimation = int(fs / min_sample_rate)
    
    if max_decimation >= 2 and fs > 8e6:
        # Use power of 2 decimation for efficiency
        decimation = min(max_decimation, 4)  # Max 4x decimation
        
        print(f"\n6️⃣  Decimating signal by {decimation}x...")
        print(f"   Original sample rate: {fs/1e6:.2f} MHz")
        
        # Apply anti-aliasing filter and decimate
        iq_data = signal.decimate(iq_data, decimation, ftype='fir', zero_phase=True)
        fs = fs / decimation
        
        print(f"   New sample rate: {fs/1e6:.2f} MHz")
    else:
        print(f"\n6️⃣  No decimation needed (fs = {fs/1e6:.2f} MHz)")
    
    print("\n" + "=" * 50)
    print(f"Preprocessing complete: {len(iq_data)} samples at {fs/1e6:.2f} MHz")
    print("=" * 50)
    
    return iq_data, fs


def scan_if_frequencies(iq_data_full, fs, if_candidates):
    """
    Scan multiple IF frequency candidates to find satellites.
    Useful when the exact IF is unknown.
    """
    print("\n" + "=" * 60)
    print("MULTI-IF FREQUENCY SCAN")
    print("=" * 60)
    
    all_detections = []
    
    for if_freq in if_candidates:
        print(f"\n📡 Trying IF = {if_freq/1e6:.3f} MHz...")
        
        iq_data, new_fs = preprocess_signal(iq_data_full.copy(), fs, if_freq=if_freq)
        
        results = acquisition.run_acquisition(
            iq_data, 
            new_fs, 
            doppler_range=10000,
            doppler_step=250,
            non_coherent_integration=10
        )
        
        if results['detections']:
            print(f"   ✅ Found {len(results['detections'])} satellites!")
            for det in results['detections']:
                det['if_freq'] = if_freq
            all_detections.extend(results['detections'])
            return results, if_freq
    
    return None, None


def main():
    print("=" * 70)
    print("GNSS SIGNAL PROCESSING - ACQUISITION PHASE (REAL DATA)")
    print("=" * 70)
    print("Optimized for real GPS L1 C/A signal acquisition")
    
    # Path to the data file
    wav_file = "/Users/alielboury/Documents/Learning/S9/Proj_GNSS/gps/groupe_20M.wav"
    
    if not os.path.exists(wav_file):
        print(f"❌ Error: File not found at {wav_file}")
        return

    # --- Step 1: Load Data ---
    print("\n" + "-" * 50)
    print("STEP 1: Loading GPS Data")
    print("-" * 50)
    
    # Load more data for better statistics (1 second for weak signals)
    try:
        iq_data_full, fs, peak_freq = setup.inspect_gps_wav(wav_file, duration_seconds=1.0)
        
        print(f"\n📊 Loaded Data Info:")
        print(f"   Samples: {len(iq_data_full):,}")
        print(f"   Sample Rate: {fs/1e6:.2f} MHz")
        print(f"   Duration: {len(iq_data_full)/fs:.3f} seconds")
        print(f"   Detected IF: {peak_freq/1e6:.4f} MHz")
        
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Step 2: Try Multiple IF Frequencies ---
    print("\n" + "-" * 50)
    print("STEP 2: Signal Preprocessing & IF Search")
    print("-" * 50)
    
    # Common IF frequencies for GPS receivers
    # Your data shows IF around 4.58 MHz, but let's try several candidates
    if_candidates = [
        peak_freq,           # Detected peak
        4.092e6,             # Common GPS IF (4x chip rate)
        4.0e6,               # 4 MHz
        0,                   # Zero-IF / baseband
        -peak_freq,          # Negative of peak (in case of sideband)
        5.0e6,               # 5 MHz
        3.78e6,              # Another common IF
    ]
    
    # Remove duplicates and sort
    if_candidates = sorted(list(set([abs(f) for f in if_candidates] + [-abs(f) for f in if_candidates])))
    
    best_results = None
    best_if = None
    best_num_detections = 0
    
    print("\n🔍 Searching with different IF frequencies...")
    
    for if_freq in [peak_freq, 0, 4.092e6, -peak_freq]:
        print(f"\n" + "=" * 60)
        print(f"Testing IF = {if_freq/1e6:.3f} MHz")
        print("=" * 60)
        
        iq_data, new_fs = preprocess_signal(iq_data_full.copy(), fs, if_freq=if_freq)

        # Run acquisition with aggressive parameters for weak signals
        results = acquisition.run_acquisition(
            iq_data, 
            new_fs, 
            doppler_range=10000,     # ±10 kHz
            doppler_step=200,        # Fine step
            non_coherent_integration=20  # Longer integration for weak signals
        )
        
        num_detections = len(results['detections'])
        if num_detections > best_num_detections:
            best_num_detections = num_detections
            best_results = results
            best_if = if_freq
        
        # Also check acquisition metrics - even without "official" detections
        # Look for promising candidates
        best_metric = np.max(results.get('best_metrics', [0]))
        print(f"\n   Best acquisition metric: {best_metric:.2f}")
        
        if num_detections > 0:
            print(f"   ✅ Found {num_detections} satellites with IF = {if_freq/1e6:.3f} MHz!")
            break

    # Use best results
    if best_results is None:
        print("\n⚠️  Using last search results...")
        best_results = results
        best_if = if_freq

    # --- Step 3: Fine Search on Top Candidates ---
    print("\n" + "-" * 50)
    print("STEP 3: Fine Search on Top Candidates")
    print("-" * 50)
    
    if best_num_detections == 0:
        print("\n🔍 No definitive detections. Performing fine search on top PRN candidates...")
        
        # Get top 5 candidates from best search
        top_prns = []
        best_metrics = best_results.get('best_metrics', np.max(best_results['peaks'], axis=1))
        sorted_indices = np.argsort(best_metrics)[::-1][:5]
        
        for idx in sorted_indices:
            prn = best_results['prn_list'][idx]
            top_prns.append(prn)
        
        print(f"   Top PRN candidates: {top_prns}")
        
        # Re-run with just top candidates, longer integration, finer Doppler
        iq_data, new_fs = preprocess_signal(iq_data_full.copy(), fs, if_freq=best_if)
        
        print("\n   Running focused search with:")
        print("   - 50ms integration (5x longer)")
        print("   - 100 Hz Doppler step (2x finer)")
        print("   - Top 5 PRN candidates only")
        
        results = acquisition.run_acquisition(
            iq_data, 
            new_fs,
            prn_list=np.array(top_prns),
            doppler_range=10000,
            doppler_step=100,  # Finer step
            non_coherent_integration=50  # Much longer integration
        )
        
        if len(results['detections']) > 0:
            best_results = results
            best_num_detections = len(results['detections'])

    # --- Step 4: Save Results ---
    print("\n" + "-" * 50)
    print("STEP 4: Saving Results")
    print("-" * 50)
    
    # Create directories
    os.makedirs("plots", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    
    # Save main plot
    save_path = "plots/acquisition_result.png"
    integration_time = best_results.get('integration_time', 10)
    acquisition.plot_acquisition_results(
        best_results, 
        title=f"GPS Acquisition - Real Data ({integration_time}ms Integration, IF={best_if/1e6:.2f}MHz)", 
        save_path=save_path
    )
    
    # Save detailed results
    results_file = "output/acquisition_results.txt"
    with open(results_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("GPS ACQUISITION RESULTS\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Data File: {wav_file}\n")
        f.write(f"Sample Rate: {fs/1e6:.2f} MHz\n")
        f.write(f"IF Frequency Used: {best_if/1e6:.4f} MHz\n")
        f.write(f"Integration Time: {integration_time} ms\n")
        f.write(f"Detection Threshold: {best_results['threshold']:.2f}\n")
        f.write(f"Metric Threshold: {best_results.get('metric_threshold', 2.5):.2f}\n\n")
        
        f.write("Detected Satellites:\n")
        f.write("-" * 60 + "\n")
        
        if best_results['detections']:
            f.write(f"{'PRN':<6}{'Doppler (Hz)':<15}{'Code Phase':<15}{'Metric':<12}{'SNR (dB)':<10}\n")
            for det in best_results['detections']:
                f.write(f"{det['prn']:<6}{det['doppler']:+12.0f}   {det['code_phase']:<15}"
                       f"{det.get('metric', 0):<12.2f}{det['snr']:<10.1f}\n")
        else:
            f.write("No satellites detected\n\n")
            f.write("Top 5 Candidates (for analysis):\n")
            best_metrics = best_results.get('best_metrics', np.max(best_results['peaks'], axis=1))
            best_dopplers = best_results.get('best_dopplers', np.zeros(len(best_results['prn_list'])))
            sorted_indices = np.argsort(best_metrics)[::-1][:5]
            
            for idx in sorted_indices:
                prn = best_results['prn_list'][idx]
                f.write(f"  PRN {prn}: Metric = {best_metrics[idx]:.2f}, Doppler = {best_dopplers[idx]:+.0f} Hz\n")
    
    print(f"   Results saved to: {results_file}")
    
    # --- Summary ---
    print("\n" + "=" * 70)
    print("ACQUISITION SUMMARY")
    print("=" * 70)
    
    if best_results['detections']:
        print(f"\n🛰️  Successfully detected {len(best_results['detections'])} satellite(s):")
        print("-" * 50)
        for det in best_results['detections']:
            code_phase_chips = det['code_phase'] / (new_fs / 1.023e6)
            print(f"   PRN {det['prn']:2d}: Doppler = {det['doppler']:+6.0f} Hz, "
                  f"Code Phase = {det['code_phase']:5d} samples ({code_phase_chips:.1f} chips)")
    else:
        print("\n❌ No satellites confidently detected.")
        print("\n📊 Analysis of your data:")
        
        # Calculate some diagnostics
        best_metric = np.max(best_results.get('best_metrics', [0]))
        
        print(f"   - Best acquisition metric: {best_metric:.2f} (need > 2.5 for detection)")
        print(f"   - This suggests the GPS signal may be very weak or absent")
        
        print("\n🔧 Troubleshooting suggestions:")
        print("   1. Verify antenna was connected and had clear sky view")
        print("   2. Check if recording was done outdoors (GPS doesn't work well indoors)")
        print("   3. Verify the receiver's local oscillator frequency")
        print("   4. Check if the data contains L1 C/A signals (1575.42 MHz)")
        print("   5. Try different starting offsets in the file (transient effects)")
        
        print("\n📁 Top PRN candidates to investigate manually:")
        best_metrics = best_results.get('best_metrics', np.max(best_results['peaks'], axis=1))
        sorted_indices = np.argsort(best_metrics)[::-1][:5]
        for idx in sorted_indices:
            prn = best_results['prn_list'][idx]
            metric = best_metrics[idx]
            print(f"      PRN {prn:2d}: metric = {metric:.2f}")
    
    print("\n✅ Process Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
