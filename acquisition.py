import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import code_ca_gen

def run_acquisition(data_chunk, sample_rate, prn_list=None, doppler_range=5000, doppler_step=500, non_coherent_integration=1):
    """
    Performs GPS satellite acquisition using Parallel Code Phase Search.
    Optimized to pre-compute Code FFTs and shift signal for Doppler.
    Supports Non-Coherent Integration for improved sensitivity.
    
    Args:
        data_chunk (np.array): Raw IF signal (1D complex array)
        sample_rate (float): Sampling frequency in Hz (e.g., 2.048e6)
        prn_list (list): PRNs to search (default 1-32)
        doppler_range (int): Doppler search range in Hz (±doppler_range)
        doppler_step (int): Doppler search step in Hz
        non_coherent_integration (int): Number of 1ms blocks to accumulate (default 1)
    
    Returns:
        results (dict): Dictionary containing:
            - 'peaks': 2D array (32, n_dopplers) of correlation peak magnitudes
            - 'dopplers': Doppler frequency array
            - 'detections': List of detected satellites with metadata
    """
    
    # Default PRN list (GPS uses 1-32)
    if prn_list is None:
        prn_list = np.arange(1, 33)
    
    # --- Setup Parameters ---
    ca_code_length = 1023
    ca_chip_rate = 1.023e6  # Hz
    
    # Upsample ratio to match sample rate
    upsample_ratio = sample_rate / ca_chip_rate
    upsampled_code_length = int(ca_code_length * upsample_ratio)
    
    # Use 1 ms of data (one complete C/A code period)
    coherent_integration_time = 1e-3  # 1 millisecond
    samples_per_ms = int(sample_rate * coherent_integration_time)
    
    # Check data length
    required_samples = samples_per_ms * non_coherent_integration
    if len(data_chunk) < required_samples:
        print(f"⚠️  Warning: data_chunk ({len(data_chunk)} samples) < required {required_samples} samples")
        # Adjust integration time to fit data
        non_coherent_integration = len(data_chunk) // samples_per_ms
        print(f"   Reduced integration to {non_coherent_integration} ms")
        if non_coherent_integration < 1:
            raise ValueError("Not enough data for even 1ms integration")
    
    # Time vector for Doppler modulation (1ms)
    # We use the same time vector for each chunk because we treat them as independent for non-coherent
    # (Phase continuity is lost in abs(), but Doppler frequency shift is constant)
    time_vector_1ms = np.arange(samples_per_ms) / sample_rate
    
    # Doppler frequency array
    dopplers = np.arange(-doppler_range, doppler_range + doppler_step, doppler_step)
    n_dopplers = len(dopplers)
    n_prns = len(prn_list)
    
    print("=" * 70)
    print("GPS ACQUISITION ALGORITHM (OPTIMIZED)")
    print("=" * 70)
    print(f"Sample rate: {sample_rate / 1e6:.3f} MHz")
    print(f"Integration time: {non_coherent_integration} ms (Non-Coherent)")
    print(f"Doppler search: ±{doppler_range} Hz with {doppler_step} Hz step")
    print(f"Number of Dopplers: {n_dopplers}")
    print(f"PRNs to search: {len(prn_list)}")
    print("=" * 70)
    
    # Storage for results
    peaks = np.zeros((n_prns, n_dopplers), dtype=float)
    code_phases = np.zeros((n_prns, n_dopplers), dtype=int)
    
    # --- Pre-compute FFT of C/A Codes ---
    print("\n⚡ Pre-computing C/A Code FFTs...")
    ca_codes_fft = {}
    
    for prn in prn_list:
        # Generate CA code for this PRN
        ca_code = code_ca_gen.generate_ca_code(prn, bipolar=True)
        
        # Upsample CA code to match sample rate
        upsampled_code = np.repeat(ca_code, int(np.ceil(upsample_ratio)))
        upsampled_code = upsampled_code[:samples_per_ms]
        
        # Compute FFT and store (conjugate for correlation)
        ca_codes_fft[prn] = np.conj(np.fft.fft(upsampled_code))
        
    print("✅ C/A Codes prepared.")
    
    # --- Main Acquisition Loop ---
    print("\n🔍 Searching for satellites (Doppler Iteration)...\n")
    
    # Loop over Doppler frequencies
    for dop_idx, doppler in enumerate(dopplers):
        if dop_idx % 5 == 0:
            print(f"Processing Doppler {doppler:+5.0f} Hz ({dop_idx+1}/{n_dopplers})...")
            
        # Pre-process signal chunks for this Doppler
        # We shift and FFT each 1ms chunk
        fft_chunks = []
        
        # Doppler removal vector (1ms)
        doppler_removal = np.exp(-1j * 2 * np.pi * doppler * time_vector_1ms)
        
        for i in range(non_coherent_integration):
            chunk = data_chunk[i*samples_per_ms : (i+1)*samples_per_ms]
            
            # Remove Doppler
            baseband_chunk = chunk * doppler_removal
            
            # FFT
            fft_chunks.append(np.fft.fft(baseband_chunk))
        
        # Correlate with all PRNs
        for prn_idx, prn in enumerate(prn_list):
            
            # Accumulate correlation power
            sum_correlation_power = np.zeros(samples_per_ms, dtype=float)
            
            for i in range(non_coherent_integration):
                # Frequency domain correlation
                correlation_spectrum = fft_chunks[i] * ca_codes_fft[prn]
                
                # IFFT
                correlation = np.fft.ifft(correlation_spectrum)
                
                # Accumulate Power (Mag^2)
                sum_correlation_power += np.abs(correlation)**2
            
            # Find peak in accumulated power
            peak_power = np.max(sum_correlation_power)
            peak_phase = np.argmax(sum_correlation_power)
            
            # Store results (take sqrt to return magnitude equivalent)
            peaks[prn_idx, dop_idx] = np.sqrt(peak_power)
            code_phases[prn_idx, dop_idx] = peak_phase

    # --- Post-Processing: Detect Satellites ---
    print("\n" + "=" * 70)
    print("DETECTION RESULTS")
    print("=" * 70)
    
    # Compute threshold (adaptive)
    mean_peak = np.mean(peaks)
    std_peak = np.std(peaks)
    threshold = mean_peak + 5 * std_peak  # 5-sigma rule
    
    print(f"Mean correlation peak: {mean_peak:.2f}")
    print(f"Std deviation: {std_peak:.2f}")
    print(f"Detection threshold: {threshold:.2f}")
    
    # Find detections - only keep the peak per PRN
    detections = []
    
    for prn_idx, prn in enumerate(prn_list):
        # Find max peak for this PRN across all dopplers
        max_peak_idx = np.argmax(peaks[prn_idx, :])
        peak = peaks[prn_idx, max_peak_idx]
        doppler = dopplers[max_peak_idx]
        code_phase = code_phases[prn_idx, max_peak_idx]
        
        if peak > threshold:
            z_score = (peak - mean_peak) / std_peak
            detections.append({
                'prn': prn,
                'doppler': doppler,
                'code_phase': code_phase,
                'peak': peak,
                'snr': z_score  # Z-score (SNR)
            })
    
    # Sort by peak magnitude (strongest first)
    detections.sort(key=lambda x: x['peak'], reverse=True)
    
    print(f"\n🛰️  Satellites Detected: {len(detections)}\n")
    
    if detections:
        print(f"{'PRN':<5} {'Doppler (Hz)':<15} {'Code Phase':<15} {'Peak':<10} {'SNR (sigma)':<10}")
        print("-" * 60)
        for det in detections:
            print(f"{det['prn']:<5} {det['doppler']:+11.0f}   {det['code_phase']:<14} {det['peak']:<10.2f} {det['snr']:<10.2f}")
    else:
        print("❌ No satellites detected.")
        # Print the strongest peak anyway for debugging
        max_peak_idx = np.argmax(peaks)
        prn_idx, dop_idx = np.unravel_index(max_peak_idx, peaks.shape)
        max_peak = peaks[prn_idx, dop_idx]
        print(f"   Strongest candidate: PRN {prn_list[prn_idx]} at {dopplers[dop_idx]} Hz (Peak: {max_peak:.2f})")
        print("   Try adjusting threshold or checking your data.")
    
    print("=" * 70)
    
    # Prepare results dictionary
    results = {
        'peaks': peaks,
        'dopplers': dopplers,
        'code_phases': code_phases,
        'detections': detections,
        'threshold': threshold,
        'prn_list': prn_list,
        'sample_rate': sample_rate,
        'mean_peak': mean_peak,
        'std_peak': std_peak
    }
    
    return results


def plot_acquisition_results(results, title="GPS Acquisition Results", save_path=None):
    """
    Create visualizations of acquisition results.
    
    Args:
        results (dict): Output from run_acquisition()
        title (str): Plot title
        save_path (str): Optional path to save the plot (e.g., 'plots/acquisition.png')
    """
    
    peaks = results['peaks']
    dopplers = results['dopplers']
    prn_list = results['prn_list']
    detections = results['detections']
    threshold = results['threshold']
    
    # --- Plot 1: 2D Heatmap (Doppler vs PRN) ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Heatmap
    ax1 = axes[0, 0]
    im = ax1.imshow(peaks, aspect='auto', cmap='hot', origin='lower')
    ax1.set_xlabel('Doppler Index')
    ax1.set_ylabel('PRN')
    ax1.set_title('Correlation Peaks: PRN vs. Doppler (Heatmap)', fontweight='bold')
    ax1.set_yticks(range(len(prn_list)))
    ax1.set_yticklabels(prn_list)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Correlation Magnitude')
    
    # Mark detections on heatmap
    for det in detections:
        prn_idx = np.where(prn_list == det['prn'])[0][0]
        dop_idx = np.where(dopplers == det['doppler'])[0][0]
        ax1.plot(dop_idx, prn_idx, 'c+', markersize=15, markeredgewidth=2)
    
    # --- Plot 2: Doppler-only view (max over all code phases) ---
    ax2 = axes[0, 1]
    max_peaks_per_doppler = np.max(peaks, axis=0)
    ax2.bar(range(len(dopplers)), max_peaks_per_doppler, color='steelblue', alpha=0.7)
    ax2.axhline(y=threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.2f}')
    ax2.set_xlabel('Doppler Index')
    ax2.set_ylabel('Max Correlation Peak')
    ax2.set_title('Maximum Peak per Doppler Frequency', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # --- Plot 3: PRN-only view (max over all dopplers) ---
    ax3 = axes[1, 0]
    max_peaks_per_prn = np.max(peaks, axis=1)
    colors = ['green' if max_peaks_per_prn[i] > threshold else 'gray' for i in range(len(prn_list))]
    ax3.bar(prn_list, max_peaks_per_prn, color=colors, alpha=0.7)
    ax3.axhline(y=threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.2f}')
    ax3.set_xlabel('PRN')
    ax3.set_ylabel('Max Correlation Peak')
    ax3.set_title('Maximum Peak per PRN', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # --- Plot 4: Detection Statistics ---
    ax4 = axes[1, 1]
    if detections:
        det_prns = [d['prn'] for d in detections]
        det_peaks = [d['peak'] for d in detections]
        det_snrs = [10 * np.log10(d['snr'] + 1e-10) for d in detections]
        
        x_pos = np.arange(len(det_prns))
        ax4.bar(x_pos, det_snrs, color='darkgreen', alpha=0.7)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels([f"PRN {p}" for p in det_prns], rotation=45)
        ax4.set_ylabel('SNR (dB)')
        ax4.set_title('Detected Satellites - SNR Estimate', fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
    else:
        ax4.text(0.5, 0.5, 'No detections', ha='center', va='center', fontsize=14)
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"✅ Plot saved to {save_path}")
    else:
        plt.show()
    
    # --- Plot 5: 3D Surface Plot (optional, slower) ---
    # Only show/save 3D plot if explicitly requested or if we are not saving (interactive mode)
    # For now, let's skip saving 3D plot to avoid complexity, or save it as a separate file if needed.
    # We will just comment it out for the save path to keep it simple, or we can save it too.
    
    if not save_path:
        fig2 = plt.figure(figsize=(14, 8))
        ax5 = fig2.add_subplot(111, projection='3d')
        
        # Create mesh grid
        doppler_indices = np.arange(len(dopplers))
        prn_indices = np.arange(len(prn_list))
        doppler_mesh, prn_mesh = np.meshgrid(doppler_indices, prn_indices)
        
        # Plot surface
        surf = ax5.plot_surface(doppler_mesh, prn_mesh, peaks, cmap='hot', alpha=0.8)
        ax5.set_xlabel('Doppler Index')
        ax5.set_ylabel('PRN')
        ax5.set_zlabel('Correlation Magnitude')
        ax5.set_title('3D View: Acquisition Search Space', fontweight='bold')
        
        fig2.colorbar(surf, ax=ax5, label='Correlation Magnitude')
        plt.show()


# --- MAIN TEST SCRIPT ---
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("GPS ACQUISITION - SIMULATION TEST")
    print("=" * 70 + "\n")
    
    # --- Simulate Received Signal ---
    # Parameters
    sample_rate = 2.048e6  # 2.048 MHz (typical for GPS receivers)
    duration = 0.001  # 1 ms (one C/A code period)
    n_samples = int(sample_rate * duration)
    
    # Create time vector
    t = np.arange(n_samples) / sample_rate
    
    # Simulate signal with PRN 5 and PRN 12 visible
    print("📡 Simulating received signal...")
    print(f"   Sample rate: {sample_rate / 1e6:.3f} MHz")
    print(f"   Duration: {duration * 1e3:.1f} ms")
    print(f"   Number of samples: {n_samples}")
    
    # Initialize signal with noise
    np.random.seed(42)
    noise = np.random.randn(n_samples) + 1j * np.random.randn(n_samples)
    noise = noise / np.sqrt(2)  # Normalize
    
    signal = noise.copy()
    
    # Add PRN 5 at doppler = -1500 Hz
    ca_code_5 = code_ca_gen.generate_ca_code(5, bipolar=True)
    upsample_ratio = sample_rate / 1.023e6
    upsampled_code_5 = np.repeat(ca_code_5, int(np.ceil(upsample_ratio)))
    upsampled_code_5 = upsampled_code_5[:n_samples]
    
    doppler_5 = -1500  # Hz
    modulation_5 = np.exp(1j * 2 * np.pi * doppler_5 * t)
    signal += 2.0 * upsampled_code_5 * modulation_5  # SNR ~ 3 dB
    
    # Add PRN 12 at doppler = +2500 Hz
    ca_code_12 = code_ca_gen.generate_ca_code(12, bipolar=True)
    upsampled_code_12 = np.repeat(ca_code_12, int(np.ceil(upsample_ratio)))
    upsampled_code_12 = upsampled_code_12[:n_samples]
    
    doppler_12 = 2500  # Hz
    modulation_12 = np.exp(1j * 2 * np.pi * doppler_12 * t)
    signal += 1.5 * upsampled_code_12 * modulation_12  # SNR ~ 2 dB
    
    print("\n✅ Signal created with:")
    print("   - PRN 5  at Doppler = -1500 Hz")
    print("   - PRN 12 at Doppler = +2500 Hz")
    print("   - AWGN background\n")
    
    # --- Run Acquisition ---
    results = run_acquisition(signal, sample_rate, doppler_range=5000, doppler_step=250)
    
    # --- Plot Results ---
    print("\n📊 Generating plots...\n")
    plot_acquisition_results(results, title="GPS Acquisition - Simulation Results", save_path="plots/acquisition_simulation.png")
    
    print("\n✅ Acquisition complete!\n")