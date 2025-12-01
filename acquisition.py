import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal as scipy_signal
import code_ca_gen

def compute_acquisition_metric(correlation_power, method='peak_to_second'):
    """
    Compute acquisition metric for detection.
    
    Methods:
    - 'peak_to_second': Ratio of peak to second highest peak (robust)
    - 'peak_to_mean': Ratio of peak to mean (simple)
    - 'peak_to_floor': Ratio of peak to noise floor
    """
    sorted_power = np.sort(correlation_power.flatten())[::-1]
    peak = sorted_power[0]
    
    if method == 'peak_to_second':
        # Find second peak that's not adjacent to first
        peak_idx = np.argmax(correlation_power)
        n = len(correlation_power)
        
        # Exclude samples near the peak (within 2 chips)
        exclude_range = max(int(n / 1023 * 2), 10)
        mask = np.ones(n, dtype=bool)
        mask[max(0, peak_idx - exclude_range):min(n, peak_idx + exclude_range + 1)] = False
        
        if np.sum(mask) > 0:
            second_peak = np.max(correlation_power[mask])
        else:
            second_peak = sorted_power[1]
        
        return peak / (second_peak + 1e-10)
        
    elif method == 'peak_to_mean':
        # Exclude top 1% for mean calculation
        exclude_count = max(1, len(sorted_power) // 100)
        mean_power = np.mean(sorted_power[exclude_count:])
        return peak / (mean_power + 1e-10)
        
    elif method == 'peak_to_floor':
        # Use median as noise floor estimate
        floor = np.median(correlation_power)
        return peak / (floor + 1e-10)
    
    return peak


def run_acquisition(data_chunk, sample_rate, prn_list=None, doppler_range=5000, doppler_step=500, non_coherent_integration=1):
    """
    Performs GPS satellite acquisition using Parallel Code Phase Search.
    Optimized for REAL GPS L1 C/A signals with robust detection.
    
    Key improvements for real data:
    - Peak-to-second-peak ratio for robust detection
    - Zero-padding for finer code phase resolution
    - Proper handling of sample rate vs chip rate
    - Adaptive threshold based on noise statistics
    
    Args:
        data_chunk (np.array): Raw IF signal (1D complex array) - should be at baseband
        sample_rate (float): Sampling frequency in Hz (e.g., 4e6)
        prn_list (list): PRNs to search (default 1-32)
        doppler_range (int): Doppler search range in Hz (±doppler_range)
        doppler_step (int): Doppler search step in Hz
        non_coherent_integration (int): Number of 1ms blocks to accumulate (default 1)
    
    Returns:
        results (dict): Dictionary containing acquisition results
    """
    
    # Default PRN list (GPS uses 1-32)
    if prn_list is None:
        prn_list = np.arange(1, 33)
    
    # --- Setup Parameters ---
    ca_code_length = 1023
    ca_chip_rate = 1.023e6  # Hz
    
    # Samples per 1ms (one C/A code period)
    samples_per_ms = int(sample_rate * 1e-3)
    
    # Samples per chip
    samples_per_chip = sample_rate / ca_chip_rate
    
    # Check data length
    required_samples = samples_per_ms * non_coherent_integration
    if len(data_chunk) < required_samples:
        print(f"⚠️  Warning: data_chunk ({len(data_chunk)} samples) < required {required_samples} samples")
        non_coherent_integration = len(data_chunk) // samples_per_ms
        print(f"   Reduced integration to {non_coherent_integration} ms")
        if non_coherent_integration < 1:
            raise ValueError("Not enough data for even 1ms integration")
    
    # Doppler frequency array
    dopplers = np.arange(-doppler_range, doppler_range + doppler_step, doppler_step)
    n_dopplers = len(dopplers)
    n_prns = len(prn_list)
    
    print("=" * 70)
    print("GPS ACQUISITION - PARALLEL CODE PHASE SEARCH")
    print("=" * 70)
    print(f"Sample rate: {sample_rate / 1e6:.3f} MHz")
    print(f"Samples per ms: {samples_per_ms}")
    print(f"Samples per chip: {samples_per_chip:.2f}")
    print(f"Integration time: {non_coherent_integration} ms (Non-Coherent)")
    print(f"Doppler search: ±{doppler_range} Hz, step {doppler_step} Hz")
    print(f"Number of Doppler bins: {n_dopplers}")
    print(f"PRNs to search: {len(prn_list)}")
    print("=" * 70)
    
    # Storage for results
    peaks = np.zeros((n_prns, n_dopplers), dtype=float)
    code_phases = np.zeros((n_prns, n_dopplers), dtype=int)
    acquisition_metrics = np.zeros((n_prns, n_dopplers), dtype=float)
    
    # Time vector for Doppler modulation (1ms)
    time_vector_1ms = np.arange(samples_per_ms) / sample_rate
    
    # --- Pre-compute C/A Codes ---
    print("\n⚡ Generating and upsampling C/A codes...")
    ca_codes_fft = {}
    
    for prn in prn_list:
        # Generate CA code for this PRN (-1/+1 bipolar)
        ca_code = code_ca_gen.generate_ca_code(prn, bipolar=True)
        
        # Upsample CA code to match sample rate
        # Use proper resampling to maintain correlation properties
        code_samples = np.zeros(samples_per_ms)
        
        for i in range(samples_per_ms):
            chip_idx = int((i * ca_chip_rate / sample_rate) % ca_code_length)
            code_samples[i] = ca_code[chip_idx]
        
        # Compute FFT of conjugate (for correlation)
        ca_codes_fft[prn] = np.conj(np.fft.fft(code_samples))
    
    print("✅ C/A Codes prepared.")
    
    # --- Main Acquisition Loop ---
    print("\n🔍 Searching for satellites...\n")
    
    # Pre-extract all data chunks for efficiency
    data_chunks = []
    for i in range(non_coherent_integration):
        chunk = data_chunk[i * samples_per_ms : (i + 1) * samples_per_ms]
        data_chunks.append(chunk)
    
    # Progress tracking
    total_iterations = n_dopplers
    progress_step = max(1, total_iterations // 10)
    
    # Loop over Doppler frequencies
    for dop_idx, doppler in enumerate(dopplers):
        if dop_idx % progress_step == 0:
            progress = (dop_idx / total_iterations) * 100
            print(f"   Progress: {progress:5.1f}% | Doppler: {doppler:+6.0f} Hz")
        
        # Doppler removal vector (1ms)
        doppler_removal = np.exp(-1j * 2 * np.pi * doppler * time_vector_1ms)
        
        # Pre-compute FFTs for all chunks at this Doppler
        fft_chunks = []
        for chunk in data_chunks:
            # Remove Doppler
            baseband_chunk = chunk * doppler_removal
            # FFT
            fft_chunks.append(np.fft.fft(baseband_chunk))
        
        # Correlate with all PRNs
        for prn_idx, prn in enumerate(prn_list):
            
            # Accumulate correlation power (non-coherent integration)
            sum_correlation_power = np.zeros(samples_per_ms, dtype=float)
            
            for i in range(non_coherent_integration):
                # Frequency domain correlation
                correlation_spectrum = fft_chunks[i] * ca_codes_fft[prn]
                
                # IFFT to get time-domain correlation
                correlation = np.fft.ifft(correlation_spectrum)
                
                # Accumulate magnitude squared (non-coherent)
                sum_correlation_power += np.abs(correlation)**2
            
            # Find peak
            peak_power = np.max(sum_correlation_power)
            peak_phase = np.argmax(sum_correlation_power)
            
            # Compute acquisition metric (peak-to-second-peak ratio)
            acq_metric = compute_acquisition_metric(sum_correlation_power, method='peak_to_second')
            
            # Store results
            peaks[prn_idx, dop_idx] = np.sqrt(peak_power)  # Return magnitude
            code_phases[prn_idx, dop_idx] = peak_phase
            acquisition_metrics[prn_idx, dop_idx] = acq_metric

    print(f"   Progress: 100.0% | Complete")
    
    # --- Post-Processing: Detect Satellites ---
    print("\n" + "=" * 70)
    print("DETECTION ANALYSIS")
    print("=" * 70)
    
    # Find best Doppler for each PRN
    best_metrics = np.zeros(n_prns)
    best_peaks = np.zeros(n_prns)
    best_dopplers = np.zeros(n_prns)
    best_code_phases = np.zeros(n_prns, dtype=int)
    
    for prn_idx in range(n_prns):
        best_dop_idx = np.argmax(peaks[prn_idx, :])
        best_metrics[prn_idx] = acquisition_metrics[prn_idx, best_dop_idx]
        best_peaks[prn_idx] = peaks[prn_idx, best_dop_idx]
        best_dopplers[prn_idx] = dopplers[best_dop_idx]
        best_code_phases[prn_idx] = code_phases[prn_idx, best_dop_idx]
    
    # Compute detection threshold
    # Use robust statistics (median absolute deviation)
    median_metric = np.median(best_metrics)
    mad = np.median(np.abs(best_metrics - median_metric))
    robust_std = 1.4826 * mad  # Scale to estimate std
    
    # Threshold based on peak-to-second-peak ratio
    # For GPS, a ratio > 2.5-3.0 typically indicates detection
    metric_threshold = max(2.5, median_metric + 3 * robust_std)
    
    # Also use peak amplitude threshold
    mean_peak = np.mean(best_peaks)
    std_peak = np.std(best_peaks)
    peak_threshold = mean_peak + 4 * std_peak
    
    print(f"\nDetection Statistics:")
    print(f"  Median acquisition metric: {median_metric:.2f}")
    print(f"  Robust std (MAD-based): {robust_std:.2f}")
    print(f"  Metric threshold: {metric_threshold:.2f}")
    print(f"  Peak threshold: {peak_threshold:.2f}")
    
    # Find detections
    detections = []
    
    for prn_idx, prn in enumerate(prn_list):
        metric = best_metrics[prn_idx]
        peak = best_peaks[prn_idx]
        
        # Detection criteria: metric > threshold OR peak > peak_threshold
        # (Use OR to catch both strong and weak signals)
        if metric > metric_threshold or peak > peak_threshold:
            snr_db = 10 * np.log10(metric) if metric > 0 else 0
            
            detections.append({
                'prn': prn,
                'doppler': best_dopplers[prn_idx],
                'code_phase': best_code_phases[prn_idx],
                'peak': peak,
                'metric': metric,
                'snr': snr_db
            })
    
    # Sort by metric (strongest first)
    detections.sort(key=lambda x: x['metric'], reverse=True)
    
    print(f"\n🛰️  Satellites Detected: {len(detections)}")
    
    if detections:
        print(f"\n{'PRN':<5} {'Doppler (Hz)':<14} {'Code Phase':<12} {'Metric':<10} {'SNR (dB)':<10}")
        print("-" * 55)
        for det in detections:
            print(f"{det['prn']:<5} {det['doppler']:+11.0f}    {det['code_phase']:<12} {det['metric']:<10.2f} {det['snr']:<10.1f}")
    else:
        print("\n❌ No satellites detected.")
        
        # Show top 5 candidates for debugging
        print("\n   Top 5 candidates (for debugging):")
        sorted_indices = np.argsort(best_metrics)[::-1][:5]
        print(f"   {'PRN':<5} {'Doppler (Hz)':<14} {'Metric':<10} {'Peak':<10}")
        print("   " + "-" * 45)
        for idx in sorted_indices:
            prn = prn_list[idx]
            print(f"   {prn:<5} {best_dopplers[idx]:+11.0f}    {best_metrics[idx]:<10.2f} {best_peaks[idx]:<10.2f}")
        
        print(f"\n   Threshold was: metric > {metric_threshold:.2f} OR peak > {peak_threshold:.2f}")
    
    print("=" * 70)
    
    # Prepare results dictionary
    results = {
        'peaks': peaks,
        'dopplers': dopplers,
        'code_phases': code_phases,
        'acquisition_metrics': acquisition_metrics,
        'detections': detections,
        'threshold': peak_threshold,
        'metric_threshold': metric_threshold,
        'prn_list': prn_list,
        'sample_rate': sample_rate,
        'mean_peak': mean_peak,
        'std_peak': std_peak,
        'integration_time': non_coherent_integration,
        'best_metrics': best_metrics,
        'best_peaks': best_peaks,
        'best_dopplers': best_dopplers
    }
    
    return results


def plot_acquisition_results(results, title="GPS Acquisition Results", save_path=None):
    """
    Create comprehensive visualizations of acquisition results.
    Enhanced for real data analysis with multiple diagnostic plots.
    
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
    
    # Get additional metrics if available
    best_metrics = results.get('best_metrics', np.max(peaks, axis=1))
    metric_threshold = results.get('metric_threshold', 2.5)
    integration_time = results.get('integration_time', 1)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    
    # --- Plot 1: 2D Heatmap (Doppler vs PRN) ---
    ax1 = fig.add_subplot(2, 3, 1)
    
    # Normalize for better visualization
    peaks_db = 10 * np.log10(peaks / np.max(peaks) + 1e-10)
    
    im = ax1.imshow(peaks_db, aspect='auto', cmap='viridis', origin='lower',
                    extent=[dopplers[0]/1000, dopplers[-1]/1000, prn_list[0]-0.5, prn_list[-1]+0.5])
    ax1.set_xlabel('Doppler (kHz)', fontsize=10)
    ax1.set_ylabel('PRN', fontsize=10)
    ax1.set_title('Correlation Peaks (dB)', fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('Relative Power (dB)', fontsize=9)
    
    # Mark detections on heatmap
    for det in detections:
        prn_idx = np.where(prn_list == det['prn'])[0][0]
        ax1.axhline(y=det['prn'], color='lime', linestyle='--', alpha=0.7, linewidth=1)
        ax1.plot(det['doppler']/1000, det['prn'], 'r*', markersize=12)
    
    # --- Plot 2: Acquisition Metric per PRN ---
    ax2 = fig.add_subplot(2, 3, 2)
    
    colors = ['limegreen' if m > metric_threshold else 'steelblue' for m in best_metrics]
    bars = ax2.bar(prn_list, best_metrics, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.axhline(y=metric_threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Threshold: {metric_threshold:.1f}')
    ax2.set_xlabel('PRN', fontsize=10)
    ax2.set_ylabel('Acquisition Metric', fontsize=10)
    ax2.set_title('Peak-to-Second-Peak Ratio', fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(prn_list[::2])  # Show every other PRN
    
    # Annotate detected PRNs
    for det in detections:
        prn_idx = np.where(prn_list == det['prn'])[0][0]
        ax2.annotate(f"PRN {det['prn']}", 
                    xy=(det['prn'], best_metrics[prn_idx]),
                    xytext=(det['prn'], best_metrics[prn_idx] + max(best_metrics)*0.1),
                    ha='center', fontsize=8, fontweight='bold')
    
    # --- Plot 3: Peak Magnitude per PRN ---
    ax3 = fig.add_subplot(2, 3, 3)
    
    best_peaks = results.get('best_peaks', np.max(peaks, axis=1))
    colors = ['limegreen' if best_peaks[i] > threshold else 'gray' for i in range(len(prn_list))]
    ax3.bar(prn_list, best_peaks, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax3.axhline(y=threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Threshold: {threshold:.1f}')
    ax3.set_xlabel('PRN', fontsize=10)
    ax3.set_ylabel('Peak Magnitude', fontsize=10)
    ax3.set_title('Maximum Correlation Peak', fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_xticks(prn_list[::2])
    
    # --- Plot 4: Doppler Spectrum (max across all PRNs) ---
    ax4 = fig.add_subplot(2, 3, 4)
    
    max_peaks_per_doppler = np.max(peaks, axis=0)
    ax4.plot(dopplers/1000, max_peaks_per_doppler, 'b-', linewidth=1.5, alpha=0.8)
    ax4.fill_between(dopplers/1000, 0, max_peaks_per_doppler, alpha=0.3)
    ax4.axhline(y=threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold')
    
    # Mark detected Dopplers
    for det in detections:
        ax4.axvline(x=det['doppler']/1000, color='limegreen', linestyle=':', 
                   alpha=0.8, linewidth=2, label=f"PRN {det['prn']}")
    
    ax4.set_xlabel('Doppler Frequency (kHz)', fontsize=10)
    ax4.set_ylabel('Max Correlation Peak', fontsize=10)
    ax4.set_title('Doppler Spectrum', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper right', fontsize=8)
    
    # --- Plot 5: Detection Summary ---
    ax5 = fig.add_subplot(2, 3, 5)
    
    if detections:
        det_prns = [f"PRN {d['prn']}" for d in detections]
        det_snrs = [d['snr'] for d in detections]
        det_dopplers = [d['doppler']/1000 for d in detections]
        
        # Create bar chart with SNR
        x_pos = np.arange(len(det_prns))
        bars = ax5.bar(x_pos, det_snrs, color='darkgreen', alpha=0.8, edgecolor='black')
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(det_prns, rotation=45, ha='right')
        ax5.set_ylabel('SNR (dB)', fontsize=10)
        ax5.set_title('Detected Satellites - Signal Quality', fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')
        
        # Add Doppler values on bars
        for i, (bar, doppler) in enumerate(zip(bars, det_dopplers)):
            height = bar.get_height()
            ax5.annotate(f'{doppler:+.1f}kHz',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight='bold')
    else:
        ax5.text(0.5, 0.5, 'No Satellites Detected', ha='center', va='center', 
                fontsize=14, fontweight='bold', color='red',
                transform=ax5.transAxes)
        ax5.set_xlim(0, 1)
        ax5.set_ylim(0, 1)
    
    # --- Plot 6: Correlation for strongest detection ---
    ax6 = fig.add_subplot(2, 3, 6)
    
    if detections:
        # Find and plot correlation for strongest detection
        det = detections[0]
        prn_idx = np.where(prn_list == det['prn'])[0][0]
        dop_idx = np.where(dopplers == det['doppler'])[0][0]
        
        # Plot correlation peak profile across all Dopplers for this PRN
        prn_peaks = peaks[prn_idx, :]
        ax6.plot(dopplers/1000, prn_peaks, 'b-', linewidth=2, label=f"PRN {det['prn']}")
        ax6.axvline(x=det['doppler']/1000, color='red', linestyle='--', 
                   label=f"Peak: {det['doppler']:+.0f} Hz")
        ax6.axhline(y=threshold, color='gray', linestyle=':', alpha=0.7)
        
        ax6.set_xlabel('Doppler Frequency (kHz)', fontsize=10)
        ax6.set_ylabel('Correlation Peak', fontsize=10)
        ax6.set_title(f'Doppler Profile - Strongest Detection (PRN {det["prn"]})', fontweight='bold')
        ax6.legend(loc='upper right')
        ax6.grid(True, alpha=0.3)
    else:
        # Show the best candidate even if not detected
        best_prn_idx = np.argmax(np.max(peaks, axis=1))
        best_prn = prn_list[best_prn_idx]
        prn_peaks = peaks[best_prn_idx, :]
        
        ax6.plot(dopplers/1000, prn_peaks, 'b-', linewidth=2, label=f"PRN {best_prn} (best candidate)")
        ax6.axhline(y=threshold, color='red', linestyle='--', label='Threshold')
        
        ax6.set_xlabel('Doppler Frequency (kHz)', fontsize=10)
        ax6.set_ylabel('Correlation Peak', fontsize=10)
        ax6.set_title(f'Doppler Profile - Best Candidate (PRN {best_prn})', fontweight='bold')
        ax6.legend(loc='upper right')
        ax6.grid(True, alpha=0.3)
    
    # Main title with info
    fig.suptitle(f'{title}\nIntegration: {integration_time}ms | Detected: {len(detections)} satellites', 
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✅ Plot saved to {save_path}")
    
    plt.show()
    
    # --- Additional: Save detailed per-PRN plots for detections ---
    if save_path and detections:
        detail_dir = os.path.dirname(save_path)
        
        for det in detections[:3]:  # Plot top 3 detections
            fig2, ax = plt.subplots(figsize=(10, 4))
            
            prn_idx = np.where(prn_list == det['prn'])[0][0]
            prn_peaks = peaks[prn_idx, :]
            
            ax.plot(dopplers, prn_peaks, 'b-', linewidth=2)
            ax.axvline(x=det['doppler'], color='red', linestyle='--', linewidth=2,
                      label=f"Detected Doppler: {det['doppler']:+.0f} Hz")
            ax.axhline(y=threshold, color='gray', linestyle=':', alpha=0.7, label='Threshold')
            
            ax.set_xlabel('Doppler Frequency (Hz)')
            ax.set_ylabel('Correlation Peak')
            ax.set_title(f"PRN {det['prn']} - Acquisition Result\nSNR: {det['snr']:.1f} dB | Code Phase: {det['code_phase']} samples")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            detail_path = os.path.join(detail_dir, f"prn_{det['prn']}_detail.png")
            plt.savefig(detail_path, dpi=100, bbox_inches='tight')
            plt.close(fig2)
            print(f"   Detail plot saved: {detail_path}")


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