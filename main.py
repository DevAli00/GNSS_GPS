import numpy as np
import setup
import acquisition
import os

def main():
    print("=" * 70)
    print("GNSS SIGNAL PROCESSING - ACQUISITION PHASE")
    print("=" * 70)
    
    # Path to the data file
    wav_file = "/Users/alielboury/Documents/Learning/S9/Proj_GNSS/gps/groupe_20M.wav"
    
    if not os.path.exists(wav_file):
        print(f"❌ Error: File not found at {wav_file}")
        return

    # --- Step 1: Load Data ---
    print("\n" + "-" * 40)
    print("1. Loading GPS Data...")
    print("-" * 40)
    
    # Load data (skip first 0.1s to avoid startup transients)
    # We load 0.2s and take the second half
    try:
        iq_data_full, fs, peak_freq = setup.inspect_gps_wav(wav_file, duration_seconds=0.2)
        # Take the second half (0.1s to 0.2s)
        iq_data = iq_data_full[int(0.1*fs):]
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return

    # Shift signal to baseband if IF is significant
    if np.abs(peak_freq) > 1000: # If IF > 1 kHz
        print(f"\n🔄 Shifting signal to baseband (IF: {peak_freq/1e6:.3f} MHz)...")
        t = np.arange(len(iq_data)) / fs
        iq_data = iq_data * np.exp(-1j * 2 * np.pi * peak_freq * t)
        
        # Remove DC (which is the shifted interference)
        print("   Removing DC bias (Notching interference)...")
        iq_data = iq_data - np.mean(iq_data)
    else:
        print(f"\n✅ Signal is already near baseband (IF: {peak_freq/1e6:.3f} MHz)")

    # Decimate signal to improve efficiency and SNR
    # 20 MHz -> 4 MHz (Factor of 5)
    decimation_factor = 5
    target_fs = fs / decimation_factor
    
    print(f"\n⬇️  Decimating signal by {decimation_factor} (Target fs: {target_fs/1e6:.2f} MHz)...")
    from scipy import signal
    iq_data = signal.decimate(iq_data, decimation_factor)
    fs = target_fs

    # --- Step 2: Run Acquisition ---
    print("\n" + "-" * 40)
    print("2. Running Acquisition...")
    print("-" * 40)
    
    # Run acquisition with wider search range and non-coherent integration
    # 20ms integration (balance between sensitivity and code drift)
    results = acquisition.run_acquisition(iq_data, fs, doppler_range=10000, doppler_step=250, non_coherent_integration=20)
    
    # --- Step 3: Save Results ---
    print("\n" + "-" * 40)
    print("3. Saving Results...")
    print("-" * 40)
    
    save_path = "plots/acquisition_result.png"
    acquisition.plot_acquisition_results(results, title="GPS Acquisition - Real Data (20ms Integration)", save_path=save_path)
    
    print("\n✅ Process Complete!")

if __name__ == "__main__":
    main()
