import pandas as pd
import numpy as np

def generate_silicon_data(n_records=500, seed=42):
    """
    Generates a high-fidelity simulated dataset for Packaged Silicon
    encompassing Environmental, SerDes, Structural, and System metrics.
    """
    np.random.seed(seed)

    # 1. Environmental & Power Data (PI: Power Integrity)
    # Reflects thermal and voltage characteristics of the packaged unit
    env_data = pd.DataFrame({
        'temp_die_center_c': np.random.normal(70, 5, n_records),
        'temp_serdes_corner_c': np.random.normal(65, 4, n_records),
        'v_core_v': np.random.normal(0.9, 0.015, n_records),
        'v_mem_v': np.random.normal(1.2, 0.01, n_records),
        'i_ddq_ma': np.random.normal(450, 25, n_records),
        'total_power_w': np.random.normal(45, 3, n_records),
        'fan_speed_rpm': np.random.normal(3200, 150, n_records)
    })

    # 2. High-Speed I/O Data (SI: Signal Integrity)
    # Capturing Eye metrics and the 15 FEC Codewords discussed for DashIT
    serdes_data = pd.DataFrame({
        'mse_db': np.random.normal(-24, 2, n_records),
        'eye_height_mv': np.random.normal(130, 12, n_records),
        'eye_width_ps': np.random.normal(16, 1.5, n_records),
        'vga_gain_db': np.random.normal(12, 1, n_records)
    })
    for i in range(1, 16):
        serdes_data[f'fec_codeword_{i}'] = np.random.poisson(0.4, n_records)

    # 3. Structural & Timing Data (Process Corners)
    # Parametrics from internal Ring Oscillators and DFT structures
    structural_data = pd.DataFrame({
        'ring_osc_speed_ghz': np.random.normal(3.3, 0.08, n_records),
        'crit_path_delay_ps': np.random.normal(145, 4, n_records),
        'leakage_current_ua': np.random.normal(12, 1.5, n_records),
        'bist_status_pass': np.random.choice([1, 0], n_records, p=[0.995, 0.005])
    })

    # 4. System & Log Data (Reliability)
    # Soft error rates and latency metrics observed during long-run stress tests
    system_data = pd.DataFrame({
        'interrupt_lat_us': np.random.normal(5, 0.5, n_records),
        'pcie_retry_count': np.random.poisson(0.05, n_records),
        'correctable_ecc_errors': np.random.poisson(1.2, n_records),
        'uptime_hrs': np.random.uniform(24, 168, n_records)
    })

    # Combine into Master Digital Twin DataFrame
    master_df = pd.concat([env_data, serdes_data, structural_data, system_data], axis=1)
    master_df.insert(0, 'unit_id', [f'SN_{1000 + i}' for i in range(n_records)])

    return master_df

# Execute and Save
df_packaged = generate_silicon_data(n_records=500)
df_packaged.to_csv('../raw_data/silicon_packaged_data_groups.csv', index=False)

print(f"Generated {df_packaged.shape[0]} records with {df_packaged.shape[1]} features.")
