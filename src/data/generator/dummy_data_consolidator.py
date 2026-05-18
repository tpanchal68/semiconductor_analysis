import os
import random
import pandas as pd
import numpy as np
from datetime import datetime


def generate_project_datasets(n_records=500, seed=42):
    """
    Consolidates and generates high-fidelity data arrays mapping precisely
    to Stage 1 (Wafer Sort) and Stage 2 (Packaged Process) Optimization views.
    """
    np.random.seed(seed)
    random.seed(seed)

    # -------------------------------------------------------------------------
    # DATASET 1: WAFER SORT METRICS (wafer_test_records_20260402.csv)
    # -------------------------------------------------------------------------
    lots = [f"LOT_{i:03d}" for i in range(101, 105)]
    date_code = "2614"  # Year 2026, Week 14
    equip_ids = ["ETCH_A1", "ETCH_B2", "LITHO_01"]

    # Calculate circular grid matrix
    radius = 15
    grid_points = int(np.sqrt(n_records) * 1.5)
    x = np.linspace(-radius, radius, grid_points)
    y = np.linspace(-radius, radius, grid_points)
    xv, yv = np.meshgrid(x, y)
    coords = np.stack([xv.flatten(), yv.flatten()], axis=1)
    mask = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2) <= radius
    wafer_coords = coords[mask]

    # Adjust indexing lengths to scale gracefully
    if len(wafer_coords) < n_records:
        extra_indices = np.random.choice(len(wafer_coords), n_records - len(wafer_coords))
        wafer_coords = np.vstack([wafer_coords, wafer_coords[extra_indices]])
    else:
        wafer_coords = wafer_coords[:n_records]

    iddq = np.random.normal(10, 1.2, n_records)
    ro_speed = np.random.normal(3.5, 0.15, n_records)
    sram_fails = np.random.poisson(0.1, n_records)
    scan_pass = np.random.choice([1, 0], n_records, p=[0.94, 0.06])

    # Inject edge leakage failure signature anomalies
    dist = np.sqrt(wafer_coords[:, 0] ** 2 + wafer_coords[:, 1] ** 2)
    edge_mask = dist > (radius * 0.85)
    iddq[edge_mask] += np.random.normal(8, 3, edge_mask.sum())
    scan_pass[edge_mask] = np.random.choice([1, 0], size=edge_mask.sum(), p=[0.75, 0.25])

    wafer_sort_df = pd.DataFrame({
        "Timestamp": pd.date_range(start="2026-04-02 08:00:00", periods=n_records, freq="min").strftime(
            "%Y-%m-%d %H:%M:%S"),
        "Date_Code": date_code,
        "Lot_ID": [random.choice(lots) for _ in range(n_records)],
        "wafer_id": np.random.choice([1, 2, 3, 4], size=n_records),
        "die_x": wafer_coords[:, 0],
        "die_y": wafer_coords[:, 1],
        "Equip_ID": [random.choice(equip_ids) for _ in range(n_records)],
        "Test_Status": ["PASS" if p == 1 else "FAIL" for p in scan_pass],
        "V_Min_Volt": round(pd.Series(np.random.normal(1.2, 0.05, n_records)), 3),
        "iddq_ua": iddq,
        "ro_speed_ghz": ro_speed,
        "sram_fails": sram_fails,
        "scan_pass": scan_pass
    })

    # Adjust V_Min for failing components
    wafer_sort_df.loc[wafer_sort_df['scan_pass'] == 0, 'V_Min_Volt'] = round(wafer_sort_df['V_Min_Volt'] * 0.5, 3)

    # -------------------------------------------------------------------------
    # DATASET 2: PACKAGED PROCESS METRICS (silicon_packaged_data_groups.csv)
    # -------------------------------------------------------------------------
    # Combine Wafer Sort traits directly into the packaged twin profile to avoid mismatch exceptions
    env_data = pd.DataFrame({
        'temp_die_center_c': np.random.normal(70, 5, n_records),
        'temp_serdes_corner_c': np.random.normal(65, 4, n_records),
        'v_core_v': np.random.normal(0.9, 0.015, n_records),
        'v_mem_v': np.random.normal(1.2, 0.01, n_records),
        'i_ddq_ma': np.random.normal(450, 25, n_records),
        'total_power_w': np.random.normal(45, 3, n_records),
        'fan_speed_rpm': np.random.normal(3200, 150, n_records)
    })

    # Reflect higher current draws on leakage exceptions
    env_data.loc[wafer_sort_df['scan_pass'] == 0, 'i_ddq_ma'] += 75.0
    env_data.loc[wafer_sort_df['scan_pass'] == 0, 'total_power_w'] += 10.0

    serdes_data = pd.DataFrame({
        'mse_db': np.random.normal(-24, 2, n_records),
        'eye_height_mv': np.random.normal(130, 12, n_records),
        'eye_width_ps': np.random.normal(16, 1.5, n_records),
        'vga_gain_db': np.random.normal(12, 1, n_records)
    })
    for i in range(1, 16):
        serdes_data[f'fec_codeword_{i}'] = np.random.poisson(0.4, n_records)

    structural_data = pd.DataFrame({
        'ring_osc_speed_ghz': np.random.normal(3.3, 0.08, n_records),
        'crit_path_delay_ps': np.random.normal(145, 4, n_records),
        'leakage_current_ua': np.random.normal(12, 1.5, n_records),
        'bist_status_pass': np.random.choice([1, 0], n_records, p=[0.995, 0.005])
    })

    system_data = pd.DataFrame({
        'interrupt_lat_us': np.random.normal(5, 0.5, n_records),
        'pcie_retry_count': np.random.poisson(0.05, n_records),
        'correctable_ecc_errors': np.random.poisson(1.2, n_records),
        'uptime_hrs': np.random.uniform(24, 168, n_records)
    })

    # Build the packaged device log table
    packaged_df = pd.concat([
        wafer_sort_df[['Timestamp', 'wafer_id', 'die_x', 'die_y', 'scan_pass']],
        env_data,
        serdes_data,
        structural_data,
        system_data
    ], axis=1)

    packaged_df.insert(0, 'unit_id', [f'SN_{1000 + i}' for i in range(n_records)])

    return wafer_sort_df, packaged_df


if __name__ == "__main__":
    # Ensure correct target file directory configurations exist
    target_dir = os.path.join("src", "data", "raw_data")
    os.makedirs(target_dir, exist_ok=True)

    # filename = f"wafer_test_records_{datetime.now().strftime('%Y%m%d')}.csv"
    wafer_file = os.path.join(target_dir, "wafer_test_records_20260402.csv")
    packaged_file = os.path.join(target_dir, "silicon_packaged_data_groups.csv")

    w_df, p_df = generate_project_datasets(n_records=600)

    # Save to final directories
    w_df.to_csv(wafer_file, index=False)
    p_df.to_csv(packaged_file, index=False)

    print("--- CONSOLIDATION SUMMARY ---")
    print(f"1. Stage 1 Log File Created: {wafer_file} ({w_df.shape})")
    print(f"2. Stage 2 Log File Created: {packaged_file} ({p_df.shape})")
    print("All optimization parameters linked successfully. Ready to run Flask routes.")
