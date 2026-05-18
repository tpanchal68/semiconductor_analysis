import pandas as pd
import numpy as np


def generate_wafer_sort_data(n_wafers=5, dies_per_wafer=800):
    np.random.seed(42)
    wafer_data = []

    # Grid for a circular wafer
    radius = 15
    grid_size = int(np.sqrt(dies_per_wafer) * 1.5)
    x = np.linspace(-radius, radius, grid_size)
    y = np.linspace(-radius, radius, grid_size)
    xv, yv = np.meshgrid(x, y)
    coords = np.stack([xv.flatten(), yv.flatten()], axis=1)
    mask = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2) <= radius
    wafer_coords = coords[mask]

    for w_id in range(1, n_wafers + 1):
        n_dies = len(wafer_coords)

        # Simulated Measurements
        iddq = np.random.normal(10, 1.2, n_dies)  # Leakage
        ro_speed = np.random.normal(3.5, 0.15, n_dies)  # Speed
        sram_fails = np.random.poisson(0.1, n_dies)  # Memory defects
        scan_pass = np.random.choice([1, 0], n_dies, p=[0.98, 0.02])  # Logic health

        # Inject Spatial Patterns (e.g., Edge Effects)
        dist = np.sqrt(wafer_coords[:, 0] ** 2 + wafer_coords[:, 1] ** 2)
        edge_mask = dist > (radius * 0.85)
        iddq[edge_mask] += np.random.normal(8, 3, edge_mask.sum())  # Edge Leakage

        w_df = pd.DataFrame({
            'wafer_id': w_id,
            'die_x': wafer_coords[:, 0],
            'die_y': wafer_coords[:, 1],
            'iddq_ua': iddq,
            'ro_speed_ghz': ro_speed,
            'sram_fails': sram_fails,
            'scan_pass': scan_pass
        })
        wafer_data.append(w_df)

    return pd.concat(wafer_data)


df_wafer = generate_wafer_sort_data()
df_wafer.to_csv('../raw_data/wafer_sort_data.csv', index=False)
print("Wafer Sort Data Generated Successfully.")
