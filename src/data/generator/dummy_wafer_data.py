import pandas as pd
import numpy as np
import random
from datetime import datetime


def generate_wafer_data(record_count=1000):
    """
    Generates dummy semiconductor test records with industry-standard parameters.
    """
    data = []

    # Industry Constants
    lots = [f"LOT_{i:03d}" for i in range(101, 105)]
    wafers = list(range(1, 26))
    date_code = "2614"  # Year 2026, Week 14
    equip_ids = ["ETCH_A1", "ETCH_B2", "LITHO_01"]

    # Simulation parameters
    fail_rate = 0.08  # 8% baseline failure

    for i in range(record_count):
        # 1. Identity & Traceability
        lot_id = random.choice(lots)
        wafer_id = random.choice(wafers)

        # 2. X, Y Coordinates (Assuming a 300mm wafer grid)
        # Using a range that approximates a circular layout
        die_x = random.randint(-50, 50)
        die_y = random.randint(-50, 50)

        # 3. Parametric Measurements (Voltage and Current)
        # Normal chips center around 1.2V
        v_min = round(np.random.normal(1.2, 0.05), 3)
        leakage = round(abs(np.random.normal(0.5, 0.1)), 2)

        # 4. Logic for Pass/Fail and Binning
        # We simulate an anomaly: if X and Y are both > 40 (the edge), failure rate triples
        current_fail_edge = fail_rate * 3 if (abs(die_x) > 40 or abs(die_y) > 40) else fail_rate

        if random.random() < current_fail_edge:
            status = "FAIL"
            bin_code = random.choice(["10_Short", "12_Open", "15_Thermal"])
            v_min = round(v_min * 0.5, 3)  # Drop voltage for failed chips
        else:
            status = "PASS"
            # Bin 01 is Premium (Higher Voltage/Efficiency), Bin 02 is Standard
            bin_code = "01_Premium" if v_min > 1.22 else "02_Standard"

        data.append({
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Date_Code": date_code,
            "Lot_ID": lot_id,
            "Wafer_ID": wafer_id,
            "Die_X": die_x,
            "Die_Y": die_y,
            "Equip_ID": random.choice(equip_ids),
            "Test_Status": status,
            "Bin_Code": bin_code,
            "V_Min_Volt": v_min,
            "Leakage_nA": leakage,
            "Test_Time_ms": random.randint(380, 520)
        })

    return pd.DataFrame(data)


# --- Execution ---
user_count = int(input("Enter number of test records to generate: ") or 1000)
df = generate_wafer_data(user_count)

# Save to CSV
filename = f"wafer_test_records_{datetime.now().strftime('%Y%m%d')}.csv"
df.to_csv(filename, index=False)

print(f"\nSuccess! Generated {len(df)} records.")
print(f"File saved as: {filename}")
print("\nSample of generated data:")
print(df.head())
