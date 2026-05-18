import pandas as pd
import numpy as np
import random
from datetime import datetime


def generate_packaging_test_data(record_count=500):
    """
    Generates dummy semiconductor packaging (Final Test) records.
    """
    data = []

    # Industry Constants
    product_lines = ["AI-Core-X1", "Eth-Switch-G5", "IoT-Sensor-A"]
    package_types = ["BGA-484", "LGA-1151", "QFN-64"]
    test_sites = ["SJC_Factory_1", "MY_Factory_4", "TW_Factory_2"]

    for i in range(record_count):
        # 1. Identity
        serial_no = f"SN-{random.randint(1000000, 9999999)}"
        product = random.choice(product_lines)

        # 2. Environmental & Stress Testing
        # Burn-in temperature in Celsius
        burn_in_temp = random.choice([85, 105, 125])
        burn_in_hours = random.choice([12, 24, 48])

        # 3. Performance Metrics (Speed Binning)
        # Clock speed in GHz
        clock_speed = round(np.random.normal(3.2, 0.4), 2)
        power_draw = round(np.random.normal(15.0, 2.0), 1)  # Watts

        # 4. Logic for Final Pass/Fail
        # High power draw or low clock speed leads to a "Downgrade" or "Fail"
        if power_draw > 20.0 or clock_speed < 2.0:
            status = "REJECT"
            bin_name = "Thermal_Failure" if power_draw > 20.0 else "Under-Performing"
        else:
            status = "PASS"
            if clock_speed > 3.6:
                bin_name = "Ultra_High_Perf"
            elif clock_speed > 3.0:
                bin_name = "Standard_Perf"
            else:
                bin_name = "Value_Tier"

        data.append({
            "Serial_Number": serial_no,
            "Product_Family": product,
            "Package_Type": random.choice(package_types),
            "Test_Site": random.choice(test_sites),
            "Date_Code": "2615",  # One week after wafer fabrication
            "Burn_In_Temp_C": burn_in_temp,
            "Burn_In_Duration_Hrs": burn_in_hours,
            "Clock_Speed_GHz": clock_speed,
            "Power_Consumption_W": power_draw,
            "Final_Status": status,
            "Market_Bin": bin_name,
            "Visual_Insp_Result": "PASS" if random.random() > 0.02 else "FAIL_SCRATCH"
        })

    return pd.DataFrame(data)


# --- Execution ---
user_count = int(input("Enter number of packaging records to generate: ") or 500)
df_pkg = generate_packaging_test_data(user_count)

# Save to CSV
filename = f"final_test_records_{datetime.now().strftime('%Y%m%d')}.csv"
df_pkg.to_csv(filename, index=False)

print(f"\nSuccess! Generated {len(df_pkg)} packaging test records.")
print(f"Sample of final units:")
print(df_pkg[['Serial_Number', 'Product_Family', 'Clock_Speed_GHz', 'Final_Status', 'Market_Bin']].head())
