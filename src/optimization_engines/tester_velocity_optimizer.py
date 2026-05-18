import pandas as pd
import numpy as np


class TesterVelocityOptimizer:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def calculate_throughput_velocity(self) -> dict:
        """
        Evaluates multi-site test cell processing velocity
        and calculates Unit-Per-Hour (UPH) tracking metrics.
        """
        # Determine performance scaling metrics based on hardware availability
        total_units = len(self.df)

        # Track simulated hardware overhead metrics
        base_test_time_per_unit = 4.2  # Seconds
        handler_index_time = 0.35  # Seconds per device relocation

        if 'interrupt_lat_us' in self.df.columns:
            # Factor in real-time hardware interrupt overheads converted to seconds
            avg_interrupt_overhead = self.df['interrupt_lat_us'].mean() / 1_000_000
        else:
            avg_interrupt_overhead = 0.005

        estimated_cycle_time = base_test_time_per_unit + handler_index_time + avg_interrupt_overhead
        raw_uph = (3600 / estimated_cycle_time) * 4  # Assuming quad-site parallel testing architecture

        return {
            "average_cycle_time_sec": round(estimated_cycle_time, 3),
            "estimated_uph_rate": round(raw_uph, 1),
            "hardware_efficiency_pct": round((base_test_time_per_unit / estimated_cycle_time) * 100, 2)
        }

    def optimize_multi_site_balancing(self) -> pd.DataFrame:
        """
        Balances site-to-site resource allocation to prevent multi-site
        hardware starvation flags on parallel test cells.
        """
        optimized_df = self.df.copy()
        if 'pcie_retry_count' in optimized_df.columns:
            # Flags site allocation shifts if interface link limits drift
            optimized_df['site_starvation_risk'] = optimized_df['pcie_retry_count'] > 2
        else:
            optimized_df['site_starvation_risk'] = False

        return optimized_df
