import pandas as pd
import numpy as np

class WaferSortEarlyExitOptimizer:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def evaluate_exit_criteria(self, gross_fail_threshold: float = 0.30) -> pd.Series:
        """Evaluates consecutive failure trends to issue tester early exit triggers."""
        status_col = 'Test_Status' if 'Test_Status' in self.df.columns else 'scan_pass'
        if status_col not in self.df.columns:
            return pd.Series(False, index=self.df.index)

        # Flag catastrophic failures dynamically
        fails = (self.df[status_col] == 'FAIL') | (self.df[status_col] == 0)
        return fails.rolling(window=10, min_periods=1).mean() > gross_fail_threshold
