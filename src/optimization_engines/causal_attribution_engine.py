import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


class CausalAttributionEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.feature_names = ['iddq_ua', 'ro_speed_ghz', 'sram_fails']

    def get_feature_importances(self) -> tuple:
        """Calculates statistical feature weights to trace root causes of failures."""
        required = self.feature_names + ['scan_pass']
        if not all(col in self.df.columns for col in required):
            return np.array([0.33, 0.33, 0.33]), self.feature_names

        X = self.df[self.feature_names]
        y = self.df['scan_pass']

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        return rf.feature_importances_, ['Iddq (Leakage)', 'RO Speed', 'SRAM Fails']
