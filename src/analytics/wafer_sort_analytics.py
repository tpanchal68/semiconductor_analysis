import io
import base64
from typing import List, Dict
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.optimization_engines import (
    VirtualMetrologyOptimizer,
    CausalAttributionEngine,
    WaferSortEarlyExitOptimizer
)


class WaferSortAnalytics:
    def __init__(self, data_path: str):
        self.df = pd.read_csv(data_path)

        # Apply standard formatting rule [cite: 2025-12-23]
        mappings = [{"column": "Timestamp", "format": "%Y-%m-%d %H:%M:%S"}]
        if "Timestamp" in self.df.columns:
            self.df = self.convert_datetime_format(self.df, mappings)

        # Bind the Inline Optimization Engines
        self.metrology_opt = VirtualMetrologyOptimizer(self.df)
        self.causal_eng = CausalAttributionEngine(self.df)
        self.early_exit_opt = WaferSortEarlyExitOptimizer(self.df)

        # Run execution pipeline
        self.wafer_df = self.metrology_opt.process_spatial_anomalies()
        self.pass_probabilities = self.metrology_opt.train_virtual_metrology()
        self.early_exit_flags = self.early_exit_opt.evaluate_exit_criteria()

    def convert_datetime_format(self, df: pd.DataFrame, mappings: List[Dict]) -> pd.DataFrame:
        for mapping in mappings:
            col = mapping['column']
            fmt = mapping['format']
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], format=fmt)
        return df

    def get_summary_stats(self) -> Dict:
        status_col = 'Test_Status' if 'Test_Status' in self.df.columns else 'scan_pass'
        yield_val = ((self.df[status_col] == 'PASS') | (
                    self.df[status_col] == 1)).mean() * 100 if status_col in self.df.columns else 0.0
        return {
            "yield": round(yield_val, 2),
            "early_exit_count": int(self.early_exit_flags.sum()),
            "total_die": len(self.df)
        }

    def _convert_to_base64(self) -> str:
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode()

    def generate_plot(self, plot_type: str) -> str:
        plt.figure(figsize=(8, 5))
        x_col = 'die_x' if 'die_x' in self.df.columns else 'x'
        y_col = 'die_y' if 'die_y' in self.df.columns else 'y'

        if plot_type == 'spatial_clusters':
            plt.scatter(self.wafer_df[x_col], self.wafer_df[y_col], c=self.wafer_df.get('cluster', -2), cmap='Set1',
                        marker='s', s=45)
            plt.title("Inline Optimizer: DBSCAN Spatial Failure Fingerprints")
        elif plot_type == 'virtual_metrology':
            sc = plt.scatter(self.wafer_df[x_col], self.wafer_df[y_col], c=self.pass_probabilities, cmap='RdYlGn',
                             marker='s', s=45)
            plt.colorbar(sc, label="Predicted Pass Likelihood")
            plt.title("Inline Optimizer: Virtual Metrology Skip-Test Map")
        elif plot_type == 'root_cause':
            importances, labels = self.causal_eng.get_feature_importances()
            plt.bar(labels, importances, color='teal')
            plt.title("Causal Attribution: Parametric Root Cause Driving Failure")
            plt.ylabel("Impact Weight")

        return self._convert_to_base64()
