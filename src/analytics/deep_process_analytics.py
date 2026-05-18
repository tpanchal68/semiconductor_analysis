import io
import base64
from typing import List, Dict
import matplotlib

# CRITICAL: Set the backend to 'Agg' before importing pyplot for Flask stability
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Import your newly integrated Unsupervised Anomaly and SPC engines
from src.optimization_engines.deep_anomaly_autoencoder import DeepAnomalyAutoencoder
from src.optimization_engines.line_control_spc_engine import LineControlSPCEngine

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)


class DeepProcessAnalytics:
    def __init__(self, data_path: str):
        self.df = pd.read_csv(data_path)

        # Apply standardized timeline rule [cite: 2025-12-23]
        mappings = [{"column": "Timestamp", "format": "%Y-%m-%d %H:%M:%S"}]
        if "Timestamp" in self.df.columns:
            self.df = self.convert_datetime_format(self.df, mappings)

        # Instantiate inline optimization engines
        self.ae_engine = DeepAnomalyAutoencoder(self.df)
        self.spc_engine = LineControlSPCEngine(self.df)

        # 1. TRIGGER OPTUNA HYPER-TUNING FIRST (Runs a lightweight 5-trial sweep on load)
        self.tuned_parameters = self.ae_engine.optimize_hyperparameters(n_trials=5)

        # 2. Train using optimal hyper-tuned weights discovered by Optuna
        self.recon_errors = self.ae_engine.train_deep_autoencoder(epochs=15)
        self.silicon_families = self.ae_engine.compute_silicon_families()

    def convert_datetime_format(self, df: pd.DataFrame, mappings: List[Dict]) -> pd.DataFrame:
        """User-defined utility to standardize datetime columns."""
        for mapping in mappings:
            col = mapping['column']
            fmt = mapping['format']
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], format=fmt)
        return df

    def get_summary_stats(self) -> Dict:
        """
        Calculates manufacturing capability indices, bottleneck states,
        and deep modeling evaluation loss parameters for the UI.
        """
        cpk_value = self.spc_engine.calculate_cpk()
        bottleneck_data = self.spc_engine.analyze_line_bottlenecks()

        # Determine average reconstruction error baseline
        mean_loss = np.mean(self.recon_errors) if len(self.recon_errors) > 0 else 0.0

        return {
            "cpk_index": cpk_value,
            "mean_reconstruction_error": round(float(mean_loss), 4),
            "bottleneck_station": bottleneck_data.get("bottleneck_station", "Unknown"),
            "station_status": bottleneck_data.get("station_status", "NOMINAL"),
            "avg_handling_delay_us": bottleneck_data.get("average_handling_delay_us", 0.0),
            "total_inspected": len(self.df)
        }

    def _convert_plt_to_base64(self) -> str:
        """Saves current active drawing canvas space directly into a base64 vector string."""
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        plt.close()
        img.seek(0)
        return base64.b64encode(img.getvalue()).decode()

    def generate_plot(self, plot_type: str = 'autoencoder_loss') -> str:
        """Generates dark-themed deep learning distribution and capability visualizations."""
        plt.figure(figsize=(8, 4.5))
        plt.style.use('dark_background')  # Seamless aesthetic integration with your template layouts

        if plot_type == 'autoencoder_loss':
            # Draw Silicon Reconstruction Loss Density Histogram
            if len(self.recon_errors) > 0:
                plt.hist(self.recon_errors, bins=30, color='#a855f7', alpha=0.7, edgecolor='#ffffff')
                plt.axvline(np.mean(self.recon_errors) + 3 * np.std(self.recon_errors),
                            color='#dc3545', linestyle='--', label='3-Sigma Alert Threshold')
                plt.title(
                    "Deep Learning Autoencoder: Reconstruction Loss Distribution\n(Isolating Electrical DNA Deviation)",
                    fontsize=11)
                plt.xlabel("Mean Squared Error (MSE) Loss")
                plt.ylabel("Device Count")
                plt.legend()
            else:
                plt.text(0.5, 0.5, 'Neural Network Evaluation Matrix Unavailable', ha='center', va='center')

        elif plot_type == 'silicon_families':
            # Maps the K-Means clusters against process speeds to showcase classification
            x_col = 'ring_osc_speed_ghz' if 'ring_osc_speed_ghz' in self.df.columns else 'v_core_v'
            y_col = 'crit_path_delay_ps' if 'crit_path_delay_ps' in self.df.columns else 'temp_die_center_c'

            if x_col in self.df.columns and y_col in self.df.columns:
                scatter = plt.scatter(self.df[x_col], self.df[y_col], c=self.silicon_families, cmap='plasma', alpha=0.6,
                                      s=30)
                plt.title(
                    "Unsupervised Fingerprinting: Silicon Families Classification\n(K-Means Electrical DNA Grouping)",
                    fontsize=11)
                plt.xlabel("Ring Oscillator Speed (GHz)" if x_col == 'ring_osc_speed_ghz' else "V_Core (V)")
                plt.ylabel("Critical Path Delay (ps)" if y_col == 'crit_path_delay_ps' else "Die Temp (°C)")
                cbar = plt.colorbar(scatter)
                cbar.set_label("Assigned Cluster Bin")
            else:
                plt.text(0.5, 0.5, 'DNA Slicing Metrics Missing', ha='center', va='center')

        return self._convert_plt_to_base64()
