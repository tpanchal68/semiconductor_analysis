import pandas as pd
import numpy as np
import statsmodels.api as sm

class ManufacturingInsightEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def plot_reliability_trendline(self):
        """Fits a Poisson Regression line to capture soft error trendlines across operational hours."""
        import matplotlib.pyplot as plt
        if 'uptime_hrs' in self.df.columns and 'correctable_ecc_errors' in self.df.columns:
            X_sys = sm.add_constant(self.df[['uptime_hrs']])
            res = sm.GLM(self.df['correctable_ecc_errors'], X_sys, family=sm.families.Poisson()).fit()
            self.df['predicted_ecc'] = res.predict(X_sys)

            sorted_sys = self.df.sort_values('uptime_hrs')
            plt.scatter(self.df['uptime_hrs'], self.df['correctable_ecc_errors'], alpha=0.3, label='Observed ECC')
            plt.plot(sorted_sys['uptime_hrs'], sorted_sys['predicted_ecc'], color='orange', linewidth=3, label='Poisson Trend')
            plt.title("Reliability Trendline Analysis")
            plt.xlabel("Uptime (Hours)")
            plt.ylabel("ECC Count")
            plt.legend()
        else:
            plt.text(0.5, 0.5, 'Reliability Metrics Unavailable', ha='center', va='center')
