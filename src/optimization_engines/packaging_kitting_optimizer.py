import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class PackagingKittingOptimizer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.struct_features = ['ring_osc_speed_ghz', 'crit_path_delay_ps']

    def compute_kitting_profiles(self) -> pd.Series:
        """Executes process corner classification via structural K-Means mapping."""
        if not all(col in self.df.columns for col in self.struct_features):
            self.df['corner_bin'] = 0
            return self.df['corner_bin']

        struct_scaled = StandardScaler().fit_transform(self.df[self.struct_features])
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        self.df['corner_bin'] = kmeans.fit_predict(struct_scaled)
        return self.df['corner_bin']

    def plot_process_corners(self):
        """Draws silicon corner grouping parameters directly onto active plots."""
        import matplotlib.pyplot as plt
        if 'corner_bin' in self.df.columns and all(col in self.df.columns for col in self.struct_features):
            plt.scatter(self.df['ring_osc_speed_ghz'], self.df['crit_path_delay_ps'], c=self.df['corner_bin'], cmap='viridis', alpha=0.6)
            plt.title("Silicon Process Binning (K-Means)")
            plt.xlabel("Ring Osc Speed (GHz)")
            plt.ylabel("Path Delay (ps)")
        else:
            plt.text(0.5, 0.5, 'Process Corner Profiles Unavailable', ha='center', va='center')
