import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import optuna

# Reduce TensorFlow verbosity during study sweeps
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class DeepAnomalyAutoencoder:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.features = ['iddq_ua', 'ro_speed_ghz', 'sram_fails'] if 'iddq_ua' in df.columns else ['i_ddq_ma',
                                                                                                   'total_power_w',
                                                                                                   'v_core_v']
        self.scaler = StandardScaler()
        self.autoencoder = None
        self.latent_dim = 2
        self.best_params = {}

    def _build_model(self, input_dim: int, hidden_dim: int, latent_dim: int) -> models.Model:
        """Helper to build the autoencoder functional architecture model."""
        encoder_input = layers.Input(shape=(input_dim,))
        enc = layers.Dense(hidden_dim, activation='relu')(encoder_input)
        latent = layers.Dense(latent_dim, activation='relu')(enc)
        encoder_model = models.Model(encoder_input, latent, name="encoder")

        decoder_input = layers.Input(shape=(latent_dim,))
        dec = layers.Dense(hidden_dim, activation='relu')(decoder_input)
        reconstructed = layers.Dense(input_dim, activation='linear')(dec)
        decoder_model = models.Model(decoder_input, reconstructed, name="decoder")

        return models.Model(encoder_input, decoder_model(encoder_model(encoder_input)), name="autoencoder")

    def optimize_hyperparameters(self, n_trials: int = 10) -> dict:
        """
        Executes an objective study sweep using Optuna to select the optimal
        network dimensions and learning rate configuration.
        """
        if not all(col in self.df.columns for col in self.features):
            return {}

        X_scaled = self.scaler.fit_transform(self.df[self.features])
        input_dim = X_scaled.shape[1]

        def objective(trial):
            # Define search space parameters
            hidden_dim = trial.suggest_int('hidden_dim', 4, 16, step=4)
            latent_dim = trial.suggest_int('latent_dim', 2, 3)
            lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

            # Build and compile model
            model = self._build_model(input_dim, hidden_dim, latent_dim)
            model.compile(optimizer=optimizers.Adam(learning_rate=lr), loss='mse')

            # Train with a validation split
            history = model.fit(
                X_scaled, X_scaled,
                validation_split=0.2,
                epochs=5,
                batch_size=16,
                verbose=0
            )

            # Target validation loss minimization
            val_loss = history.history['val_loss'][-1]
            return val_loss

        # Suppress Optuna logging noise during real-time study execution
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(direction='minimize')
        # FIX: Remove the unreferenced ternary operator block
        study.optimize(objective, n_trials=n_trials)

        self.best_params = study.best_params
        self.latent_dim = self.best_params.get('latent_dim', 2)
        return self.best_params

    def train_deep_autoencoder(self, epochs: int = 15) -> np.ndarray:
        """Trains the deep autoencoder using tuned or optimized hyperparameters."""
        if not all(col in self.df.columns for col in self.features):
            return np.zeros(len(self.df))

        X_scaled = self.scaler.fit_transform(self.df[self.features])
        input_dim = X_scaled.shape[1]

        # Use tuned values if optimize_hyperparameters was run, else use robust defaults
        hidden_dim = self.best_params.get('hidden_dim', 8)
        latent_dim = self.best_params.get('latent_dim', self.latent_dim)
        lr = self.best_params.get('lr', 0.01)

        self.autoencoder = self._build_model(input_dim, hidden_dim, latent_dim)
        self.autoencoder.compile(optimizer=optimizers.Adam(learning_rate=lr), loss='mse')

        # Fit the production model configuration
        self.autoencoder.fit(X_scaled, X_scaled, epochs=epochs, batch_size=16, verbose=0)

        predictions = self.autoencoder.predict(X_scaled, verbose=0)
        recon_errors = np.mean(np.power(X_scaled - predictions, 2), axis=1)
        return recon_errors

    def compute_silicon_families(self, n_clusters: int = 3) -> np.ndarray:
        """Groups devices by their underlying latent 'Electrical DNA' via K-Means."""
        if not all(col in self.df.columns for col in self.features):
            return np.zeros(len(self.df))

        X_scaled = self.scaler.fit_transform(self.df[self.features])
        kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
        return kmeans.fit_predict(X_scaled)
