# Wafer Process Analysis Platform

An enterprise-grade, data-driven Flask web application designed for high-fidelity semiconductor manufacturing analytics and inline optimization. This platform integrates statistical processing, unsupervised anomaly detection via deep learning architectures, automated process corner binning, and hyperparameter tuning to evaluate and optimize device yields through multi-stage production contexts.

The codebase is organized into a clean, decoupled architecture:
- **Optimization Engines (`src/optimization_engines/`)**: Low-level algorithmic blocks containing model architectures, training routines, and mathematical estimators.
- **Middle-Tier Analytics (`src/analytics/`)**: Orchestration modules that clean timelines, manage state transformations, execute modeling loops, and serialize visualizations into memory-optimized vector strings.
- **Routing & Views (`src/routes.py` & `src/templates/`)**: Lightweight controller mappings and dark-themed structural layouts that present real-time dashboards to the end-user.

---

## Technical Architecture & Core Pillars

### 1. Stage 1: Wafer Sort Optimization Layer
Monitors semiconductor characteristics directly at the probe and wafer test insertions to flag manufacturing escapes and reclaim valuable tester cell capacity.
- **Inline Metrology & Spatial Profiling**: Utilizes a density-based clustering algorithm (**DBSCAN**) on early parametric signals (`iddq_ua`, `ro_speed_ghz`, `sram_fails`) to isolate structural wafer-level defects and capture spatial failure patterns (e.g., edge effects) in real-time.
- **Virtual Metrology Skip-Test Map**: Trains an inline **Random Forest Classifier** on early electrical sensor matrices to output predictive functional yield pass probabilities, allowing premium parts to skip redundant insertions.
- **Catastrophic Early Exit Triggering**: Evaluates failure trend limits across continuous moving windows. If moving failure rates breach guardbands, a structural "early exit" sequence is commanded to instantly abort the tester routine on defective lots.
- **Causal Attribution & Failure Mapping**: Automatically extracts non-linear random forest feature importances to establish statistical weightings back to specific fab tool operations, isolating the root causes driving physical variance.

### 2. Stage 2: Packaged Process & Adaptive Test Layer
Tracks the performance of singulated, packaged silicon dies across high-speed I/O testing, assembly configurations, and long-run thermal stresses.
- **Environmental Power Integrity Isolation**: Implements an **Isolation Forest** outlier model to scan multi-domain environmental profiles (`temp_die_center_c`, `v_core_v`, `i_ddq_ma`, `total_power_w`), automatically flagging high-risk thermal and power escapes.
- **Signal Integrity Anomaly Tracking**: Utilizes a deep learning **PyTorch Autoencoder** to evaluate multi-lane high-speed SerDes metrics (`mse_db`, `eye_height_mv`, `eye_width_ps`) along with 15 sequential forward error correction (**FEC**) codewords, identifying degradation through MSE reconstruction loss tracking.
- **Corner Kitting and Binning**: Employs **K-Means Clustering** on internal ring oscillator speeds and path delays to segment devices into physical process corners (e.g., SS, TT, FF), enabling optimal pairing of silicon dies with substrate packages.
- **Reliability Lifecycle Projections**: Fits a **Poisson Regression Generalized Linear Model (GLM)** to correctable ECC memory errors over uptime operational hours, mapping out soft error rates (SER) and reliability trendlines.
- **Tester Velocity Tuning**: Evaluates parallel multi-site test cell performance and hardware interrupt overhead to continuously compute Unit-Per-Hour (UPH) capacity and prevent site starvation during quad-site insertions.

### 3. Stage 3: Deep Learning Optimization & SPC Layer
Bridges advanced machine learning with formal statistical process boundaries to maintain long-term process capability.
- **Optuna Hyperparameter Tuning**: Dynamically optimizes network dimensions and execution parameters on initialization. Optuna conducts an objective search across dense hidden layer configurations and log-spaced learning rates to minimize validation reconstruction loss.
- **Unsupervised Electrical DNA Fingerprinting**: Uses the latent space vectors from hyper-tuned **TensorFlow Keras Autoencoders** to profile underlying electrical traits and segment device lineages.
- **Statistical Process Control (SPC)**: Computes the formal **Cpk Process Capability Index** relative to upper and lower engineering specification limits (USL/LSL) to measure manufacturing stability against a baseline target of >= 1.33.
- **Line Control Bottleneck Analysis**: Tracks real-time machine indexing latency to spot cell friction points and issue alert flags when handling delays drift out of limits.

---

## Project Structure

```text
src/
├── analytics/
│   ├── deep_process_analytics.py       # Orchestrator for Stage 3 ML and SPC models
│   ├── packaged_process_analytics.py   # Orchestrator for final packaged test metrics
│   └── wafer_sort_analytics.py         # Orchestrator for early sort mapping and probe telemetry
├── data/
│   ├── dummy_data_consolidator.py      # Unified simulation data generator engine
│   └── raw_data/
│       ├── silicon_packaged_data_groups.csv  # High-dimensional packaged device logs
│       └── wafer_test_records_20260402.csv   # Early sort and coordinate records
├── optimization_engines/
│   ├── __init__.py                     # Package entry point exposing core modules
│   ├── adaptive_routing_optimizer.py   # Isolation Forest power outlier routing blocks
│   ├── causal_attribution_engine.py    # Random Forest failure root-cause analysis
│   ├── deep_anomaly_autoencoder.py     # Optuna-tuned TensorFlow Keras Autoencoder models
│   ├── line_control_spc_engine.py      # SPC capability Cpk calculator and bottleneck logic
│   ├── packaging_kitting_optimizer.py  # K-Means process corner clustering blocks
│   ├── predictive_suite_coordinator.py # Global pipeline runner and execution director
│   ├── tester_velocity_optimizer.py    # Parallel testing UPH and capacity metrics calculator
│   ├── virtual_metrology_optimizer.py  # DBSCAN and Random Forest inline yield models
│   └── wafer_sort_early_exit_optimizer.py # Rolling window fail evaluation controllers
├── routes.py                           # Blueprint routing controllers and path resolvers
├── templates/
│   ├── base.html                       # Core dark-themed layout wrapper
│   ├── wafer_dashboard.html            # UI grid for Stage 1 sort and causal mapping
│   ├── package_dashboard.html          # UI grid for Stage 2 signal integrity and power mapping
│   └── deep_dashboard.html             # UI grid for Stage 3 Neural Network losses and Cpk metrics
├── config.py                           # App development and production configuration environments
├── models.py                           # SQLAlchemy database tracking schemas
└── __init__.py                         # Application factory block and extension initialization
