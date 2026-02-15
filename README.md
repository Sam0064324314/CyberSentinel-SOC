<div align="center">

# 🛡️ CyberSentinel

### Enterprise Security Operations Center — Intrusion Detection Dashboard

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-006CB4?style=for-the-badge)](https://xgboost.readthedocs.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-00FF9F?style=for-the-badge)](LICENSE)

**A production-grade, real-time intrusion detection system with multi-model ML benchmarking, live packet simulation, and explainable AI — built for security professionals and ML engineers.**

[Quick Start](#-quick-start) · [Features](#-features) · [Architecture](#-architecture) · [Models](#-models) · [Deployment](#-deployment) · [Contributing](#-contributing)

</div>

---

## 🎯 Overview

CyberSentinel is an enterprise-grade Security Operations Center (SOC) dashboard that combines machine learning-based intrusion detection with real-time traffic simulation and explainable AI. It processes network flow data to classify traffic as normal or malicious across multiple attack categories.

### What It Does

| Capability | Description |
|:---|:---|
| **Multi-Model Training** | Train and compare 5 ML architectures on the same dataset split |
| **Real-Time Simulation** | Live packet stream with dynamic threat level assessment |
| **Model Comparison** | Side-by-side benchmarking with accuracy, F1, ROC-AUC, and training time charts |
| **Explainable AI** | Feature importance analysis, local explanations, and confidence interpretation |
| **SOC Dashboard** | Enterprise-grade dark theme with synchronized cross-tab state |

### Who It's For

- **Security Analysts** — Monitor simulated network traffic and understand threat patterns
- **ML Engineers** — Benchmark classification models on network intrusion data
- **Students & Researchers** — Learn about cybersecurity ML with beginner-friendly explanations
- **DevOps / SRE** — Deploy via Docker with built-in health checks

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- pip (Python package manager)
- ~2 GB RAM (recommended)

### 1. Clone & Install

```bash
git clone https://github.com/your-repo/cybersentinel.git
cd cybersentinel/IntrusionDetectionDashboard

pip install -r requirements.txt
```

### 2. Add Dataset

Place the dataset CSV files in the **parent directory** of the dashboard folder:

```
Detect cyber Intrusion/
├── dataset-part1.csv          ← Data files here
├── dataset-part2.csv
├── dataset-part3.csv
├── dataset-part4.csv
└── IntrusionDetectionDashboard/
    ├── app.py
    └── ...
```

> **Dataset:** This project uses the [NF-UQ-NIDS](https://staff.itee.uq.edu.au/marius/NIDS_datasets/) network flow dataset. Each CSV contains labeled network flows with 15 feature columns and a `LABEL` column for classification.

### 3. Launch

```bash
streamlit run app.py
```

The dashboard opens at **http://localhost:8501**.

### 4. Train & Explore

1. **Sidebar → ⚡ TRAIN ALL MODELS** — Trains all 5 architectures
2. **📈 MODEL COMPARISON** — View benchmarks and select best model
3. **🚨 REAL-TIME OPS → ▶ START** — Launch live simulation
4. **🧠 INTELLIGENCE** — Explore model reasoning

---

## ✨ Features

### 📊 Overview Dashboard
- Live traffic counters synced with simulation engine
- Attack rate (rolling 30-packet window)
- Unique IP tracking
- Dynamic threat level indicator (LOW / MODERATE / HIGH)
- Traffic class distribution chart
- Active model performance snapshot

### 🚨 Real-Time Operations
- **Live Packet Stream** — Scrolling log with color-coded severity
- **Threat Intelligence Panel** — Dynamic threat gauge, session metrics, attack frequency
- **Simulation Controls** — Start / Stop / Reset with no UI flicker
- **Cross-Tab Sync** — All tabs read from a single source of truth

### 📈 Model Comparison
- **Train All Models** — One-click benchmark across 5 architectures
- **Best Model Auto-Selection** — Automatically identifies highest F1 score
- **Visual Comparison Charts:**
  - Accuracy bar chart
  - F1 Score bar chart
  - ROC-AUC bar chart
  - Training time comparison
- **Summary Table** — All metrics with highlighted maxima
- **Per-Model Confusion Matrix** — Dropdown selector

### 🧠 Intelligence (XAI Engine)
- **Global Feature Importance** — Bar chart with Plotly
- **Beginner Panel** — "What Is Feature Importance?" explainer
- **Impact Scale Legend** — Dominant / Strong / Weak interpretation table
- **Auto-Generated Behavior Summary** — Dynamic narrative based on top-3 feature categories
- **Local Explanation** — Live prediction on random sample with:
  - Per-feature reasoning
  - Confidence gauge (HIGH / MODERATE / LOW)
  - Class probability distribution chart
- **Responsible AI** — Model limitation awareness and human-in-the-loop recommendations

### 🖥️ Resource Monitor
- Real-time RAM and CPU usage
- Progress bar indicators
- High-memory warning threshold

---

## 🏗️ Architecture

```
IntrusionDetectionDashboard/
│
├── app.py                    # Main application (single source of truth)
├── config.py                 # Centralized configuration & constants
├── requirements.txt          # Python dependencies
├── Dockerfile                # Production container
│
├── assets/
│   └── style.css             # Enterprise dark theme (SOC styling)
│
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py      # Data loading, scaling, splitting
│   ├── training.py           # Model training (5 architectures)
│   ├── evaluation.py         # Metrics, confusion matrix, ROC, feature importance
│   ├── model_io.py           # Model save/load (joblib)
│   ├── explainability.py     # SHAP integration
│   └── logger.py             # Structured logging
│
├── models/                   # Saved model artifacts (.joblib)
└── logs/                     # Application logs
```

### State Management

All live data flows through `st.session_state` as a **single source of truth**:

```python
# Simulation state (shared across ALL tabs)
'simulation_running': bool
'total_packets': int
'blocked_packets': int
'attack_history': list      # Rolling window, max 30
'packet_log': list           # Max 50 entries
'unique_ips': set
'threat_level': str          # LOW / MODERATE / HIGH

# Model registry
'model_registry': dict       # {name: {model, accuracy, f1, roc_auc, ...}}
'active_model_name': str
'best_model_name': str
```

### Data Pipeline

```
Raw CSVs → load_data() → preprocess_data() → split_data() → train_model()
                ↓                ↓                              ↓
          Sampling         LabelEncoder              Model + Metrics
          (configurable)   StandardScaler             → Registry
```

---

## 🤖 Models

| Model | Type | Strengths |
|:---|:---|:---|
| **Random Forest** | Ensemble (Bagging) | Robust, interpretable, handles imbalanced data |
| **Decision Tree** | Tree-based | Fast, fully interpretable, low memory |
| **Gaussian NB** | Probabilistic | Extremely fast training, good baseline |
| **XGBoost** | Ensemble (Boosting) | State-of-the-art accuracy, handles complex patterns |
| **MLP** | Neural Network | Captures non-linear relationships |

### Feature Set (15 Network Flow Features)

| Feature | Category | Description |
|:---|:---|:---|
| `DST_TOS` | Protocol | Destination Type of Service |
| `SRC_TOS` | Protocol | Source Type of Service |
| `TCP_WIN_SCALE_OUT` | TCP | Outbound window scale factor |
| `TCP_WIN_SCALE_IN` | TCP | Inbound window scale factor |
| `TCP_FLAGS` | TCP | TCP flag combination |
| `TCP_WIN_MAX_OUT` | TCP | Max outbound window size |
| `PROTOCOL` | Protocol | IP protocol number |
| `TCP_WIN_MIN_OUT` | TCP | Min outbound window size |
| `TCP_WIN_MIN_IN` | TCP | Min inbound window size |
| `TCP_WIN_MAX_IN` | TCP | Max inbound window size |
| `LAST_SWITCHED` | Timing | Last packet timestamp |
| `TCP_WIN_MSS_IN` | TCP | Maximum segment size (inbound) |
| `TOTAL_FLOWS_EXP` | Flow | Total exported flows |
| `FIRST_SWITCHED` | Timing | First packet timestamp |
| `FLOW_DURATION_MILLISECONDS` | Timing | Total flow duration |

### Threat Level Logic

```
attack_rate = sum(last_30_predictions) / 30

if attack_rate > 0.20  → HIGH    🔴
if attack_rate > 0.05  → MODERATE 🟡
else                   → LOW      🟢
```

---

## 🐳 Deployment

### Docker

```bash
# Build
docker build -t cybersentinel .

# Run
docker run -p 8501:8501 cybersentinel
```

Access at **http://localhost:8501**

### Docker Compose (Optional)

```yaml
version: '3.8'
services:
  cybersentinel:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ../:/app/data   # Mount dataset directory
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Streamlit Cloud

1. Push to GitHub
2. Connect at [share.streamlit.io](https://share.streamlit.io)
3. Set main file path to `IntrusionDetectionDashboard/app.py`
4. Upload dataset files or mount from cloud storage

---

## ⚙️ Configuration

All settings are centralized in `config.py`:

| Parameter | Default | Description |
|:---|:---|:---|
| `SAMPLE_SIZE` | 100,000 | Default training sample size |
| `MAX_SAMPLE_SIZE` | 500,000 | Maximum allowed via UI slider |
| `SHAP_SAMPLE_SIZE` | 100 | Samples for SHAP computation |
| `RAM_WARNING_THRESHOLD` | 80% | RAM usage warning trigger |
| `TEST_SIZE` | 0.3 | Train/test split ratio |
| `RANDOM_STATE` | 42 | Reproducibility seed |

---

## 📊 Performance

Typical benchmark results on NF-UQ-NIDS dataset (100K sample):

| Model | Accuracy | F1 Score | ROC-AUC | Train Time |
|:---|:---:|:---:|:---:|:---:|
| Random Forest | ~97% | ~96% | ~99% | ~5s |
| XGBoost | ~96% | ~95% | ~98% | ~8s |
| Decision Tree | ~95% | ~94% | ~93% | ~1s |
| MLP | ~93% | ~92% | ~97% | ~30s |
| Gaussian NB | ~65% | ~60% | ~85% | <1s |

> **Note:** Results vary based on sample size, class distribution, and hardware.

---

## 🛠️ Tech Stack

| Layer | Technology |
|:---|:---|
| **Frontend** | Streamlit, Plotly, Custom CSS |
| **ML Framework** | scikit-learn, XGBoost |
| **XAI** | SHAP, Feature Importance |
| **Data** | Pandas, NumPy |
| **System** | psutil (RAM/CPU monitoring) |
| **Persistence** | joblib (model serialization) |
| **Container** | Docker (Python 3.9-slim) |
| **Logging** | Python logging module |

---

## 🧪 Development

### Running Locally

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Launch with auto-reload
streamlit run app.py
```

### Project Standards

- **State Management:** Single source of truth via `st.session_state`
- **No Global Variables:** All state is scoped to the session
- **Memory Safety:** Rolling windows (max 30), log trimming (max 50), configurable sampling
- **No Blocking Loops:** `st.rerun()` for controlled re-execution
- **Auto-Reset:** Model switching resets simulation state to prevent contamination

---

## 📁 Dataset

This project uses the **NF-UQ-NIDS** (NetFlow University of Queensland Network Intrusion Detection System) dataset.

- **Source:** [UQ NIDS Datasets](https://staff.itee.uq.edu.au/marius/NIDS_datasets/)
- **Format:** CSV with NetFlow v9 features
- **Size:** ~1.6M records across 4 files
- **Classes:** Normal traffic + multiple attack categories (DoS, DDoS, Reconnaissance, etc.)

Place all 4 CSV files in the parent directory of the dashboard folder.

---

## 🤝 Contributing

Contributions are welcome. Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Areas for Contribution

- [ ] Deep learning models (CNN, LSTM on raw packet data)
- [ ] Real network interface capture (pcap integration)
- [ ] Multi-user authentication
- [ ] Alerting system (email/Slack notifications)
- [ ] Historical analysis and trend reporting
- [ ] MITRE ATT&CK framework mapping

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with 🛡️ by Security Engineers, for Security Engineers**

CyberSentinel v3.0 Enterprise · Multi-Model SOC · © 2026

</div>
