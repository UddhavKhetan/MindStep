
# 🚦 Machine Learning–Driven Intention-Aware Smart Crosswalk
**Simulation Study, Equity Analysis, and Policy Optimization**

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)

## 📖 Overview
This repository contains a complete, time-stepped 2D kinematic simulation and machine learning pipeline for evaluating an **Intention-Aware Smart Crosswalk**. 

Traditional traffic controllers rely on fixed timers or physical push-buttons, often leading to pedestrian starvation or unnecessary vehicle delays. This project implements a **SmartController** that utilizes simulated wearable IMU data to predict pedestrian crossing intent in real-time. By propagating ML uncertainty and combining it with strict kinematic safety rules (Dilemma Zone protection), the system achieves Pareto-optimal trade-offs between pedestrian wait times and vehicle flow.

### 🌟 Key Accomplishments
* **Optimized Traffic Flow:** Reduced average pedestrian wait times by ~60% compared to fixed-time baselines, while maintaining safe vehicle throughput.
* **Demographic Equity:** Modeled diverse pedestrian profiles (Elderly, Child, Distracted, Slow Walker). Achieved a near-perfect **Gap Ratio (1.0)**, ensuring vulnerable pedestrians receive the same quality of service as normal walkers without being penalized for slower speeds or erratic movement.
* **Crash-Proof Safety Constraints:** Implemented strict physical Dilemma Zone checks that override ML predictions if a vehicle physically cannot stop in time, dropping simulated near-misses (TTC < 1.5s) to near zero.
* **Uncertainty-Aware ML:** The controller calculates the Shannon Entropy of the Random Forest predictions and penalizes high-uncertainty events, preventing "phantom" extensions.

---

## 🏗️ Project Architecture

```text
smart_crosswalk/
├── analysis/                 # Evaluation scripts
│   ├── equity_analysis.py    # Demographic Gap Ratio calculations
│   └── grid_search.py        # Pareto-optimization of controller parameters
├── control_policy/           # Traffic Controllers
│   ├── base_controller.py    # Fixed-Time Baseline
│   ├── baselines.py          # Heuristic & Push-Button Baselines
│   └── smart_controller.py   # ML-Aware, Uncertainty-Propagated Controller
├── ml_intention/             # Machine Learning Pipeline
│   ├── features.py           # Sliding-window IMU feature extraction
│   ├── model.py              # Random Forest Classifier training
│   └── diagnostics.py        # ROC, AUC, and Classification Reports
├── sensors/
│   └── imu_generator.py      # Synthetic wearable IMU noise generation
├── simulation/               # Kinematic Physics Engine
│   ├── agents.py             # Pedestrian (with profiles) & Vehicle classes
│   └── environment.py        # Intersection geometry, scenarios, TTC/PET proxies
├── outputs/                  # Auto-generated plots, CSVs, and reports
├── main.py                   # Data generation script
├── run_full_pipeline.py      # Master execution script (Monte Carlo + T-Tests)
├── visualize_run.py          # Interactive Pygame 2D Simulator
├── demo_app.py               # Streamlit Analytics Dashboard
└── smart_crosswalk_dashboard.html # Standalone JS/HTML Browser Demo
```

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/smart_crosswalk.git](https://github.com/yourusername/smart_crosswalk.git)
   cd smart_crosswalk
   ```

2. **Create a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies:**
   ```bash
   pip install numpy pandas scikit-learn scipy matplotlib seaborn plotly pygame streamlit joblib nbformat
   ```

---

## 🚀 Usage & Execution Pipeline

The project is designed to be run sequentially from data generation to final presentation.

### 1. Data Generation & Model Training
First, generate the synthetic IMU dataset and train the Random Forest Intention Predictor.
```bash
# Generates `imu_intention_dataset.csv`
python main.py 

# Trains the model and outputs `intention_model.joblib`
python -m ml_intention.model
```

### 2. Live Simulation & Visualization (Pygame)
Launch the interactive 2D simulation to watch the trained controller in action. Features a heads-up display (HUD), agent click-inspection, and a live event log.
```bash
python visualize_run.py
```
* **Controls:** `[SPACE]` to Pause/Play, `[UP/DOWN]` to adjust sim speed, `[C]` to cycle controllers.

### 3. Full Research Evaluation (Monte Carlo)
Run the master pipeline to execute Monte Carlo simulations (multiple seeds) across Normal and Heavy Traffic scenarios. This script conducts Equity Analysis, Grid Search Optimization, and generates Matplotlib trade-off charts.
```bash
python run_full_pipeline.py
```
*Results, CSVs, and plots will be saved in the `/outputs/` directory.*

### 4. Interactive Analytics Dashboard (Streamlit)
*Note: Run Step 3 first to generate the necessary data.*
Explore the results of your Monte Carlo runs, view interactive Plotly trade-off scatter plots, and analyze the Demographic Equity Heatmap.
```bash
streamlit run demo_app.py
```

### 5. Web Dashboard (Standalone HTML)
For quick demonstrations without a Python backend, open the standalone HTML file in any modern web browser. It features a lightweight embedded kinematic engine mirroring the Python logic.
```bash
# Double-click in your file explorer, or run:
open smart_crosswalk_dashboard.html  # macOS
start smart_crosswalk_dashboard.html # Windows
```

---

## 🔬 Methodology Highlights

* **Predictive Features:** Mean, standard deviation, min, max, and energy of 6-axis IMU (Accelerometer + Gyroscope) over a sliding time window.
* **Safety Proxies:** Instead of binary collision checks, the environment calculates continuous **Time-To-Collision (TTC)** vectors, flagging events where TTC drops below 1.5s as severe conflicts.
* **Controller Fallback:** In heavy traffic, vehicle platoons can cause "Pedestrian Starvation" if the controller waits for a perfectly clear dilemma zone. The system utilizes an *Absolute Max-Green* fallback, ensuring pedestrians are eventually served even under peak loads.

---

## 🤝 Acknowledgments
* Developed as part of a Course-Based Design Project.
