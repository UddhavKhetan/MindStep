import os
import pandas as pd
import numpy as np  # <-- Added missing import
from run_all_experiments import run_mc
from analysis.equity_analysis import run_equity_analysis
from analysis.grid_search import grid_search_optimization
from ml_intention.diagnostics import generate_ml_diagnostics
from control_policy.base_controller import TrafficController
from control_policy.smart_controller import SmartController
from control_policy.baselines import HeuristicPedestrianController, PushButtonController

def extract_metrics_enhanced(env):
    """Updated to include ped_type_waits and TTC proxies."""
    all_p_waits = env.ped_wait_times + [p.wait_time for p in env.pedestrians]
    all_v_delays = env.veh_delay_times + [v.delay_time for v in env.vehicles]
    
    return {
        "avg_wait": np.mean(all_p_waits) if all_p_waits else 0,
        "p95_wait": np.percentile(all_p_waits, 95) if all_p_waits else 0,
        "avg_delay": np.mean(all_v_delays) if all_v_delays else 0,
        "p95_delay": np.percentile(all_v_delays, 95) if all_v_delays else 0,
        "near_misses": env.near_misses,
        "severe_conflicts": getattr(env, 'severe_conflicts', 0),
        "min_ttc_observed": getattr(env, 'min_ttc_observed', float('inf')),
        "extensions": getattr(env.controller, 'extensions_count', 0),
        "ped_type_waits": env.ped_type_waits # Pass through for equity analysis
    }

# Inject the enhanced metric extractor into the existing run_mc scope
import run_all_experiments
run_all_experiments.extract_metrics = extract_metrics_enhanced

if __name__ == "__main__":
    print("==================================================")
    print(" SMART CROSSWALK - PUBLICATION PIPELINE")
    print("==================================================")
    os.makedirs("outputs", exist_ok=True)
    
    # 1. ML Diagnostics
    generate_ml_diagnostics()
    
    # 2. Define Controller Array
    ctrls = {
        "FixedTime": lambda: TrafficController(),
        "PushButton": lambda: PushButtonController(),
        "Heuristic": lambda: HeuristicPedestrianController(),
        "Smart (Full)": lambda: SmartController(),
        "Smart (Risk-Averse)": lambda: SmartController(uncertainty_penalty_weight=0.8),
        "Smart (NoSpatial)": lambda: SmartController(ablate_spatial=True)
    }

    # 3. Monte Carlo & Equity Extraction
    df_norm = run_all_experiments.run_mc(ctrls, num_seeds=10, scenario="normal")
    df_heavy = run_all_experiments.run_mc(ctrls, num_seeds=10, scenario="heavy_traffic")
    
    run_equity_analysis(pd.concat([df_norm, df_heavy]))
    
    # 4. Grid Search Optimization
    grid_search_optimization(seeds=2)
    
    print("\n✅ PIPELINE COMPLETE. Launching Streamlit Demo...")
    os.system("streamlit run demo_app.py")