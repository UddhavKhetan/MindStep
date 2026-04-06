import pandas as pd
import numpy as np
from simulation.environment import IntersectionEnv
from control_policy.heuristic_controller import HeuristicPedestrianController

def extract_metrics(env):
    peds = env.cleared_peds + len(env.pedestrians)
    vehs = env.cleared_vehs + len(env.vehicles)
    total_wait = env.total_ped_wait_time + sum(p.wait_time for p in env.pedestrians)
    total_delay = env.total_veh_delay_time + sum(v.delay_time for v in env.vehicles)
    
    return {
        "peds": peds,
        "vehs": vehs,
        "near_misses": env.near_misses,
        "avg_wait": total_wait / peds if peds > 0 else 0,
        "avg_delay": total_delay / vehs if vehs > 0 else 0,
        "extensions": getattr(env.controller, 'extensions_count', 0)
    }

def run_episode(duration, dt, ctrl_type, seed, intent_thresh=0.5):
    env = IntersectionEnv(use_smart_controller=(ctrl_type=="smart"), seed=seed)
    
    # Inject heuristic controller if requested
    if ctrl_type == "heuristic":
        env.controller = HeuristicPedestrianController()
    elif ctrl_type == "smart":
        env.controller.intent_threshold = intent_thresh
        
    for _ in range(int(duration / dt)):
        env.step(dt)
        
    return extract_metrics(env)

def run_monte_carlo(num_seeds=5, duration=300.0):
    print(f"\n--- Running Monte Carlo Evaluation ({num_seeds} seeds) ---")
    results = []
    
    for c_type in ["fixed", "heuristic", "smart"]:
        for seed in range(num_seeds):
            m = run_episode(duration, 0.1, c_type, seed=seed+42)
            m['controller'] = c_type
            results.append(m)
            
    df = pd.DataFrame(results)
    summary = df.groupby('controller').agg({
        'avg_wait': ['mean', 'std'],
        'avg_delay': ['mean', 'std'],
        'near_misses': ['mean'],
        'extensions': ['mean']
    }).round(2)
    print(summary)
    return df

def run_sensitivity(seeds=3):
    print(f"\n--- Sensitivity Analysis (SmartController) ---")
    thresholds = [0.4, 0.5, 0.6]
    for thresh in thresholds:
        waits = []
        for seed in range(seeds):
            m = run_episode(300.0, 0.1, "smart", seed=seed+42, intent_thresh=thresh)
            waits.append(m['avg_wait'])
        print(f"Threshold: {thresh} -> Avg Wait: {np.mean(waits):.2f}s (±{np.std(waits):.2f})")

if __name__ == "__main__":
    run_monte_carlo(num_seeds=10)
    run_sensitivity(seeds=5)
    print("\nEvaluations complete. Ready for publication.")