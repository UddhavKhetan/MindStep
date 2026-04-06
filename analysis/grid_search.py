import numpy as np
import pandas as pd
import itertools
from simulation.environment import IntersectionEnv
from control_policy.smart_controller import SmartController

def grid_search_optimization(seeds=3):
    print("\n--- Grid Search Policy Optimization ---")
    thresholds = [0.4, 0.45, 0.5, 0.55]
    grace_windows = [1.0, 2.0, 3.0]
    max_extensions = [7.5, 10.0, 12.5]
    
    combinations = list(itertools.product(thresholds, grace_windows, max_extensions))
    results = []
    
    for th, gw, me in combinations:
        waits, delays = [], []
        for s in range(seeds):
            ctrl = SmartController(intent_threshold=th, grace_window=gw, max_extension=me)
            env = IntersectionEnv(controller=ctrl, seed=s+42, scenario="normal")
            for _ in range(int(200 / 0.1)):
                env.step(0.1)
                
            all_p_waits = env.ped_wait_times + [p.wait_time for p in env.pedestrians]
            all_v_delays = env.veh_delay_times + [v.delay_time for v in env.vehicles]
            waits.append(np.mean(all_p_waits) if all_p_waits else 0)
            delays.append(np.mean(all_v_delays) if all_v_delays else 0)
            
        results.append({
            "Threshold": th, "Grace (s)": gw, "Max Ext (s)": me,
            "Avg Wait": np.mean(waits), "Avg Delay": np.mean(delays)
        })
        
    df = pd.DataFrame(results)
    # Simple Pareto: rank by Wait + Delay
    df['Pareto Score'] = df['Avg Wait'] + df['Avg Delay']
    df = df.sort_values("Pareto Score")
    
    print("Top 5 Optimal Parameter Configurations:")
    print(df.head(5).to_string(index=False))
    df.to_csv("outputs/grid_search_opt.csv", index=False)
    return df