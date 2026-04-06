import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import stats
import seaborn as sns
from adjustText import adjust_text

from simulation.environment import IntersectionEnv
from control_policy.base_controller import TrafficController
from control_policy.smart_controller import SmartController
from control_policy.baselines import HeuristicPedestrianController, PushButtonController
from ml_intention.diagnostics import generate_ml_diagnostics

os.makedirs("outputs", exist_ok=True)

def extract_metrics(env):
    all_p_waits = env.ped_wait_times + [p.wait_time for p in env.pedestrians]
    all_v_delays = env.veh_delay_times + [v.delay_time for v in env.vehicles]
    
    return {
        "avg_wait": np.mean(all_p_waits) if all_p_waits else 0,
        "p95_wait": np.percentile(all_p_waits, 95) if all_p_waits else 0,
        "avg_delay": np.mean(all_v_delays) if all_v_delays else 0,
        "p95_delay": np.percentile(all_v_delays, 95) if all_v_delays else 0,
        "near_misses": env.near_misses,
        "extensions": getattr(env.controller, 'extensions_count', 0)
    }

def run_mc(controllers_dict, num_seeds=10, scenario="normal"):
    results = []
    print(f"\n--- Monte Carlo: {scenario.upper()} ({num_seeds} seeds) ---")
    
    for name, ctrl_class in controllers_dict.items():
        for seed in range(num_seeds):
            env = IntersectionEnv(controller=ctrl_class(), seed=seed+42, scenario=scenario)
            for _ in range(int(300 / 0.1)):
                env.step(0.1)
            
            m = extract_metrics(env)
            m['Controller'] = name
            results.append(m)
            
    df = pd.DataFrame(results)
    
    # Statistical Summary
    summary = df.groupby('Controller').agg({
        'avg_wait': ['mean', 'std'], 'p95_wait': ['mean'],
        'avg_delay': ['mean', 'std'], 'p95_delay': ['mean'],
        'near_misses': ['mean'], 'extensions': ['mean']
    }).round(2)
    print(summary)
    
    # T-Test (Smart vs Fixed)
    smart_waits = df[df['Controller']=='Smart']['avg_wait']
    fixed_waits = df[df['Controller']=='FixedTime']['avg_wait']
    if len(smart_waits) > 0 and len(fixed_waits) > 0:
        t_stat, p_val = stats.ttest_ind(smart_waits, fixed_waits)
        print(f"\nStatistical Significance (Smart vs Fixed Wait Time): p-value = {p_val:.4f}")
    
    df.to_csv(f"outputs/mc_results_{scenario}.csv", index=False)
    return df

def plot_tradeoffs(df, filename="outputs/tradeoff_plot.png"):
    plt.figure(figsize=(10, 7))
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    means = df.groupby('Controller').mean(numeric_only=True)
    
    # Plot points
    ax = sns.scatterplot(
        x=means['avg_delay'], 
        y=means['avg_wait'], 
        hue=means.index, 
        s=250, 
        edgecolor='black', 
        linewidth=1.5,
        palette="tab10",
        legend=False
    )
    
    # Create smart repelling labels
    texts = []
    for idx, row in means.iterrows():
        texts.append(plt.text(row['avg_delay'], row['avg_wait'], idx, 
                              fontdict={'weight': 'bold', 'size': 10}))
    
    # adjust_text prevents overlapping and draws arrows
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', lw=1.5),
                expand_points=(1.5, 1.5), expand_text=(1.5, 1.5))
        
    plt.title('Performance Trade-off: Vehicle Delay vs Pedestrian Wait', pad=20, weight='bold')
    plt.xlabel('Average Vehicle Delay (seconds)', weight='bold')
    plt.ylabel('Average Pedestrian Wait (seconds)', weight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Initiating Smart Crosswalk Evaluation Pipeline...")
    
    # 1. ML Diagnostics
    generate_ml_diagnostics()

    # 2. Define Controllers to test
    ctrls = {
        "FixedTime": lambda: TrafficController(),
        "PushButton": lambda: PushButtonController(),
        "Heuristic": lambda: HeuristicPedestrianController(),
        "Smart": lambda: SmartController(),
        "Smart (NoSpatial)": lambda: SmartController(ablate_spatial=True)
    }

    # 3. Run Experiments
    df_normal = run_mc(ctrls, num_seeds=15, scenario="normal")
    df_heavy = run_mc(ctrls, num_seeds=15, scenario="heavy_traffic")

    # 4. Generate Plots
    plot_tradeoffs(df_normal, "outputs/tradeoff_normal.png")
    plot_tradeoffs(df_heavy, "outputs/tradeoff_heavy_traffic.png")
    
    print("\n✅ All experiments complete. Check the 'outputs/' folder for plots and CSVs.")