import pandas as pd
from simulation.environment import IntersectionEnv

def extract_metrics(env):
    """Safely extracts average metrics taking into account uncleared agents."""
    peds = env.cleared_peds + len(env.pedestrians)
    vehs = env.cleared_vehs + len(env.vehicles)
    
    total_wait = env.total_ped_wait_time + sum(p.wait_time for p in env.pedestrians)
    total_delay = env.total_veh_delay_time + sum(v.delay_time for v in env.vehicles)
    
    avg_wait = total_wait / peds if peds > 0 else 0
    avg_delay = total_delay / vehs if vehs > 0 else 0

    return {
        "time": env.time,
        "peds": peds,
        "vehs": vehs,
        "near_misses": env.near_misses,
        "avg_wait": avg_wait,
        "avg_delay": avg_delay
    }

def run_simulation(duration=300.0, dt=0.1, use_smart=False, seed=42):
    env = IntersectionEnv(use_smart_controller=use_smart, seed=seed)
    total_steps = int(duration / dt)
    
    for _ in range(total_steps):
        env.step(dt)

    metrics = extract_metrics(env)
    
    if use_smart:
        metrics.update({
            "extensions": env.controller.extensions_count,
            "extra_green": env.controller.total_extension_time,
            "high_intent": env.controller.high_intent_events,
            "opportunities": env.controller.extension_opportunities
        })
        
        if env.intent_log:
            df = pd.DataFrame(env.intent_log)
            df.to_csv("intent_predictions_smart.csv", index=False)
    else:
        if env.intent_log:
            df = pd.DataFrame(env.intent_log)
            df.to_csv("intent_predictions_fixed.csv", index=False)

    return metrics

if __name__ == "__main__":
    duration = 300.0
    seed = 42
    
    print(f"Running Fixed-Time Baseline (Seed={seed})...")
    base_metrics = run_simulation(duration, use_smart=False, seed=seed)
    
    print(f"Running ML-Aware SmartController (Seed={seed})...")
    smart_metrics = run_simulation(duration, use_smart=True, seed=seed)
    
    print("\n" + "="*50)
    print("CONTROLLER COMPARISON REPORT")
    print("="*50)
    print(f"{'Metric':<25} | {'Fixed-Time':<10} | {'SmartController':<10}")
    print("-" * 50)
    print(f"{'Total Peds/Vehs':<25} | {base_metrics['peds']}/{base_metrics['vehs']:<10} | {smart_metrics['peds']}/{smart_metrics['vehs']:<10}")
    print(f"{'Near-Miss Events':<25} | {base_metrics['near_misses']:<10} | {smart_metrics['near_misses']:<10}")
    print(f"{'Avg Ped Wait Time (s)':<25} | {base_metrics['avg_wait']:<10.2f} | {smart_metrics['avg_wait']:<10.2f}")
    print(f"{'Avg Veh Delay Time (s)':<25} | {base_metrics['avg_delay']:<10.2f} | {smart_metrics['avg_delay']:<10.2f}")
    print("-" * 50)
    print(f"SmartController Specifics:")
    print(f"  Extensions triggered    : {smart_metrics['extensions']}")
    print(f"  Total extra green time  : {smart_metrics['extra_green']:.1f} s")
    print(f"  High-intent events      : {smart_metrics['high_intent']}")
    print(f"  Extension opportunities : {smart_metrics['opportunities']}")
    print("="*50)