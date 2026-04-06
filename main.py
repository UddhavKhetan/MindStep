import pandas as pd
from simulation.environment import IntersectionEnv

def run_simulation(duration=60.0, dt=0.1, use_smart_controller=False):
    controller_name = "Smart" if use_smart_controller else "Fixed-time"
    print(f"Starting simulation for {duration} seconds (dt={dt}s) with {controller_name} controller...")
    
    env = IntersectionEnv(use_smart_controller=use_smart_controller)
    
    total_steps = int(duration / dt)
    last_log_time = 0.0

    for step in range(total_steps):
        env.step(dt)
        
        if env.time - last_log_time >= 10.0:
            active_peds = len(env.pedestrians)
            active_vehs = len(env.vehicles)
            phase = env.controller.current_phase
            print(f"[Time: {env.time:04.1f}s] Phase: {phase} | "
                  f"Active Peds: {active_peds} | Active Vehs: {active_vehs}")
            last_log_time = env.time

    total_peds_spawned = env._next_ped_id - 1
    total_vehs_spawned = env._next_veh_id - 1
    
    print("\n=== Simulation Complete ===")
    print(f"Controller type      : {controller_name}")
    print(f"Total simulated time : {env.time:.1f} s")
    print(f"Total pedestrians    : {total_peds_spawned}")
    print(f"Total vehicles       : {total_vehs_spawned}")
    print(f"Near-miss events     : {env.near_misses}")
    
    if use_smart_controller:
        print(f"SmartController      : Pedestrian phase extensions count = {env.controller.extensions_count}")
        print(f"SmartController      : Total extra green time = {env.controller.total_extension_time:.1f} s")
        print(f"SmartController      : High-intent events = {env.controller.high_intent_events}")
        print(f"SmartController      : Extension opportunities = {env.controller.extension_opportunities}")

    # Write prediction log to disk for analysis
    if env.intent_log:
        df = pd.DataFrame(env.intent_log)
        filename = f"intent_predictions_{controller_name.lower()}.csv"
        df.to_csv(filename, index=False)
        print(f"\nSaved {len(env.intent_log)} online prediction records to {filename}")

if __name__ == "__main__":
    # Run Baseline
    run_simulation(duration=120.0, dt=0.1, use_smart_controller=False)
    
    print("\n" + "="*40 + "\n")
    
    # Run ML-Aware
    run_simulation(duration=120.0, dt=0.1, use_smart_controller=True)