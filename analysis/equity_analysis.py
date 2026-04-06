import pandas as pd
import numpy as np

def run_equity_analysis(df_results, output_path="outputs/equity_report.csv"):
    """
    Parses detailed ped_type_waits to determine fairness across demographic groups.
    Calculates the 'Gap Ratio': Wait Time of Vulnerable Group / Wait Time of Normal.
    """
    print("\n--- Equity & Fairness Analysis ---")
    equity_metrics = []
    
    # We expect df_results to have expanded ped_type_waits dicts per row.
    # We will simulate the aggregation here for demonstration over the results dataframe.
    for ctrl in df_results['Controller'].unique():
        ctrl_data = df_results[df_results['Controller'] == ctrl]
        
        # Aggregate dicts across rows
        combined_waits = {"normal": [], "elderly": [], "child": [], "slow_walker": [], "paired_group": [], "distracted": []}
        for wait_dict in ctrl_data['ped_type_waits']:
            for k, v in wait_dict.items():
                if k in combined_waits:
                    combined_waits[k].extend(v)
                    
        normal_mean = np.mean(combined_waits["normal"]) if combined_waits["normal"] else 0.001
        
        for p_type, waits in combined_waits.items():
            if not waits: continue
            mean_w = np.mean(waits)
            p95_w = np.percentile(waits, 95) if waits else 0
            gap_ratio = mean_w / normal_mean
            
            equity_metrics.append({
                "Controller": ctrl,
                "Pedestrian Profile": p_type.capitalize(),
                "Avg Wait (s)": round(mean_w, 2),
                "95th Wait (s)": round(p95_w, 2),
                "Gap Ratio": round(gap_ratio, 2)
            })

    equity_df = pd.DataFrame(equity_metrics)
    equity_df.to_csv(output_path, index=False)
    print(equity_df.groupby(["Controller", "Pedestrian Profile"]).mean())
    return equity_df