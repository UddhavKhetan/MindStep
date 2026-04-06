import nbformat as nbf

nb = nbf.v4.new_notebook()
nb['cells'] = [
    nbf.v4.new_markdown_cell("# Machine Learning–Driven Intention-Aware Smart Crosswalk\n## Simulation Study & Equity Analysis\n\nThis notebook reproduces the core findings of the paper, detailing the trade-offs between heuristic and uncertainty-propagated ML control policies."),
    nbf.v4.new_code_cell("import pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\n# Load Data\ndf_normal = pd.read_csv('outputs/mc_results_normal.csv')\ndf_equity = pd.read_csv('outputs/equity_report.csv')"),
    nbf.v4.new_markdown_cell("### 1. Performance Trade-offs\nComparing Vehicle Delay vs Pedestrian Wait Times across baselines."),
    nbf.v4.new_code_cell("means = df_normal.groupby('Controller').mean(numeric_only=True)\nsns.scatterplot(data=means, x='avg_delay', y='avg_wait', hue=means.index, s=200)\nplt.title('Delay vs Wait Time')\nplt.show()"),
    nbf.v4.new_markdown_cell("### 2. Equity & Fairness Analysis\nThe Gap Ratio compares vulnerable pedestrian wait times to normal pedestrians (Target = 1.0)."),
    nbf.v4.new_code_cell("pivot = df_equity.pivot(index='Controller', columns='Pedestrian Profile', values='Gap Ratio')\nsns.heatmap(pivot, annot=True, cmap='coolwarm', center=1.0)\nplt.title('Equity Gap Ratio by Profile')\nplt.show()")
]
with open('SmartCrosswalk_Paper.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Notebook generated: SmartCrosswalk_Paper.ipynb")