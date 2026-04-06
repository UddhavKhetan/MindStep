import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Smart Crosswalk Lab", page_icon="🚦", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for polished KPI cards and dark theme tweaks
st.markdown("""
<style>
    .metric-card {
        background-color: #1E2127; border-radius: 8px; padding: 15px; border: 1px solid #333;
        text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-val { font-size: 24px; font-weight: bold; color: #4CAF50; }
    .metric-label { font-size: 14px; color: #A0AAB5; text-transform: uppercase; letter-spacing: 1px;}
    .stTabs [data-baseweb="tab-list"] { gap: 20px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; padding: 10px 20px; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(scenario):
    filename = "outputs/mc_results_normal.csv" if scenario == "Normal Traffic" else "outputs/mc_results_heavy_traffic.csv"
    if os.path.exists(filename):
        return pd.read_csv(filename)
    return None

@st.cache_data
def load_equity():
    if os.path.exists("outputs/equity_report.csv"):
        return pd.read_csv("outputs/equity_report.csv")
    return None

# --- SIDEBAR ---
st.sidebar.title("🚦 Configuration")
scenario = st.sidebar.radio("Select Scenario", ["Normal Traffic", "Heavy Traffic"])
df = load_data(scenario)
df_eq = load_equity()

st.sidebar.markdown("---")
st.sidebar.markdown("**Download Datasets**")
if df is not None:
    st.sidebar.download_button("📥 Download Simulation CSV", df.to_csv(index=False), file_name=f"{scenario.replace(' ', '_')}.csv", mime="text/csv")
if df_eq is not None:
    st.sidebar.download_button("📥 Download Equity CSV", df_eq.to_csv(index=False), file_name="equity_report.csv", mime="text/csv")

if df is None:
    st.error(f"Data for {scenario} not found. Please run the experiment pipeline first.")
    st.stop()

# --- HEADER & KPIs ---
st.title("Machine Learning–Driven Intention-Aware Crosswalk")
st.markdown("Interactive analysis of simulation outputs, performance trade-offs, and demographic equity.")

selected_controller = st.selectbox("Highlight Controller for KPIs", options=df['Controller'].unique(), index=list(df['Controller'].unique()).index("Smart (Full)") if "Smart (Full)" in df['Controller'].unique() else 0)

# Filter for KPI averages
kpi_data = df[df['Controller'] == selected_controller].mean(numeric_only=True)

col1, col2, col3, col4, col5 = st.columns(5)
col1.markdown(f"<div class='metric-card'><div class='metric-label'>Avg Ped Wait</div><div class='metric-val'>{kpi_data['avg_wait']:.2f} s</div></div>", unsafe_allow_html=True)
col2.markdown(f"<div class='metric-card'><div class='metric-label'>95th% Ped Wait</div><div class='metric-val'>{kpi_data['p95_wait']:.2f} s</div></div>", unsafe_allow_html=True)
col3.markdown(f"<div class='metric-card'><div class='metric-label'>Avg Veh Delay</div><div class='metric-val'>{kpi_data['avg_delay']:.2f} s</div></div>", unsafe_allow_html=True)
col4.markdown(f"<div class='metric-card'><div class='metric-label'>Near Misses</div><div class='metric-val'>{kpi_data['near_misses']:.1f}</div></div>", unsafe_allow_html=True)
col5.markdown(f"<div class='metric-card'><div class='metric-label'>Extensions</div><div class='metric-val'>{kpi_data['extensions']:.1f}</div></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Performance Trade-Offs", "⚖️ Equity & Fairness", "🛡️ Advanced Safety"])

with tab1:
    st.subheader("Pareto Analysis: Delay vs Wait Time")
    st.markdown("Ideal controllers push towards the bottom-left quadrant (low wait, low delay).")
    
    means = df.groupby('Controller').mean(numeric_only=True).reset_index()
    fig = px.scatter(
        means, x='avg_delay', y='avg_wait', color='Controller', size=[20]*len(means),
        hover_data=['near_misses', 'extensions'],
        labels={"avg_delay": "Avg Vehicle Delay (s)", "avg_wait": "Avg Pedestrian Wait (s)"},
        template="plotly_dark", height=500
    )
    fig.update_traces(marker=dict(line=dict(width=2, color='white')))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Demographic Gap Ratio Heatmap")
    st.markdown("Displays the ratio of wait times for vulnerable groups compared to normal walkers. **Target = 1.0** (perfect equity).")
    if df_eq is not None:
        pivot_eq = df_eq.pivot(index="Controller", columns="Pedestrian Profile", values="Gap Ratio")
        fig_heat = px.imshow(
            pivot_eq, text_auto=".2f", color_continuous_scale="RdBu_r", 
            color_continuous_midpoint=1.0, aspect="auto", template="plotly_dark", height=500
        )
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.warning("Equity data not generated yet.")

with tab3:
    st.subheader("Safety Proxies")
    st.markdown("Advanced traffic engineering metrics capturing conflict severity.")
    cols = st.columns(2)
    fig_severe = px.bar(
        means, x='Controller', y='severe_conflicts', color='Controller',
        title="Avg Severe Conflicts (Time-To-Collision < 1.5s)", template="plotly_dark"
    )
    cols[0].plotly_chart(fig_severe, use_container_width=True)
    
    fig_ttc = px.bar(
        means, x='Controller', y='min_ttc_observed', color='Controller',
        title="Minimum Time-To-Collision Observed (Higher is Safer)", template="plotly_dark"
    )
    cols[1].plotly_chart(fig_ttc, use_container_width=True)