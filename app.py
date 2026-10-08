"""
app.py
Frontend Web Application: Green Fleet Dispatch & Maritime Decarbonization
Powered by Streamlit, QIRF Digital Twin, and Quantum-Inspired Optimization (QGA).
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from app_backend import (
    load_digital_twin,
    optimize_fleet_schedule,
    optimize_fleet_classical_milp,
    run_dynamic_benchmark,
    run_fuel_scenario_analysis,
    DEFAULT_VESSELS,
    DEFAULT_ROUTES,
    FUEL_SPECS
)

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Green Fleet Dispatch | Quantum-Inspired DSS",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 14px;
        border: 1px solid #e9ecef;
    }
    .status-optimal { color: #198754; font-weight: bold; }
    .status-slack { color: #dc3545; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# CACHED SURROGATE MODEL LOADER
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading QIRF Digital Twin & Preprocessor...")
def get_model():
    return load_digital_twin('qirf_fuel_predictor.joblib')

try:
    preprocessor, qirf_model, feature_cols = get_model()
except Exception as e:
    st.error(f"Failed to load model artifact: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# SIDEBAR: MULTI-OBJECTIVE WEIGHTS & HYPERPARAMETERS
# -----------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/cargo-ship.png", width=64)
st.sidebar.title("Operational Controls")

st.sidebar.subheader("Multi-Objective Policy")
w_energy = st.sidebar.slider("Energy Weight (w_energy)", 0.0, 1.0, 0.40, 0.05)
w_emissions = st.sidebar.slider("Emissions Weight (w_emissions)", 0.0, 1.0, 0.40, 0.05)
w_cost = st.sidebar.slider("Cost Weight (w_cost)", 0.0, 1.0, 0.20, 0.05)

# Normalise weights
total_w = w_energy + w_emissions + w_cost
if total_w > 0:
    w_norm = (w_energy / total_w, w_emissions / total_w, w_cost / total_w)
else:
    w_norm = (0.4, 0.4, 0.2)

st.sidebar.caption(f"Normalized Weights: **{w_norm[0]:.2f}** | **{w_norm[1]:.2f}** | **{w_norm[2]:.2f}**")

st.sidebar.markdown("---")
st.sidebar.subheader("Carbon Tax")
# Carbon Tax Slider
carbon_tax = st.sidebar.slider(
    "Simulated Carbon Tax / ETS Penalty ($ / metric ton CO2e)",
    min_value=0,
    max_value=300,
    value=100,
    step=10,
    help="Simulates EU ETS allowances or IMO carbon levy pricing."
)

st.sidebar.markdown("---")

st.sidebar.subheader("QGA Hyperparameters")
qga_generations = st.sidebar.slider("Generations", 5, 100, 40, 5)
qga_pop_size = st.sidebar.slider("Population Size", 5, 100, 30, 5)

# -----------------------------------------------------------------------------
# MAIN APP HEADER
# -----------------------------------------------------------------------------
st.title("⚓ Green Fleet Dispatch & Decarbonization DSS")
st.markdown(
    "**Decision Support System (Deliverable 4)** combining a **Quantum-Inspired Random Feature (QIRF) Digital Twin** "
    "with a **Quantum-Inspired Genetic Algorithm (QGA)** for multi-objective maritime voyage optimization."
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 1. Fleet & Mission Config",
    "🚀 2. Live QGA Dispatch",
    "⚖️ 3. Classical MILP Benchmark",
    "⚡ 4. Dynamic Dispatch (Non-Linear Co-Optimization)",
    "🌱 5. Alternative Fuel Pathways"
])

# -----------------------------------------------------------------------------
# TAB 1: FLEET & MISSION CONFIGURATION (FULLY EDITABLE & CSV READY)
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("1. Active Fleet Registry")
    st.caption("Edit ship attributes below, or add new vessels using the '+' at the bottom of the table:")

    # Initialize fleet registry in session state as a DataFrame if not present
    if 'vessel_data' not in st.session_state:
        init_vessels = []
        for ship_name, specs in DEFAULT_VESSELS.items():
            row = {'Ship_Type': ship_name}
            row.update(specs)
            init_vessels.append(row)
        st.session_state.vessel_data = pd.DataFrame(init_vessels)

    # Optional CSV uploader for Fleet
    fleet_file = st.file_uploader("Upload Fleet Registry CSV (optional)", type=['csv'], key='fleet_uploader')
    if fleet_file is not None:
        try:
            st.session_state.vessel_data = pd.read_csv(fleet_file)
            st.success("Fleet registry loaded from uploaded CSV!")
        except Exception as e:
            st.error(f"Error loading fleet CSV: {e}")

    # Editable fleet table
    edited_vessels = st.data_editor(
        st.session_state.vessel_data,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Ship_Type": st.column_config.SelectboxColumn("Ship Type", options=["Bulk Carrier", "Tanker", "Container Ship", "Fish Carrier", "General Cargo"], required=True),
            "Engine_Type": st.column_config.SelectboxColumn("Engine Type", options=["Two-Stroke", "Four-Stroke", "Gas Turbine"], required=True),
            "Maintenance_Status": st.column_config.SelectboxColumn("Status", options=["Good", "Fair", "Critical"], required=True),
            "Fuel_Type": st.column_config.SelectboxColumn("Fuel Type", options=list(FUEL_SPECS.keys()), required=True),
            "Draft_meters": st.column_config.NumberColumn("Draft (m)", min_value=1.0, max_value=30.0, step=0.1),
            "Max_Capacity": st.column_config.NumberColumn("Deadweight (tons)", min_value=500.0, step=1000.0)
        }
    )
    st.session_state.vessel_data = edited_vessels

    st.markdown("---")
    st.subheader("2. Mission Schedule Builder")
    st.caption("Define route requirements, payloads, and deadlines. Click '+' to insert a new voyage:")

    # Initialize voyage missions in session state if not present
    if 'voyage_data' not in st.session_state:
        st.session_state.voyage_data = pd.DataFrame(DEFAULT_ROUTES)

    # Optional CSV uploader for Missions
    voyage_file = st.file_uploader("Upload Voyage Demand CSV (optional)", type=['csv'], key='voyage_uploader')
    if voyage_file is not None:
        try:
            st.session_state.voyage_data = pd.read_csv(voyage_file)
            st.success("Voyage schedule loaded from uploaded CSV!")
        except Exception as e:
            st.error(f"Error loading voyage CSV: {e}")

    # Editable voyage table
    edited_voyages = st.data_editor(
        st.session_state.voyage_data,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "route_id": st.column_config.TextColumn("Route ID", required=True),
            "route_type": st.column_config.SelectboxColumn("Route Type", options=["Transoceanic", "Long-haul", "Coastal"], required=True),
            "distance_nm": st.column_config.NumberColumn("Distance (nm)", min_value=10.0, step=50.0),
            "cargo_weight_tons": st.column_config.NumberColumn("Cargo (tons)", min_value=10.0, step=500.0),
            "weather_condition": st.column_config.SelectboxColumn("Weather", options=["Calm", "Moderate", "Rough"], required=True),
            "max_arrival_hours": st.column_config.NumberColumn("Deadline (hrs)", min_value=1.0, step=1.0)
        }
    )
    st.session_state.voyage_data = edited_voyages
# -----------------------------------------------------------------------------
# TAB 2: LIVE QGA OPTIMIZATION & DISPATCH
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Automated Fleet Dispatch (QGA Engine)")
    
    if st.button("🚀 Run Quantum-Inspired Fleet Optimization", type="primary"):
        # Convert the edited fleet table into the required vessel_registry dictionary:
        # active_vessels_dict = {
        #     row['Ship_Type']: {
        #         'Engine_Type': row['Engine_Type'],
        #         'Maintenance_Status': row['Maintenance_Status'],
        #         'Fuel_Type': row['Fuel_Type'],
        #         'Draft_meters': float(row['Draft_meters']),
        #         'Max_Capacity': float(row['Max_Capacity'])
        #     }
        #     for _, row in st.session_state.vessel_data.iterrows()
        # }
        active_vessels_dict = {
            f"Vessel_{i+1:03d} ({row['Ship_Type']})": {
                'Ship_Type': row['Ship_Type'],
                'Engine_Type': row['Engine_Type'],
                'Maintenance_Status': row['Maintenance_Status'],
                'Fuel_Type': row['Fuel_Type'],
                'Draft_meters': float(row['Draft_meters']),
                'Max_Capacity': float(row['Max_Capacity'])
            }
            for i, row in st.session_state.vessel_data.reset_index(drop=True).iterrows()
        }
        routes_list = st.session_state.voyage_data.to_dict(orient='records')
        
        with st.spinner("Simulating Born-rule measurement & Ry rotation gates..."):
            dispatch_df, conv_hist, runtime = optimize_fleet_schedule(
                voyage_schedule=routes_list,
                vessel_registry=active_vessels_dict,
                preprocessor=preprocessor,
                qirf_model=qirf_model,
                feature_cols=feature_cols,
                weights=w_norm,
                n_generations=qga_generations,
                pop_size=qga_pop_size,
                carbon_tax_per_ton=carbon_tax
            )

            if 'Feasible' in dispatch_df.columns and not dispatch_df['Feasible'].all():
              infeasible_count = int((~dispatch_df['Feasible']).sum())
              st.warning(f"⚠️ {infeasible_count} route(s) exceeded their arrival deadlines with available vessels.")

            st.session_state.qga_dispatch = dispatch_df
            st.session_state.qga_hist = conv_hist
            st.session_state.qga_runtime = runtime
    # # In app.py under the dispatch table:

    if 'qga_dispatch' in st.session_state:
        df = st.session_state.qga_dispatch
        runtime = st.session_state.qga_runtime
        conv_hist = st.session_state.qga_hist

        # Display 5 distinct KPI cards
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Execution Time", f"{runtime:.3f} s")
        col2.metric("Fleet Energy", f"{df['Energy (kWh)'].sum():,.1f} kWh")
        col3.metric("Fleet Emissions", f"{df['Emissions (kg CO2e)'].sum() / 1000:,.2f} t CO2e")
        col4.metric("Bunker Fuel Cost", f"${df['Bunker Cost (USD)'].sum():,.2f}")
        col5.metric("Total Effective Cost", f"${df['Total Cost (USD)'].sum():,.2f}",  
                    delta=f"+${df['Carbon Penalty (USD)'].sum():,.2f} Tax" if df['Carbon Penalty (USD)'].sum() > 0 else None,
                    delta_color="inverse")

        st.markdown("#### Optimal Voyage Manifest")
        st.dataframe(df, use_container_width=True)

        # Plot QGA Convergence
        st.markdown("#### QGA Multi-Objective Convergence History")
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(range(1, len(conv_hist) + 1), conv_hist, marker='o', markersize=3, color='#0d6efd')
        ax.set_xlabel("Generation")
        ax.set_ylabel("Best Objective Cost")
        ax.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig)
        plt.close()

        # CSV Export Download (Deliverable 4 requirement)
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Dispatch Manifest (CSV)",
            data=csv_data,
            file_name="optimal_fleet_dispatch_manifest.csv",
            mime="text/csv"
        )
    else:
        st.info("Click 'Run Quantum-Inspired Fleet Optimization' to compute the optimal dispatch plan.")

# -----------------------------------------------------------------------------
# TAB 3: CLASSICAL BENCHMARK (MILP VS QGA)
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("Classical Deterministic Benchmark: HiGHS Branch-and-Cut MILP")
    st.markdown("Validating heuristic convergence against exact mathematical baseline.")

    if st.button("⚖️ Run Baseline & Compare"):
        # Convert the edited fleet table into the required vessel_registry dictionary:
        active_vessels_dict = {
            f"Vessel_{i+1:03d} ({row['Ship_Type']})": {
                'Ship_Type': row['Ship_Type'],
                'Engine_Type': row['Engine_Type'],
                'Maintenance_Status': row['Maintenance_Status'],
                'Fuel_Type': row['Fuel_Type'],
                'Draft_meters': float(row['Draft_meters']),
                'Max_Capacity': float(row['Max_Capacity'])
            }
            for i, row in st.session_state.vessel_data.reset_index(drop=True).iterrows()
        }
        routes_list = st.session_state.voyage_data.to_dict(orient='records')
        
        with st.spinner("Solving classical Mixed-Integer Linear Program..."):
            milp_df, milp_obj, milp_time = optimize_fleet_classical_milp(
                voyage_schedule=routes_list,
                vessel_registry=active_vessels_dict,
                preprocessor=preprocessor,
                qirf_model=qirf_model,
                feature_cols=feature_cols,
                weights=w_norm
            )
            st.session_state.milp_dispatch = milp_df
            st.session_state.milp_time = milp_time
            st.session_state.milp_obj = milp_obj

    if 'milp_dispatch' in st.session_state and 'qga_dispatch' in st.session_state:
        milp_df = st.session_state.milp_dispatch
        qga_df = st.session_state.qga_dispatch

        m_emissions = milp_df['Emissions (kg CO2e)'].sum()
        q_emissions = qga_df['Emissions (kg CO2e)'].sum()
        m_energy = milp_df['Energy (kWh)'].sum()
        q_energy = qga_df['Energy (kWh)'].sum()
        m_time = st.session_state.milp_time
        q_time = st.session_state.qga_runtime

        comp_df = pd.DataFrame({
            'Metric': ['Total Emissions (t CO2e)', 'Total Energy (MWh)', 'Execution Time (s)', 'Status'],
            'Classical MILP (HiGHS)': [f"{m_emissions/1000:.2f}", f"{m_energy/1000:.2f}", f"{m_time:.4f}", "Exact Optimum"],
            'Quantum-Inspired GA': [f"{q_emissions/1000:.2f}", f"{q_energy/1000:.2f}", f"{q_time:.4f}", "Pareto Heuristic"]
        })
        st.table(comp_df)

        # 3-Panel Comparison Figure
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        
        # Panel 1: Emissions
        axes[0].bar(['MILP', 'QGA'], [m_emissions/1000, q_emissions/1000], color=['#6c757d', '#198754'], width=0.5)
        axes[0].set_title("Emissions (t CO2e)")
        axes[0].set_ylabel("Metric Tons")
        
        # Panel 2: Energy
        axes[1].bar(['MILP', 'QGA'], [m_energy/1000, q_energy/1000], color=['#6c757d', '#0d6efd'], width=0.5)
        axes[1].set_title("Energy (MWh)")
        
        # Panel 3: Runtime
        axes[2].bar(['MILP', 'QGA'], [m_time, q_time], color=['#6c757d', '#ffc107'], width=0.5)
        axes[2].set_title("Runtime (seconds)")

        for ax in axes:
            ax.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.info("Run both the QGA in Tab 2 and the Baseline here to visualize the comparative benchmark.")
# -----------------------------------------------------------------------------
# Stress Test
# -----------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("⚡ Combinatorial Scaling & Stress Test")
    st.caption("Evaluate solver runtimes, solution gap, and memory stability across scaling problem dimensions.")

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        test_routes_n = st.slider("Number of Voyage Routes", min_value=5, max_value=50, value=12, step=1)
    with col_s2:
        test_vessels_n = st.slider("Number of Available Vessels", min_value=5, max_value=50, value=12, step=1)
    with col_s3:
        test_seed = st.number_input("RNG Seed", value=42, step=1)

    combinations_space = test_vessels_n ** test_routes_n
    st.info(f"📐 Theoretical Unconstrained Assignment Space: **~{combinations_space:.2e}** combinations")

    if st.button("🔥 Execute Combinatorial Benchmark", type="secondary"):
        from app_backend import generate_stress_test_data
        
        with st.spinner("Generating scenario & benchmarking solvers..."):
            stress_routes, stress_fleet = generate_stress_test_data(
                n_routes=test_routes_n, 
                n_vessels=test_vessels_n, 
                seed=test_seed
            )

            # 1. Classical MILP Run
            milp_stress_df, milp_stress_obj, milp_stress_t = optimize_fleet_classical_milp(
                voyage_schedule=stress_routes,
                vessel_registry=stress_fleet,
                preprocessor=preprocessor,
                qirf_model=qirf_model,
                feature_cols=feature_cols,
                weights=w_norm,
                carbon_tax_per_ton=carbon_tax if 'carbon_tax' in locals() else 0.0
            )

            # 2. QGA Engine Run
            qga_stress_df, qga_stress_hist, qga_stress_t = optimize_fleet_schedule(
                voyage_schedule=stress_routes,
                vessel_registry=stress_fleet,
                preprocessor=preprocessor,
                qirf_model=qirf_model,
                feature_cols=feature_cols,
                weights=w_norm,
                n_generations=qga_generations,
                pop_size=qga_pop_size,
                carbon_tax_per_ton=carbon_tax if 'carbon_tax' in locals() else 0.0
            )

            # Store in session state
            st.session_state.stress_results = {
                'routes_n': test_routes_n,
                'vessels_n': test_vessels_n,
                'milp_t': milp_stress_t,
                'qga_t': qga_stress_t,
                'milp_em': milp_stress_df['Emissions (kg CO2e)'].sum() / 1000.0,
                'qga_em': qga_stress_df['Emissions (kg CO2e)'].sum() / 1000.0,
                'milp_cost': milp_stress_df['Total Cost (USD)'].sum() if 'Total Operating Cost (USD)' in milp_stress_df.columns else milp_stress_df['Bunker Cost (USD)'].sum(),
                'qga_cost': qga_stress_df['Total Cost (USD)'].sum() if 'Total Operating Cost (USD)' in qga_stress_df.columns else qga_stress_df['Bunker Cost (USD)'].sum(),
                'qga_hist': qga_stress_hist
            }

    if 'stress_results' in st.session_state:
        sr = st.session_state.stress_results
        
        # Display Comparative KPI Cards
        st.markdown(f"#### Benchmark Results ({sr['routes_n']} Routes × {sr['vessels_n']} Vessels)")
        scol1, scol2, scol3, scol4 = st.columns(4)
        scol1.metric("MILP Runtime", f"{sr['milp_t']:.3f} s")
        scol2.metric("QGA Runtime", f"{sr['qga_t']:.3f} s")
        
        gap = ((sr['qga_cost'] - sr['milp_cost']) / sr['milp_cost']) * 100 if sr['milp_cost'] > 0 else 0
        scol3.metric("MILP Total Cost", f"${sr['milp_cost']:,.0f}")
        scol4.metric("QGA Total Cost", f"${sr['qga_cost']:,.0f}", delta=f"{gap:+.2f}% vs MILP", delta_color="inverse")

        # Convergence of QGA on the complex problem
        fig_s, ax_s = plt.subplots(figsize=(8, 3))
        ax_s.plot(range(1, len(sr['qga_hist']) + 1), sr['qga_hist'], color='#0d6efd', lw=2, marker='.', markersize=4)
        ax_s.set_title(f"QGA Search Dynamics Across {combinations_space:.1e} Combinations")
        ax_s.set_xlabel("Generation")
        ax_s.set_ylabel("Fitness (Weighted Objective)")
        ax_s.grid(True, linestyle="--", alpha=0.5)
        # Dynamic zoom to highlight subtle generation deltas
        hist = sr['qga_hist']
        if len(hist) > 0:
            y_min = min(hist)
            y_max = max(hist)
            margin = max((y_max - y_min) * 0.15, 100.0)
            ax_s.set_ylim(y_min - margin, y_max + margin)

        ax_s.grid(True, linestyle="--", alpha=0.5)

        st.pyplot(fig_s)
        plt.close()
        
# -----------------------------------------------------------------------------
# Tab 4: DYNAMIC NON-LINEAR BENCHMARK (CONTINUOUS SPEED & BERTH SCHEDULING)
# -----------------------------------------------------------------------------
with tab4:
    st.markdown("### ⚡ Dynamic Non-Linear Benchmark (Continuous Speed & Berth Scheduling)")
    st.caption("Here, speed is chosen jointly across 5 tiers and berth windows are evaluated on the fly. This breaks Total Unimodularity and forces MILP to branch on $(N \times S \times K)$ binary variables.")
    col_d1, col_d2 = st.columns(2)
    dyn_routes_n = col_d1.slider("Dynamic Benchmark Routes", 10,100, 30, step=5)
    dyn_ships_n = col_d2.slider("Dynamic Benchmark Vessels", 10, 100, 30, step=5)
    
    if st.button("🚀 Run Dynamic Non-Linear Benchmark"):
        # 1. Safely resolve vessel_registry and available ships from session_state or globals
        registry = st.session_state.get('vessel_registry', globals().get('vessel_registry', {}))
        all_ships = list(registry.keys())
        
        # Fallback if registry is empty
        if not all_ships:
            all_ships = [f"VESSEL_{i:03d}" for i in range(dyn_ships_n)]
            registry = {s: {'Base_Fuel_Rate': 0.045 + (i % 5) * 0.005} for i, s in enumerate(all_ships)}
            
        dyn_ships = all_ships[:dyn_ships_n]

        # 2. Build or slice synthetic route schedule with explicit column keys
        np.random.seed(42)
        dyn_schedule = []
        for r_id in range(dyn_routes_n):
            dist = float(np.random.uniform(1200, 4500))
            # Required deadline so speeds between 12-18 knots can make it
            deadline = float(dist / np.random.uniform(13.0, 16.5))
            dyn_schedule.append({
                'distance_nm': dist,
                'max_arrival_hours': deadline
            })

        # 3. Execute solver comparison
        with st.spinner("Executing non-linear solver comparison..."):
            m_t, m_cost, q_t, q_cost, hist, num_vars = run_dynamic_benchmark(
                dyn_schedule, dyn_ships, registry
            )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Decision Variables (MILP)", f"{num_vars:,}")
            c2.metric("MILP Solve Time", f"{m_t:.3f} s")
            c3.metric("QGA Solve Time", f"{q_t:.3f} s", delta=f"{(m_t - q_t):+.2f}s vs MILP", delta_color="normal")

            gap = ((q_cost - m_cost) / m_cost) * 100 if m_cost > 0 else 0
            c4.metric("QGA Cost vs MILP", f"${q_cost:,.0f}", delta=f"{gap:+.2f}%")

            # Dynamic Convergence Plot
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(range(1, len(hist) + 1), hist, color='#00d26a', lw=2, marker='.')
            ax.set_title("QGA Convergence on Non-Linear Dynamic Surface")
            ax.set_xlabel("Generation")
            ax.set_ylabel("Fitness (Fuel + Demurrage)")
            ax.grid(True, linestyle="--", alpha=0.5)

            y_min, y_max = min(hist), max(hist)
            margin = max((y_max - y_min) * 0.15, 100.0)
            ax.set_ylim(y_min - margin, y_max + margin)

            st.pyplot(fig)
            plt.close()

# -----------------------------------------------------------------------------
# TAB 5: ALTERNATIVE FUEL PATHWAYS & SCENARIO ANALYSIS
# -----------------------------------------------------------------------------
with tab5:
    st.subheader("Maritime Fuel Transition & IMO/EU ETS Scenario Simulator")
    st.markdown("Simulate fleet-wide lifecycle emissions and economic trade-offs across clean fuel pathways.")

    if st.button("🌱 Run Alternative Fuel Scenario Analysis"):
        # Convert the edited fleet table into the required vessel_registry dictionary:
        active_vessels_dict = {
            f"Vessel_{i+1:03d} ({row['Ship_Type']})": {
                'Ship_Type': row['Ship_Type'],
                'Engine_Type': row['Engine_Type'],
                'Maintenance_Status': row['Maintenance_Status'],
                'Fuel_Type': row['Fuel_Type'],
                'Draft_meters': float(row['Draft_meters']),
                'Max_Capacity': float(row['Max_Capacity'])
            }
            for i, row in st.session_state.vessel_data.reset_index(drop=True).iterrows()
        }        
        routes_list = st.session_state.voyage_data.to_dict(orient='records')
        
        with st.spinner("Evaluating multi-fuel lifecycle scenarios..."):
            scenario_df = run_fuel_scenario_analysis(
                voyage_schedule=routes_list,
                vessel_registry=active_vessels_dict ,
                preprocessor=preprocessor,
                qirf_model=qirf_model,
                feature_cols=feature_cols
            )
            st.session_state.scenario_summary = scenario_df

    if 'scenario_summary' in st.session_state:
        sdf = st.session_state.scenario_summary.copy()

        # -------------------------------------------------------------
        # 1. ORIGINAL ABATEMENT BAR CHART (PRESERVED)
        # -------------------------------------------------------------
        st.dataframe(sdf, use_container_width=True)

        fig, ax1 = plt.subplots(figsize=(9, 3.8))
        colors = ['#dc3545', '#ffc107', '#0dcaf0', '#198754', '#0d6efd']
        bars = ax1.bar(sdf['Fuel Scenario'], sdf['Emissions Abatement vs HFO (%)'], color=colors, width=0.5)
        ax1.set_ylabel("Emissions Abatement vs HFO (%)")
        ax1.set_ylim(0, 100)
        ax1.grid(True, linestyle="--", alpha=0.4)

        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2.0, height + 1.5, f"{height:.1f}%", ha='center', va='bottom', fontsize=9)

        st.pyplot(fig)
        plt.close()
        
        st.success("Analysis complete. Clean fuels (Ammonia/Hydrogen) deliver up to 90%+ lifecycle abatement.")

        # -------------------------------------------------------------
        # 2. COST VS EMISSIONS TRADE-OFF & CARBON TAX COMPARISON
        # -------------------------------------------------------------
        st.markdown("---")
        st.subheader("📊 Optimal Fuel Trade-Off: Cost vs. Emissions Tax")

        hfo_cost = sdf.loc[sdf['Fuel Scenario'] == 'Heavy Fuel Oil (HFO)', 'Total Operating Cost (USD)'].values[0]
        hfo_emissions = sdf.loc[sdf['Fuel Scenario'] == 'Heavy Fuel Oil (HFO)', 'Total Emissions (kg CO2e)'].values[0]

        sdf['Emissions (t CO2e)'] = sdf['Total Emissions (kg CO2e)'] / 1000.0
        sdf['Carbon Tax Liability ($)'] = sdf['Emissions (t CO2e)'] * carbon_tax
        sdf['Effective Total Cost ($)'] = sdf['Total Operating Cost (USD)'] + sdf['Carbon Tax Liability ($)']

        # Marginal Abatement Cost ($ / t CO2e abated relative to baseline HFO)
        tonnes_abated = (hfo_emissions - sdf['Total Emissions (kg CO2e)']) / 1000.0
        cost_diff = sdf['Total Operating Cost (USD)'] - hfo_cost

        sdf['Marginal Abatement Cost ($/t CO2e)'] = np.where(
            tonnes_abated > 0.1,
            cost_diff / tonnes_abated,
            0.0
        )

        best_fuel = sdf.sort_values(by='Effective Total Cost ($)').iloc[0]['Fuel Scenario']
        st.info(f"🏆 **Lowest Total Cost Option at ${carbon_tax}/t CO₂e Tax**: **{best_fuel}** (Fuel Cost + Emissions Levy)")

        display_cols = [
            'Fuel Scenario',
            'Total Operating Cost (USD)',
            'Emissions (t CO2e)',
            'Carbon Tax Liability ($)',
            'Effective Total Cost ($)',
            'Marginal Abatement Cost ($/t CO2e)'
        ]
        st.dataframe(
            sdf[display_cols].style.format({
                'Total Operating Cost (USD)': '${:,.2f}',
                'Emissions (t CO2e)': '{:,.2f}',
                'Carbon Tax Liability ($)': '${:,.2f}',
                'Effective Total Cost ($)': '${:,.2f}',
                'Marginal Abatement Cost ($/t CO2e)': '${:,.1f}'
            }),
            use_container_width=True
        )

        # 2D Trade-Off Scatter Landscape
        fig_scatter, ax_scatter = plt.subplots(figsize=(9, 4.2))

        for _, row in sdf.iterrows():
            ax_scatter.scatter(row['Emissions (t CO2e)'], row['Effective Total Cost ($)'], s=130, zorder=5)
            ax_scatter.annotate(
                f" {row['Fuel Scenario']}\n (Eff: ${row['Effective Total Cost ($)']:,.0f})",
                (row['Emissions (t CO2e)'], row['Effective Total Cost ($)']),
                fontsize=8.5,
                xytext=(8, -5 if row['Effective Total Cost ($)'] > 70000 else 5),
                textcoords='offset points'
            )

        ax_scatter.set_xlabel("Fleet Emissions (Metric Tons CO2e) → [Lower is Cleaner]")
        ax_scatter.set_ylabel("Total Fuel Cost (USD) → [Lower is Cheaper]")
        ax_scatter.grid(True, linestyle="--", alpha=0.4)

        # Add 10-15% headroom to y-axis limits to prevent text clipping
        y_vals = sdf['Effective Total Cost ($)'].values
        ax_scatter.set_ylim(min(y_vals) * 0.9, max(y_vals) * 1.12)
        ax_scatter.set_xlim(-10, max(sdf['Emissions (t CO2e)']) * 1.08)

        plt.tight_layout()
        st.pyplot(fig_scatter)
        plt.close()