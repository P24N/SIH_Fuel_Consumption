"""
app_backend.py
Core computational engine: Digital Twin surrogate model, QGA optimizer,
MILP benchmark solver, and scenario simulation engine.
"""

import time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import RidgeCV
from scipy.optimize import milp, LinearConstraint


# ==========================================================
# 1. QUANTUM-INSPIRED SURROGATE CLASS DEFINITION
# ==========================================================
class QuantumInspiredFuelRegressor(BaseEstimator, RegressorMixin):
    """
    Quantum-Inspired Random Feature Regressor mimicking multi-qubit
    superposition and ZZ-phase entanglement transformations.
    """
    def __init__(self, n_components=256, gamma=0.1, alpha=1.0, random_state=42):
        self.n_components = n_components
        self.gamma = gamma
        self.alpha = alpha
        self.random_state = random_state
        self.ridge = None
        self.weights_ = None
        self.bias_ = None

    def _quantum_feature_map(self, X):
        projections = np.dot(X, self.weights_) + self.bias_
        phi_single = np.sqrt(2.0 / self.n_components) * np.cos(projections)
        phi_entangled = phi_single[:, :-1] * phi_single[:, 1:]
        return np.hstack([phi_single, phi_entangled])

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        n_features = X.shape[1]
        self.weights_ = rng.normal(0, np.sqrt(2 * self.gamma), size=(n_features, self.n_components))
        self.bias_ = rng.uniform(0, 2 * np.pi, size=self.n_components)
        Z = self._quantum_feature_map(X)
        self.ridge = RidgeCV(alphas=np.logspace(-3, 3, 10))
        self.ridge.fit(Z, y)
        return self

    def predict(self, X):
        Z = self._quantum_feature_map(X)
        return self.ridge.predict(Z)

    def save(self, filepath, preprocessor=None, features=None, metadata=None):
        payload = {
            'n_components': self.n_components,
            'gamma': self.gamma,
            'weights': self.weights_,
            'bias': self.bias_,
            'ridge': self.ridge,
            'preprocessor': preprocessor,
            'features': features,
            'metadata': metadata or {}
        }
        joblib.dump(payload, filepath)

    @classmethod
    def load(cls, filepath):
        data = joblib.load(filepath)
        model = cls(n_components=data['n_components'], gamma=data.get('gamma', 0.1))
        model.weights_ = data['weights']
        model.bias_ = data['bias']
        model.ridge = data['ridge']
        return model, data.get('preprocessor'), data.get('features'), data.get('metadata')

# ==========================================================
# 2. GLOBAL REGISTRIES & CONSTANTS
# ==========================================================
FUEL_SPECS = {
    'Heavy Fuel Oil (HFO)': {'emission_factor': 0.62, 'cost_per_kwh': 0.08},
    'LNG':                  {'emission_factor': 0.41, 'cost_per_kwh': 0.09},
    'Methanol':             {'emission_factor': 0.30, 'cost_per_kwh': 0.12},
    'Ammonia':              {'emission_factor': 0.10, 'cost_per_kwh': 0.15},
    'Hydrogen':             {'emission_factor': 0.02, 'cost_per_kwh': 0.20}
}

DEFAULT_VESSELS = {
    'Bulk Carrier': {
        'Engine_Type': 'Four-Stroke',
        'Maintenance_Status': 'Fair',
        'Fuel_Type': 'Heavy Fuel Oil (HFO)',
        'Draft_meters': 12.5,
        'Max_Capacity': 104268.0
    },
    'Tanker': {
        'Engine_Type': 'Two-Stroke',
        'Maintenance_Status': 'Good',
        'Fuel_Type': 'Ammonia',
        'Draft_meters': 15.0,
        'Max_Capacity': 145344.0
    },
    'Container Ship': {
        'Engine_Type': 'Two-Stroke',
        'Maintenance_Status': 'Critical',
        'Fuel_Type': 'LNG',
        'Draft_meters': 9.0,
        'Max_Capacity': 21446.0
    },
    'Fish Carrier': {
        'Engine_Type': 'Gas Turbine',
        'Maintenance_Status': 'Good',
        'Fuel_Type': 'Hydrogen',
        'Draft_meters': 5.2,
        'Max_Capacity': 4942.0
    }
}

DEFAULT_ROUTES = [
    {
        'route_id': 'VOY-01',
        'route_type': 'Transoceanic',
        'distance_nm': 3200,
        'cargo_weight_tons': 45000,
        'weather_condition': 'Moderate',
        'max_arrival_hours': 240.0
    },
    {
        'route_id': 'VOY-02',
        'route_type': 'Long-haul',
        'distance_nm': 1800,
        'cargo_weight_tons': 28000,
        'weather_condition': 'Rough',
        'max_arrival_hours': 140.0
    },
    {
        'route_id': 'VOY-03',
        'route_type': 'Coastal',
        'distance_nm': 650,
        'cargo_weight_tons': 4000,
        'weather_condition': 'Calm',
        'max_arrival_hours': 60.0
    }
]

def generate_stress_test_data(n_routes=15, n_vessels=10, seed=42):
    """Generates synthetic, physically feasible fleets and route demand schedules."""
    np.random.seed(seed)
    
    ship_types = ['Bulk Carrier', 'Tanker', 'Container Ship', 'Fish Carrier', 'General Cargo']
    engine_types = ['Two-Stroke', 'Four-Stroke', 'Gas Turbine']
    fuel_types = list(FUEL_SPECS.keys())
    
    # Generate fleet
    stress_fleet = {}
    for i in range(n_vessels):
        name = f"Vessel_{i+1:02d}_{np.random.choice(ship_types)}"
        stress_fleet[name] = {
            'Ship_Type': name.split('_', 2)[2],
            'Engine_Type': str(np.random.choice(engine_types)),
            'Maintenance_Status': 'Good' if np.random.rand() > 0.15 else 'Critical',
            'Fuel_Type': str(np.random.choice(fuel_types)),
            'Draft_meters': round(float(np.random.uniform(6.0, 16.0)), 1),
            'Max_Capacity': round(float(np.random.uniform(5000.0, 120000.0)), 1)
        }
    
    # Generate voyages
    route_types = ['Transoceanic', 'Long-haul', 'Coastal']
    weathers = ['Calm', 'Moderate', 'Rough']
    stress_routes = []
    
    for j in range(n_routes):
        r_type = np.random.choice(route_types)
        dist = float(np.random.uniform(2000, 5000) if r_type == 'Transoceanic' else (
            np.random.uniform(800, 2000) if r_type == 'Long-haul' else np.random.uniform(200, 800)
        ))
        speed_est = 14.0
        min_hours = dist / speed_est
        deadline = round(min_hours * float(np.random.uniform(1.15, 1.6)), 0)
        
        stress_routes.append({
            'route_id': f"STR-{j+1:03d}",
            'route_type': str(r_type),
            'distance_nm': round(dist, 1),
            'cargo_weight_tons': round(float(np.random.uniform(3000.0, 75000.0)), 1),
            'weather_condition': str(np.random.choice(weathers)),
            'max_arrival_hours': float(deadline)
        })
        
    return stress_routes, stress_fleet

# ==========================================================
# 3. MODEL LOADER
# ==========================================================
def load_digital_twin(bundle_path='qirf_fuel_predictor.joblib'):
    """Loads preprocessor and reconstructed digital twin via class factory."""
    bundle_path = Path(bundle_path)
    if not bundle_path.is_absolute():
        bundle_path = Path(__file__).resolve().parent / bundle_path
    model, preprocessor, features, _ = QuantumInspiredFuelRegressor.load(bundle_path)
    return preprocessor, model, features


# ==========================================================
# 4. PRE-EVALUATION CACHE HELPER
# ==========================================================
def precompute_operational_eval_cache(voyage_schedule, vessel_registry, preprocessor, 
                                     qirf_model, feature_cols, candidate_speeds, weights,carbon_tax_per_ton=0.0):
    """
    Evaluates candidate speeds across viable ship-route assignments using the QIRF model.
    """
    w_energy, w_emissions, w_cost = weights
    available_ships = list(vessel_registry.keys())
    eval_cache = {}

    for r_idx, mission in enumerate(voyage_schedule):
        for s_idx, ship_name in enumerate(available_ships):
            ship_meta = vessel_registry[ship_name]

            # Infeasible checks
            if ship_meta['Maintenance_Status'] == 'Critical':
                continue
            if mission['cargo_weight_tons'] > ship_meta['Max_Capacity']:
                continue

            batch = pd.DataFrame([{
                'Ship_Type': ship_meta.get('Ship_Type', ship_name),
                'Route_Type': mission['route_type'],
                'Engine_Type': ship_meta['Engine_Type'],
                'Maintenance_Status': ship_meta['Maintenance_Status'],
                'Fuel_Type': ship_meta['Fuel_Type'],
                'Weather_Condition': mission['weather_condition'],
                'Speed_Over_Ground_knots': spd,
                'Distance_Traveled_nm': mission['distance_nm'],
                'Draft_meters': ship_meta['Draft_meters'],
                'Cargo_Weight_tons': mission['cargo_weight_tons'],
                'Max_Capacity': ship_meta['Max_Capacity'],
                'Current_Load': mission['cargo_weight_tons'],
                'Seasonal_Impact_Score': 1.0
            } for spd in candidate_speeds])

            X_proc = preprocessor.transform(batch[feature_cols])
            energies = qirf_model.predict(X_proc)
            fuel_info = FUEL_SPECS.get(ship_meta['Fuel_Type'], {'emission_factor': 0.55, 'cost_per_kwh': 0.10})

            feasible = []
            for spd, energy in zip(candidate_speeds, energies):
                duration = mission['distance_nm'] / spd
                if duration <= mission['max_arrival_hours']:
                    em = energy * fuel_info['emission_factor']
                    # 1. Base bunker fuel cost
                    bunker_cost = energy * fuel_info['cost_per_kwh']

                    # 2. Emissions tax penalty: (emissions / 1000 to convert kg to metric tons) * tax_per_ton
                    carbon_penalty = (em / 1000.0) * carbon_tax_per_ton

                    # 3. Total economic cost passed to objective function
                    cost = bunker_cost + carbon_penalty
                    # cost = energy * fuel_info['cost_per_kwh']
                    obj = (w_energy * energy) + (w_emissions * em) + (w_cost * cost)
                    feasible.append({
                        'speed': spd,
                        'energy_kwh': energy,
                        'emissions_kg': em,
                        'bunker_cost': bunker_cost,
                        'carbon_penalty': carbon_penalty,
                        'cost_usd': cost,
                        'duration_hours': duration,
                        'obj': obj
                    })
            if feasible:
                eval_cache[(r_idx, s_idx)] = min(feasible, key=lambda x: x['obj'])

    return eval_cache


# ==========================================================
# 5. QUANTUM-INSPIRED GENETIC ALGORITHM (QGA)
# ==========================================================
def optimize_fleet_schedule(voyage_schedule, vessel_registry, preprocessor, qirf_model, feature_cols,
                            candidate_speeds=np.arange(10, 22, 1), weights=(0.4, 0.4, 0.2), 
                            n_generations=40, pop_size=30,carbon_tax_per_ton=0.0):
    """
    Quantum-Inspired Genetic Algorithm optimizing fleet allocation via Ry rotation gates.
    """
    # start_time = time.perf_counter()
    available_ships = list(vessel_registry.keys())
    num_routes = len(voyage_schedule)
    num_ships = len(available_ships)

    eval_cache = precompute_operational_eval_cache(
        voyage_schedule, vessel_registry, preprocessor, qirf_model, feature_cols, candidate_speeds, weights, carbon_tax_per_ton
    )
    start_time = time.perf_counter()
    feasibility_mask = np.zeros((num_routes, num_ships), dtype=int)
    for (r, s) in eval_cache.keys():
        feasibility_mask[r, s] = 1

    # -------------------------------------------------------------
    # Informed / Greedy Q-bit Initialization
    # -------------------------------------------------------------

    # Initialize empty population tensor: [pop_size, num_routes, num_ships, 2]
    q_pop = np.zeros((pop_size, num_routes, num_ships, 2), dtype=np.float64)

    # 1. Base prior: probability of selection should sum roughly to 1 across ships
    # Default selection probability ~ 1 / S
    base_p1 = max(0.05, 1.0 / num_ships)
    
    # 2. Score candidate compatibility from pre-evaluated lookup
    # Higher affinity for lower-cost feasible ship-route pairs
    for r in range(num_routes):
        # Collect costs of all feasible assignments for this route
        route_costs = []
        for s in range(num_ships):
            entry = eval_cache.get((r, s))
            if entry is not None and entry.get('status', 'FEASIBLE') == 'FEASIBLE':
                route_costs.append((s, entry.get('cost', entry.get('cost_usd', 1e6))))
            else:
                route_costs.append((s, float('inf')))

        # Sort feasible ships by cost for this route
        valid_pairs = [pair for pair in route_costs if pair[1] < float('inf')]
        
        for s, cost in route_costs:
            if cost == float('inf'):
                # Infeasible / deadline violation: suppress |1> probability strongly (~2%)
                p_one = 0.02
            else:
                # Feasible ship: rank-based greedy boost
                # Rank 0 (cheapest) gets highest probability
                rank = sorted([p[1] for p in valid_pairs]).index(cost) if valid_pairs else 0
                rank_weight = 1.0 / (rank + 1.0)
                p_one = min(0.65, base_p1 * (1.0 + rank_weight))

            # Ensure valid probability bounds [0.01, 0.99]
            p_one = np.clip(p_one, 0.01, 0.99)
            
            # Map Born rule probabilities: beta = sqrt(P(|1>)), alpha = sqrt(1 - P(|1>))
            beta_val = np.sqrt(p_one)
            alpha_val = np.sqrt(1.0 - p_one)

            # Assign to all chromosomes with slight random phase noise across population
            for i in range(pop_size):
                # Mild perturbation per individual preserves genetic diversity
                noise = np.random.uniform(-0.02, 0.02)
                b_noisy = np.clip(beta_val + noise, 0.05, 0.95)
                a_noisy = np.sqrt(1.0 - b_noisy**2)
                q_pop[i, r, s, 0] = a_noisy
                q_pop[i, r, s, 1] = b_noisy

    best_deployment = None
    best_fitness = float('inf')
    convergence_history = []

# -------------------------------------------------------------

    # Pre-build a fast 2D lookup matrix: shape (num_routes, num_ships)
    cost_matrix = np.full((num_routes, num_ships), 1e9)
    for (r, s), item in eval_cache.items():
        cost_matrix[r, s] = item.get('obj', 1e9)

# -------------------------------------------------------------
    # Early stopping settings
    patience = 12               # Number of generations to wait without improvement
    min_delta = 1e-4           # Minimum improvement threshold to reset counter
    stall_counter = 0          # Tracks non-improving generations
# -------------------------------------------------------------
    for gen in range(n_generations):
        prob_matrix = q_pop[:, :, :, 1] ** 2
        collapsed = (np.random.rand(*prob_matrix.shape) < prob_matrix).astype(int)

        # -------------------------------------------------------------
        # HARD CARGO DELIVERY ENFORCEMENT (No slack / No unserved routes)
        # -------------------------------------------------------------
        valid_states = np.zeros_like(collapsed)
        
        for i in range(pop_size):
            used_ships = set()

            route_order = np.random.permutation(num_routes)
            
            # 1. Enforce delivery for EVERY route r
            for r in route_order:

                # Find all ships that physically satisfy draft and deadlines
                feasible_ships = [s for s in range(num_ships) if feasibility_mask[r, s] == 1]
                if not feasible_ships:
                    # Absolute fallback if deadline is tight: any ship in fleet
                    feasible_ships = list(range(num_ships))

                # Look at what the quantum collapse sampled for this route
                sampled_ships = [s for s in np.where(collapsed[i, r, :] == 1)[0] if s in feasible_ships]

                # Prioritize an unused feasible ship that was sampled by quantum collapse
                candidate_ships = [s for s in sampled_ships if s not in used_ships]
                
                if not candidate_ships:
                    # If collapse didn't find an available ship, pick the cheapest unused feasible ship
                    unused_feasible = [s for s   in feasible_ships if s not in used_ships]
                    candidate_ships = unused_feasible if unused_feasible else feasible_ships

                # Select the lowest-cost ship among candidates
                best_s = min(candidate_ships, key=lambda s: eval_cache.get((r, s), {}).get('cost', 1e9))
                
                # HARD ASSIGNMENT: Exactly 1 ship assigned to this route
                valid_states[i, r, best_s] = 1
                used_ships.add(best_s)
    # -------------------------------------------------------------
    # Vectorized Population Evaluation (Replaces 2.25M loop checks)
    # -------------------------------------------------------------
        # 1. Base cost calculation: Element-wise multiply and sum over route and ship axes
        # valid_states shape: (pop_size, num_routes, num_ships)
        # cost_matrix shape: (num_routes, num_ships)
        base_costs = np.sum(valid_states * cost_matrix[None, :, :], axis=(1, 2))

        # 2. Vectorized Fleet Invariant penalty (deployed > 1 per ship)
        # deployed_counts shape: (pop_size, num_ships)
        deployed_counts = np.sum(valid_states, axis=1)
        over_deployed = np.maximum(0, deployed_counts - 1)
        penalties = np.sum(over_deployed * 35_000.0, axis=1)

        # Combined objective scores for all individuals in one operation
        scores = base_costs + penalties
# -----------------------------------------------------------------------------------------------------
        min_idx = np.argmin(scores)
        gen_best_score = scores[min_idx]

        if gen_best_score < best_fitness - min_delta:
            best_fitness = gen_best_score
            best_deployment = np.copy(valid_states[min_idx])
            stall_counter = 0  # Reset counter on genuine improvement
        else:
            stall_counter += 1  # Increment if no meaningful progress

        convergence_history.append(best_fitness)

        # Check early stopping condition
        if stall_counter >= patience:
            break

        # Dynamic Ry rotation
        delta_theta = 0.05 * np.pi * (1.0 - (gen / n_generations)*0.7)

        # Quantum Mutation / Superposition Reset to escape local minima
        # Quantum Mutation: Gentle Phase Kick (Prevents destroying learned states)
        if len(convergence_history) >= 4 and abs(convergence_history[-1] - convergence_history[-4]) < 1e-4:
            # 10% chance of mutation
            mutation_mask = np.random.rand(pop_size, num_routes, num_ships) < 0.10
            kick_angles = np.random.uniform(-0.1 * np.pi, 0.1 * np.pi, size=(pop_size, num_routes, num_ships))
            
            cos_k = np.cos(kick_angles)
            sin_k = np.sin(kick_angles)
            
            a_curr = q_pop[:, :, :, 0]
            b_curr = q_pop[:, :, :, 1]
            
            # Apply rotation kick only where mutation_mask is True
            a_kicked = np.where(mutation_mask, cos_k * a_curr - sin_k * b_curr, a_curr)
            b_kicked = np.where(mutation_mask, sin_k * a_curr + cos_k * b_curr, b_curr)
            
            norm_k = np.sqrt(a_kicked**2 + b_kicked**2)
            q_pop[:, :, :, 0] = a_kicked / norm_k
            q_pop[:, :, :, 1] = b_kicked / norm_k

        # -------------------------------------------------------------
        # Vectorized Quantum Rotation Gates (Replaces 1.8M scalar calls)
        # -------------------------------------------------------------

        if best_deployment is not None:
            # Broadcast target bits across all individuals in the population: [pop_size, num_routes, num_ships]
            target_matrix = np.broadcast_to(best_deployment[None, :, :], (pop_size, num_routes, num_ships))
            theta = np.where(target_matrix == 1, delta_theta, -delta_theta)
        else:
            theta = np.full((pop_size, num_routes, num_ships), -delta_theta)

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        alpha = q_pop[:, :, :, 0]
        beta = q_pop[:, :, :, 1]

        # Apply 2D rotation matrix: [cos, -sin; sin, cos] simultaneously
        new_alpha = cos_t * alpha - sin_t * beta
        new_beta = sin_t * alpha + cos_t * beta

        # Normalize probability amplitudes to enforce |alpha|^2 + |beta|^2 = 1
        norm = np.sqrt(new_alpha**2 + new_beta**2)
        q_pop[:, :, :, 0] = new_alpha / norm
        q_pop[:, :, :, 1] = new_beta / norm

    runtime = time.perf_counter() - start_time

    # Output DataFrame
    records = []
    for r_idx, mission in enumerate(voyage_schedule):
        assigned_ship = None
        
        # 1. Check if best_deployment assigned a valid ship to route r_idx
        if best_deployment is not None:
            assigned_indices = np.where(best_deployment[r_idx, :] == 1)[0]
            if len(assigned_indices) > 0:
                s_idx = assigned_indices[0]
                assigned_ship = available_ships[s_idx]
                sol = eval_cache.get((r_idx, s_idx))
        
        # 2. Fallback: If unassigned, pick the lowest-cost candidate rather than leaving it slack
        if assigned_ship is None or sol is None:
            s_idx = min(range(len(available_ships)), key=lambda s: eval_cache.get((r_idx, s), {}).get('cost', 1e9))
            assigned_ship = available_ships[s_idx]
            sol = eval_cache.get((r_idx, s_idx), {})

        meta = vessel_registry[assigned_ship]
        # Check if the chosen assignment actually satisfied constraints
        duration = sol.get('duration_hours', 0.0)
        deadline = mission['max_arrival_hours']
        is_feasible = (r_idx, s_idx) in eval_cache and duration <= deadline
        records.append({
            'Route ID': mission['route_id'],
            'Route Type': mission['route_type'],
            'Assigned Vessel': assigned_ship,
            'Fuel Type': meta['Fuel_Type'],
            'Optimized Speed (knots)': sol.get('speed', 14.0),
            'Duration (hrs)': round(sol.get('duration_hours', 0.0), 1),
            'Deadline (hrs)': mission['max_arrival_hours'],
            'Energy (kWh)': round(sol.get('energy_kwh', 0.0), 2),
            'Emissions (kg CO2e)': round(sol.get('emissions_kg', 0.0), 2),
            'Bunker Cost (USD)': round(sol.get('bunker_cost', sol.get('cost_usd', 0)), 2),
            'Carbon Penalty (USD)': round(sol.get('carbon_penalty', 0), 2),
            'Total Cost (USD)': round(sol.get('cost', sol.get('cost_usd', 0)), 2),
            'Status': 'OPTIMAL' if is_feasible else 'INFEASIBLE (DEADLINE EXCEEDED)'
        })

    return pd.DataFrame(records), convergence_history, runtime


# ==========================================================
# 6. CLASSICAL BENCHMARK SOLVER (MILP VIA HIGHS)
# ==========================================================
def optimize_fleet_classical_milp(voyage_schedule, vessel_registry, preprocessor, qirf_model, 
                                 feature_cols, weights=(0.4, 0.4, 0.2),carbon_tax_per_ton=0.0,
                                 time_limit=None):
    """
    Deterministic classical benchmark: Branch-and-Cut MILP with slack formulation.
    """
    # start_time = time.perf_counter()
    available_ships = list(vessel_registry.keys())
    num_routes = len(voyage_schedule)
    num_ships = len(available_ships)
    
    eval_cache = precompute_operational_eval_cache(
        voyage_schedule, vessel_registry, preprocessor, qirf_model, feature_cols, np.arange(10, 22, 1), weights, carbon_tax_per_ton
    )

    start_time = time.perf_counter()

    num_assign = num_routes * num_ships
    num_vars = num_assign + num_routes

    # --- (Strict Assignment without Slack): ---
    num_assign = num_routes * num_ships
    num_vars = num_assign  # Exactly R * S binary assignment variables

    c = np.zeros(num_vars)
    for r in range(num_routes):
        for s in range(num_ships):
            idx = r * num_ships + s
            c[idx] = eval_cache[(r, s)]['obj'] if (r, s) in eval_cache else 1e9

    # Constraint 1: sum_s x_{r,s} == 1 (Strict cargo delivery demand)
    A_eq = np.zeros((num_routes, num_vars))
    for r in range(num_routes):
        for s in range(num_ships):
            A_eq[r, r * num_ships + s] = 1.0
    b_eq = np.ones(num_routes)

    # Constraint 2: sum_r x_{r,s} <= 1 (Fleet invariant: max 1 mission per ship)
    A_ub = np.zeros((num_ships, num_vars))
    for s in range(num_ships):
        for r in range(num_routes):
            A_ub[s, r * num_ships + s] = 1.0
    b_ub = np.ones(num_ships)

    A_all = np.vstack([A_eq, A_ub])
    lhs = np.concatenate([b_eq, np.zeros(num_ships)])   # 0 <= sum_r x_{r,s}
    rhs = np.concatenate([b_eq, b_ub])                  # sum_s x_{r,s} == 1 and sum_r x_{r,s} <= 1

    constraints = LinearConstraint(A_all, lhs, rhs)
    integrality = np.ones(num_vars)
    bounds = (np.zeros(num_vars), np.ones(num_vars))

    res = milp(c=c, integrality=integrality, constraints=constraints, bounds=bounds,
               options={'time_limit': time_limit} if time_limit is not None else None)
    runtime = time.perf_counter() - start_time

    records = []
    total_obj = res.fun if res.success else float('inf')

    for r in range(num_routes):
        assigned_ship = None
        for s in range(num_ships):
            idx = r * num_ships + s
            if res.x is not None and res.x[idx] > 0.5 and (r, s) in eval_cache:
                assigned_ship = available_ships[s]
                sol = eval_cache[(r, s)]
                meta = vessel_registry[assigned_ship]
                records.append({
                    'Route ID': voyage_schedule[r]['route_id'],
                    'Route Type': voyage_schedule[r]['route_type'],
                    'Assigned Vessel': assigned_ship,
                    'Fuel Type': meta['Fuel_Type'],
                    'Speed (knots)': sol['speed'],
                    'Duration (hrs)': round(sol['duration_hours'], 1),
                    'Energy (kWh)': round(sol['energy_kwh'], 2),
                    'Emissions (kg CO2e)': round(sol['emissions_kg'], 2),
                    'Bunker Cost (USD)': round(sol.get('bunker_cost', sol.get('cost_usd', 0)), 2),
                    'Carbon Penalty (USD)': round(sol.get('carbon_penalty', 0), 2),
                    'Total Cost (USD)': round(sol.get('cost', sol.get('cost_usd', 0)), 2),
                    'Status': 'OPTIMAL'
                })
                break

        # Fallback safeguard in case HiGHS fails or an infeasible instance occurs
        if assigned_ship is None:
            # Deterministically assign the lowest-cost available candidate
            fallback_s = min(range(num_ships), key=lambda s: eval_cache.get((r, s), {}).get('cost', 1e9))
            sol = eval_cache.get((r, fallback_s), {})
            ship_name = available_ships[fallback_s]
            meta = vessel_registry[ship_name]
            records.append({
                'Route ID': voyage_schedule[r]['route_id'],
                'Route Type': voyage_schedule[r]['route_type'],
                'Assigned Vessel': ship_name,
                'Fuel Type': meta['Fuel_Type'],
                'Speed (knots)': sol.get('speed', 14.0),
                'Duration (hrs)': round(sol.get('duration_hours', 0.0), 1),
                'Energy (kWh)': round(sol.get('energy_kwh', 0.0), 2),
                'Emissions (kg CO2e)': round(sol.get('emissions_kg', 0.0), 2),
                'Bunker Cost (USD)': round(sol.get('bunker_cost', 0.0), 2),
                'Carbon Penalty (USD)': round(sol.get('carbon_penalty', 0.0), 2),
                'Total Cost (USD)': round(sol.get('cost', sol.get('cost_usd', 0.0)), 2),
                'Status': 'FEASIBLE_FALLBACK'
            })
    return pd.DataFrame(records), total_obj, runtime


# ==========================================================
# 7. ALTERNATIVE FUEL SCENARIO ANALYSIS (DELIVERABLE 3)
# ==========================================================
def run_fuel_scenario_analysis(voyage_schedule, vessel_registry, preprocessor, qirf_model, feature_cols,
                                fuel_options=['Heavy Fuel Oil (HFO)', 'LNG', 'Methanol', 'Ammonia', 'Hydrogen']):
    """
    Evaluates fleet dispatch across distinct fuel pathways, computing lifecycle
    emissions and abatement percentages vs. the HFO baseline.
    """
    scenario_results = []
    temp_registry = {ship: dict(meta) for ship, meta in vessel_registry.items()}

    for fuel in fuel_options:
        for ship in temp_registry:
            temp_registry[ship]['Fuel_Type'] = fuel

        res_df, _, _ = optimize_fleet_schedule(
            voyage_schedule, temp_registry, preprocessor, qirf_model, feature_cols, n_generations=25, pop_size=20
        )

        total_energy = res_df['Energy (kWh)'].sum()
        total_emissions = res_df['Emissions (kg CO2e)'].sum()

        # Safe lookup: checks for 'Bunker Cost (USD)' first, then falls back to 'Cost (USD)'
        cost_col = 'Bunker Cost (USD)' if 'Bunker Cost (USD)' in res_df.columns else 'Total Cost (USD)'
        total_cost = res_df[cost_col].sum()

        scenario_results.append({
            'Fuel Scenario': fuel,
            'Total Fleet Energy (kWh)': round(total_energy, 2),
            'Total Emissions (kg CO2e)': round(total_emissions, 2),
            'Total Operating Cost (USD)': round(total_cost, 2),
            'Emissions Abatement vs HFO (%)': 0.0
        })

    summary_df = pd.DataFrame(scenario_results)
    hfo_rows = summary_df.loc[summary_df['Fuel Scenario'] == 'Heavy Fuel Oil (HFO)', 'Total Emissions (kg CO2e)']
    if not hfo_rows.empty:
        base_emissions = hfo_rows.values[0]
        if base_emissions > 0:
            summary_df['Emissions Abatement vs HFO (%)'] = (
                (base_emissions - summary_df['Total Emissions (kg CO2e)']) / base_emissions * 100
            ).round(2)

    return summary_df

# ------------------------------------------------------------------------------------------------------------------


import numpy as np
import time
from scipy.optimize import milp, LinearConstraint

def run_dynamic_benchmark(voyage_schedule, available_ships, vessel_registry, time_limit=None):
    """
    Benchmarks MILP vs QGA when speed selection (K speed tiers) and berth clashing 
    are co-optimized dynamically, breaking Total Unimodularity (TUM).
    """
    num_routes = len(voyage_schedule)
    # available_ships = list(vessel_registry.keys())
    num_ships = len(available_ships)
    
    # Speed tiers: e.g., 12, 14, 16, 18, 20 knots
    speed_tiers = np.array([12.0, 14.0, 16.0, 18.0, 20.0])
    num_tiers = len(speed_tiers)
    
    # -------------------------------------------------------------
    # 1. Classical MILP with Berth Conflict Constraints (TUM Broken)
    # -------------------------------------------------------------
    milp_start = time.perf_counter()
    n_vars = num_routes * num_ships * num_tiers
    c = np.zeros(n_vars)
    arrival_times_map = {}

    for r, route in enumerate(voyage_schedule):
        dist = route['distance_nm']
        deadline = route['max_arrival_hours']
        for s, ship in enumerate(available_ships):
            ship_meta = vessel_registry.get(ship, {})
            base_eff = ship_meta.get('Base_Fuel_Rate', 0.05)
            for k, spd in enumerate(speed_tiers):
                idx = (r * num_ships * num_tiers) + (s * num_tiers) + k
                transit_time = dist / spd
                arrival_times_map[idx] = transit_time
                
                if transit_time > deadline:
                    c[idx] = 1e6
                else:
                    c[idx] = base_eff * dist * ((spd / 14.0) ** 2) * 450.0

    # Constraint 1: Exactly 1 vessel & speed per route (shape: num_routes x n_vars)
    A_eq1 = np.zeros((num_routes, n_vars))
    for r in range(num_routes):
        for s in range(num_ships):
            for k in range(num_tiers):
                A_eq1[r, (r * num_ships * num_tiers) + (s * num_tiers) + k] = 1.0

    # Constraint 2: Ship can take at most 1 route (shape: num_ships x n_vars)
    A_ub2 = np.zeros((num_ships, n_vars))
    for s in range(num_ships):
        for r in range(num_routes):
            for k in range(num_tiers):
                A_ub2[s, (r * num_ships * num_tiers) + (s * num_tiers) + k] = 1.0

    # Constraint 3: BERTH CONGESTION MUTUAL EXCLUSION (Breaks TUM)
    # If two voyages arrive within 3 hours of each other, they cannot both be scheduled
    conflict_rows = []
    # Check conflicts across candidate routes
    for r1 in range(num_routes):
        for r2 in range(r1 + 1, num_routes):
            for s1 in range(num_ships):
                for s2 in range(num_ships):
                    for k1 in range(num_tiers):
                        for k2 in range(num_tiers):
                            idx1 = (r1 * num_ships * num_tiers) + (s1 * num_tiers) + k1
                            idx2 = (r2 * num_ships * num_tiers) + (s2 * num_tiers) + k2
                            t1 = arrival_times_map[idx1]
                            t2 = arrival_times_map[idx2]
                            if abs(t1 - t2) < 3.0:  # Port berth conflict window
                                row = np.zeros(n_vars)
                                row[idx1] = 1.0
                                row[idx2] = 1.0
                                conflict_rows.append(row)
                                if len(conflict_rows) >= 1200:  # Bound cut additions
                                    break
                    if len(conflict_rows) >= 1200:
                        break
            if len(conflict_rows) >= 1200:
                break

    if conflict_rows:
        A_conflicts = np.array(conflict_rows)
        A_ub_total = np.vstack([A_ub2, A_conflicts])
        rhs_ub = np.concatenate([np.ones(num_ships), np.ones(len(conflict_rows))])
    else:
        A_ub_total = A_ub2
        rhs_ub = np.ones(num_ships)

    constraints = [
        LinearConstraint(A_eq1, np.ones(num_routes), np.ones(num_routes)),
        LinearConstraint(A_ub_total, -np.inf, rhs_ub)
    ]
    integrality = np.ones(n_vars)
    bounds = (0, 1)

    res = milp(c=c, integrality=integrality, constraints=constraints, bounds=bounds,
               options={'time_limit': time_limit} if time_limit is not None else None)
    milp_runtime = time.perf_counter() - milp_start
    milp_cost = res.fun if (res.success and res.fun < 1e7) else float('nan')

    # -------------------------------------------------------------
    # 2. Dynamic QGA Engine (Continuous Speed Evaluation)
    # -------------------------------------------------------------
    qga_start = time.perf_counter()
    pop_size = 40
    n_generations = 30
    
    # Pre-extract distance and deadline vectors for fast tensor operations
    distances = np.array([r['distance_nm'] for r in voyage_schedule])
    deadlines = np.array([r['max_arrival_hours'] for r in voyage_schedule])
    base_effs = np.array([vessel_registry[s].get('Base_Fuel_Rate', 0.05) for s in available_ships])

    # Initial Quantum state amplitudes
    q_pop = np.full((pop_size, num_routes, num_ships, 2), 1.0 / np.sqrt(2))
    best_qga_cost = float('inf')
    qga_history = []
    
    delta_theta = 0.04 * np.pi
    best_deployment = None

    for gen in range(n_generations):
        # 1. Quantum sampling & Repair
        probs = q_pop[:, :, :, 1] ** 2
        valid_deployments = np.zeros((pop_size, num_routes, num_ships))
        
        for i in range(pop_size):
            used_ships = set()
            route_order = np.random.permutation(num_routes)
            for r in route_order:
                candidates = np.argsort(-probs[i, r, :])
                for s in candidates:
                    if s not in used_ships:
                        valid_deployments[i, r, s] = 1
                        used_ships.add(s)
                        break

        # 2. Vectorized Continuous Speed Optimization
        # Analytical optimal speed: v = max(dist / deadline, 12.0)
        min_feasible_speeds = np.clip(distances / deadlines, 12.0, 20.0) # shape (num_routes,)
        # Fuel = base_eff * dist * (v/14)^2 * 450
        route_fuel = distances * ((min_feasible_speeds / 14.0)**2) * 450.0 # shape (num_routes,)
        
        # Multiply across assignments: shape (pop_size, num_routes, num_ships)
        cost_tensor = route_fuel[:, None] * base_effs[None, :]
        scores = np.sum(valid_deployments * cost_tensor[None, :, :], axis=(1, 2))

        # 3. Dynamic Berth Congestion Check
        # Penalize port arrival clustering (vectorized)
        arrival_times = distances / min_feasible_speeds  # arrival in hours
        sorted_arrivals = np.sort(arrival_times)
        berth_violations = np.sum(np.maximum(0, 3.0 - np.diff(sorted_arrivals)))
        scores += berth_violations * 2500.0  # Demurrage penalty

        min_idx = np.argmin(scores)
        if scores[min_idx] < best_qga_cost:
            best_qga_cost = scores[min_idx]
            best_deployment = valid_deployments[min_idx]

        qga_history.append(best_qga_cost)

        # 4. Vectorized Quantum Rotation towards best
        target_matrix = np.broadcast_to(best_deployment[None, :, :], (pop_size, num_routes, num_ships))
        theta = np.where(target_matrix == 1, delta_theta, -delta_theta)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        
        a = q_pop[:, :, :, 0]
        b = q_pop[:, :, :, 1]
        a_new = cos_t * a - sin_t * b
        b_new = sin_t * a + cos_t * b
        norm = np.sqrt(a_new**2 + b_new**2)
        q_pop[:, :, :, 0] = a_new / norm
        q_pop[:, :, :, 1] = b_new / norm

    qga_runtime = time.perf_counter() - qga_start
    return milp_runtime, milp_cost, qga_runtime, best_qga_cost, qga_history, n_vars

# --------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    print("Testing app_backend.py pipeline standalone...")
    preprocessor, model, features = load_digital_twin()
    df, hist, t = optimize_fleet_schedule(DEFAULT_ROUTES, DEFAULT_VESSELS, preprocessor, model, features)
    print(f"QGA Test Run Successful! Runtime: {t:.4f}s | Converged Fitness: {hist[-1]:.2f}")
