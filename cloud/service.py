"""Stateless request boundary shared by Vercel and the local web server."""
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import warnings
from functools import lru_cache
from importlib.metadata import version

# Limit BLAS workers before importing the scientific stack on a small cloud CPU.
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning
import app_backend as backend

ROOT = Path(__file__).resolve().parents[1]
SOLVER_LOCK = threading.Lock()  # Existing optimizers use NumPy's global RNG.
SHIP_TYPES = ['Bulk Carrier', 'Tanker', 'Container Ship', 'Fish Carrier', 'General Cargo']
CHOICES = {
    'Ship_Type': SHIP_TYPES,
    'Engine_Type': ['Two-Stroke', 'Four-Stroke', 'Gas Turbine'],
    'Maintenance_Status': ['Good', 'Fair', 'Critical'],
    'Fuel_Type': list(backend.FUEL_SPECS),
    'route_type': ['Transoceanic', 'Long-haul', 'Coastal'],
    'weather_condition': ['Calm', 'Moderate', 'Rough'],
}


class BusyError(Exception):
    pass


@lru_cache(maxsize=1)
def model():
    """Fail closed on a changed artifact or incompatible sklearn; never download it."""
    manifest = json.loads((ROOT / 'model_manifest.json').read_text())
    path = ROOT / 'qirf_fuel_predictor.joblib'
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['sha256']:
        raise RuntimeError('Model checksum mismatch; regenerate and verify the model manifest.')
    if version('scikit-learn') != manifest['scikit_learn']:
        raise RuntimeError('The model requires the pinned scikit-learn version.')
    with warnings.catch_warnings():
        warnings.simplefilter('error', InconsistentVersionWarning)
        pre, reg, features = backend.load_digital_twin(path)
    sample = pd.DataFrame(manifest['smoke_inputs'])
    prediction = reg.predict(pre.transform(sample[features]))
    if not np.allclose(prediction, manifest['smoke_predictions'], rtol=1e-7, atol=1e-6):
        raise RuntimeError('Model inference self-test failed.')
    return pre, reg, features


def number(value, name, low, high, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a number.')
    if not math.isfinite(value) or not low <= value <= high or (integer and int(value) != value):
        raise ValueError(f'{name} must be {"an integer " if integer else ""}between {low} and {high}.')
    return int(value) if integer else float(value)


def text(value, name):
    if not isinstance(value, str) or not value.strip() or len(value) > 80:
        raise ValueError(f'{name} must be 1–80 characters.')
    return value.strip()


def choice(row, key):
    value = row.get(key)
    if value not in CHOICES[key]:
        raise ValueError(f'Invalid {key}. Choose one of: {", ".join(CHOICES[key])}.')
    return value


def scenario(data):
    fleet = data.get('fleet')
    routes = data.get('routes')
    if not isinstance(fleet, list) or not 1 <= len(fleet) <= 20:
        raise ValueError('Provide between 1 and 20 vessels.')
    if not isinstance(routes, list) or not 1 <= len(routes) <= 20:
        raise ValueError('Provide between 1 and 20 routes.')
    registry, schedule, ids = {}, [], set()
    for row in fleet:
        if not isinstance(row, dict):
            raise ValueError('Each vessel must be an object.')
        name = text(row.get('Vessel'), 'Vessel')
        if name in registry:
            raise ValueError('Vessel names must be unique.')
        registry[name] = {key: choice(row, key) for key in
                          ['Ship_Type', 'Engine_Type', 'Maintenance_Status', 'Fuel_Type']}
        registry[name].update(
            Draft_meters=number(row.get('Draft_meters'), 'Draft_meters', 1, 30),
            Max_Capacity=number(row.get('Max_Capacity'), 'Max_Capacity', 500, 500000))
    for row in routes:
        if not isinstance(row, dict):
            raise ValueError('Each route must be an object.')
        rid = text(row.get('route_id'), 'route_id')
        if rid in ids:
            raise ValueError('Route IDs must be unique.')
        ids.add(rid)
        schedule.append(dict(route_id=rid, route_type=choice(row, 'route_type'),
            weather_condition=choice(row, 'weather_condition'),
            distance_nm=number(row.get('distance_nm'), 'distance_nm', 10, 20000),
            cargo_weight_tons=number(row.get('cargo_weight_tons'), 'cargo_weight_tons', 10, 500000),
            max_arrival_hours=number(row.get('max_arrival_hours'), 'max_arrival_hours', 1, 5000)))
    return schedule, registry


def controls(data):
    raw = data.get('weights', [0.4, 0.4, 0.2])
    if not isinstance(raw, list) or len(raw) != 3:
        raise ValueError('weights must contain three numbers.')
    weights = [number(v, 'weight', 0, 1) for v in raw]
    total = sum(weights)
    return dict(weights=tuple(v / total for v in weights) if total else (0.4, 0.4, 0.2),
        carbon_tax_per_ton=number(data.get('carbon_tax', 100), 'carbon_tax', 0, 300)), dict(
        n_generations=number(data.get('generations', 40), 'generations', 5, 100, True),
        pop_size=number(data.get('population', 30), 'population', 5, 100, True))


def records(frame):
    return json.loads(frame.to_json(orient='records'))


def dispatch_result(frame, registry, schedule, runtime, history=None, objective=None):
    rows = records(frame)
    counts = frame['Assigned Vessel'].value_counts()
    issues = []
    for row, route in zip(rows, schedule):
        ship = registry[row['Assigned Vessel']]
        feasible = (ship['Maintenance_Status'] != 'Critical'
            and ship['Max_Capacity'] >= route['cargo_weight_tons']
            and 0 < row['Duration (hrs)'] <= route['max_arrival_hours'] + 0.05
            and counts[row['Assigned Vessel']] == 1
            and row.get('Status') == 'OPTIMAL')
        row['Feasible'] = bool(feasible)
        if not feasible:
            row['Status'] = 'INFEASIBLE / FALLBACK'
            issues.append(route['route_id'])
    return dict(rows=rows, runtime=runtime, history=history or [], objective=objective,
        warnings=([f'Not a feasible fleet plan: check routes {", ".join(issues)}. '
                   'Fallback assignments may reuse vessels or violate constraints.'] if issues else []))


def execute(action, data):
    if not isinstance(data, dict):
        raise ValueError('Request must be a JSON object.')
    if action == 'defaults':
        return dict(fleet=[dict(Vessel=k, Ship_Type=k, **v) for k, v in backend.DEFAULT_VESSELS.items()],
                    routes=backend.DEFAULT_ROUTES, choices=CHOICES)
    if action == 'health':
        model()
        return dict(status='ok', model='verified', sklearn=version('scikit-learn'))
    if action not in {'qga', 'milp', 'fuels', 'dynamic', 'stress'}:
        raise ValueError('Unknown operation.')
    settings, qga = controls(data)
    seed = number(data.get('seed', 42), 'seed', 0, 2**32 - 1, True)
    if action in {'stress', 'dynamic'}:
        n_routes = number(data.get('n_routes', 8), 'n_routes', 2, 12, True)
        n_vessels = number(data.get('n_vessels', 8), 'n_vessels', 2, 12, True)
    else:
        schedule, registry = scenario(data)
    # Fail fast rather than queuing expensive work within one warm function.
    if not SOLVER_LOCK.acquire(blocking=False):
        raise BusyError('This worker is busy. Please retry shortly.')
    try:
        np.random.seed(seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', UserWarning)
            pre, reg, features = model()
            if action == 'dynamic':
                rng = np.random.default_rng(seed)
                schedule = [dict(distance_nm=float(d), max_arrival_hours=float(d / rng.uniform(13, 16.5)))
                            for d in rng.uniform(1200, 4500, n_routes)]
                registry = {f'VESSEL_{i:03d}': {'Base_Fuel_Rate': 0.045 + i % 5 * 0.005}
                            for i in range(n_vessels)}
                mt, mc, qt, qc, hist, variables = backend.run_dynamic_benchmark(
                    schedule, list(registry), registry, time_limit=15)
                result = dict(rows=[dict(Solver='MILP', Runtime=mt, Cost=mc),
                                    dict(Solver='QGA', Runtime=qt, Cost=qc)], history=hist,
                              variables=variables, warnings=['Synthetic benchmark: QGA uses soft berth penalties; '
                                  'its objective is not an exact optimality gap against MILP.'])
            else:
                if action == 'stress':
                    schedule, registry = backend.generate_stress_test_data(n_routes, n_vessels, seed)
                args = (schedule, registry, pre, reg, features)
                def run_qga():
                    frame, hist, elapsed = backend.optimize_fleet_schedule(*args, **settings, **qga)
                    return dispatch_result(frame, registry, schedule, elapsed, hist)
                def run_milp():
                    frame, obj, elapsed = backend.optimize_fleet_classical_milp(*args, **settings, time_limit=15)
                    result = dispatch_result(frame, registry, schedule, elapsed, objective=obj)
                    if not math.isfinite(obj):
                        result['warnings'].append('MILP did not establish an optimum within the constraints/time limit.')
                    return result
                if action == 'qga':
                    result = run_qga()
                elif action == 'milp':
                    result = run_milp()
                elif action == 'stress':
                    result = dict(qga=run_qga(), milp=run_milp(), warnings=[])
                else:
                    rows, notes = [], []
                    for fuel in backend.FUEL_SPECS:
                        fleet = {k: dict(v, Fuel_Type=fuel) for k, v in registry.items()}
                        frame, hist, elapsed = backend.optimize_fleet_schedule(schedule, fleet, pre, reg, features,
                                                                              **settings, **qga)
                        report = dispatch_result(frame, fleet, schedule, elapsed, hist)
                        notes.extend(f'{fuel}: {w}' for w in report['warnings'])
                        rows.append({'Fuel Scenario': fuel, 'Energy (kWh)': float(frame['Energy (kWh)'].sum()),
                                     'Emissions (t CO2e)': float(frame['Emissions (kg CO2e)'].sum()) / 1000,
                                     'Bunker Cost (USD)': float(frame['Bunker Cost (USD)'].sum()),
                                     'Total Cost (USD)': float(frame['Total Cost (USD)'].sum())})
                    baseline = rows[0]['Emissions (t CO2e)']
                    for row in rows:
                        row['Abatement (%)'] = 100 * (baseline - row['Emissions (t CO2e)']) / baseline if baseline else 0
                    result = dict(rows=rows, warnings=list(dict.fromkeys(notes)))
            if caught:
                result.setdefault('warnings', []).extend(sorted(set(str(w.message) for w in caught)))
            return clean(result)
    finally:
        SOLVER_LOCK.release()


def clean(value):
    """Strict JSON: SciPy's infeasible objectives can be NaN/Infinity."""
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
