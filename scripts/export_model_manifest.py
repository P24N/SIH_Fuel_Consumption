"""Run after deliberately retraining the trusted model; commit both artifacts."""
import hashlib
import json
from pathlib import Path
import sys
from importlib.metadata import version

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from app_backend import load_digital_twin

if __name__ == '__main__':
    pre, model, features = load_digital_twin()
    sample = pd.read_csv(ROOT / 'datasets/augmented_green_fleet_dataset.csv')[features].head(8)
    payload = dict(sha256=hashlib.sha256((ROOT / 'qirf_fuel_predictor.joblib').read_bytes()).hexdigest(),
                   scikit_learn=version('scikit-learn'), features=list(features),
                   smoke_inputs=json.loads(sample.to_json(orient='records')),
                   smoke_predictions=model.predict(pre.transform(sample)).tolist())
    (ROOT / 'model_manifest.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    print('Exported model checksum and 8 reference predictions.')
