# Green Fleet Dispatch

Streamlit dashboard with a fuel-prediction model, fleet optimizers, and analysis notebooks.

For the Vercel web frontend and cloud model API, see [DEPLOYMENT.md](DEPLOYMENT.md).
Run its local preview with `.\.venv\Scripts\python.exe scripts/serve.py` at
http://127.0.0.1:8000. The instructions below are for the original Streamlit app.

## Windows / PowerShell setup

Run these commands from the project directory. Python 3.13 is used for this setup.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

The development requirements include the dashboard and all notebook dependencies.
The root `requirements.txt` is the minimal cloud runtime. Scikit-learn
is pinned to the version recorded in the supplied model artifact.

## Run the dashboard

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open http://127.0.0.1:8502. The backend runs inside Streamlit; no separate backend
server is required. Keep `qirf_fuel_predictor.joblib` in the project directory.
Use the dashboard buttons to run dispatch, benchmarks, and fuel scenarios.
Press Ctrl+C in the terminal to stop the server.

For a standalone backend check:

```powershell
.\.venv\Scripts\python.exe app_backend.py
```

## Run the notebooks

```powershell
.\.venv\Scripts\python.exe -m jupyterlab
```

Open the local URL printed by Jupyter. Run notebook cells from top to bottom:

- `EDA.ipynb`: dataset exploration, augmentation, and quantum experiments.
- `Fuel_Predictor.ipynb`: model training and export.
- `Fuel_Opitmization.ipynb`: optimization experiments (filename as supplied).

The supplied datasets and model let you launch the app without retraining.
EDA and training cells can overwrite generated datasets or the model artifact;
quantum simulations and tuning can take substantial time.

## Optional activation

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run app.py
# When finished:
deactivate
```

If PowerShell blocks activation, use the explicit `.venv` Python commands above;
activation is not required.

## Browser module-loading errors

The project uses `127.0.0.1:8502` in `.streamlit/config.toml` to give it a
dedicated browser origin, separate from earlier sessions on `localhost:8501`.
Close old dashboard tabs and use the URL above. If a browser reports
"Failed to fetch dynamically imported module", try Ctrl+Shift+R or a private
window. If only the normal window fails, clear site data for this local app
and check browser extensions that block requests. Do not clear unrelated sites.
