# Vercel deployment

The Vercel version consists of `public/` (the web dashboard) and `api/index.py`
(the Python function at `/api`). It runs the existing model and Python optimizers
on the server. `app.py` remains the separate, local Streamlit interface.

## Deploy from Git

1. Commit the new deployment files, `app_backend.py`, **both**
   `qirf_fuel_predictor.joblib` and `model_manifest.json`, and the runtime requirements.
2. Import this repository in Vercel. Select the repository root as Root Directory
   and **Other** as Framework Preset. Use Python **3.13**, as specified in
   `.python-version`.
3. Let `vercel.json` supply the settings: no frontend build command, output
   directory `public`, and a 60-second Python function limit. Do not set the
   install command to `requirements-dev.txt` or start Streamlit in a build command.
   Vercel's Python builder installs `requirements.txt` automatically.
4. Deploy a Preview. Open `/api?action=health`; it must return HTTP 200 with
   `"model": "verified"`. This checks the bundled model's checksum, sklearn
   version, deserialization, preprocessing, and eight reference predictions.
5. Open `/` and run QGA, MILP, fuel scenarios, and the two benchmarks. Confirm
   warnings are visible for infeasible assignments. Promote the verified Preview
   to Production.

No model URL, external storage, API key, training step, or writable persistent
directory is required. Do not upload a virtual environment. The model is bundled
outside `public/`; the static frontend cannot download it.

The resolved Linux Python 3.13 wheels contain approximately 231 MiB uncompressed
before runtime overhead/bytecode. This is an estimate, not a completed Vercel build.
The dependency wheel download was verified locally; the actual Linux execution
check is supplied by the GitHub Actions workflow.

## Local cloud-app preview

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/serve.py
```

Open http://127.0.0.1:8000. This serves the same frontend and API handler as Vercel.
For a fresh environment, first run `py -3.13 -m venv .venv`.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
node --test tests/frontend.test.cjs
```

GitHub Actions repeats these checks on Ubuntu with only production Python
dependencies. Its Linux result must pass before production deployment. A local
Windows run alone does not prove Linux execution or Vercel routing.

## Model compatibility and updates

The trusted joblib artifact contains scikit-learn objects, so sklearn is pinned
to its training version, 1.9.0. Scientific packages are pinned too. The loader
resolves files relative to source code, verifies SHA-256 before deserializing,
and caches the verified model per warm worker. It does not mutate the estimator
or fit it during requests. A fresh/cold worker repeats verification.

After intentionally retraining the model in the development environment:

```powershell
.\.venv\Scripts\python.exe scripts/export_model_manifest.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Commit the model, regenerated manifest, and any corresponding dependency pins
together. Never regenerate the manifest simply to bypass a failed integrity check.
The prediction fixture verifies deployment consistency, not scientific accuracy.
The existing artifact has limited category coverage and can emit unknown-category
warnings; the API returns these warnings in results. Retraining/repairing the
underlying training data is a separate model-quality task.

## Execution limits and behavior

- Fleet analysis accepts at most 20 vessels and 20 routes. Synthetic benchmarks
  accept at most 12 of each. Generations/population are limited to 100.
- Classical solvers have a 15-second solve budget. A missing optimum is reported;
  NaN/Infinity become JSON `null` rather than breaking the frontend.
- Existing optimizer fallback assignments are checked for duplicated vessels,
  capacity, maintenance, and arrival constraints and flagged as infeasible.
- The dynamic benchmark uses its original simplified fuel equation, not the
  trained model. Its QGA and MILP formulations differ, so their costs do not
  constitute an exact optimality gap.
- NumPy RNG use is serialized per worker to avoid cross-request interference.
  A busy worker returns HTTP 429; retry after the current request completes.
- Inputs/results live in browser memory and reset on reload. Export CSV to save
  them. The app has no account or persistence layer.
- This is a public computation API. For a restricted demonstration, use Vercel
  Deployment Protection; configure platform rate limits/spending controls for
  your expected traffic before a broad public launch.

## Troubleshooting

- `503`: read the Vercel function log. Check both model files are committed, the
  Python version is 3.13, and only the production requirements are installed.
- `404` at `/api`: check Framework Preset is Other, Root Directory is the repo
  root, and `api/index.py` is included. Do not select Streamlit or a Python
  framework preset for this static-plus-function layout.
- A function-size failure usually means notebooks, `.venv`, or development
  packages were included. Keep `.vercelignore` and function exclusions in place.
- A timeout: reduce dimensions, generations, or population. Do not silently
  display an incomplete response as a successful dispatch.
- An old frontend after redeploy: reload once. Frontend files use revalidation
  headers, and this UI has no Streamlit dynamic JavaScript chunks.

References: [Python file-based functions](https://vercel.com/docs/functions/runtimes/python/api-directory),
[Python versions and bundling](https://vercel.com/docs/functions/runtimes/python),
[function limits](https://vercel.com/docs/functions/limitations).
