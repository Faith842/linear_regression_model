# Newborn Birth Weight Prediction

**Mission and problem**
Reduce preventable maternal and newborn deaths by helping health facilities spot high-risk births early.
Problem: estimate a newborn's birth weight (kg) from fields on the maternity register, so midwives can flag likely low birth weight (< 2.5 kg) babies
and data teams can catch data-entry errors (for example a 10 kg baby) in the register.

## Dataset

De-identified maternity register data from 23 facilities in Uganda (Busoga region) and Kenya (Migori county), collected by the East Africa Preterm Birth Initiative (PTBi-EA), March to October 2016.
One row is one delivery. After cleaning: 17,180 rows, 9 input features (3 numeric, 6 categorical), target is birth weight in kg.

Source: Dryad, "East Africa preterm birth initiative birth register data (March 2016 - October 2016)"
https://datadryad.org/dataset/doi:10.7272/Q6833Q63

## Public Swagger UI (the API is tested here)

https://birthweight-api.onrender.com/docs

The service runs on the Render free plan and sleeps after 15 minutes without traffic, so the first request can take up to a minute.

## YouTube video demo

`<PASTE YOUR YOUTUBE LINK>`

## Run the mobile app

You need the Flutter SDK (3.22 or newer) and an Android emulator, an iOS simulator or a phone.

1. Clone the repo and go to the app folder:
   ```
   git clone <your-repo-url>
   cd linear_regression_model/summative/FlutterApp
   ```
2. Generate the platform folders (android, ios). This keeps `lib/main.dart` and `pubspec.yaml` as they are:
   ```
   flutter create --project-name birthweight_app --platforms=android,ios .
   ```
   If it ever overwrites `lib/main.dart`, restore it with `git checkout lib/main.dart`.
3. Install packages:
   ```
   flutter pub get
   ```
4. The app already points to the deployed API (`https://birthweight-api.onrender.com`, set as the default `API_BASE_URL` in `lib/main.dart`). You can pass a different address with `--dart-define` when you run, see step 6.
5. Start an emulator (or plug in a phone with USB debugging) and check it shows up:
   ```
   flutter devices
   ```
6. Run the app:
   ```
   flutter run
   ```
   To use an API running on your own computer from the Android emulator:
   ```
   uvicorn prediction:app --host 0.0.0.0 --port 8000      (in summative/API)
   flutter run --dart-define=API_BASE_URL=http://10.0.2.2:8000
   ```
   Plain http needs `android:usesCleartextTraffic="true"` on the `<application>` tag in `android/app/src/main/AndroidManifest.xml`. The Render URL is https, so it does not need this.
7. Fill in the 9 fields and tap **Predict**. Try a bad value (for example gestational age 60, or leave a field empty) to see the error message.

To build a release APK, add `<uses-permission android:name="android.permission.INTERNET"/>` to `android/app/src/main/AndroidManifest.xml` first (debug builds already have it).

---

## Repository layout

```
linear_regression_model/
├── README.md
└── summative/
    ├── pyproject.toml, uv.lock           notebook environment
    ├── linear_regression/
    │   ├── multivariate.ipynb            data prep, 4 models, comparison, saved model
    │   └── maternity_register.xlsx       raw data used by the notebook
    ├── API/
    │   ├── prediction.py                 FastAPI app: /predict, /retrain, /docs, CORS
    │   ├── retrain.py                    schema, upload validation, retraining
    │   ├── best_model.pkl                saved pipeline (preprocessing + RandomForest)
    │   ├── training_data.csv             original training rows, used by /retrain
    │   ├── pyproject.toml, uv.lock, requirements.txt
    └── FlutterApp/                       one-page Flutter app
```

## Model results (same 20% test set, MSE in kg²)

| Model | Test MSE |
|---|---|
| RandomForestRegressor (best, saved) | 0.2552 |
| LinearRegression (closed form) | 0.2628 |
| SGDRegressor (gradient descent) | 0.2635 |
| DecisionTreeRegressor | 0.2675 |

Best model = lowest test MSE, about 505 g RMSE.

## Run the API locally

```
cd summative/API
uv sync
uv run uvicorn prediction:app --reload
```
Open http://127.0.0.1:8000/docs

scikit-learn is pinned to 1.6.1 because `best_model.pkl` was saved with that version. A different version can fail to load the pickle.

### Endpoints

| Route | Method | What it does |
|---|---|---|
| `/predict` | POST | JSON with 9 fields, returns predicted weight in kg and g, a low birth weight flag and the model version. Wrong type, missing or out-of-range values return a 422 error. |
| `/retrain` | POST | Upload a CSV (9 feature columns + `birth_weight_kg`). Rows are validated, the model is refit on original data + new rows and used by `/predict` straight away, no redeploy. |
| `/docs` | GET | Swagger UI |

Retraining details: a RandomForest has no `partial_fit`, so the pipeline is refit on original + new rows with the same hyperparameters. The response reports rows used and rejected (with reasons) and how the old model scored on the new rows. `model_version` goes up by 1 on every successful retrain.
On the Render free plan the retrained model is lost when the service restarts or sleeps, and it falls back to the `best_model.pkl` in the repo. This is accepted by the assignment.

## Deploy on Render

1. Push the repo to GitHub.
2. Render dashboard, New, Web Service, connect the repo.
3. Settings:
   - Root Directory: `summative/API`
   - Build Command: `uv sync --locked`
   - Start Command: `uv run uvicorn prediction:app --host 0.0.0.0 --port $PORT`
   - Instance type: Free
   - Environment variable: `PYTHON_VERSION` = `3.12.8`
4. After the deploy, open https://birthweight-api.onrender.com/docs

To refresh `requirements.txt` after changing dependencies:
```
cd summative/API
uv lock
uv export --format requirements-txt --no-hashes --no-emit-project -o requirements.txt
```
