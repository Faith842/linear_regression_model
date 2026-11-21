"""Schema, model loading and retraining logic for the birth weight API.

prediction.py imports from this file (never the other way round), so there is
no circular import. Nothing here knows about FastAPI routes.

Retraining rule (from the assignment):
  - The saved best model is a RandomForestRegressor, not an SGDRegressor, so we
    cannot continue it with partial_fit. Instead we refit a fresh copy of the
    same pipeline (same hyperparameters) on ORIGINAL data + NEW rows.
"""

import io
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_squared_error

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.pkl"
TRAINING_DATA_PATH = BASE_DIR / "training_data.csv"

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB is plenty for a register extract

# ---------------------------------------------------------------------------
# Schema: the single source of truth for column names, allowed values, ranges.
# Order matters, it is the column order the saved pipeline was trained with.
# ---------------------------------------------------------------------------
FEATURES_NUM = ["ga_weeks", "mothers_age", "multiple"]
FEATURES_CAT = ["sex", "country", "facility", "mode_of_delivery", "viability", "facility_birth"]
FEATURES = FEATURES_NUM + FEATURES_CAT
TARGET = "birth_weight_kg"

# Realistic ranges (also enforced in the Pydantic request model)
RANGES = {
    "ga_weeks": (20, 44),       # weeks of gestation
    "mothers_age": (12, 55),    # years
    "multiple": (1, 5),         # 1 = singleton, 2 = twins, ...
}
TARGET_RANGE = (0.5, 6.0)       # kg, same cut used in the notebook

CATEGORIES = {
    "sex": ["Male", "Female", "Not_indicated"],
    "country": ["Uganda", "Kenya"],
    "facility": [
        "Iganga_Hospital", "Jinja_Regional_Ref_Hospital", "Bugiri_Hospital",
        "Kamuli_Mission_Hospital", "Migori_County_Referral_Hospital", "Kamuli_Hospital",
        "Buluba_Hospital", "Rongo_Sub-County_Hospital", "Isebania_Sub-County_Hospital",
        "Ntimaru_Sub-County_Hospital", "Kehancha_Sub-County_Hospital",
        "St_Joseph-Ombo_Mission_Hospital", "Awendo_Sub-County_Hospital",
        "Uriri_Health_Center", "Macalder_Sub-County_Hospital", "Tisinye_Health_Center",
        "Kegonga_Sub-County_Hospital", "Karungu_Sub-County_Hospital",
        "Bugumbe_Health_Center", "Othoro_Sub-County_Hospital", "Muhuru_Health_Center",
        "Nyamaraga_Sub-County_Hospital", "St_Camillus_Mission_Hospital",
    ],
    "mode_of_delivery": ["Normal_Vaginal_Delivery", "Caesarean_Section", "Born_Before_Arrival", "Other"],
    "viability": ["No Indication", "IUFD", "Abortion"],
    "facility_birth": ["Facility birth", "BBA"],
}


def _key(text: str) -> str:
    """Lower-case and treat spaces/underscores as the same, so 'kamuli hospital' == 'Kamuli_Hospital'."""
    return str(text).strip().lower().replace(" ", "_")


_LOOKUP = {col: {_key(v): v for v in values} for col, values in CATEGORIES.items()}


def canonical_category(column: str, value):
    """Return the canonical spelling of a category, or None if it is not a known value."""
    if not isinstance(value, str):
        return None
    return _LOOKUP[column].get(_key(value))


# ---------------------------------------------------------------------------
# Model I/O
# ---------------------------------------------------------------------------
def load_pipeline(path: Path = MODEL_PATH):
    """Load the saved sklearn Pipeline (preprocessing + best model)."""
    return joblib.load(path)


def _atomic_dump(obj, path: Path):
    """Write to a temp file in the same folder, then swap it in, so a crash never leaves a half-written model."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        joblib.dump(obj, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _atomic_csv(df: pd.DataFrame, path: Path):
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# ---------------------------------------------------------------------------
# Upload validation
# ---------------------------------------------------------------------------
class RetrainError(ValueError):
    """Raised for a bad upload. The API turns it into a 4xx response."""


def validate_upload(csv_bytes: bytes):
    """Parse and clean an uploaded CSV.

    Returns (clean_df, n_received, rejection_reasons).
    Rules:
      - all 9 feature columns and the target column must exist
      - numeric columns must be numbers inside the allowed range
        (ga_weeks and mothers_age may be blank, the pipeline imputes them like in training)
      - `multiple` and the target are required
      - categorical columns must use known values (blank is allowed, imputed in the pipeline)
      - a row that breaks a rule is rejected, the rest are kept
    """
    if len(csv_bytes) == 0:
        raise RetrainError("The uploaded file is empty.")
    if len(csv_bytes) > MAX_UPLOAD_BYTES:
        raise RetrainError(f"File is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")

    try:
        raw = pd.read_csv(io.BytesIO(csv_bytes))
    except Exception as exc:  # malformed CSV, wrong encoding, not a CSV at all
        raise RetrainError(f"Could not read the file as CSV: {exc}")

    raw.columns = [str(c).strip() for c in raw.columns]
    missing_cols = [c for c in FEATURES + [TARGET] if c not in raw.columns]
    if missing_cols:
        raise RetrainError(f"Missing required column(s): {', '.join(missing_cols)}")
    if raw.empty:
        raise RetrainError("The CSV has the right columns but no rows.")

    n_received = len(raw)
    df = raw[FEATURES + [TARGET]].copy()
    bad = pd.Series(False, index=df.index)
    reasons = {}

    def flag(mask, why):
        nonlocal bad
        if mask.sum():
            reasons[why] = reasons.get(why, 0) + int(mask.sum())
            bad = bad | mask

    # numeric features
    for col in FEATURES_NUM:
        original = df[col]
        num = pd.to_numeric(original, errors="coerce")
        lo, hi = RANGES[col]
        flag(original.notna() & num.isna(), f"{col}: not a number")
        flag(num.notna() & ~num.between(lo, hi), f"{col}: outside {lo} to {hi}")
        if col == "multiple":
            flag(original.isna(), "multiple: missing")
        df[col] = num

    # target
    target_raw = df[TARGET]
    target_num = pd.to_numeric(target_raw, errors="coerce")
    lo, hi = TARGET_RANGE
    flag(target_raw.isna(), f"{TARGET}: missing")
    flag(target_raw.notna() & target_num.isna(), f"{TARGET}: not a number")
    flag(target_num.notna() & ~target_num.between(lo, hi), f"{TARGET}: outside {lo} to {hi} kg")
    df[TARGET] = target_num

    # categorical features
    for col in FEATURES_CAT:
        original = df[col].astype("object")
        canon = original.map(lambda v: canonical_category(col, v) if pd.notna(v) else None)
        flag(original.notna() & canon.isna(), f"{col}: unknown value")
        df[col] = canon

    clean = df[~bad].reset_index(drop=True)
    return clean, n_received, reasons


# ---------------------------------------------------------------------------
# Retraining
# ---------------------------------------------------------------------------
def retrain(csv_bytes: bytes, current_pipeline, model_path: Path = MODEL_PATH,
            data_path: Path = TRAINING_DATA_PATH):
    """Validate the upload, refit on original + new rows, save, and return (new_pipeline, report)."""
    new_rows, n_received, reasons = validate_upload(csv_bytes)
    if new_rows.empty:
        raise RetrainError(f"No usable rows in the upload. Reasons: {reasons}")

    original = pd.read_csv(data_path)
    combined = pd.concat([original[FEATURES + [TARGET]], new_rows], ignore_index=True)

    # Score the current model on the new rows before retraining, to see how it does on unseen data
    old_mse = float(mean_squared_error(new_rows[TARGET], current_pipeline.predict(new_rows[FEATURES])))

    # Fresh copy of the same pipeline (same hyperparameters), refit on everything
    new_pipeline = clone(current_pipeline)
    new_pipeline.fit(combined[FEATURES], combined[TARGET])

    # Persist, so the next restart in the same container keeps the update
    _atomic_dump(new_pipeline, model_path)
    _atomic_csv(combined, data_path)

    report = {
        "rows_received": n_received,
        "rows_used": int(len(new_rows)),
        "rows_rejected": int(n_received - len(new_rows)),
        "rejection_reasons": reasons,
        "training_rows_before": int(len(original)),
        "training_rows_after": int(len(combined)),
        "old_model_mse_on_new_rows": round(old_mse, 4),
        "retrained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "strategy": "refit on original data + new rows (RandomForest has no partial_fit)",
    }
    return new_pipeline, report
