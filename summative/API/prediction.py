"""FastAPI service for newborn birth weight prediction.

Run locally:   uvicorn prediction:app --reload      (from summative/API)
Swagger UI:    /docs
Routes:        POST /predict   POST /retrain   GET /docs   (plus GET / as a health check)
"""

import os
import threading
from typing import Literal

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator

from retrain import (
    CATEGORIES,
    FEATURES,
    MAX_UPLOAD_BYTES,
    RANGES,
    RetrainError,
    canonical_category,
    load_pipeline,
    retrain,
)

app = FastAPI(
    title="Newborn Birth Weight Prediction API",
    description=(
        "Predicts a newborn's birth weight (kg) from maternity register fields. "
        "Use /predict for a single delivery and /retrain to upload new rows as a CSV."
    ),
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# CORS
# What is allowed: only the browser origins listed below (the deployed Render URL and the
#   local Flutter web / dev server addresses), only GET and POST (the methods this API uses),
#   and only the Content-Type, Accept and Authorization request headers.
# What is restricted: every other website. A page on a random domain cannot call this API
#   from a browser. There is no wildcard (*) for origins, methods or headers.
# Credentials are off (allow_credentials=False) because the API uses no cookies or login,
#   so there is nothing for a website to send on a user's behalf. Keeping it off also
#   means a mistake in the origin list cannot leak a logged-in session.
# Why: a wildcard origin lets any site read responses from the API in a user's browser.
#   Listing exact origins keeps the surface small. A native Flutter mobile app is not a
#   browser, so CORS does not apply to it: this setting only controls which WEBSITES may call us.
# ---------------------------------------------------------------------------
ALLOWED_ORIGINS = [
    "http://localhost:3000",      # local web front end
    "http://127.0.0.1:3000",
    "http://localhost:8080",      # flutter run -d chrome --web-port 8080
    "http://127.0.0.1:8080",
    "http://localhost:8000",      # Swagger UI served by uvicorn locally
    "http://127.0.0.1:8000",
]
# Render sets RENDER_EXTERNAL_URL (https://your-service.onrender.com) on every web service,
# so the deployed Swagger UI is allowed without hard-coding the name.
if os.getenv("RENDER_EXTERNAL_URL"):
    ALLOWED_ORIGINS.append(os.environ["RENDER_EXTERNAL_URL"].rstrip("/"))
# Optional extra origins, comma separated, e.g. a hosted web build of the app
ALLOWED_ORIGINS += [o.strip().rstrip("/") for o in os.getenv("EXTRA_ALLOWED_ORIGINS", "").split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept", "Authorization"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
def _literal(column: str):
    return Literal[tuple(CATEGORIES[column])]


class BirthInput(BaseModel):
    """One delivery, exactly the 9 columns the model was trained on. Every field is required."""

    model_config = ConfigDict(extra="forbid")

    ga_weeks: float = Field(..., ge=RANGES["ga_weeks"][0], le=RANGES["ga_weeks"][1],
                            description="Gestational age in weeks (20 to 44)", examples=[38])
    mothers_age: int = Field(..., ge=RANGES["mothers_age"][0], le=RANGES["mothers_age"][1],
                             description="Mother's age in years (12 to 55)", examples=[27])
    multiple: int = Field(..., ge=RANGES["multiple"][0], le=RANGES["multiple"][1],
                          description="Plurality: 1 = singleton, 2 = twins, 3 = triplets ...", examples=[1])
    sex: _literal("sex") = Field(..., description="Baby's sex", examples=["Male"])
    country: _literal("country") = Field(..., description="Country of the facility", examples=["Uganda"])
    facility: _literal("facility") = Field(..., description="Facility where the delivery was recorded",
                                           examples=["Kamuli_Hospital"])
    mode_of_delivery: _literal("mode_of_delivery") = Field(..., description="Mode of delivery",
                                                           examples=["Normal_Vaginal_Delivery"])
    viability: _literal("viability") = Field(..., description="Viability recorded in the register",
                                             examples=["No Indication"])
    facility_birth: _literal("facility_birth") = Field(..., description="Facility birth or born before arrival (BBA)",
                                                       examples=["Facility birth"])

    # Accept 'male', ' Kamuli hospital ' and so on, but only if it maps to a known category.
    # Non-strings and unknown values are passed through unchanged so Pydantic rejects them.
    @field_validator("sex", "country", "facility", "mode_of_delivery", "viability", "facility_birth",
                     mode="before")
    @classmethod
    def _normalise_category(cls, value, info):
        canon = canonical_category(info.field_name, value)
        return canon if canon is not None else value


class PredictionOut(BaseModel):
    predicted_birth_weight_kg: float
    predicted_birth_weight_g: int
    low_birth_weight: bool = Field(description="True if the predicted weight is below 2.5 kg")
    model_version: int = Field(description="Starts at 1, goes up by 1 after every successful /retrain")


class RetrainOut(BaseModel):
    message: str
    model_version: int
    report: dict


# ---------------------------------------------------------------------------
# Model state. The pipeline is loaded once at startup from the repository copy.
# /retrain swaps the reference, so /predict uses the new model with no redeploy.
# ---------------------------------------------------------------------------
class _State:
    pipeline = load_pipeline()
    version = 1


state = _State()
_retrain_lock = threading.Lock()  # one retrain at a time


@app.get("/", include_in_schema=False)
def health():
    return {"status": "ok", "docs": "/docs", "model_version": state.version}


@app.post("/predict", response_model=PredictionOut, summary="Predict birth weight for one delivery")
def predict(body: BirthInput):
    row = pd.DataFrame([body.model_dump()])[FEATURES]
    kg = float(state.pipeline.predict(row)[0])
    return PredictionOut(
        predicted_birth_weight_kg=round(kg, 3),
        predicted_birth_weight_g=int(round(kg * 1000)),
        low_birth_weight=kg < 2.5,
        model_version=state.version,
    )


@app.post("/retrain", response_model=RetrainOut, summary="Upload a CSV of new deliveries and retrain the model")
def retrain_endpoint(file: UploadFile = File(..., description="CSV with the 9 feature columns plus birth_weight_kg")):
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")

    data = file.file.read(MAX_UPLOAD_BYTES + 1)

    if not _retrain_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="A retrain is already running. Try again in a moment.")
    try:
        new_pipeline, report = retrain(data, state.pipeline)
        # Swap in the new model. Requests already running finish on the old one, the next ones use the new one.
        state.pipeline = new_pipeline
        state.version += 1
    except RetrainError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    finally:
        _retrain_lock.release()

    return RetrainOut(
        message="Model retrained on original data + uploaded rows and is now live.",
        model_version=state.version,
        report=report,
    )
