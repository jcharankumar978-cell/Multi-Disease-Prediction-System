"""FastAPI inference service for the single-disease diabetes XGBoost model.
Run the notebook first to create diabetes_xgb_model.joblib.
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).with_name("diabetes_xgb_model.joblib")
app = FastAPI(title="Diabetes Risk Prediction API", version="1.0.0")
FEATURES = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigree","Age"]

class DiabetesInput(BaseModel):
    Pregnancies: int = Field(..., ge=0, le=20)
    Glucose: float = Field(..., ge=0, le=400)
    BloodPressure: float = Field(..., ge=0, le=200)
    SkinThickness: float = Field(..., ge=0, le=100)
    Insulin: float = Field(..., ge=0, le=1000)
    BMI: float = Field(..., ge=0, le=80)
    DiabetesPedigree: float = Field(..., ge=0, le=3)
    Age: int = Field(..., ge=1, le=120)

def get_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Run the notebook first to create diabetes_xgb_model.joblib.")
    return joblib.load(MODEL_PATH)

@app.get("/health")
def health():
    return {"status":"ok","model_loaded":MODEL_PATH.exists(),"target":"diabetes"}

@app.post("/api/v1/predict")
def predict(payload: DiabetesInput):
    try:
        bundle = get_bundle()
        row = pd.DataFrame([payload.model_dump() if hasattr(payload,"model_dump") else payload.dict()], columns=FEATURES)
        model = bundle["model"]
        probability = float(model.predict_proba(row)[0,1])
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(row)
        if isinstance(sv,list): values = np.asarray(sv[-1])[0]
        elif np.asarray(sv).ndim == 3: values = np.asarray(sv)[0,:,1]
        else: values = np.asarray(sv)[0]
        contributions = {name:float(value) for name,value in zip(FEATURES,values)}
        return {"condition":"diabetes","probability":probability,"probability_percent":round(probability*100,2),"predicted_class":int(probability>=0.5),"shap_contributions":contributions,"disclaimer":"Educational model only; not a medical diagnosis."}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
