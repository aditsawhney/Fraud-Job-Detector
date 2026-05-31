"""
app.py - FraudScan FastAPI backend.

POST /predict  - runs the ML model on a job posting
GET  /health   - Render health check

Run locally from project root:
    uvicorn backend.app:app --reload
"""

import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))
from model.features import CombinedFeatureExtractor  # noqa: F401 - needed for pickle

MODEL_PATH    = BASE_DIR / "model" / "model.pkl"
FEATURES_PATH = BASE_DIR / "model" / "top_features.json"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    raise RuntimeError(f"model.pkl not found at {MODEL_PATH}. Run model/train.py first.")

with open(FEATURES_PATH) as f:
    top_features_data = json.load(f)

app = FastAPI(
    title="FraudScan API",
    description="Detects fraudulent job postings targeting the Indian job market.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class JobPosting(BaseModel):
    title: str = ""
    company: str = ""
    location: str = ""
    salary: str = ""
    contact: str = ""
    description: str = ""


class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    risk_level: str
    reasons: list[str]
    top_model_features: list[dict]


SIGNAL_EXPLANATIONS = {
    "fee_language":         "Mentions registration, processing, or training fees",
    "upfront_payment":      "Asks for upfront payment before joining",
    "unrealistic_salary":   "Promises unrealistically high daily earnings",
    "guaranteed_job":       "Claims 100% job placement guarantee",
    "whatsapp_only":        "Contact is WhatsApp-only (no official channel)",
    "gmail_contact":        "Uses personal Gmail/Yahoo email instead of company domain",
    "personal_phone":       "Provides personal phone number as primary contact",
    "aadhaar_request":      "Requests Aadhaar card details early in the process",
    "pan_request":          "Requests PAN card details early in the process",
    "bank_details_early":   "Asks for bank account details before joining",
    "mnc_impersonation":    "Claims to be hiring for a major MNC (TCS, Infosys, etc.)",
    "background_check_fee": "Charges a fee for background verification",
    "overseas_visa_fee":    "Overseas job offer with visa/processing fee",
    "simple_task_scam":     "Offers money for simple tasks (like, share, click)",
    "vague_company":        "Company name is suspiciously short or vague",
}

PATTERNS = {
    "fee_language":         r"registration fee|processing fee|consultancy fee|security deposit|refundable deposit|training fee|joining fee",
    "upfront_payment":      r"pay.{0,20}(first|advance|before joining|before training)|deposit.{0,20}refund",
    "unrealistic_salary":   r"₹\s*[5-9]\d{3,}[\s/]*(day|daily)|earn.{0,15}₹\s*[5-9]\d{3,}|daily earning|per day earning",
    "guaranteed_job":       r"100%\s*(job\s*)?(guarantee|guaranteed)|placement guarantee|guaranteed placement|assured job",
    "whatsapp_only":        r"whatsapp\s*(only|me|us|number|contact)|contact.{0,20}whatsapp",
    "gmail_contact":        r"[\w.\-]+@gmail\.com|[\w.\-]+@yahoo\.com|[\w.\-]+@hotmail\.com",
    "personal_phone":       r"(call|contact|whatsapp).{0,20}\+?91[-\s]?[6-9]\d{9}",
    "aadhaar_request":      r"aadhaar|aadhar|uid number",
    "pan_request":          r"\bpan\s*(card|number|details)\b",
    "bank_details_early":   r"bank\s*(account|details|statement).{0,40}(apply|before|joining|interview|send)",
    "mnc_impersonation":    r"\b(tcs|infosys|wipro|accenture|amazon|flipkart|hcl|tech mahindra)\b.{0,60}(hiring|recruit|opening|vacancy)",
    "background_check_fee": r"background\s*(check|verification).{0,30}(fee|charge|pay|₹)",
    "overseas_visa_fee":    r"(dubai|singapore|canada|uk|abroad|overseas).{0,60}(visa|processing).{0,30}(fee|charge|₹)",
    "simple_task_scam":     r"(like|share|follow|click).{0,20}(earn|money|₹|income)|work from (home|mobile).{0,30}(₹|earn|income).{0,20}(day|daily|hour)",
    "vague_company":        r"^.{0,30}$",
}


def get_fired_signals(job: JobPosting) -> list[str]:
    full_text = f"{job.title} {job.company} {job.description}".lower()
    company = job.company.lower().strip()
    reasons = []
    for name, pat in PATTERNS.items():
        if name == "vague_company":
            if re.match(pat, company):
                reasons.append(SIGNAL_EXPLANATIONS[name])
        else:
            if re.search(pat, full_text, re.IGNORECASE):
                reasons.append(SIGNAL_EXPLANATIONS[name])
    return reasons


def confidence_to_risk(confidence: float, prediction: int) -> str:
    if prediction == 0:
        return "LOW"
    if confidence >= 0.80:
        return "HIGH"
    if confidence >= 0.55:
        return "MEDIUM"
    return "LOW"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(job: JobPosting):
    full_text = f"{job.title} {job.company} {job.description}".strip()

    input_df = pd.DataFrame([{
        "title":     job.title,
        "company":   job.company,
        "location":  job.location,
        "salary":    job.salary,
        "contact":   job.contact,
        "full_text": full_text,
    }])

    try:
        proba = model.predict_proba(input_df)[0]
        prediction = int(np.argmax(proba))
        confidence = float(proba[prediction])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    return PredictionResponse(
        prediction=prediction,
        confidence=round(confidence, 4),
        risk_level=confidence_to_risk(confidence, prediction),
        reasons=get_fired_signals(job),
        top_model_features=[
            f for f in top_features_data["top_fraud_signals"]
            if f["feature"].startswith("indian_")
        ][:5],
    )