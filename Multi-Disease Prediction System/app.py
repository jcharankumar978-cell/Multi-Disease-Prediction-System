import os
import io
import pickle
import numpy as np
import xgboost as xgb
import shap
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

app = FastAPI(
    title="AI-Powered Multi-Disease Prediction & Explainability API",
    description="Backend service providing simultaneous risk evaluation and SHAP explanations for Diabetes, Heart Disease, Liver Disease, and Kidney Failure.",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- REQUEST AND RESPONSE SCHEMAS ---
class HealthMetricsInput(BaseModel):
    age: int = Field(..., ge=1, le=120, description="Age of the patient", example=45)
    gender: int = Field(..., description="0 for Female, 1 for Male", example=1)
    bmi: float = Field(..., ge=10.0, le=60.0, description="Body Mass Index", example=28.4)
    blood_pressure_systolic: int = Field(..., ge=70, le=220, example=130)
    blood_pressure_diastolic: int = Field(..., ge=40, le=130, example=85)
    glucose: float = Field(..., ge=50, le=400, description="Fasting Blood Glucose (mg/dL)", example=115)
    cholesterol: float = Field(..., ge=100, le=400, description="Total Cholesterol (mg/dL)", example=210)
    smoking_status: int = Field(..., description="0 = Never, 1 = Former, 2 = Current", example=0)
    physical_activity_hours: float = Field(..., ge=0, le=40, description="Exercise hours per week", example=2.5)

class DiseaseExplanation(BaseModel):
    base_value: float
    prediction_risk: float
    top_contributors: dict

class AssessmentResponse(BaseModel):
    patient_metrics: dict
    predictions: dict

# --- MOCK ML PIPELINE ENGINE FOR DEMONSTRATION ---
# In production, replace these with real loaded model binaries (.pkl/.json):
# example: model = pickle.load(open("models/diabetes_xgb.pkl", "rb"))
def mock_predict_and_explain(metrics: HealthMetricsInput, disease: str):
    # Simulated logical risk matrices based on features
    base_risks = {"diabetes": 0.15, "heart_disease": 0.20, "liver_disease": 0.10, "kidney_failure": 0.08}
    risk = base_risks[disease]
    
    # Feature influences
    contribs = {}
    if metrics.bmi > 25:
        risk += 0.12
        contribs["High BMI"] = 0.12
    if metrics.age > 40:
        risk += 0.08
        contribs["Age > 40"] = 0.08
    if metrics.glucose > 100 and disease == "diabetes":
        risk += 0.25
        contribs["Elevated Glucose"] = 0.25
    if metrics.blood_pressure_systolic > 120 and disease == "heart_disease":
        risk += 0.18
        contribs["Elevated Blood Pressure"] = 0.18
        
    risk = min(max(risk, 0.01), 0.99) # Clip between 1% and 99%
    
    return {
        "risk_percentage": round(risk * 100, 1),
        "explanation": {
            "base_value": base_risks[disease],
            "top_contributors": contribs if contribs else {"All metrics optimal": 0.0}
        }
    }

@app.post("/api/v1/assess", response_model=AssessmentResponse)
async def assess_health_risk(metrics: HealthMetricsInput):
    try:
        diseases = ["diabetes", "heart_disease", "liver_disease", "kidney_failure"]
        results = {}
        for disease in diseases:
            results[disease] = mock_predict_and_explain(metrics, disease)
            
        return {
            "patient_metrics": metrics.dict(),
            "predictions": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/report/pdf")
async def generate_pdf_report(metrics: HealthMetricsInput):
    # Generate predictive dataset internally
    diseases = ["diabetes", "heart_disease", "liver_disease", "kidney_failure"]
    results = {d: mock_predict_and_explain(metrics, d) for d in diseases}
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=letter,
        rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36,
        title="AI Health Analytics Summary"
    )
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle', parent=styles['Heading1'], fontSize=24, textColor=colors.HexColor("#1A365D"), spaceAfter=15
    )
    section_heading = ParagraphStyle(
        'SectionH', parent=styles['Heading2'], fontSize=14, textColor=colors.HexColor("#2B6CB0"), spaceBefore=12, spaceAfter=8
    )
    body_style = ParagraphStyle(
        'BodyTextCustom', parent=styles['Normal'], fontSize=10, leading=14, textColor=colors.HexColor("#2D3748")
    )
    disclaimer_style = ParagraphStyle(
        'Disclaimer', parent=styles['Normal'], fontSize=8, leading=11, textColor=colors.HexColor("#718096"), alignment=1
    )

    story = []
    
    # Title & Header
    story.append(Paragraph("AI-POWERED MULTI-DISEASE RISK ASSESSMENT REPORT", title_style))
    story.append(Paragraph("Generated by Clinical Intelligence Core Engine Architecture", body_style))
    story.append(Spacer(1, 15))
    
    # Demographics & Vitals Table
    story.append(Paragraph("1. Patient Baseline Metrics & Vitals", section_heading))
    vitals_data = [
        [Paragraph("<b>Metric</b>", body_style), Paragraph("<b>Value Submitted</b>", body_style), Paragraph("<b>Metric</b>", body_style), Paragraph("<b>Value Submitted</b>", body_style)],
        ["Age", f"{metrics.age} years", "Gender", "Male" if metrics.gender == 1 else "Female"],
        ["BMI", f"{metrics.bmi} kg/m²", "Blood Pressure", f"{metrics.blood_pressure_systolic}/{metrics.blood_pressure_diastolic} mmHg"],
        ["Fasting Glucose", f"{metrics.glucose} mg/dL", "Total Cholesterol", f"{metrics.cholesterol} mg/dL"],
        ["Smoking Status", "Current" if metrics.smoking_status==2 else "Former" if metrics.smoking_status==1 else "Never", "Physical Activity", f"{metrics.physical_activity_hours} hrs/wk"]
    ]
    t_vitals = Table(vitals_data, colWidths=[130, 130, 130, 130])
    t_vitals.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#E2E8F0")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('PADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_vitals)
    story.append(Spacer(1, 15))
    
    # Predictions Table
    story.append(Paragraph("2. Consolidated Disease Risk Matrix", section_heading))
    matrix_data = [
        [Paragraph("<b>Target Disease</b>", body_style), Paragraph("<b>Calculated Risk</b>", body_style), Paragraph("<b>Primary Risk Drivers (SHAP Value Shifts)</b>", body_style)]
    ]
    for d, data in results.items():
        disease_name = d.replace("_", " ").title()
        risk_str = f"{data['risk_percentage']}%"
        drivers = ", ".join([f"{k} (+{v*100:.1f}%)" if v>0 else f"{k} ({v*100:.1f}%)" for k, v in data['explanation']['top_contributors'].items()])
        matrix_data.append([disease_name, risk_str, drivers])
        
    t_matrix = Table(matrix_data, colWidths=[120, 100, 320])
    t_matrix.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2B6CB0")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('PADDING', (0,0), (-1,-1), 8),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    # Quick fix for text colors inside table headers
    for i in range(3):
        matrix_data[0][i].style.textColor = colors.white
    story.append(t_matrix)
    story.append(Spacer(1, 15))
    
    # Lifestyle Recommendations
    story.append(Paragraph("3. AI-Driven Preventative Health Strategies", section_heading))
    recs = []
    if metrics.bmi > 25:
        recs.append("• <b>Weight Management:</b> Your BMI falls in an elevated range. Integrating dietary modifications and calorie deficit planning can scale down overall cardiac stress metrics.")
    if metrics.glucose > 100:
        recs.append("• <b>Glycemic Control:</b> Fasting blood sugar shows pre-diabetic or elevated patterns. Restructure carbohydrate loading cycles and track postprandial glucose levels.")
    if metrics.physical_activity_hours < 2.5:
        recs.append("• <b>Cardio Acceleration:</b> Elevate metabolic output to minimum 150 minutes of zone-2 aerobic activity per week to optimize arterial elasticity and lower multi-disease indices.")
    if not recs:
        recs.append("• All metric categories are sitting within healthy equilibrium baselines. Keep maintaining structural dietary and physical patterns.")
        
    rec_text = "<br/><br/>".join(recs)
    story.append(Paragraph(rec_text, body_style))
    story.append(Spacer(1, 40))
    
    # Legal Disclaimer
    story.append(Paragraph("<b>REGULATORY & MEDICAL DISCLAIMER:</b> This is for informational purposes only. For medical advice or diagnosis, consult a professional. AI responses may include mistakes. This automated layout does not replace an official in-person physician's biochemical clinical analysis.", disclaimer_style))
    
    doc.build(story)
    buffer.seek(0)
    
    return Response(
        content=buffer.getvalue(),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=health_summary_report.pdf"}
    )
