from pydantic import BaseModel, Field
from typing import List, Optional

class PredictIn(BaseModel):
    age: int = Field(..., ge=18, le=85)
    sex: str = Field(..., pattern="^(F|M)$")
    height_cm: float = Field(..., ge=120, le=220)
    weight_kg: float = Field(..., ge=30, le=220)
    waist_cm: float = Field(..., ge=40, le=170)
    sleep_hours: Optional[float] = Field(None, ge=3, le=14)
    smokes_cig_day: Optional[int] = Field(0, ge=0, le=60)
    days_mvpa_week: Optional[int] = Field(0, ge=0, le=7)
    fruit_veg_portions_day: Optional[float] = Field(0, ge=0, le=12)

class Driver(BaseModel):
    feature: str
    contribution: float

class PredictOut(BaseModel):
    score: float
    calibrated: float
    drivers: List[Driver]
    referral_flag: bool

class CoachIn(PredictIn):
    score: float

class Citation(BaseModel):
    kb_doc: str
    section: str

class CoachOut(BaseModel):
    plan_text: str
    steps: List[str]
    citations: List[Citation]
    disclaimer: str
