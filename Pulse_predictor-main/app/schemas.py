from pydantic import BaseModel
from typing import Optional


class UserCreate(BaseModel):
    name: str
    email: str
    password: str
    role: str = "VIEWER"


class UserLogin(BaseModel):
    email: str
    password: str


class ProjectCreate(BaseModel):
    name: str
    planned_cost: Optional[float] = 0
    actual_cost: Optional[float] = 0
    planned_effort: Optional[float] = 0
    actual_effort: Optional[float] = 0
    resource_count: Optional[int] = 1
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    tech_stack: Optional[str] = ""
    status: Optional[str] = "Active"
    mcc: Optional[str] = ""
    service_line: Optional[str] = ""
    segment: Optional[str] = ""
    service_offering: Optional[str] = ""
    contract_type: Optional[str] = ""
    baseline_rpp: Optional[float] = None
    latest_rpp: Optional[float] = None
    dollar_impact: Optional[float] = None
    project_margin_baseline: Optional[float] = None
    project_margin_latest: Optional[float] = None
    onsite_mix_pct: Optional[float] = None


class ProjectUpdate(ProjectCreate):
    pass


class TokenData(BaseModel):
    user_id: int
    role: str
