"""
Pydantic Request & Response Schemas for JobGuard AI FastAPI service.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class JobPostingInput(BaseModel):
    """Input payload representing a job posting to classify."""
    title: str = Field(..., description="Job Title", min_length=2)
    company_profile: Optional[str] = Field(default="", description="Company Profile / About Us")
    description: Optional[str] = Field(default="", description="Job Description")
    requirements: Optional[str] = Field(default="", description="Candidate Requirements")
    benefits: Optional[str] = Field(default="", description="Benefits Package")
    telecommuting: Optional[int] = Field(default=0, description="1 if telecommuting allowed, else 0")
    has_company_logo: Optional[int] = Field(default=1, description="1 if logo present, else 0")
    has_questions: Optional[int] = Field(default=0, description="1 if screening questions present, else 0")
    employment_type: Optional[str] = Field(default="Full-time", description="Employment type (e.g. Full-time, Contract, Part-time)")
    required_experience: Optional[str] = Field(default="Entry level", description="Required experience level")
    required_education: Optional[str] = Field(default="Bachelor's Degree", description="Required education level")
    industry: Optional[str] = Field(default="Information Technology", description="Industry domain")
    function: Optional[str] = Field(default="Engineering", description="Job functional area")
    salary_range: Optional[str] = Field(default=None, description="Salary range string (e.g., '60000-80000')")

    model_config = {
        "json_schema_extra": {
            "example": {
                "title": "Junior Python Backend Developer",
                "company_profile": "Innovatech is an established SaaS provider with 200+ employees globally.",
                "description": "Build high-throughput REST APIs and microservices using FastAPI and PostgreSQL.",
                "requirements": "Proficiency in Python 3, relational databases, git version control, and unit testing.",
                "benefits": "Competitive salary, 401(k) matching, comprehensive health coverage, flexible PTO.",
                "telecommuting": 1,
                "has_company_logo": 1,
                "has_questions": 1,
                "employment_type": "Full-time",
                "required_experience": "Entry level",
                "required_education": "Bachelor's Degree",
                "industry": "Information Technology",
                "function": "Engineering",
                "salary_range": "75000-95000",
            }
        }
    }


class PredictionResponse(BaseModel):
    """Output prediction and diagnostic reasoning."""
    fraud_probability: float = Field(..., description="Estimated probability that posting is fraudulent [0.0 - 1.0]")
    risk_level: str = Field(..., description="Categorical risk rating: 'Low Risk', 'Moderate Risk', 'High Risk'")
    is_fraudulent: bool = Field(..., description="Binary decision classification at selected operating threshold")
    decision_threshold: float = Field(..., description="Configured decision threshold used for classification")
    reasons: List[str] = Field(..., description="Human-readable plain English reasons explaining the risk rating")
    model_name: str = Field(..., description="Name of the production model artifact that generated prediction")


class ModelInfoResponse(BaseModel):
    """Model architecture metadata, training timestamp, and verified test metrics."""
    model_name: str
    selected_architecture: str
    training_date: str
    headline_metrics: Dict[str, float]
    operating_point: Dict[str, Any]
    confusion_matrix: Dict[str, int]
    class_support: Dict[str, Any]
    comparison_summary: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """System and model readiness health status."""
    status: str
    model_loaded: bool
    model_path: str
    version: str


class LoginRequest(BaseModel):
    """User credentials payload for authentication."""
    email: str = Field(..., description="Analyst or Auditor email address", min_length=3)
    password: str = Field(..., description="User password", min_length=4)


class UserProfile(BaseModel):
    """Authenticated user profile metadata."""
    id: str
    email: str
    name: str
    role: str
    organization: str = "JobGuard Security Operations"


class LoginResponse(BaseModel):
    """Authentication token and authenticated user profile."""
    access_token: str
    token_type: str = "bearer"
    user: UserProfile

