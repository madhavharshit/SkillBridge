from pydantic import BaseModel, Field
from typing import Any, Literal

class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    source_type: str | None = None
    response_language: Literal["en", "ja"] = "en"

class ProcedureStep(BaseModel):
    instruction: str
    timestamp: str | None = None
    source_ref: str | None = None
    evidence_type: Literal["source_supported", "visual_observation", "expert_confirmed", "approved_sop_reference", "ai_inferred", "verification_required"] = "verification_required"

class ProcedureUpdate(BaseModel):
    title: str
    description: str = ""
    tools: list[str] = []
    warnings: list[str] = []
    steps: list[ProcedureStep] = []

class ReviewRequest(BaseModel):
    action: str
    note: str = ""

class ProgressRequest(BaseModel):
    procedure_id: int
    trainee_name: str = "Demo Trainee"
    completed_steps: int = 0
    total_steps: int = 0
    quiz_score: int = 0
    completed: bool = False

class DiscoveryAnswerRequest(BaseModel):
    answer: str = ""
    status: Literal["answered", "uncertain", "not_applicable", "unanswered"] = "answered"
    correction: str = ""

class DecisionKnowledgeRequest(BaseModel):
    question_ids: list[int]
    context: str = ""
    observed_cue: str = ""
    interpretation: str = ""
    decision: str = ""
    action: str = ""
    outcome: str = ""
    exceptions: str = ""
    limits: str = ""

class DecisionKnowledgeUpdate(BaseModel):
    context: str = ""
    observed_cue: str = ""
    interpretation: str = ""
    decision: str = ""
    action: str = ""
    outcome: str = ""
    exceptions: str = ""
    limits: str = ""

class VideoExpertAnswerRequest(BaseModel):
    answer: str = Field(default="", max_length=12000)
    status: Literal["pending", "answered", "dismissed"] = "answered"
    dismissal_reason: str = Field(default="", max_length=2000)
