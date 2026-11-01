from pydantic import BaseModel, Field


class StartRequest(BaseModel):
    dataset_id: str = 'demo'
    budget: int = Field(60, ge=1, le=50000)
    review_cost: float = Field(15.0, ge=0, le=1000000)


class EvaluationRequest(BaseModel):
    job_id: str
    model: str
    threshold: float | None = None
    budget: int | None = Field(None, ge=1)
    review_cost: float | None = Field(None, ge=0)


class ExplainRequest(BaseModel):
    job_id: str
    model: str
    observation: int = Field(0, ge=0)
    global_count: int = Field(8, ge=1, le=18)
    background_size: int = Field(12, ge=5, le=25)


class PermutationRequest(BaseModel):
    job_id: str
    model: str
    sample_size: int = Field(200, ge=30, le=800)
    repeats: int = Field(3, ge=2, le=8)


class PredictRequest(BaseModel):
    job_id: str
    model: str
    features: dict
