from typing import List, Optional

from pydantic import BaseModel, Field


class NodeInput(BaseModel):
    id: str = Field(..., description="Unique transaction identifier")
    features: List[float] = Field(
        ..., description="RAW (unscaled) Elliptic feature vector, length 165. "
        "The API applies the same StandardScaler used at training time."
    )


class EdgeInput(BaseModel):
    source: str
    target: str


class SubgraphRequest(BaseModel):
    nodes: List[NodeInput]
    edges: List[EdgeInput] = []
    model_type: Optional[str] = Field(
        default="graphsage",
        description="graphsage | gat | mlp | xgboost. "
        "Note: on this dataset's realistic time-based test split, xgboost "
        "currently generalizes best (see README comparison table) -- the "
        "graph models are kept as the primary research artifact of this "
        "project, not because they win on raw accuracy.",
    )


class PredictionResult(BaseModel):
    id: str
    fraud_probability: float
    flagged: bool


class SubgraphResponse(BaseModel):
    predictions: List[PredictionResult]
    threshold: float
    model_type: str