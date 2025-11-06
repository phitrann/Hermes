"""
Data models for validating and structuring Hermes response outputs.

Provides Pydantic models for type-safe validation of handler responses,
ensuring consistent structure across prediction, recommendation, visualization,
statistics, and general query handlers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, Literal
from datetime import datetime
from pathlib import Path

try:
    from pydantic import BaseModel, Field, validator, field_validator
    PYDANTIC_V2 = True
except ImportError:
    from pydantic import BaseModel, Field, validator
    PYDANTIC_V2 = False


# ============================================================================
# Core Response Models
# ============================================================================

class ChartData(BaseModel):
    """Represents a generated chart/visualization."""
    
    path: Optional[str] = Field(None, description="File system path to chart image")
    mime_type: str = Field(default="image/png", description="MIME type of chart")
    caption: Optional[str] = Field(None, description="Chart description/caption")
    
    if PYDANTIC_V2:
        @field_validator("path")
        @classmethod
        def validate_path_exists(cls, v: Optional[str]) -> Optional[str]:
            if v and not Path(v).exists():
                raise ValueError(f"Chart file does not exist: {v}")
            return v
    else:
        @validator("path")
        def validate_path_exists(cls, v: Optional[str]) -> Optional[str]:
            if v and not Path(v).exists():
                raise ValueError(f"Chart file does not exist: {v}")
            return v
    
    @property
    def exists(self) -> bool:
        """Check if chart file exists."""
        return self.path is not None and Path(self.path).exists()


class MetricsData(BaseModel):
    """ML model performance metrics."""
    
    r2_score: float = Field(..., ge=0.0, le=1.0, description="R² coefficient of determination")
    rmse: float = Field(..., ge=0.0, description="Root Mean Squared Error")
    mae: Optional[float] = Field(None, ge=0.0, description="Mean Absolute Error")
    model_type: str = Field(default="Linear Regression", description="Model algorithm used")
    training_samples: Optional[int] = Field(None, ge=0, description="Number of training samples")


class PredictionData(BaseModel):
    """Prediction forecast results."""
    
    predicted_avg_delay: float = Field(..., description="Predicted average delay (minutes)")
    predicted_median: float = Field(..., description="Predicted median delay (minutes)")
    forecast_period: str = Field(..., description="Time period for forecast")
    confidence_interval: Optional[Dict[str, float]] = Field(None, description="95% confidence bounds")


class RecommendationItem(BaseModel):
    """Single actionable recommendation."""
    
    category: str = Field(..., description="Recommendation category")
    priority: str = Field(..., description="Priority level (Critical/High/Medium/Low)")
    finding: str = Field(..., description="Key finding/observation")
    action: str = Field(..., description="Recommended action")
    impact: Optional[str] = Field(None, description="Expected impact")
    
    if PYDANTIC_V2:
        @field_validator("priority")
        @classmethod
        def validate_priority(cls, v: str) -> str:
            allowed = {"Critical", "High", "Medium", "Low"}
            if v not in allowed:
                raise ValueError(f"Priority must be one of {allowed}")
            return v
    else:
        @validator("priority")
        def validate_priority(cls, v: str) -> str:
            allowed = {"Critical", "High", "Medium", "Low"}
            if v not in allowed:
                raise ValueError(f"Priority must be one of {allowed}")
            return v


class StatsSummary(BaseModel):
    """Summary statistics for dataset."""
    
    total_shipments: int = Field(..., ge=0)
    delayed_shipments: int = Field(..., ge=0)
    on_time_shipments: int = Field(..., ge=0)
    avg_delay_minutes: float = Field(..., ge=0.0)
    median_delay_minutes: Optional[float] = Field(None, ge=0.0)
    delay_rate: float = Field(..., ge=0.0, le=1.0, description="Proportion of delayed shipments")
    date_range: Optional[str] = Field(None, description="Data date range")


class DataFramePreview(BaseModel):
    """Structured dataframe preview for UI display."""
    
    columns: List[str] = Field(..., description="Column names")
    data: List[List[Any]] = Field(..., description="Row data")
    shape: tuple[int, int] = Field(..., description="(rows, cols)")
    dtypes: Optional[Dict[str, str]] = Field(None, description="Column data types")


# ============================================================================
# Handler Response Models
# ============================================================================

class BaseResponse(BaseModel):
    """Base response structure for all handlers."""
    
    text: str = Field(..., description="Primary response text (markdown/HTML)")
    chart: Optional[ChartData] = Field(None, description="Generated chart data")
    intent: str = Field(..., description="Classified query intent")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    success: bool = Field(default=True, description="Whether request succeeded")
    error: Optional[str] = Field(None, description="Error message if failed")


class PredictionResponse(BaseResponse):
    """Response for prediction queries."""
    
    intent: Literal["prediction"]
    metrics: Optional[MetricsData] = Field(None, description="Model performance metrics")
    prediction: Optional[PredictionData] = Field(None, description="Forecast results")
    
    if PYDANTIC_V2:
        @field_validator("metadata")
        @classmethod
        def populate_metadata(cls, v: Dict[str, Any], values) -> Dict[str, Any]:
            # Auto-populate metadata from metrics and prediction
            if "metrics" in values and values["metrics"]:
                v["metrics"] = values["metrics"].dict()
            if "prediction" in values and values["prediction"]:
                v["prediction"] = values["prediction"].dict()
            return v
    else:
        @validator("metadata", always=True)
        def populate_metadata(cls, v: Dict[str, Any], values) -> Dict[str, Any]:
            if "metrics" in values and values["metrics"]:
                v["metrics"] = values["metrics"].dict()
            if "prediction" in values and values["prediction"]:
                v["prediction"] = values["prediction"].dict()
            return v


class RecommendationResponse(BaseResponse):
    """Response for recommendation queries."""

    intent: Literal["recommendation"]
    recommendations: List[RecommendationItem] = Field(default_factory=list)
    summary_stats: Optional[StatsSummary] = Field(None, description="Supporting statistics")


class VisualizationResponse(BaseResponse):
    """Response for visualization queries."""
    
    intent: Literal["visualization"]
    chart_type: Optional[str] = Field(None, description="Type of chart generated")
    data_preview: Optional[DataFramePreview] = Field(None, description="Underlying data")
    
    if PYDANTIC_V2:
        @field_validator("chart")
        @classmethod
        def ensure_chart_exists(cls, v: Optional[ChartData]) -> Optional[ChartData]:
            if v and not v.exists:
                raise ValueError("Visualization response must include valid chart")
            return v
    else:
        @validator("chart")
        def ensure_chart_exists(cls, v: Optional[ChartData]) -> Optional[ChartData]:
            if v and not v.exists:
                raise ValueError("Visualization response must include valid chart")
            return v


class StatisticsResponse(BaseResponse):
    """Response for statistics queries."""

    intent: Literal["statistics"]
    stats: Optional[StatsSummary] = Field(None, description="Computed statistics")
    data_preview: Optional[DataFramePreview] = Field(None, description="Sample data")


class GeneralResponse(BaseResponse):
    """Response for general queries."""
    
    intent: Literal["general"]
    data_type: Optional[str] = Field(None, description="Type of data returned (text/number/dataframe)")
    raw_result: Optional[Any] = Field(None, description="Raw LLM response")


# ============================================================================
# Response Factory & Helpers
# ============================================================================

class ResponseFactory:
    """Factory for creating typed responses from handler outputs."""
    
    @staticmethod
    def from_dict(response_dict: Dict[str, Any]) -> BaseResponse:
        """
        Create appropriate response model from dictionary.
        
        Args:
            response_dict: Raw handler output dictionary
            
        Returns:
            Typed response model instance
        """
        intent = response_dict.get("intent", "general")
        
        # Parse chart if present
        chart_data = None
        if response_dict.get("chart"):
            chart_path = response_dict["chart"]
            chart_data = ChartData(
                path=chart_path,
                mime_type="image/png",
                caption=response_dict.get("metadata", {}).get("chart_caption")
            )
        
        # Route to appropriate model
        if intent == "prediction":
            return PredictionResponse(
                text=response_dict.get("text", ""),
                chart=chart_data,
                metadata=response_dict.get("metadata", {}),
                success=response_dict.get("success", True),
                error=response_dict.get("error"),
                metrics=MetricsData(**response_dict["metadata"]["metrics"]) 
                    if "metrics" in response_dict.get("metadata", {}) else None,
                prediction=PredictionData(**response_dict["metadata"]["prediction"])
                    if "prediction" in response_dict.get("metadata", {}) else None,
            )
        
        elif intent == "recommendation":
            recs = [
                RecommendationItem(**r) 
                for r in response_dict.get("metadata", {}).get("recommendations", [])
            ]
            return RecommendationResponse(
                text=response_dict.get("text", ""),
                chart=chart_data,
                metadata=response_dict.get("metadata", {}),
                recommendations=recs,
            )
        
        elif intent == "visualization":
            return VisualizationResponse(
                text=response_dict.get("text", ""),
                chart=chart_data,
                metadata=response_dict.get("metadata", {}),
                chart_type=response_dict.get("metadata", {}).get("chart_type"),
            )
        
        elif intent == "statistics":
            stats = response_dict.get("metadata", {}).get("stats")
            return StatisticsResponse(
                text=response_dict.get("text", ""),
                chart=chart_data,
                metadata=response_dict.get("metadata", {}),
                stats=StatsSummary(**stats) if stats else None,
            )
        
        else:
            return GeneralResponse(
                text=response_dict.get("text", ""),
                chart=chart_data,
                metadata=response_dict.get("metadata", {}),
                raw_result=response_dict.get("metadata", {}).get("raw"),
            )


# ============================================================================
# Gradio Display Helpers
# ============================================================================

class GradioFormatter:
    """Format typed responses for Gradio component display."""
    
    @staticmethod
    def format_for_chatbot(response: BaseResponse) -> Dict[str, Any]:
        """
        Format response for Gradio Chatbot component.
        
        Returns:
            Dict with 'text' and optional 'files' for multimodal display
        """
        result = {
            "text": response.text,
            "metadata": {
                "intent": response.intent,
                "timestamp": response.timestamp.isoformat(),
            }
        }
        
        if response.chart and response.chart.exists:
            result["files"] = [response.chart.path]
        
        return result
    
    @staticmethod
    def format_dataframe(data: Optional[DataFramePreview]) -> Optional[List[List[Any]]]:
        """Format DataFramePreview for Gradio DataFrame component."""
        if not data:
            return None
        return [data.columns] + data.data
    
    @staticmethod
    def format_metrics(metrics: Optional[MetricsData]) -> Optional[str]:
        """Format metrics as markdown table."""
        if not metrics:
            return None
        
        return f"""
| Metric | Value |
|--------|-------|
| R² Score | {metrics.r2_score:.3f} |
| RMSE | {metrics.rmse:.2f} min |
| Model | {metrics.model_type} |
"""
    
    @staticmethod
    def format_recommendations(recs: List[RecommendationItem]) -> str:
        """Format recommendations as markdown list."""
        if not recs:
            return "*No recommendations available*"
        
        priority_emoji = {
            "Critical": "🔴",
            "High": "🟠", 
            "Medium": "🟡",
            "Low": "🟢"
        }
        
        result = ""
        for i, rec in enumerate(recs, 1):
            emoji = priority_emoji.get(rec.priority, "⚪")
            result += f"\n**{i}. {rec.category}** {emoji}\n"
            result += f"- *Finding:* {rec.finding}\n"
            result += f"- *Action:* {rec.action}\n"
        
        return result