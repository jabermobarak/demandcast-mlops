from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .features import HISTORY_FEATURES


class PredictionRow(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    timestamp: datetime
    lag_24: float | None = Field(ge=0, le=1_000_000)
    lag_48: float | None = Field(ge=0, le=1_000_000)
    lag_168: float | None = Field(ge=0, le=1_000_000)
    rolling_mean_24: float | None = Field(ge=0, le=1_000_000)
    rolling_mean_168: float | None = Field(ge=0, le=1_000_000)

    @field_validator("timestamp")
    @classmethod
    def exact_hour(cls, value):
        if value.tzinfo is not None or value.minute or value.second or value.microsecond:
            raise ValueError("Use an exact, timezone-naive dataset-local hour")
        if not 2011 <= value.year <= 2100:
            raise ValueError("Timestamp year must be between 2011 and 2100")
        return value

    @model_validator(mode="after")
    def require_some_history(self):
        if all(getattr(self, name) is None for name in HISTORY_FEATURES):
            raise ValueError("At least one historical demand feature is required")
        return self


class PredictionBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rows: list[PredictionRow] = Field(min_length=1, max_length=168)


class ShiftRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    factor: float = Field(default=1.4, ge=0.5, le=2.0)
