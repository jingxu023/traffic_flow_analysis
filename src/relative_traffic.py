"""V2: interpretable relative traffic-volume classification.

Important: output measures traffic volume RELATIVE TO its reference hour,
not observed traffic congestion, travel speed, or future traffic conditions.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd

VEHICLE_COLUMNS = ("CarCount", "BikeCount", "BusCount", "TruckCount")
LABELS = ("low", "normal", "high", "very_high")


@dataclass(frozen=True)
class RelativeResult:
    hour: int
    total_vehicles: int
    reference_vehicles: float
    ratio: float
    category: str
    description: str


def prepare_data(csv_path: str | Path) -> pd.DataFrame:
    """Check and transform the raw Kaggle-format CSV without using target labels."""
    df = pd.read_csv(csv_path)
    needed = {"Time", "Date", "Day of the week", *VEHICLE_COLUMNS, "Total"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df[list(needed)].isnull().any().any():
        raise ValueError("Null values in required input fields")
    for col in (*VEHICLE_COLUMNS, "Total"):
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"{col} must be numeric")
        if (df[col] < 0).any():
            raise ValueError(f"{col} cannot be negative")
    if not (df[list(VEHICLE_COLUMNS)].sum(axis=1) == df["Total"]).all():
        raise ValueError("Total does not equal the sum of vehicle counts")
    df = df.copy()
    df["Hour"] = pd.to_datetime(df["Time"], format="%I:%M:%S %p", errors="raise").dt.hour
    # Do NOT derive weekdays from Date (it has no month/year).
    return df


def grouped_date_split(df: pd.DataFrame, holdout_dates: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out entire Date groups. Not a chronological evaluation: no month/year."""
    dates = sorted(df["Date"].unique())
    if len(dates) <= holdout_dates or holdout_dates < 1:
        raise ValueError("Not enough distinct Date values")
    # Deterministic, spaced-out date groups to cover the month rather than random rows.
    holdout = {dates[i] for i in range(3, len(dates), 4)}
    train = df.loc[~df["Date"].isin(holdout)].copy()
    test = df.loc[df["Date"].isin(holdout)].copy()
    if train.empty or test.empty:
        raise ValueError("Empty train or holdout split")
    return train, test


@dataclass(frozen=True)
class RelativeTrafficModel:
    hourly_medians: dict[int, float]
    ratio_cutoffs: tuple[float, float, float]

    @classmethod
    def fit(cls, train: pd.DataFrame) -> "RelativeTrafficModel":
        medians = train.groupby("Hour")["Total"].median()
        if set(medians.index) != set(range(24)):
            raise ValueError("Training set must cover all 24 hours")
        if (medians <= 0).any():
            raise ValueError("Nonpositive baseline in training data")
        ratios = train["Total"] / train["Hour"].map(medians)
        cutoffs = tuple(float(x) for x in ratios.quantile([0.25, 0.5, 0.75]).to_numpy())
        if not (cutoffs[0] < cutoffs[1] < cutoffs[2]):
            raise ValueError("Quantile boundaries not distinct")
        return cls({int(h): float(v) for h,v in medians.items()}, cutoffs)

    def classify(self, hour: int, total_vehicles: int) -> RelativeResult:
        if isinstance(hour, bool) or not isinstance(hour, (int, np.integer)) or hour not in range(24):
            raise ValueError("hour must be an integer between 0 and 23")
        if isinstance(total_vehicles, bool) or not isinstance(total_vehicles, (int, np.integer)) or total_vehicles < 0:
            raise ValueError("total_vehicles must be a nonnegative integer")
        baseline = self.hourly_medians[hour]
        ratio = float(total_vehicles / baseline)
        a,b,c = self.ratio_cutoffs
        category = LABELS[0] if ratio < a else LABELS[1] if ratio < b else LABELS[2] if ratio < c else LABELS[3]
        explanations = {"low":"Below typical relative volume", "normal":"Around/below the typical range", "high":"Above typical relative volume", "very_high":"Far above typical relative volume"}
        return RelativeResult(hour,total_vehicles,baseline,ratio,category,explanations[category])

    def to_json(self, filepath: str | Path) -> None:
        Path(filepath).write_text(json.dumps({"hourly_medians":self.hourly_medians,"ratio_cutoffs":self.ratio_cutoffs,"method":"hour median + training-set ratio quartiles", "meaning":"relative traffic volume, NOT congestion"},indent=2),encoding="utf-8")

    @classmethod
    def from_json(cls, filepath: str | Path) -> "RelativeTrafficModel":
        content=json.loads(Path(filepath).read_text(encoding="utf-8"))
        return cls({int(k):float(v) for k,v in content["hourly_medians"].items()},tuple(map(float,content["ratio_cutoffs"])))


def classify_vehicle_counts(model: RelativeTrafficModel, *, hour: int, car_count: int, bike_count: int, bus_count: int, truck_count: int) -> RelativeResult:
    counts = (car_count,bike_count,bus_count,truck_count)
    if any(isinstance(x,bool) or not isinstance(x,(int,np.integer)) or x < 0 for x in counts):
        raise ValueError("Vehicle counts must be nonnegative integers")
    return model.classify(hour, sum(counts))
