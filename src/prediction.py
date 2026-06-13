import logging
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


LOGGER = logging.getLogger(__name__)
FEATURE_COLUMNS = [
    "Hour",
    "Day of the week",
    "CarCount",
    "BikeCount",
    "BusCount",
    "TruckCount",
]


@dataclass(frozen=True)
class ModelBundle:
    model: object
    day_encoder: object
    target_encoder: object


def load_model_bundle(model_dir: str | Path = ".") -> ModelBundle:
    model_path = Path(model_dir)
    with (model_path / "traffic_model.pkl").open("rb") as file:
        model = pickle.load(file)
    with (model_path / "label_encoder_day.pkl").open("rb") as file:
        day_encoder = pickle.load(file)
    with (model_path / "label_encoder_target.pkl").open("rb") as file:
        target_encoder = pickle.load(file)

    LOGGER.info("event=model_loaded model_type=%s", type(model).__name__)
    return ModelBundle(model, day_encoder, target_encoder)


def build_features(
    bundle: ModelBundle,
    *,
    hour: int,
    day: str,
    car_count: int,
    bike_count: int,
    bus_count: int,
    truck_count: int,
) -> pd.DataFrame:
    if not 0 <= hour <= 23:
        raise ValueError("hour must be between 0 and 23")

    counts = [car_count, bike_count, bus_count, truck_count]
    if any(count < 0 for count in counts):
        raise ValueError("vehicle counts cannot be negative")

    encoded_day = bundle.day_encoder.transform([day])[0]
    return pd.DataFrame(
        [[hour, encoded_day, *counts]],
        columns=FEATURE_COLUMNS,
    )


def predict_traffic(bundle: ModelBundle, features: pd.DataFrame) -> str:
    started_at = time.perf_counter()
    prediction = bundle.model.predict(features)[0]
    label = bundle.target_encoder.inverse_transform([prediction])[0]
    duration_ms = (time.perf_counter() - started_at) * 1000
    LOGGER.info(
        "event=prediction_completed result=%s duration_ms=%.2f",
        label,
        duration_ms,
    )
    return str(label)
