import pytest

from src.prediction import (
    FEATURE_COLUMNS,
    build_features,
    load_model_bundle,
    predict_traffic,
)


@pytest.fixture(scope="module")
def bundle():
    return load_model_bundle()


def test_build_features_matches_training_schema(bundle):
    features = build_features(
        bundle,
        hour=8,
        day="Monday",
        car_count=50,
        bike_count=10,
        bus_count=5,
        truck_count=10,
    )

    assert features.columns.tolist() == FEATURE_COLUMNS
    assert features.iloc[0]["Hour"] == 8


def test_prediction_returns_known_label(bundle):
    features = build_features(
        bundle,
        hour=17,
        day="Friday",
        car_count=100,
        bike_count=20,
        bus_count=10,
        truck_count=15,
    )

    result = predict_traffic(bundle, features)
    assert result in set(bundle.target_encoder.classes_)


def test_invalid_hour_is_rejected(bundle):
    with pytest.raises(ValueError, match="hour"):
        build_features(
            bundle,
            hour=24,
            day="Monday",
            car_count=1,
            bike_count=1,
            bus_count=1,
            truck_count=1,
        )
