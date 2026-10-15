from pathlib import Path
import pytest
from src.relative_traffic import prepare_data, grouped_date_split, RelativeTrafficModel, classify_vehicle_counts

CSV = Path(__file__).parents[1] / 'Traffic.csv'

@pytest.fixture(scope='module')
def model():
    df=prepare_data(CSV)
    train,test=grouped_date_split(df)
    assert set(train['Date']).isdisjoint(set(test['Date']))
    assert len(train)+len(test)==len(df)
    return RelativeTrafficModel.fit(train)

def test_real_csv_and_time(model):
    df=prepare_data(CSV)
    assert len(df)==2976
    assert df.iloc[0]['Hour']==0
    assert df['Hour'].between(0,23).all()

def test_same_count_different_hour_changes_result(model):
    morning=model.classify(8,150)
    night=model.classify(23,150)
    assert morning.reference_vehicles != night.reference_vehicles
    assert morning.ratio != night.ratio
    assert morning.category != night.category

def test_vehicle_count_sum(model):
    a=classify_vehicle_counts(model,hour=8,car_count=110,bike_count=15,bus_count=15,truck_count=10)
    assert a.total_vehicles==150
    assert a==model.classify(8,150)

@pytest.mark.parametrize('hour,total', [(-1,100),(24,100),(8,-2),(8,1.5),(True,5)])
def test_invalid_inputs(model,hour,total):
    with pytest.raises(ValueError): model.classify(hour,total)

def test_serialization_roundtrip(model,tmp_path):
    path=tmp_path/'reference.json'
    model.to_json(path)
    loaded=RelativeTrafficModel.from_json(path)
    assert loaded.classify(8,150)==model.classify(8,150)
