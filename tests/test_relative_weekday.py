import pytest
from src.relative_traffic import prepare_data, grouped_date_split
from src.relative_weekday import WeekdayRelativeTrafficModel

@pytest.fixture(scope='module')
def m():
    df=prepare_data('Traffic.csv')
    train,test=grouped_date_split(df)
    return WeekdayRelativeTrafficModel.fit(train)

def test_same_count_weekday_can_change_reference(m):
    mon=m.classify('Monday',8,150)
    sun=m.classify('Sunday',8,150)
    assert mon.reference_type=='weekday + hour'
    assert sun.reference_type=='weekday + hour'
    assert mon.reference_vehicles != sun.reference_vehicles

def test_hour_can_change_category(m):
    a=m.classify('Monday',8,150)
    b=m.classify('Monday',23,150)
    assert a.category != b.category

def test_round_trip_json(tmp_path,m):
    path=tmp_path/'model.json'
    m.to_json(path)
    recovered=WeekdayRelativeTrafficModel.from_json(path)
    assert recovered.classify('Tuesday',8,150)==m.classify('Tuesday',8,150)

@pytest.mark.parametrize('weekday,hour,total', [('Funday',8,100),('Monday',24,100),('Monday',8,-1)])
def test_invalid_inputs(m,weekday,hour,total):
    with pytest.raises(ValueError):m.classify(weekday,hour,total)

def test_fallback(m):
    patched=WeekdayRelativeTrafficModel(m.weekday_hour_medians,m.weekday_hour_counts,m.hourly_medians,m.ratio_cutoffs,10000)
    result=patched.classify('Monday',8,150)
    assert result.reference_type=='hour-only fallback'
    assert result.reference_vehicles==m.hourly_medians[8]
