"""Experimental weekday + hour relative traffic volume, not actual congestion.

Original Kaggle weekday labels lack a verifiable complete calendar date; treat as
source-provided categories, not validated chronology. Reference derived on train only.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.relative_traffic import LABELS

WEEKDAYS = ('Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday')

@dataclass(frozen=True)
class WeekdayResult:
    weekday: str
    hour: int
    total_vehicles: int
    reference_vehicles: float
    ratio: float
    category: str
    reference_type: str
    baseline_observations: int

@dataclass(frozen=True)
class WeekdayRelativeTrafficModel:
    # Keys are 'Monday|08' to make JSON stable and unambiguous.
    weekday_hour_medians: dict[str,float]
    weekday_hour_counts: dict[str,int]
    hourly_medians: dict[int,float]
    ratio_cutoffs: tuple[float,float,float]
    min_observations: int = 8

    @staticmethod
    def key(weekday: str, hour: int) -> str:
        return f'{weekday}|{hour:02d}'

    @classmethod
    def fit(cls, train: pd.DataFrame, min_observations: int=8) -> 'WeekdayRelativeTrafficModel':
        if min_observations < 1:
            raise ValueError('min_observations must be >= 1')
        if not {'Hour','Day of the week','Total'}.issubset(train.columns):
            raise ValueError('Training data is missing required columns')
        if train.empty or not train['Day of the week'].isin(WEEKDAYS).all():
            raise ValueError('Missing data or invalid weekday names')
        hour_medians=train.groupby('Hour')['Total'].median()
        if set(hour_medians.index)!=set(range(24)) or (hour_medians <= 0).any():
            raise ValueError('Training data requires positive baselines in all 24 hours')
        grouped=train.groupby(['Day of the week','Hour'])['Total'].agg(['median','count'])
        medians={cls.key(d,int(h)):float(row['median']) for (d,h),row in grouped.iterrows()}
        counts={cls.key(d,int(h)):int(row['count']) for (d,h),row in grouped.iterrows()}
        hourly={int(h):float(m) for h,m in hour_medians.items()}
        ref=np.array([medians[cls.key(d,int(h))] if counts.get(cls.key(d,int(h)),0)>=min_observations else hourly[int(h)] for d,h in zip(train['Day of the week'],train['Hour'])])
        if (ref <= 0).any():
            raise ValueError('Reference baseline must be positive')
        ratios=train['Total'].to_numpy()/ref
        cutoffs=tuple(float(x) for x in np.quantile(ratios,[.25,.5,.75]))
        if not cutoffs[0]<cutoffs[1]<cutoffs[2]:
            raise ValueError('Quartile thresholds are not distinct')
        return cls(medians,counts,hourly,cutoffs,min_observations)

    def classify(self,weekday: str,hour: int,total_vehicles: int) -> WeekdayResult:
        if weekday not in WEEKDAYS:
            raise ValueError('Unknown weekday')
        if isinstance(hour,bool) or not isinstance(hour,(int,np.integer)) or not 0<=hour<=23:
            raise ValueError('Hour must be an integer in 0..23')
        if isinstance(total_vehicles,bool) or not isinstance(total_vehicles,(int,np.integer)) or total_vehicles<0:
            raise ValueError('Total must be a nonnegative integer')
        key=self.key(weekday,hour)
        n=self.weekday_hour_counts.get(key,0)
        available=n>=self.min_observations
        baseline=self.weekday_hour_medians[key] if available else self.hourly_medians[hour]
        ratio=float(total_vehicles/baseline)
        a,b,c=self.ratio_cutoffs
        label=LABELS[0] if ratio<a else LABELS[1] if ratio<b else LABELS[2] if ratio<c else LABELS[3]
        return WeekdayResult(weekday,hour,total_vehicles,baseline,ratio,label,'weekday + hour' if available else 'hour-only fallback',n)

    def to_json(self,path: str | Path) -> None:
        data={'weekday_hour_medians':self.weekday_hour_medians,'weekday_hour_counts':self.weekday_hour_counts,'hourly_medians':self.hourly_medians,'ratio_cutoffs':self.ratio_cutoffs,'min_observations':self.min_observations,'method':'weekday-hour median (fallback to hour), ratio quartiles on training only','limitation':'Unverified source weekday chronology; relative traffic volume, not congestion'}
        Path(path).write_text(json.dumps(data,indent=2),encoding='utf-8')

    @classmethod
    def from_json(cls,path: str | Path) -> 'WeekdayRelativeTrafficModel':
        x=json.loads(Path(path).read_text(encoding='utf-8'))
        return cls({k:float(v) for k,v in x['weekday_hour_medians'].items()},{k:int(v) for k,v in x['weekday_hour_counts'].items()},{int(k):float(v) for k,v in x['hourly_medians'].items()},tuple(map(float,x['ratio_cutoffs'])),int(x['min_observations']))
