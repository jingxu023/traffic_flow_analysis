"""Run from project root: python build_relative_v2.py --csv Traffic.csv"""
import argparse
from pathlib import Path
import json
from src.relative_traffic import prepare_data, grouped_date_split, RelativeTrafficModel


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--csv',default='Traffic.csv')
    parser.add_argument('--output',default='relative_traffic_v2.json')
    args=parser.parse_args()
    df=prepare_data(args.csv)
    train,holdout=grouped_date_split(df)
    model=RelativeTrafficModel.fit(train)
    model.to_json(args.output)
    print(f'Rows: {len(df)}, Train: {len(train)}, Holdout: {len(holdout)}')
    print('Training-only relative-ratio quartiles:',tuple(round(x,4) for x in model.ratio_cutoffs))
    print('Per-hour training medians:',model.hourly_medians)
    for hour in [8,23]:
        result=model.classify(hour,150)
        print(f'Hour {hour:02d}, 150 vehicles => {result.category}, reference={result.reference_vehicles:.1f}, ratio={result.ratio:.3f}')
    # Holdout category distribution; unlike supervised labels, there is NO accuracy claim.
    cats=holdout.apply(lambda row:model.classify(int(row.Hour),int(row.Total)).category,axis=1)
    print('Holdout V2 category distribution:',cats.value_counts().to_dict())
    print('This is a DESCRIPTIVE classification, not a validated congestion forecast.')

if __name__=='__main__':main()
