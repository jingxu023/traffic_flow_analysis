"""From project root: python build_weekday_v2.py --csv Traffic.csv"""
import argparse
from src.relative_traffic import prepare_data,grouped_date_split
from src.relative_weekday import WeekdayRelativeTrafficModel

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--csv',default='Traffic.csv')
    p.add_argument('--output',default='relative_traffic_weekday_v2.json')
    a=p.parse_args()
    df=prepare_data(a.csv)
    train,holdout=grouped_date_split(df)
    model=WeekdayRelativeTrafficModel.fit(train)
    model.to_json(a.output)
    print(f'Train={len(train)}, held-out={len(holdout)}')
    print('Reference groups:',len(model.weekday_hour_counts),'(minimum 8 training observations required)')
    print('Training ratio quartiles:',model.ratio_cutoffs)
    for day in ['Monday','Sunday']:
        for hour in [8,23]:
            print(day,hour,model.classify(day,hour,150))
    heldout_categories=holdout.apply(lambda r:model.classify(r['Day of the week'],int(r['Hour']),int(r['Total'])).category,axis=1)
    print('Held-out category distribution:',heldout_categories.value_counts().to_dict())
    print('No forecast accuracy claim: categories are descriptive, not independently measured congestion labels.')

if __name__=='__main__':main()
