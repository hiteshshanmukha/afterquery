import pandas as pd

train = pd.read_csv('train.csv')

# Check all categorical column value ranges
cols_to_check = [
    'CRASH_HOUR', 'ROAD_CLASS', 'ROAD_ALIGN', 'ROAD_SURF_COND',
    'LIGHT_COND', 'WEATHER', 'REGION', 'RURAL_URBAN', 'SPEED_LIMIT',
    'MAN_COLL', 'TYP_INT', 'NUM_VEHICLES', 'NUM_PERSONS',
    'BODY_TYP', 'VEH_MAKE', 'VSPD_EST', 'FIRE_EXP',
    'DRIVER_AGE', 'DRIVER_SEX', 'DRINKING', 'DRUG_INVOLVEMENT',
    'DISTRACTED', 'RESTRAINT_USE'
]

for col in cols_to_check:
    vc = train[col].value_counts().sort_index()
    print(f"\n{col}:")
    print(vc.to_string())
    print(f"  Unique values: {sorted(train[col].unique())}")
