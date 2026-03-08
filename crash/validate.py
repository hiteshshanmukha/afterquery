import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sol = pd.read_csv('solution.csv')
perf = pd.read_csv('perfect_submission.csv')
samp = pd.read_csv('sample_submission.csv')

print(f"Train: {train.shape}")
print(f"Test: {test.shape}")
print(f"Solution: {sol.shape}")
print(f"Perfect: {perf.shape}")
print(f"Sample: {samp.shape}")

print(f"\nTrain columns: {list(train.columns)}")
print(f"Test columns: {list(test.columns)}")
print(f"Solution columns: {list(sol.columns)}")

print(f"\nIDs match (test/solution): {set(test.id) == set(sol.id)}")
print(f"IDs match (test/perfect): {set(test.id) == set(perf.id)}")
print(f"IDs match (test/sample): {set(test.id) == set(samp.id)}")

print(f"\nTrain has INJ_SEV: {'INJ_SEV' in train.columns}")
print(f"Test has INJ_SEV: {'INJ_SEV' in test.columns}")
print(f"Test has RATWGT: {'RATWGT' in test.columns}")
print(f"Solution has RATWGT: {'RATWGT' in sol.columns}")

print(f"\nPerfect == Solution INJ_SEV: {(perf.INJ_SEV == sol.INJ_SEV).all()}")
print(f"Sample all zeros: {(samp.INJ_SEV == 0).all()}")

# Check RATWGT distribution
print(f"\nRATWGT stats:")
print(train['RATWGT'].describe())

# Check for NaN in key columns
print(f"\nNaN counts in train:")
print(train.isnull().sum()[train.isnull().sum() > 0])

# PREV_DWI and PREV_ACC removed from schema - confirm
print(f"\nPREV_DWI in train: {'PREV_DWI' in train.columns}")
print(f"PREV_ACC in train: {'PREV_ACC' in train.columns}")

# Check some column value ranges
print(f"\nCRASH_YEAR range: {train.CRASH_YEAR.min()}-{train.CRASH_YEAR.max()}")
print(f"CRASH_HOUR unique: {sorted(train.CRASH_HOUR.unique())}")
print(f"REGION unique: {sorted(train.REGION.unique())}")
print(f"ROLLOVER unique: {sorted(train.ROLLOVER.unique())}")
print(f"INJ_SEV unique: {sorted(train.INJ_SEV.unique())}")
print(f"BODY_TYP sample: {sorted(train.BODY_TYP.unique())[:15]}")
print(f"VSPD_EST range: {train.VSPD_EST.min()}-{train.VSPD_EST.max()}")
print(f"DRIVER_AGE range: {train.DRIVER_AGE.min()}-{train.DRIVER_AGE.max()}")
print(f"SPEED_LIMIT sample: {sorted(train.SPEED_LIMIT.unique())[:15]}")
