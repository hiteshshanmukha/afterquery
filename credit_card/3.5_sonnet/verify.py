import os
import pandas as pd

print('Verifying submission file...')

# Check file exists
if not os.path.exists('/app/submission.csv'):
    print('ERROR: submission.csv does not exist!')
    exit(1)

# Load files
submission = pd.read_csv('/app/submission.csv')
sample = pd.read_csv('/app/sample_submission.csv')
test = pd.read_csv('/app/test.csv')

print('\nBasic checks:')
print(f'File size: {os.path.getsize("/app/submission.csv")} bytes')
print(f'Row count: {len(submission)} (should be {len(test)})')
print(f'Columns: {list(submission.columns)}')
print(f'Column names match sample: {list(submission.columns) == list(sample.columns)}')

print('\nValue checks:')
print(f'Prediction range: [{submission.default_prob.min():.3f}, {submission.default_prob.max():.3f}]')
print(f'Any missing values: {submission.isnull().any().any()}')

print('\nFirst 5 rows:')
print(submission.head())

print('\nVerification complete!')