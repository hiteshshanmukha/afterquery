import pandas as pd
import numpy as np

print('Validating submission file...')

# Load files
submission = pd.read_csv('/app/submission.csv')
test = pd.read_csv('/app/test.csv')
sample = pd.read_csv('/app/sample_submission.csv')

# Basic checks
print('\nFile validation:')
print(f'Submission shape: {submission.shape}')
print(f'Expected shape: ({len(test)}, 2)')
print(f'Columns present: {list(submission.columns)}')
print(f'Columns match sample: {list(submission.columns) == list(sample.columns)}')

# Value checks
print('\nValue validation:')
print(f'Prediction range: [{submission.default_prob.min():.3f}, {submission.default_prob.max():.3f}]')
print(f'Missing values: {submission.isnull().sum().sum()}')

# ID checks
print('\nID validation:')
print(f'All test IDs present: {set(test.id) == set(submission.id)}')

print('\nFirst few predictions:')
print(submission.head())