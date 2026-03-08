import os
import pandas as pd
import numpy as np
from datetime import datetime

# Create validation report
with open('validation_report.txt', 'w') as f:
    f.write(f'Validation Report - {datetime.now()}\n')
    f.write('=' * 50 + '\n\n')
    
    # Check file exists
    if not os.path.exists('/app/submission.csv'):
        f.write('ERROR: submission.csv does not exist!\n')
        exit(1)
    
    # Load files
    submission = pd.read_csv('/app/submission.csv')
    test = pd.read_csv('/app/test.csv')
    sample = pd.read_csv('/app/sample_submission.csv')
    
    # File checks
    f.write('File Validation:\n')
    f.write(f'- File size: {os.path.getsize("/app/submission.csv")} bytes\n')
    f.write(f'- Row count: {len(submission)} (expected: {len(test)})\n')
    f.write(f'- Columns: {list(submission.columns)}\n')
    f.write(f'- Matches sample: {list(submission.columns) == list(sample.columns)}\n\n')
    
    # Value checks
    f.write('Prediction Validation:\n')
    f.write(f'- Min value: {submission.default_prob.min():.4f}\n')
    f.write(f'- Max value: {submission.default_prob.max():.4f}\n')
    f.write(f'- Mean value: {submission.default_prob.mean():.4f}\n')
    f.write(f'- Missing values: {submission.isnull().sum().sum()}\n\n')
    
    # Content preview
    f.write('First 5 Predictions:\n')
    f.write(submission.head().to_string())
    
    f.write('\n\nValidation Complete!')

# Print the report
with open('validation_report.txt', 'r') as f:
    print(f.read())