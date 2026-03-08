import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Prepare features
X = train.drop(['id', 'default_payment_next_month'], axis=1)
y = train['default_payment_next_month']
X_test = test.drop(['id'], axis=1)

# Build pipeline with scaling
model = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1))
])
model.fit(X, y)

# Make predictions
preds = model.predict_proba(X_test)[:, 1]

# Create submission
submission = pd.DataFrame({'id': test['id'], 'default_prob': preds})
submission.to_csv('submission.csv', index=False)
print('Submission file created.')