import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load data
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# Prepare features and target
X = df_train.drop(['id', 'default_payment_next_month'], axis=1)
y = df_train['default_payment_next_month']
X_test = df_test.drop('id', axis=1)

# Train a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X, y)

# Predict probabilities for the positive class
probs = model.predict_proba(X_test)[:, 1]

# Create submission DataFrame
submission = pd.DataFrame({'id': df_test['id'], 'default_prob': probs})
submission.to_csv('/app/submission.csv', index=False)