import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

# Load data
train = pd.read_csv('/app/train.csv')
test = pd.read_csv('/app/test.csv')

# Prepare features and target
X_train = train.drop(['id', 'default_payment_next_month'], axis=1)
y_train = train['default_payment_next_month']
X_test = test.drop(['id'], axis=1)

# Train a scikit-learn histogram-based gradient boosting model
model = HistGradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict probabilities for the positive class
probs = model.predict_proba(X_test)[:, 1]

# Build submission DataFrame and save
submission = pd.DataFrame({
    'id': test['id'],
    'default_prob': probs
})
submission.to_csv('/app/submission.csv', index=False)