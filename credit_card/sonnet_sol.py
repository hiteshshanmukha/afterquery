import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# Load data
print("Loading data...")
train = pd.read_csv('/app/train.csv')
test = pd.read_csv('/app/test.csv')

# Separate features and target
print("Preparing features...")
feature_cols = [col for col in train.columns if col not in ['id', 'default_payment_next_month']]
X = train[feature_cols]
y = train['default_payment_next_month']
X_test = test[feature_cols]
test_ids = test['id']

print(f"Training set shape: {X.shape}")
print(f"Test set shape: {X_test.shape}")
print(f"Default rate in training: {y.mean():.3f}")

# Split for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Gradient Boosting model
print("\nTraining Gradient Boosting model...")
model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    random_state=42,
    verbose=1
)

model.fit(X_train, y_train)

# Validate
print("\nValidating...")
val_preds = model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, val_preds)
print(f"Validation AUC: {val_auc:.4f}")

# Make predictions on test set
print("\nGenerating predictions for test set...")
test_preds = model.predict_proba(X_test)[:, 1]

# Create submission file
submission = pd.DataFrame({
    'id': test_ids,
    'default_prob': test_preds
})

submission.to_csv('/app/submission.csv', index=False)
print(f"\nSubmission saved to /app/submission.csv")
print(f"Submission shape: {submission.shape}")
print(f"\nFirst few predictions:")
print(submission.head(10))
print(f"\nPrediction statistics:")
print(submission['default_prob'].describe())