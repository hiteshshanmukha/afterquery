import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold

print('Loading data...')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

def engineer_features(df):
    # Payment history features
    df['avg_pay_delay'] = df[[f'PAY_{i}' for i in [0] + list(range(2,7))]].mean(axis=1)
    df['max_pay_delay'] = df[[f'PAY_{i}' for i in [0] + list(range(2,7))]].max(axis=1)
    df['num_delays'] = (df[[f'PAY_{i}' for i in [0] + list(range(2,7))]] > 0).sum(axis=1)
    df['num_paid_on_time'] = (df[[f'PAY_{i}' for i in [0] + list(range(2,7))]] <= 0).sum(axis=1)
    
    # Bill amount features
    df['avg_bill'] = df[[f'BILL_AMT{i}' for i in range(1,7)]].mean(axis=1)
    df['max_bill'] = df[[f'BILL_AMT{i}' for i in range(1,7)]].max(axis=1)
    df['std_bill'] = df[[f'BILL_AMT{i}' for i in range(1,7)]].std(axis=1)
    df['bill_growth'] = df['BILL_AMT1'] / (df['BILL_AMT6'] + 1) - 1
    df['bill_utilization'] = df['max_bill'] / (df['LIMIT_BAL'] + 1)
    
    # Payment amount features
    df['avg_payment'] = df[[f'PAY_AMT{i}' for i in range(1,7)]].mean(axis=1)
    df['max_payment'] = df[[f'PAY_AMT{i}' for i in range(1,7)]].max(axis=1)
    df['std_payment'] = df[[f'PAY_AMT{i}' for i in range(1,7)]].std(axis=1)
    df['payment_ratio'] = df['avg_payment'] / (df['avg_bill'] + 1)
    
    # Credit limit utilization
    df['util_ratio'] = df['avg_bill'] / (df['LIMIT_BAL'] + 1)
    
    # Payment status patterns
    df['recent_delay'] = (df[[f'PAY_{i}' for i in [0, 2, 3]]].mean(axis=1) > 0).astype(int)
    df['historical_delay'] = (df[[f'PAY_{i}' for i in [4, 5, 6]]].mean(axis=1) > 0).astype(int)
    df['worsening_status'] = (df['recent_delay'] > df['historical_delay']).astype(int)
    
    # Payment trend
    df['payment_trend'] = df[[f'PAY_AMT{i}' for i in range(1,7)]].apply(lambda x: np.polyfit(range(6), x, 1)[0], axis=1)
    
    # Feature interactions
    df['delay_util_interaction'] = df['avg_pay_delay'] * df['util_ratio']
    df['payment_util_ratio'] = df['payment_ratio'] * df['util_ratio']
    df['age_util_interaction'] = df['AGE'] * df['util_ratio']
    
    return df

print('Engineering features...')
train = engineer_features(train)
test = engineer_features(test)

# Process categorical variables
cat_cols = ['SEX', 'EDUCATION', 'MARRIAGE']
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])

# Prepare features
feature_cols = [col for col in train.columns if col not in ['id', 'default_payment_next_month']]
X = train[feature_cols]
y = train['default_payment_next_month']
X_test = test[feature_cols]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Configure model
model = GradientBoostingClassifier(
    n_estimators=400,
    learning_rate=0.03,
    max_depth=7,
    min_samples_split=100,
    min_samples_leaf=50,
    subsample=0.85,
    random_state=42
)

# Cross-validation with stratification
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_scaled, y, cv=skf, scoring='roc_auc')
print(f'CV ROC-AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})')

# Train final model
print('Training final model...')
model.fit(X_scaled, y)

# Feature importance analysis
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print('\nTop 10 most important features:')
print(feature_importance.head(10))

# Make predictions
preds = model.predict_proba(X_test_scaled)[:, 1]

# Create submission
submission = pd.DataFrame({
    'id': test['id'],
    'default_prob': preds
})

# Save submission
submission.to_csv('/app/submission.csv', index=False)
print('\nSubmission saved to /app/submission.csv')

# Verify submission
final = pd.read_csv('/app/submission.csv')
print(f'Submission shape: {final.shape}')
print('First few rows:')
print(final.head())