import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Load data
train = pd.read_csv('/app/train.csv')
test = pd.read_csv('/app/test.csv')

# Prepare features
features = [col for col in train.columns if col not in ['id', 'default_payment_next_month']]
X = train[features]
y = train['default_payment_next_month']

# Train model
model = XGBClassifier(scale_pos_weight=(len(y)-sum(y))/sum(y), use_label_encoder=False, eval_metric='auc')
model.fit(X, y)

# Predict
test['default_prob'] = model.predict_proba(test[features])[:,1]
test[['id', 'default_prob']].to_csv('/app/submission.csv', index=False)