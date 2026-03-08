import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Load data
train = pd.read_csv('/app/train.csv')
test = pd.read_csv('/app/test.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Default rate: {train['default_payment_next_month'].mean():.4f}")

# Save IDs
train_ids = train['id'].values
test_ids = test['id'].values

# Target
y = train['default_payment_next_month'].values

# Drop id and target
train = train.drop(columns=['id', 'default_payment_next_month'])
test = test.drop(columns=['id'])

# Feature engineering
def create_features(df):
    df = df.copy()
    
    pay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
    pay_amt_cols = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    
    # PAY status features
    df['PAY_MAX'] = df[pay_cols].max(axis=1)
    df['PAY_MIN'] = df[pay_cols].min(axis=1)
    df['PAY_MEAN'] = df[pay_cols].mean(axis=1)
    df['PAY_STD'] = df[pay_cols].std(axis=1)
    df['PAY_SUM'] = df[pay_cols].sum(axis=1)
    
    df['NUM_LATE'] = (df[pay_cols] > 0).sum(axis=1)
    df['NUM_NO_USE'] = (df[pay_cols] == -2).sum(axis=1)
    df['NUM_PAID_FULL'] = (df[pay_cols] == -1).sum(axis=1)
    df['NUM_MIN_PAY'] = (df[pay_cols] == 0).sum(axis=1)
    df['NUM_SEVERE_LATE'] = (df[pay_cols] >= 2).sum(axis=1)
    
    df['PAY_RECENT_MEAN'] = df[['PAY_0', 'PAY_2', 'PAY_3']].mean(axis=1)
    df['PAY_OLD_MEAN'] = df[['PAY_4', 'PAY_5', 'PAY_6']].mean(axis=1)
    df['PAY_TREND'] = df['PAY_RECENT_MEAN'] - df['PAY_OLD_MEAN']
    
    df['BILL_MEAN'] = df[bill_cols].mean(axis=1)
    df['BILL_MAX'] = df[bill_cols].max(axis=1)
    df['BILL_MIN'] = df[bill_cols].min(axis=1)
    df['BILL_STD'] = df[bill_cols].std(axis=1)
    df['BILL_SUM'] = df[bill_cols].sum(axis=1)
    df['BILL_RANGE'] = df['BILL_MAX'] - df['BILL_MIN']
    
    df['BILL_RECENT_MEAN'] = df[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3']].mean(axis=1)
    df['BILL_OLD_MEAN'] = df[['BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']].mean(axis=1)
    df['BILL_TREND'] = df['BILL_RECENT_MEAN'] - df['BILL_OLD_MEAN']
    
    df['PAY_AMT_MEAN'] = df[pay_amt_cols].mean(axis=1)
    df['PAY_AMT_MAX'] = df[pay_amt_cols].max(axis=1)
    df['PAY_AMT_MIN'] = df[pay_amt_cols].min(axis=1)
    df['PAY_AMT_STD'] = df[pay_amt_cols].std(axis=1)
    df['PAY_AMT_SUM'] = df[pay_amt_cols].sum(axis=1)
    
    for i in range(1, 7):
        bill = df[f'BILL_AMT{i}']
        pay = df[f'PAY_AMT{i}']
        df[f'PAY_RATIO_{i}'] = pay / (bill.abs() + 1)
        df[f'PAY_BILL_DIFF_{i}'] = pay - bill
    
    df['TOTAL_PAY_RATIO'] = df['PAY_AMT_SUM'] / (df['BILL_SUM'].abs() + 1)
    
    df['UTIL_RATE'] = df['BILL_AMT1'] / (df['LIMIT_BAL'] + 1)
    df['UTIL_RATE_MEAN'] = df['BILL_MEAN'] / (df['LIMIT_BAL'] + 1)
    df['UTIL_RATE_MAX'] = df['BILL_MAX'] / (df['LIMIT_BAL'] + 1)
    
    df['BALANCE_1'] = df['BILL_AMT1'] - df['PAY_AMT1']
    df['BALANCE_GROWTH'] = df['BILL_AMT1'] - df['BILL_AMT6']
    
    df['HAS_NEG_BILL'] = (df[bill_cols] < 0).any(axis=1).astype(int)
    df['NUM_NEG_BILL'] = (df[bill_cols] < 0).sum(axis=1)
    df['NUM_ZERO_PAY'] = (df[pay_amt_cols] == 0).sum(axis=1)
    
    df['LOG_LIMIT_BAL'] = np.log1p(df['LIMIT_BAL'])
    df['LOG_BILL_MEAN'] = np.log1p(df['BILL_MEAN'].clip(lower=0))
    df['LOG_PAY_AMT_MEAN'] = np.log1p(df['PAY_AMT_MEAN'])
    
    df['AGE_LIMIT'] = df['AGE'] * df['LIMIT_BAL']
    df['PAY0_LIMIT'] = df['PAY_0'] * df['LIMIT_BAL']
    df['PAY0_BILL1'] = df['PAY_0'] * df['BILL_AMT1']
    df['LATE_UTIL'] = df['NUM_LATE'] * df['UTIL_RATE']
    
    df['EDU_GROUPED'] = df['EDUCATION'].apply(lambda x: x if x in [1,2,3] else 4)
    df['MAR_GROUPED'] = df['MARRIAGE'].apply(lambda x: x if x in [1,2,3] else 0)
    
    df['PAY0_LATE'] = (df['PAY_0'] > 0).astype(int)
    df['PAY0_SEVERE'] = (df['PAY_0'] >= 2).astype(int)
    
    for i in range(1, 6):
        df[f'BILL_CHANGE_{i}'] = df[f'BILL_AMT{i}'] - df[f'BILL_AMT{i+1}']
        df[f'PAY_AMT_CHANGE_{i}'] = df[f'PAY_AMT{i}'] - df[f'PAY_AMT{i+1}']
    
    return df

train = create_features(train)
test = create_features(test)

print(f"Features: {train.shape[1]}")

n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# === LightGBM Model 1 ===
lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.02,
    'num_leaves': 31,
    'max_depth': 6,
    'min_child_samples': 50,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'n_estimators': 2000,
    'verbose': -1,
    'random_state': 42,
}

lgb_oof = np.zeros(len(train))
lgb_preds = np.zeros(len(test))
lgb_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(train, y)):
    X_tr, X_val = train.iloc[train_idx], train.iloc[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    
    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )
    
    lgb_oof[val_idx] = model.predict_proba(X_val)[:, 1]
    lgb_preds += model.predict_proba(test)[:, 1] / n_splits
    
    score = roc_auc_score(y_val, lgb_oof[val_idx])
    lgb_scores.append(score)

print(f"LGB1 Mean AUC: {np.mean(lgb_scores):.6f} (+/- {np.std(lgb_scores):.6f})")
print(f"LGB1 OOF AUC: {roc_auc_score(y, lgb_oof):.6f}")

# === XGBoost ===
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'learning_rate': 0.02,
    'max_depth': 5,
    'min_child_weight': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'n_estimators': 2000,
    'verbosity': 0,
    'random_state': 42,
    'early_stopping_rounds': 100,
}

xgb_oof = np.zeros(len(train))
xgb_preds = np.zeros(len(test))
xgb_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(train, y)):
    X_tr, X_val = train.iloc[train_idx], train.iloc[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    
    xgb_oof[val_idx] = model.predict_proba(X_val)[:, 1]
    xgb_preds += model.predict_proba(test)[:, 1] / n_splits
    
    score = roc_auc_score(y_val, xgb_oof[val_idx])
    xgb_scores.append(score)

print(f"XGB Mean AUC: {np.mean(xgb_scores):.6f} (+/- {np.std(xgb_scores):.6f})")
print(f"XGB OOF AUC: {roc_auc_score(y, xgb_oof):.6f}")

# === LightGBM Model 2 (different hyperparams) ===
lgb_params2 = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.01,
    'num_leaves': 63,
    'max_depth': 8,
    'min_child_samples': 30,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.5,
    'reg_lambda': 2.0,
    'n_estimators': 3000,
    'verbose': -1,
    'random_state': 123,
}

lgb2_oof = np.zeros(len(train))
lgb2_preds = np.zeros(len(test))
lgb2_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(train, y)):
    X_tr, X_val = train.iloc[train_idx], train.iloc[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    
    model = lgb.LGBMClassifier(**lgb_params2)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )
    
    lgb2_oof[val_idx] = model.predict_proba(X_val)[:, 1]
    lgb2_preds += model.predict_proba(test)[:, 1] / n_splits
    
    score = roc_auc_score(y_val, lgb2_oof[val_idx])
    lgb2_scores.append(score)

print(f"LGB2 Mean AUC: {np.mean(lgb2_scores):.6f} (+/- {np.std(lgb2_scores):.6f})")
print(f"LGB2 OOF AUC: {roc_auc_score(y, lgb2_oof):.6f}")

# === LightGBM Model 3 (DART) ===
lgb_params3 = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'dart',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': 6,
    'min_child_samples': 40,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.2,
    'reg_lambda': 1.5,
    'n_estimators': 500,
    'verbose': -1,
    'random_state': 456,
}

lgb3_oof = np.zeros(len(train))
lgb3_preds = np.zeros(len(test))
lgb3_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(train, y)):
    X_tr, X_val = train.iloc[train_idx], train.iloc[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    
    model = lgb.LGBMClassifier(**lgb_params3)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              callbacks=[lgb.log_evaluation(0)])
    
    lgb3_oof[val_idx] = model.predict_proba(X_val)[:, 1]
    lgb3_preds += model.predict_proba(test)[:, 1] / n_splits
    
    score = roc_auc_score(y_val, lgb3_oof[val_idx])
    lgb3_scores.append(score)

print(f"LGB3 (DART) Mean AUC: {np.mean(lgb3_scores):.6f} (+/- {np.std(lgb3_scores):.6f})")
print(f"LGB3 (DART) OOF AUC: {roc_auc_score(y, lgb3_oof):.6f}")

# === Optimal ensemble weights ===
from scipy.optimize import minimize

def neg_auc(weights):
    w = np.array(weights)
    w = w / w.sum()
    blend = w[0]*lgb_oof + w[1]*xgb_oof + w[2]*lgb2_oof + w[3]*lgb3_oof
    return -roc_auc_score(y, blend)

result = minimize(neg_auc, x0=[0.25, 0.25, 0.25, 0.25], 
                  bounds=[(0,1)]*4,
                  method='Nelder-Mead')

opt_weights = result.x / result.x.sum()
print(f"\nOptimal weights: LGB1={opt_weights[0]:.4f}, XGB={opt_weights[1]:.4f}, LGB2={opt_weights[2]:.4f}, LGB3={opt_weights[3]:.4f}")

blend_oof = opt_weights[0]*lgb_oof + opt_weights[1]*xgb_oof + opt_weights[2]*lgb2_oof + opt_weights[3]*lgb3_oof
print(f"Ensemble OOF AUC: {roc_auc_score(y, blend_oof):.6f}")

# Final predictions
final_preds = opt_weights[0]*lgb_preds + opt_weights[1]*xgb_preds + opt_weights[2]*lgb2_preds + opt_weights[3]*lgb3_preds

# Create submission
submission = pd.DataFrame({
    'id': test_ids,
    'default_prob': final_preds
})

submission.to_csv('/app/submission.csv', index=False)
print(f"\nSubmission saved. Shape: {submission.shape}")
print(submission.head())
print(f"\nPrediction stats:")
print(submission['default_prob'].describe())