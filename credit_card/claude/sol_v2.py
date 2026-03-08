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

train_ids = train['id'].values
test_ids = test['id'].values
y = train['default_payment_next_month'].values

train = train.drop(columns=['id', 'default_payment_next_month'])
test = test.drop(columns=['id'])

def create_features(df):
    df = df.copy()
    pay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
    pay_amt_cols = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    
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

# Use multiple seeds for more stable predictions
all_oof = []
all_preds = []
model_names = []

seeds = [42, 123, 456, 789, 2024]

for seed_idx, seed in enumerate(seeds):
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    # LightGBM
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
        'random_state': seed,
    }
    
    oof = np.zeros(len(train))
    preds = np.zeros(len(test))
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train, y)):
        X_tr, X_val = train.iloc[train_idx], train.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        oof[val_idx] = model.predict_proba(X_val)[:, 1]
        preds += model.predict_proba(test)[:, 1] / n_splits
        scores.append(roc_auc_score(y_val, oof[val_idx]))
    
    print(f"LGB seed={seed}: OOF AUC = {roc_auc_score(y, oof):.6f}, Mean = {np.mean(scores):.6f}")
    all_oof.append(oof)
    all_preds.append(preds)
    model_names.append(f'lgb_s{seed}')
    
    # XGBoost
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
        'random_state': seed,
        'early_stopping_rounds': 100,
    }
    
    oof = np.zeros(len(train))
    preds = np.zeros(len(test))
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train, y)):
        X_tr, X_val = train.iloc[train_idx], train.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        oof[val_idx] = model.predict_proba(X_val)[:, 1]
        preds += model.predict_proba(test)[:, 1] / n_splits
        scores.append(roc_auc_score(y_val, oof[val_idx]))
    
    print(f"XGB seed={seed}: OOF AUC = {roc_auc_score(y, oof):.6f}, Mean = {np.mean(scores):.6f}")
    all_oof.append(oof)
    all_preds.append(preds)
    model_names.append(f'xgb_s{seed}')

# Simple average ensemble
ensemble_oof = np.mean(all_oof, axis=0)
ensemble_preds = np.mean(all_preds, axis=0)
print(f"\nSimple Average Ensemble OOF AUC: {roc_auc_score(y, ensemble_oof):.6f}")

# Optimize weights
from scipy.optimize import minimize

def neg_auc(weights):
    w = np.array(weights)
    w = w / w.sum()
    blend = sum(w[i] * all_oof[i] for i in range(len(all_oof)))
    return -roc_auc_score(y, blend)

n_models = len(all_oof)
result = minimize(neg_auc, x0=[1/n_models]*n_models,
                  bounds=[(0,1)]*n_models,
                  method='Nelder-Mead')
opt_w = result.x / result.x.sum()

blend_oof = sum(opt_w[i] * all_oof[i] for i in range(n_models))
blend_preds = sum(opt_w[i] * all_preds[i] for i in range(n_models))
print(f"Optimized Ensemble OOF AUC: {roc_auc_score(y, blend_oof):.6f}")

for name, w in zip(model_names, opt_w):
    print(f"  {name}: {w:.4f}")

# Use the better ensemble
if roc_auc_score(y, blend_oof) > roc_auc_score(y, ensemble_oof):
    final_preds = blend_preds
    print("Using optimized weights")
else:
    final_preds = ensemble_preds
    print("Using simple average")

submission = pd.DataFrame({
    'id': test_ids,
    'default_prob': final_preds
})
submission.to_csv('/app/submission.csv', index=False)
print(f"\nSubmission saved. Shape: {submission.shape}")
print(submission.head())
print(f"\nPrediction stats:")
print(submission['default_prob'].describe())