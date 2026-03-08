import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

RANDOM_STATE = 42


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering to capture repayment behavior and amount dynamics."""
    out = df.copy()

    pay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
    pay_amt_cols = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

    # Repayment status aggregates
    out['pay_mean'] = out[pay_cols].mean(axis=1)
    out['pay_max'] = out[pay_cols].max(axis=1)
    out['pay_min'] = out[pay_cols].min(axis=1)
    out['pay_std'] = out[pay_cols].std(axis=1)
    out['pay_delinq_count_ge1'] = (out[pay_cols] >= 1).sum(axis=1)
    out['pay_delinq_count_ge2'] = (out[pay_cols] >= 2).sum(axis=1)
    out['pay_no_use_count'] = (out[pay_cols] == -2).sum(axis=1)
    out['pay_paid_full_count'] = (out[pay_cols] == -1).sum(axis=1)

    # Bill and payment aggregates
    out['bill_sum'] = out[bill_cols].sum(axis=1)
    out['bill_mean'] = out[bill_cols].mean(axis=1)
    out['bill_std'] = out[bill_cols].std(axis=1)
    out['bill_max'] = out[bill_cols].max(axis=1)

    out['pay_amt_sum'] = out[pay_amt_cols].sum(axis=1)
    out['pay_amt_mean'] = out[pay_amt_cols].mean(axis=1)
    out['pay_amt_std'] = out[pay_amt_cols].std(axis=1)
    out['pay_amt_max'] = out[pay_amt_cols].max(axis=1)

    # Ratios and trends
    out['pay_to_bill_ratio_total'] = out['pay_amt_sum'] / (np.abs(out['bill_sum']) + 1.0)
    out['bill_to_limit_ratio_mean'] = out['bill_mean'] / (out['LIMIT_BAL'] + 1.0)
    out['bill1_to_limit'] = out['BILL_AMT1'] / (out['LIMIT_BAL'] + 1.0)
    out['pay1_to_bill1'] = out['PAY_AMT1'] / (np.abs(out['BILL_AMT1']) + 1.0)

    for i in range(1, 6):
        out[f'bill_diff_{i}_{i+1}'] = out[f'BILL_AMT{i}'] - out[f'BILL_AMT{i+1}']
        out[f'pay_amt_diff_{i}_{i+1}'] = out[f'PAY_AMT{i}'] - out[f'PAY_AMT{i+1}']

    return out


def main():
    train = pd.read_csv('/app/train.csv')
    test = pd.read_csv('/app/test.csv')

    y = train['default_payment_next_month'].astype(int)

    X_train = train.drop(columns=['default_payment_next_month', 'id'])
    X_test = test.drop(columns=['id'])

    X_train = add_features(X_train)
    X_test = add_features(X_test)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    models = {
        'logreg': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=2500, class_weight='balanced', random_state=RANDOM_STATE))
        ]),
        'random_forest': RandomForestClassifier(
            n_estimators=600,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE
        ),
        'hist_gb': HistGradientBoostingClassifier(
            learning_rate=0.03,
            max_depth=5,
            max_iter=500,
            l2_regularization=1.0,
            random_state=RANDOM_STATE
        ),
    }

    # Optional boosted tree libraries
    try:
        from xgboost import XGBClassifier
        models['xgboost'] = XGBClassifier(
            n_estimators=700,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective='binary:logistic',
            eval_metric='auc',
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    except Exception:
        pass

    try:
        from lightgbm import LGBMClassifier
        models['lightgbm'] = LGBMClassifier(
            n_estimators=900,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    except Exception:
        pass

    results = {}
    best_name = None
    best_score = -1.0

    for name, model in models.items():
        scores = cross_val_score(model, X_train, y, cv=cv, scoring='roc_auc', n_jobs=-1)
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))
        results[name] = {
            'fold_scores': [float(s) for s in scores],
            'mean_auc': mean_score,
            'std_auc': std_score,
        }
        print(f"{name}: mean_auc={mean_score:.6f} std={std_score:.6f} folds={scores}")
        if mean_score > best_score:
            best_score = mean_score
            best_name = name

    print(f"Best model: {best_name} (CV AUC={best_score:.6f})")

    best_model = models[best_name]
    best_model.fit(X_train, y)
    test_pred = best_model.predict_proba(X_test)[:, 1]

    submission = pd.DataFrame({
        'id': test['id'],
        'default_prob': test_pred
    })
    submission.to_csv('/app/submission.csv', index=False)

    with open('/app/solution_v0_cv_results.json', 'w') as f:
        json.dump({'best_model': best_name, 'results': results}, f, indent=2)


if __name__ == '__main__':
    main()