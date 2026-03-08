import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import HistGradientBoostingClassifier

RANDOM_STATE = 42


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
    pay_amt_cols = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

    # Repayment behavior summaries
    out['pay_mean'] = out[pay_cols].mean(axis=1)
    out['pay_max'] = out[pay_cols].max(axis=1)
    out['pay_min'] = out[pay_cols].min(axis=1)
    out['pay_std'] = out[pay_cols].std(axis=1)
    out['pay_delinq_count_ge1'] = (out[pay_cols] >= 1).sum(axis=1)
    out['pay_delinq_count_ge2'] = (out[pay_cols] >= 2).sum(axis=1)
    out['pay_no_use_count'] = (out[pay_cols] == -2).sum(axis=1)
    out['pay_paid_full_count'] = (out[pay_cols] == -1).sum(axis=1)

    # Amount summaries
    out['bill_sum'] = out[bill_cols].sum(axis=1)
    out['bill_mean'] = out[bill_cols].mean(axis=1)
    out['bill_std'] = out[bill_cols].std(axis=1)
    out['bill_max'] = out[bill_cols].max(axis=1)

    out['pay_amt_sum'] = out[pay_amt_cols].sum(axis=1)
    out['pay_amt_mean'] = out[pay_amt_cols].mean(axis=1)
    out['pay_amt_std'] = out[pay_amt_cols].std(axis=1)
    out['pay_amt_max'] = out[pay_amt_cols].max(axis=1)

    # Ratios and short-term changes
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
    X_train = add_features(train.drop(columns=['default_payment_next_month', 'id']))
    X_test = add_features(test.drop(columns=['id']))

    model = HistGradientBoostingClassifier(
        learning_rate=0.02,
        max_depth=4,
        max_iter=400,
        l2_regularization=1.0,
        min_samples_leaf=20,
        random_state=RANDOM_STATE,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X_train, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    print(f"solution_v1 HistGB CV AUC: mean={cv_scores.mean():.6f} std={cv_scores.std():.6f} folds={cv_scores}")

    model.fit(X_train, y)
    preds = model.predict_proba(X_test)[:, 1]

    submission = pd.DataFrame({'id': test['id'], 'default_prob': preds})
    submission.to_csv('/app/submission.csv', index=False)


if __name__ == '__main__':
    main()