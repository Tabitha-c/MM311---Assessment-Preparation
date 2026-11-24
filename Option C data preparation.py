## MM311 option c
## Plan: inspect each file, fit preprocessing and models on development.csv only, then freeze the method, test review policies on validation.csv, and apply it to holdout.csv.

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import average_precision_score, roc_auc_score
from IPython.display import display

sns.set_theme(style='whitegrid')
Path('results').mkdir(exist_ok=True)
def finish(fig, name):
    fig.tight_layout()
    fig.savefig(f'results/{name}.png', dpi=160, bbox_inches='tight')
    plt.show()
    plt.close(fig)

##Load the data
dev = pd.read_csv('development.csv', dtype={'case_id': str})
val = pd.read_csv('validation.csv', dtype={'case_id': str})
hold = pd.read_csv('holdout.csv', dtype={'case_id': str})
context = pd.read_csv('review_context_SIMULATED.csv', dtype={'case_id': str})
scenarios = pd.read_csv('review_scenarios_SIMULATED.csv')
print('Development:', dev.shape, 'Validation:', val.shape, 'Holdout:', hold.shape)
display(dev.head(), scenarios)

## Class imbalance
counts = dev['backorder'].value_counts()
share = dev['backorder'].value_counts(normalize=True)
display(pd.DataFrame({'Cases': counts, 'Share': share.round(4)}))
print('Always-No accuracy:', round(share['No'], 4))
print('Validation share of Yes:', round((val['backorder'] == 'Yes').mean(), 4))
fig, ax = plt.subplots(figsize=(5, 3))
counts.plot.bar(ax=ax, color='#326779')
ax.set(title='Backorder classes (development)', ylabel='Cases', xlabel='')
finish(fig, 'class_balance')

## missing values & -99 performance code
perf_cols = ['perf_6_month_avg', 'perf_12_month_avg']
for name, df in [('development', dev), ('validation', val), ('holdout', hold)]:
    print(name)
    missing = df.isna().sum()
    print('  NaN:', missing[missing > 0].to_dict())
    for col in perf_cols:
        print('  -99 in', col, ':', (df[col] == -99).sum(), f'({(df[col] == -99).mean():.1%})')
# -99 only appears in the performance columns
print('Other columns containing -99:', [c for c in dev.columns if dev[c].dtype != object and c not in perf_cols and (dev[c] == -99).any()])
# range of the performance columns once -99 is treated as missing
perf_clean = dev[perf_cols].replace(-99, np.nan)
display(perf_clean.describe().round(3))

## is missingness related to the target - backorder rate of each group
dev['perf6_is_99'] = dev['perf_6_month_avg'] == -99
dev['lead_missing'] = dev['lead_time'].isna()
display((dev.groupby('perf6_is_99')['backorder'].apply(lambda s: (s == 'Yes').mean())).round(4))
display((dev.groupby('lead_missing')['backorder'].apply(lambda s: (s == 'Yes').mean())).round(4))
dev = dev.drop(columns=['perf6_is_99', 'lead_missing'])

## other checks - negative inventory and overlapping ids
print('Negative national_inv rows:', (dev['national_inv'] < 0).sum())
flags = ['potential_issue', 'deck_risk', 'oe_constraint', 'ppap_risk', 'stop_auto_buy', 'rev_stop']
display(pd.DataFrame({c: dev[c].value_counts() for c in flags}).T)
print('Overlapping IDs dev/val:', len(set(dev['case_id']) & set(val['case_id'])))
print('Overlapping IDs dev/holdout:', len(set(dev['case_id']) & set(hold['case_id'])))
print('Same columns in validation:', list(val.columns) == list(dev.columns))
# context must join one-to-one with validation and holdout
for name, df in [('validation', val), ('holdout', hold)]:
    matched = df['case_id'].isin(context['case_id']).sum()
    print(name, 'cases with context:', matched, 'of', len(df))
display(context.groupby('review_minutes')['missed_event_loss_gbp'].agg(['min', 'max', 'size']))

## preprocessing
quantity_cols = ['national_inv', 'lead_time', 'in_transit_qty', 'forecast_3_month',
                 'forecast_6_month', 'forecast_9_month', 'sales_1_month', 'sales_3_month',
                 'sales_6_month', 'sales_9_month', 'min_bank', 'pieces_past_due']
def prepare(df):
    X = pd.DataFrame(index=df.index)
    for col in perf_cols:
        X[col] = df[col].replace(-99, np.nan)
        X[col + '_missing'] = X[col].isna().astype(int)
    X['lead_time_missing'] = df['lead_time'].isna().astype(int)
    for col in quantity_cols:
        X[col] = np.sign(df[col]) * np.log1p(df[col].abs())
    for col in flags:
        X[col] = (df[col] == 'Yes').astype(int)
    return X
X_dev = prepare(dev)
y_dev = (dev['backorder'] == 'Yes').astype(int)
print('Predictors used:', list(X_dev.columns))
print('ID or context column used?', any(c in X_dev.columns for c in ['case_id', 'batch_id', 'review_minutes', 'missed_event_loss_gbp']))

## fit and compare models - development only
models = {
    'Logistic': make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                              LogisticRegression(class_weight='balanced', C=0.1, max_iter=2000)),
    'GradBoost': make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                               HistGradientBoostingClassifier(class_weight='balanced', learning_rate=0.05,
                                   max_depth=4, max_iter=200, min_samples_leaf=40, l2_regularization=1.0,
                                   random_state=42)),
}
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = {}
rows = []
for name, model in models.items():
    cv_scores[name] = cross_val_predict(model, X_dev, y_dev, cv=folds, method='predict_proba')[:, 1]
    p = cv_scores[name]
    # share of positives found when we look at the top 5% highest scores
    top = np.argsort(-p)[:int(0.05 * len(p))]
    rows.append({'Model': name, 'PR-AUC': average_precision_score(y_dev, p),
                 'ROC-AUC': roc_auc_score(y_dev, p),
                 'Recall in top 5%': y_dev.iloc[top].sum() / y_dev.sum(),
                 'Precision in top 5%': y_dev.iloc[top].mean()})
cv_table = pd.DataFrame(rows).set_index('Model')
print('Prevalence (no-skill PR-AUC):', round(y_dev.mean(), 4))
display(cv_table.round(3))
best_name = cv_table['PR-AUC'].idxmax()
print('Chosen model (development CV PR-AUC):', best_name)

## calibrate the scores
def to_logit(p):
    p = np.clip(p, 0.000001, 0.999999)
    return np.log(p / (1 - p)).reshape(-1, 1)
calibrator = LogisticRegression(C=1000000, max_iter=1000)
calibrator.fit(to_logit(cv_scores[best_name]), y_dev)
final_model = models[best_name].fit(X_dev, y_dev)
def risk(df):
    # calibrated probability of backorder for any dataset
    raw = final_model.predict_proba(prepare(df))[:, 1]
    return calibrator.predict_proba(to_logit(raw))[:, 1]
print('Mean calibrated risk on development:', round(risk(dev).mean(), 4), 'vs prevalence', round(y_dev.mean(), 4))

## decision rule for reviews
def pick_cases(values, minutes, capacity):
    # best[i, t] = best total value using the first i cases and at most t minutes
    n = len(values)
    best = np.zeros((n + 1, capacity + 1))
    for i in range(1, n + 1):
        v = values[i - 1]
        w = minutes[i - 1]
        best[i] = best[i - 1]
        if v > 0 and w <= capacity:
            with_case = best[i - 1, :capacity + 1 - w] + v
            best[i, w:] = np.maximum(best[i - 1, w:], with_case)
    # walk backwards to find which cases were chosen
    chosen = np.zeros(n, dtype=int)
    t = capacity
    for i in range(n, 0, -1):
        if best[i, t] > best[i - 1, t]:
            chosen[i - 1] = 1
            t = t - minutes[i - 1]
    return chosen

def allocate(df, scenario):
    # df needs: case_id, batch_id, risk, review_minutes, missed_event_loss_gbp
    df = df.sort_values(['batch_id', 'case_id']).reset_index(drop=True)
    df['priority_score'] = (df['risk'] * df['missed_event_loss_gbp'] * scenario['missed_loss_multiplier']
                            - df['review_minutes'] * scenario['cost_gbp_per_minute'])
    df['selected'] = 0
    for batch in df['batch_id'].unique():
        rows = df.index[df['batch_id'] == batch]
        chosen = pick_cases(df.loc[rows, 'priority_score'].values,
                            df.loc[rows, 'review_minutes'].values.astype(int),
                            int(scenario['minutes_per_batch']))
        df.loc[rows, 'selected'] = chosen
        assert df.loc[rows, 'review_minutes'][chosen == 1].sum() <= scenario['minutes_per_batch']
    return df

def risk_only(df, scenario):
    # comparison policy: highest risk first, add a case if it still fits in the batch
    df = df.sort_values(['batch_id', 'risk', 'case_id'], ascending=[True, False, True]).reset_index(drop=True)
    df['selected'] = 0
    used = {}
    for i in df.index:
        batch = df.loc[i, 'batch_id']
        if used.get(batch, 0) + df.loc[i, 'review_minutes'] <= scenario['minutes_per_batch']:
            df.loc[i, 'selected'] = 1
            used[batch] = used.get(batch, 0) + df.loc[i, 'review_minutes']
    return df

def total_loss(df, scenario):
    chosen = df['selected'] == 1
    positive = df['actual'] == 1
    review_cost = (df.loc[chosen, 'review_minutes'] * scenario['cost_gbp_per_minute']).sum()
    missed = (df.loc[~chosen & positive, 'missed_event_loss_gbp'] * scenario['missed_loss_multiplier']).sum()
    found = (chosen & positive).sum()
    precision = found / chosen.sum() if chosen.sum() > 0 else np.nan   # undefined if nothing selected
    return {'Reviewed': chosen.sum(), 'Minutes': df.loc[chosen, 'review_minutes'].sum(),
            'Positives found': found, 'Precision': precision, 'Recall': found / positive.sum(),
            'Review cost': review_cost, 'Missed loss': missed, 'Total loss': review_cost + missed}

## check the rule
test = np.array([300, 400, 150, 50])
minutes = np.array([25, 15, 8, 8])
print(pick_cases(test, minutes, 30))   # expect [0 1 1 0]

## validation - compare policies
V = val.merge(context, on='case_id', how='left', validate='one_to_one')
print('Missing context values:', V[['batch_id', 'review_minutes', 'missed_event_loss_gbp']].isna().sum().sum())
V['risk'] = risk(val)
V['actual'] = (V['backorder'] == 'Yes').astype(int)
print('Validation PR-AUC:', round(average_precision_score(V['actual'], V['risk']), 4),
      'ROC-AUC:', round(roc_auc_score(V['actual'], V['risk']), 4), 'Positives:', V['actual'].sum())
V = V[['case_id', 'batch_id', 'risk', 'review_minutes', 'missed_event_loss_gbp', 'actual']]

results = []
batch_results = []
for _, sc in scenarios.iterrows():
    no_review = V.assign(selected=0)
    greedy = risk_only(V, sc)
    knap = allocate(V, sc)
    for name, d in [('No review', no_review), ('Risk-only', greedy), ('Expected-value knapsack', knap)]:
        row = total_loss(d, sc)
        row.update({'Scenario': sc['scenario'], 'Policy': name})
        results.append(row)
    for batch in knap['batch_id'].unique():
        row = total_loss(knap[knap['batch_id'] == batch], sc)
        row.update({'Scenario': sc['scenario'], 'Batch': batch})
        batch_results.append(row)
results = pd.DataFrame(results)[['Scenario', 'Policy', 'Reviewed', 'Minutes', 'Positives found',
                                 'Precision', 'Recall', 'Review cost', 'Missed loss', 'Total loss']]
batch_results = pd.DataFrame(batch_results)
display(results.round(3))
display(batch_results.pivot(index='Batch', columns='Scenario', values='Total loss').round(0))
## plot
fig, ax = plt.subplots(figsize=(9, 4))
sns.barplot(data=results, x='Scenario', y='Total loss', hue='Policy', ax=ax)
ax.set(title='Validation total loss by scenario (£, 10 batches)', xlabel='')
finish(fig, 'validation_loss')

## freeze the method and score the holdout
H = hold.merge(context, on='case_id', how='left', validate='one_to_one')
H['risk'] = risk(hold)
H = H[['case_id', 'batch_id', 'risk', 'review_minutes', 'missed_event_loss_gbp']]
pieces = []
for _, sc in scenarios.iterrows():
    out = allocate(H, sc)
    out['scenario'] = sc['scenario']
    pieces.append(out[['case_id', 'scenario', 'priority_score', 'selected']])
submission = pd.concat(pieces, ignore_index=True)
print('Rows:', len(submission), 'expected:', len(hold) * len(scenarios))
print('Duplicate case/scenario pairs:', submission.duplicated(['case_id', 'scenario']).sum())
display(submission.groupby('scenario')['selected'].agg(['sum', 'count']))

## save results
submission.to_csv('results/holdout_submission.csv', index=False)
results.to_csv('results/validation_policy_comparison.csv', index=False)
batch_results.to_csv('results/validation_loss_by_batch.csv', index=False)
print('Frozen method:')
print('  model:', best_name, '| preprocessing: -99 to missing, missing flags, signed log1p, median fill, scaling')
print('  calibration: logistic on development cross-validation scores')
print('  priority_score: risk x missed loss x multiplier - review minutes x cost rate')
print('  allocation: best set per batch within the minute limit, positive scores only, ties by case_id')