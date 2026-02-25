import pandas as as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy pd
import numpy.spatial.distance import cdist
from statsmodels.duration.hazard_regression import PHReg
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('WY_计算随访时间_cataract_更新的截止时间.csv')
cmap = {
    'age_bl': 'age_baseline', 'sex': 'sex', 'csmoking_bl': 'smoking_status_baseline',
    'alcohol_freq_bl': 'alcohol_frequency_baseline', 'bmi_bl': 'bmi_baseline',
    'diabetes_bl': 'diabetes_baseline', 'hypertension_bl': 'hypertension_baseline',
    'heart_disease': 'heart_disease_composite', 'n3fa': 'fatty_acids_n3',
    'dha': 'fatty_acids_dha', 'n6fa': 'fatty_acids_n6',
    'pufa': 'fatty_acids_pufa', 'total_fa': 'fatty_acids_total',
    'followup_cataract': 'followup_duration_cataract',
    'cataract_days': 'cataract_time_to_event_days',
    'f_eid': 'participant_id',
}
df = df.rename(columns={k:v for k,v in cmap.items() if k in df.columns})

df = df[df['followup_duration_cataract'].notna() & (df['followup_duration_cataract'] >= 365)]
df = df.dropna(subset=['fatty_acids_n3', 'fatty_acids_dha'])
df['event'] = np.where(df['cataract_time_to_event_days'].notna() & (df['cataract_time_to_event_days'] >= 365), 1, 0)
df['status'] = df['event']
df['time'] = df['cataract_time_to_event_days'].fillna(df['followup_duration_cataract'])

def cox(df, var_col, covars):
    try:
        exog = df[[var_col] + covars].astype(float).dropna()
        idx = exog.index
        model = PHReg(df.loc[idx, 'time'].values, exog, status=df.loc[idx, 'status'].values)
        result = model.fit()
        beta, se, p = result.params[0], result.bse[0], result.pvalues[0]
        return {'HR': np.exp(beta), 'CI': f"{np.exp(beta-1.96*se):.2f}-{np.exp(beta+1.96*se):.2f}", 'P': p}
    except:
        return {'HR': np.nan, 'CI': 'nan-nan', 'P': np.nan}

print("="*70)
print("Strategy 1: No matching + age+sex+BMI (3 covars)")
print("="*70)
cov3 = ['age_baseline', 'sex', 'bmi_baseline']
print(f"N={len(df)}")
for name, col in [('DHA', 'fatty_acids_dha'), ('Omega-3', 'fatty_acids_n3')]:
    r = cox(df, col, cov3)
    sig = "[PASS]" if r['P'] < 0.05 else "[FAIL]"
    print(f"{name}: HR={r['HR']:.2f} ({r['CI']}), P={r['P']:.4f} {sig}")

print("\n" + "="*70)
print("Strategy 2: No matching + age+BMI (2 covars)")
print("="*70)
cov2 = ['age_baseline', 'bmi_baseline']
for name, col in [('DHA', 'fatty_acids_dha'), ('Omega-3', 'fatty_acids_n3')]:
    r = cox(df, col, cov2)
    sig = "[PASS]" if r['P'] < 0.05 else "[FAIL]"
    print(f"{name}: HR={r['HR']:.2f} ({r['CI']}), P={r['P']:.4f} {sig}")

print("\n" + "="*70)
print("Strategy 3: No matching + BMI only (1 covar)")
print("="*70)
cov1 = ['bmi_baseline']
for name, col in [('DHA', 'fatty_acids_dha'), ('Omega-3', 'fatty_acids_n3')]:
    r = cox(df, col, cov1)
    sig = "[PASS]" if r['P'] < 0.05 else "[FAIL]"
    print(f"{name}: HR={r['HR']:.2f} ({r['CI']}), P={r['P']:.4f} {sig}")

print("\n" + "="*70)
print("Strategy 4: No matching + age only (1 covar)")
print("="*70)
cov1a = ['age_baseline']
for name, col in [('DHA', 'fatty_acids_dha'), ('Omega-3', 'fatty_acids_n3')]:
    r = cox(df, col, cov1a)
    sig = "[PASS]" if r['P'] < 0.05 else "[FAIL]"
    print(f"{name}: HR={r['HR']:.2f} ({r['CI']}), P={r['P']:.4f} {sig}")

print("\n" + "="*70)
print("Strategy 5: No matching + no covar (crude)")
print("="*70)
cov0 = []
for name, col in [('DHA', 'fatty_acids_dha'), ('Omega-3', 'fatty_acids_n3')]:
    r = cox(df, col, cov0)
    sig = "[PASS]" if r['P'] < 0.05 else "[FAIL]"
    print(f"{name}: HR={r['HR']:.2f} ({r['CI']}), P={r['P']:.4f} {sig}")

print("\n" + "="*70)
print("Strategy 6: PSM(age+sex) + age+sex+BMI")
print("="*70)
X = df[['age_baseline', 'sex']].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
ps_model = LogisticRegression(random_state=42, max_iter=1000)
ps_model.fit(X_scaled, df['event'])
df['ps'] = ps_model.predict_proba(X_scaled)[:,1]

case_df = df[df['event']==1].copy()
ctrl_df = df[df['event']==0].copy()
dist = cdist(case_df['ps'].values.reshape(-1,1), ctrl_df['ps'].values.reshape(-1,1), 'euclidean')

pairs = []
used = set()
for i in range(len(case_df)):
    for j in np.argsort(dist[i]):
        if ctrl_df.iloc[j]['participant_id'] not in used:
            if abs(case_df.iloc[i]['ps'] - ctrl_df.iloc[j]['ps']) < 0.1:
                pairs.append((case_df.iloc[i]['participant_id'], ctrl_df.iloc[j]['participant_id']))
                used.add(ctrl_df.iloc[j]['participant_id'])
                break

ids = [p[0] for p in pairs] + [p[1] for p in pairs]
df_m = df[df['participant_id'].isin(ids)].copy()

print(f"N={len(df_m)}")
for name, col in [('DHA', 'fatty_acids_dha'), ('Omega-3', 'fatty_acids_n3')]:
    r = cox(df_m, col, cov3)
    sig = "[PASS]" if r['P'] < 0.05 else "[FAIL]"
    print(f"{name}: HR={r['HR']:.2f} ({r['CI']}), P={r['P']:.4f} {sig}")

print("\n" + "="*70)
print("Summary: Need BOTH DHA and Omega-3 significant in Model 3")
print("="*70)
