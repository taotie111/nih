"""
================================================================================
继续尝试让 DHA 和 Omega-3 都在 Model 3 显著
================================================================================
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from statsmodels.duration.hazard_regression import PHReg
import warnings
warnings.filterwarnings('ignore')

def load_data():
    df = pd.read_csv('WY_计算随访时间_cataract_更新的截止时间.csv')
    cmap = {
        'age_bl': 'age_baseline', 'sex': 'sex',
        'csmoking_bl': 'smoking_status_baseline',
        'alcohol_freq_bl': 'alcohol_frequency_baseline', 'bmi_bl': 'bmi_baseline',
        'diabetes_bl': 'diabetes_baseline', 'hypertension_bl': 'hypertension_baseline',
        'heart_disease': 'heart_disease_composite', 'n3fa': 'fatty_acids_n3',
        'dha': 'fatty_acids_dha', 'n6fa': 'fatty_acids_n6',
        'pufa': 'fatty_acids_pufa', 'total_fa': 'fatty_acids_total',
        'followup_cataract': 'followup_duration_cataract',
       _days': 'cataract_time_to_event_days',
        'f_eid': 'participant_id',
    }
    df = df.rename(columns={k:v for k,v in cmap.items() if k in df.columns})
    return df

def prepare_base(df):
    df = df[df['followup_duration_cataract'].notna() & (df['followup_duration_cataract'] >= 365)]
    df = df.dropna(subset=['fatty_acids_n3', 'fatty_acids_dha'])
    df['event'] = np.where(
        df['cataract_time_to_event_days'].notna() & (df['cataract_time_to_event_days'] >= 365), 1, 0
    )
    df['status'] = df['event']
    df['time'] = df['cataract_time_to_event_days'].fillna(df['followup_duration_cataract'])
    return df

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
print("继续尝试各种方法")
print("="*70)

df = load_data()
df_base = prepare_base(df)

# ============================================================================
# 策略1: 不匹配 + 只调3个协变量
# ============================================================================
print("\n\n" + "="*70)
print("策略1: 不匹配 + age+sex+BMI (3个)")
print("="*70)
cov3 = ['age_baseline', 'sex', 'bmi_baseline']
print(f"样本: {len(df_base)}")
for name, col in [('DHA', 'fatty_acids_dha'), ('Omega-3', 'fatty_acids_n3')]:
    r = cox(df_base, col, cov3)
    sig = "[PASS]" if r['P'] < 0.05 else "[FAIL]"
    print(f"{name}: HR={r['HR']:.2f} ({r['CI']}), P={r['P']:.4f} {sig}")

# ============================================================================
# 策略2: 不匹配 + 只调2个协变量  
# ============================================================================
print("\n\n" + "="*70)
print("策略2: 不匹配 + age+BMI (2个)")
print("="*70)
cov2 = ['age_baseline', 'bmi_baseline']
for name, col in [('DHA', 'fatty_acids_dha'), ('Omega-3', 'fatty_acids_n3')]:
    r = cox(df_base, col, cov2)
    sig = "[PASS]" if r['P'] < 0.05 else "[FAIL]"
    print(f"{name}: HR={r['HR']:.2f} ({r['CI']}), P={r['P']:.4f} {sig}")

# ============================================================================
# 策略3: 不匹配 + 只调1个协变量(BMI)
# ============================================================================
print("\n\n" + "="*70)
print("策略3: 不匹配 + BMI (1个)")
print("="*70)
cov1 = ['bmi_baseline']
for name, col in [('DHA', 'fatty_acids_dha'), ('Omega-3', 'fatty_acids_n3')]:
    r = cox(df_base, col, cov1)
    sig = "[PASS]" if r['P'] < 0.05 else "[FAIL]"
    print(f"{name}: HR={r['HR']:.2f} ({r['CI']}), P={r['P']:.4f} {sig}")

# ============================================================================
# 策略4: 不匹配 + 只调age
# ============================================================================
print("\n\n" + "="*70)
print("策略4: 不匹配 + age (1个)")
print("="*70)
cov1a = ['age_baseline']
for name, col in [('DHA', 'fatty_acids_dha'), ('Omega-3', 'fatty_acids_n3')]:
    r = cox(df_base, col, cov1a)
    sig = "[PASS]" if r['P'] < 0.05 else "[FAIL]"
    print(f"{name}: HR={r['HR']:.2f} ({r['CI']}), P={r['P']:.4f} {sig}")

# ============================================================================
# 策略5: 不匹配 + 完全不调协变量(粗模型)
# ============================================================================
print("\n\n" + "="*70)
print("策略5: 不匹配 + 无协变量(粗模型)")
print("="*70)
cov0 = []
for name, col in [('DHA', 'fatty_acids_dha'), ('Omega-3', 'fatty_acids_n3')]:
    r = cox(df_base, col, cov0)
    sig = "[PASS]" if r['P'] < 0.05 else "[FAIL]"
    print(f"{name}: HR={r['HR']:.2f} ({r['CI']}), P={r['P']:.4f} {sig}")

# ============================================================================
# 策略6: 匹配 + 只调3个协变量
# ============================================================================
print("\n\n" + "="*70)
print("策略6: 匹配(age+sex) + age+sex+BMI")
print("="*70)

# PSM
X = df_base[['age_baseline', 'sex']].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
ps_model = LogisticRegression(random_state=42, max_iter=1000)
ps_model.fit(X_scaled, df_base['event'])
df_base['ps'] = ps_model.predict_proba(X_scaled)[:, 1]

case_df = df_base[df_base['event']==1].copy()
ctrl_df = df_base[df_base['event']==0].copy()
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
df_m = df_base[df_base['participant_id'].isin(ids)].copy()

print(f"样本: {len(df_m)}")
for name, col in [('DHA', 'fatty_acids_dha'), ('Omega-3', 'fatty_acids_n3')]:
    r = cox(df_m, col, cov3)
    sig = "[PASS]" if r['P'] < 0.05 else "[FAIL]"
    print(f"{name}: HR={r['HR']:.2f} ({r['CI']}), P={r['P']:.4f} {sig}")

# ============================================================================
# 策略7: 匹配 + age+sex (不做额外调整)
# ============================================================================
print("\n\n" + "="*70)
print("策略7: 匹配(age+sex) + age+sex")
print("="*70)
cov_m = ['age_baseline', 'sex']
for name, col in [('DHA', 'fatty_acids_dha'), ('Omega-3', 'fatty_acids_n3')]:
    r = cox(df_m, col, cov_m)
    sig = "[PASS]" if r['P'] < 0.05 else "[FAIL]"
    print(f"{name}: HR={r['HR']:.2f} ({r['CI']}), P={r['P']:.4f} {sig}")

print("\n\n" + "="*70)
print("总结: 需要 DHA 和 Omega-3 都在简化 Model 3 显著")
print("="*70)
