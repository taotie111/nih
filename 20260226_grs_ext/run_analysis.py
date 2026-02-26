import pandas as pd
import numpy as np
from statsmodels.duration.hazard_regression import PHReg
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('WY_计算随访时间_cataract_更新的截止时间.csv')
df = df.rename(columns={
    'age_bl': 'age_baseline', 'sex': 'sex', 
    'csmoking_bl': 'smoking_status_baseline',
    'alcohol_freq_bl': 'alcohol_frequency_baseline', 
    'bmi_bl': 'bmi_baseline', 
    'diabetes_bl': 'diabetes_baseline', 
    'hypertension_bl': 'hypertension_baseline',
    'glaucoma_bl': 'glaucoma_baseline',
    'amd_bl': 'amd_baseline'
})

df = df[df['glaucoma_baseline'] == 0]
df = df[df['followup_cataract'].notna()]

covars = ['age_baseline', 'sex', 'bmi_baseline', 'smoking_status_baseline', 'alcohol_frequency_baseline', 'hypertension_baseline']
min_days = 730

# Sensitivity
sens_results = []
for excl_name, excl_col in [('AMD', 'amd_baseline'), ('Diabetes', 'diabetes_baseline'), ('Cancer', 'cancer_bl'), ('Stroke', 'stroke_bl')]:
    df_sens = df[df[excl_col] == 0] if excl_col in df.columns else df
    for exp in ['n3fa_grs', 'n6fa_grs', 'pufa_grs', 'tfa_grs']:
        if exp not in df_sens.columns: continue
        df2 = df_sens.dropna(subset=[exp])
        df2['event'] = (df2['cataract_days'].notna()) & (df2['cataract_days'] >= min_days)
        df2['event'] = df2['event'].astype(int)
        df2['time'] = df2['cataract_days'].fillna(df2['followup_cataract']*365)
        exog = df2[[exp] + covars].astype(float).dropna()
        if exog.shape[0] < 50: continue
        idx = exog.index
        try:
            model = PHReg(df2.loc[idx, 'time'].values, exog, status=df2.loc[idx, 'event'].values)
            res = model.fit()
            beta, se, p = res.params[0], res.bse[0], res.pvalues[0]
            HR = np.exp(beta)
            CI_l, CI_u = np.exp(beta - 1.96 * se), np.exp(beta + 1.96 * se)
            sens_results.append({'Exposure': exp, 'Exclusion': excl_name, 'N': len(exog), 'HR': round(HR,3), 'CI_low': round(CI_l,3), 'CI_high': round(CI_u,3), 'P': round(p,4)})
        except: pass

pd.DataFrame(sens_results).to_csv('20260226_grs_ext/grs_ext_results_sensitivity.csv', index=False)
print('Saved sensitivity results')
print(pd.DataFrame(sens_results).to_string())

# Interaction
print('\n=== INTERACTION ANALYSIS ===')
int_results = []
df_i = df.dropna(subset=['n3fa_grs', 'sex']).copy()
df_i['int_sex'] = df_i['n3fa_grs'] * df_i['sex']
df_i['event'] = (df_i['cataract_days'].notna()) & (df_i['cataract_days'] >= min_days)
df_i['event'] = df_i['event'].astype(int)
df_i['time'] = df_i['cataract_days'].fillna(df_i['followup_cataract']*365)
try:
    exog = df_i[['n3fa_grs', 'sex', 'int_sex'] + covars].astype(float).dropna()
    model = PHReg(df_i.loc[exog.index, 'time'].values, exog, status=df_i.loc[exog.index, 'event'].values)
    res = model.fit()
    idx = list(exog.columns).index('int_sex')
    beta, se, p = res.params[idx], res.bse[idx], res.pvalues[idx]
    HR = np.exp(beta)
    CI_l, CI_u = np.exp(beta - 1.96 * se), np.exp(beta + 1.96 * se)
    int_results.append({'Exposure': 'n3fa_grs', 'Interaction': 'sex', 'HR': round(HR,3), 'CI_low': round(CI_l,3), 'CI_high': round(CI_u,3), 'P': round(p,4)})
except Exception as e:
    print(f'Sex interaction error: {e}')

# Age interaction
df_i['age_grp'] = pd.cut(df_i['age_baseline'], bins=[0,55,65,100], labels=['<55','55-64','>=65'])
for ag in ['<55','55-64','>=65']:
    df_i[f'int_age_{ag}'] = (df_i['age_grp'] == ag).astype(int) * df_i['n3fa_grs']
    try:
        exog = df_i[['n3fa_grs', f'int_age_{ag}'] + covars].astype(float).dropna()
        model = PHReg(df_i.loc[exog.index, 'time'].values, exog, status=df_i.loc[exog.index, 'event'].values)
        res = model.fit()
        idx = list(exog.columns).index(f'int_age_{ag}')
        beta, se, p = res.params[idx], res.bse[idx], res.pvalues[idx]
        HR = np.exp(beta)
        CI_l, CI_u = np.exp(beta - 1.96 * se), np.exp(beta + 1.96 * se)
        int_results.append({'Exposure': 'n3fa_grs', 'Interaction': f'age_{ag}', 'HR': round(HR,3), 'CI_low': round(CI_l,3), 'CI_high': round(CI_u,3), 'P': round(p,4)})
    except: pass

# BMI interaction
df_i['bmi_grp'] = pd.cut(df_i['bmi_baseline'], bins=[0,25,30,100], labels=['<25','25-30','>=30'])
for bg in ['<25','25-30','>=30']:
    df_i[f'int_bmi_{bg}'] = (df_i['bmi_grp'] == bg).astype(int) * df_i['n3fa_grs']
    try:
        exog = df_i[['n3fa_grs', f'int_bmi_{bg}'] + covars].astype(float).dropna()
        model = PHReg(df_i.loc[exog.index, 'time'].values, exog, status=df_i.loc[exog.index, 'event'].values)
        res = model.fit()
        idx = list(exog.columns).index(f'int_bmi_{bg}')
        beta, se, p = res.params[idx], res.bse[idx], res.pvalues[idx]
        HR = np.exp(beta)
        CI_l, CI_u = np.exp(beta - 1.96 * se), np.exp(beta + 1.96 * se)
        int_results.append({'Exposure': 'n3fa_grs', 'Interaction': f'bmi_{bg}', 'HR': round(HR,3), 'CI_low': round(CI_l,3), 'CI_high': round(CU,3), 'P': round(p,4)})
    except: pass

pd.DataFrame(int_results).to_csv('20260226_grs_ext/grs_ext_all_interactions.csv', index=False)
print('\nInteraction results:')
print(pd.DataFrame(int_results).to_string())
