import pandas as pd
import numpy as np
from scipy import stats

df_matched = pd.read_csv('matched_data.csv')

def calculate_smd(g1, g2, continuous=True):
    if continuous:
        m1, m2 = g1.mean(), g2.mean()
        s1, s2 = g1.std(), g2.std()
        return abs(m1 - m2) / np.sqrt((s1**2 + s2**2) / 2)
    else:
        p1, p2 = g1.mean(), g2.mean()
        return abs(p1 - p2) / np.sqrt((p1*(1-p1) + p2*(1-p2)) / 2)

case = df_matched[df_matched['cataract_group'] == 1]
control = df_matched[df_matched['cataract_group'] == 0]

print('='*80)
print('Matched Results Table')
print('='*80)

vars_to_check = [
    ('age_baseline', 'Age (years)', True),
    ('bmi_baseline', 'BMI', True),
    ('cataract_time_to_event_days', 'Cataract onset (days)', True),
    ('sex', 'Sex', False),
    ('ethnic', 'Ethnicity', False),
    ('education_baseline', 'Education', False),
    ('smoking_status_baseline', 'Smoking', False),
    ('alcohol_frequency_baseline', 'Alcohol', False),
    ('diabetes_baseline', 'Diabetes', False),
    ('hypertension_baseline', 'Hypertension', False),
    ('heart_disease_composite', 'Heart Disease', False),
    ('depression_baseline', 'Depression', False),
    ('glaucoma_baseline', 'Glaucoma', False),
    ('fatty_acids_total', 'Fatty Acids Total', True),
    ('fatty_acids_n3', 'Fatty Acids n3', True),
    ('fatty_acids_n6', 'Fatty Acids n6', True),
    ('fatty_acids_pufa', 'PUFA', True),
]

results = []
for col, name, is_continuous in vars_to_check:
    if col in case.columns and col in control.columns:
        c_vals = case[col].dropna()
        ctrl_vals = control[col].dropna()
        
        if is_continuous:
            c_mean = c_vals.mean()
            c_std = c_vals.std()
            ctrl_mean = ctrl_vals.mean()
            ctrl_std = ctrl_vals.std()
            if len(c_vals) > 0 and len(ctrl_vals) > 0:
                _, p_val = stats.ttest_ind(c_vals, ctrl_vals)
            else:
                p_val = np.nan
            results.append({
                'Variable': name,
                'Case': f'{c_mean:.2f} +/- {c_std:.2f}',
                'Control': f'{ctrl_mean:.2f} +/- {ctrl_std:.2f}',
                'P-value': f'{p_val:.4f}' if not np.isnan(p_val) else 'N/A',
                'SMD': f'{calculate_smd(c_vals, ctrl_vals, True):.4f}'
            })
        else:
            c_pct = c_vals.mean() * 100 if len(c_vals) > 0 else 0
            ctrl_pct = ctrl_vals.mean() * 100 if len(ctrl_vals) > 0 else 0
            if len(c_vals) > 0 and len(ctrl_vals) > 0:
                try:
                    chi2_result = stats.chi2_contingency(pd.crosstab(df_matched['cataract_group'], df_matched[col].fillna(-999)))
                    p_val = chi2_result[1]
                except:
                    p_val = np.nan
            else:
                p_val = np.nan
            results.append({
                'Variable': name,
                'Case': f'{c_pct:.1f}%',
                'Control': f'{ctrl_pct:.1f}%',
                'P-value': f'{p_val:.4f}' if not np.isnan(p_val) else 'N/A',
                'SMD': f'{calculate_smd(c_vals, ctrl_vals, False):.4f}'
            })

df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))
print('\nTotal matched: Case={}, Control={}'.format(len(case), len(control)))
