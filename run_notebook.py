import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

file_path = 'WY_计算随访时间_cataract_更新的截止时间.csv'
df = pd.read_csv(file_path)

column_mapping = {
    'sex': 'sex', 'ethnic_background': 'ethnic_background', 'ethnic_l': 'ethnic_l', 'ethnic': 'ethnic',
    'education_bl': 'education_baseline', 'age_bl': 'age_baseline', 'csmoking_bl': 'smoking_status_baseline',
    'alcohol_freq_bl': 'alcohol_frequency_baseline', 'alcohol_bl': 'alcohol_frequency_baseline1',
    'sleep_dur_bl': 'sleep_duration_baseline', 'bmi_bl': 'bmi_baseline', 'obesity': 'obesity_status',
    'diabetes_bl': 'diabetes_baseline', 'hypertension_bl': 'hypertension_baseline',
    'heart_attack_bl': 'myocardial_infarction_baseline', 'heart_failure_bl': 'heart_failure_baseline',
    'stroke_bl': 'stroke_baseline', 'kidney_stone_bl': 'kidney_stone_baseline', 'depression_bl': 'depression_baseline',
    'cancer_bl': 'cancer_baseline', 'heart_disease': 'heart_disease_composite',
    'amd_bl': 'amd_baseline', 'amd_bl1': 'amd_baseline1', 'amd_blt': 'amd_baseline2',
    'cataract_bl': 'cataract_baseline', 'glaucoma_bl': 'glaucoma_baseline', 'gla_blt': 'glaucoma_baseline1',
    'diabetic_eye_bl': 'diabetic_retinopathy_baseline', 'dr_bl': 'diabetic_retinopathy_baseline', 'dr_blt': 'diabetic_retinopathy_baseline',
    'hba1c_bl': 'hba1c_baseline', 'total_cholesterol': 'total_cholesterol', 'ldl_cholesterol': 'ldl_cholesterol',
    'hdl_cholesterol': 'hdl_cholesterol', 'triglycerides': 'triglycerides',
    'total_fa': 'fatty_acids_total', 'n3fa': 'fatty_acids_n3', 'n6fa': 'fatty_acids_n6',
    'pufa': 'fatty_acids_pufa', 'mufa': 'fatty_acids_mufa', 'sfa': 'fatty_acids_sfa',
    'la': 'fatty_acids_la', 'dha': 'fatty_acids_dha',
    'n3fa_grs': 'n3fa_grs', 'n6fa_grs': 'n6fa_grs', 'pufa_grs': 'pufa_grs', 'tfa_grs': 'tfa_grs',
    'age_layer2': 'age_quantile_2', 'age50': 'age_threshold_50', 'age57': 'age_threshold_57', 'age60': 'age_threshold_60',
    'date_interview': 'interview_date', 'date_interview_time': 'interview_date', 'lastdate': 'last_followup_date',
    'followup_cataract_yrs': 'followup_duration_cataract_yrs', 'followup_cataract': 'followup_duration_cataract',
    'followup_cataract_182': 'followup_duration_cataract_182', 'followdate_cataract': 'followup_date_cataract', 'eligible': 'eligible_status',
    'amd_onset': 'amd_onset_date', 'amd_onset_date': 'amd_onset_date', 'amd_onset_time': 'amd_onset_date', 'amd_days': 'amd_time_to_event_days',
    'cataract_onset': 'cataract_onset_date', 'cataract_onset_date': 'cataract_onset_date', 'cataract_days': 'cataract_time_to_event_days',
    'incident_cataract': 'cataract_incident',
    'diabetic_eye_onset': 'diabetic_eye_onset_date', 'diabetic_eye_onset_date': 'diabetic_eye_onset_date', 'diabetic_eye_days': 'diabetic_eye_time_to_event_days',
    'glaucoma_onset': 'glaucoma_onset_date', 'glaucoma_onset_date': 'glaucoma_onset_date', 'glaucoma_days': 'glaucoma_time_to_event_days',
    'amd_prediction': 'amd_risk_prediction', 'cataract_prediction': 'cataract_risk_prediction',
    'diabetic_eye_prediction': 'diabetic_eye_risk_prediction', 'glaucoma_prediction': 'glaucoma_risk_prediction',
    'f_eid': 'participant_id', 'analysis_fa': 'analysis_flag_fatty_acids', 'metabolomic_age': 'metabolomic_age',
}
df = df.rename(columns={orig: new for orig, new in column_mapping.items() if orig in df.columns})

initial_n = len(df)
print('=' * 65)
print(f'Step 0 - Initial: {initial_n:,} people')
print('=' * 65)

cataract_days_col = 'cataract_time_to_event_days'

print('\n--- Step 1: Exclude follow-up < 365 days ---')
pre_step1_n = len(df)
df_step1 = df.loc[df['followup_duration_cataract'].notna() & (df['followup_duration_cataract'] >= 365)].copy()
after_step1_n = len(df_step1)
print(f'Before: {pre_step1_n:,} | After: {after_step1_n:,} | Removed: {pre_step1_n - after_step1_n:,}')

print('\n--- Step 1.5: Exclude missing alcohol_frequency_baseline ---')
pre_step15_n = len(df_step1)
n_missing = df_step1['alcohol_frequency_baseline'].isna().sum()
print(f'Missing: {n_missing:,}')
df_step15 = df_step1.dropna(subset=['alcohol_frequency_baseline']).copy()
print(f'Before: {pre_step15_n:,} | After: {len(df_step15):,}')

print('\n--- Step 2: Complete fatty acids data ---')
fa_core_cols = ['fatty_acids_total', 'fatty_acids_n3', 'fatty_acids_n6', 'fatty_acids_pufa', 'fatty_acids_mufa', 'fatty_acids_sfa', 'fatty_acids_la', 'fatty_acids_dha']
existing_fa_core = [col for col in fa_core_cols if col in df_step15.columns]
pre_step2_n = len(df_step15)
df_fa_complete = df_step15.dropna(subset=existing_fa_core).copy()
print(f'Before: {pre_step2_n:,} | After: {len(df_fa_complete):,}')

print('\n--- Step 3: Filter valid population ---')
valid_mask = (df_fa_complete[cataract_days_col] >= 365) | (df_fa_complete[cataract_days_col].isna())
df_fa_complete = df_fa_complete[valid_mask].copy()
print(f'After filtering: {len(df_fa_complete):,}')
print(f'Cases (>=365d): {(df_fa_complete[cataract_days_col] >= 365).sum():,}')
print(f'Controls (NaN): {df_fa_complete[cataract_days_col].isna().sum():,}')

print('\n--- Step 4: Create group variable ---')
df_fa_complete['cataract_event'] = np.where(df_fa_complete[cataract_days_col] > 0, 1, 0)
df_psm = df_fa_complete.copy()
df_psm['cataract_group'] = df_psm['cataract_event']
n_case = df_psm['cataract_group'].sum()
n_control = (df_psm['cataract_group'] == 0).sum()
print(f'Cases (>=365d): {n_case:,} | Controls (NaN): {n_control:,}')

match_vars = ['age_baseline', 'sex']
clean_mask = df_psm[match_vars].notna().all(axis=1)
df_psm = df_psm[clean_mask].copy()

X = df_psm[match_vars].copy()
y = df_psm['cataract_group']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
ps_model = LogisticRegression(random_state=42, max_iter=1000)
ps_model.fit(X_scaled, y)
df_psm['propensity_score'] = ps_model.predict_proba(X_scaled)[:, 1]
print('\n--- PS Model ---')
print(f'Coefficients: age={ps_model.coef_[0][0]:.4f}, sex={ps_model.coef_[0][1]:.4f}')
print(f'Intercept: {ps_model.intercept_[0]:.4f}')

print('\n--- 1:1 Nearest Neighbor Matching (caliper=0.1) ---')
case_df = df_psm[df_psm['cataract_group'] == 1].copy()
control_df = df_psm[df_psm['cataract_group'] == 0].copy()

case_ps = case_df['propensity_score'].values.reshape(-1, 1)
control_ps = control_df['propensity_score'].values.reshape(-1, 1)
distances = cdist(case_ps, control_ps, metric='euclidean')

matched_pairs = []
used_controls = set()
for i, case_id in enumerate(case_df['participant_id']):
    case_score = case_df.iloc[i]['propensity_score']
    sorted_indices = np.argsort(distances[i])
    for j in sorted_indices:
        control_id = control_df.iloc[j]['participant_id']
        if control_id not in used_controls:
            if abs(case_score - control_df.iloc[j]['propensity_score']) < 0.1:
                matched_pairs.append({'case_id': case_id, 'control_id': control_id})
                used_controls.add(control_id)
                break

print(f'Successfully matched pairs: {len(matched_pairs)}')
print(f'Cases: {len(case_df):,} | Matched: {len(matched_pairs):,}')

print('\n--- SMD Assessment ---')
matched_case_ids = [p['case_id'] for p in matched_pairs]
matched_control_ids = [p['control_id'] for p in matched_pairs]
df_matched = df_psm[df_psm['participant_id'].isin(matched_case_ids + matched_control_ids)].copy()

def calculate_smd(g1, g2):
    m1, m2 = g1.mean(), g2.mean()
    s1, s2 = g1.std(), g2.std()
    return abs(m1 - m2) / np.sqrt((s1**2 + s2**2) / 2)

for var in match_vars:
    case_b = df_psm[df_psm['cataract_group'] == 1][var]
    ctrl_b = df_psm[df_psm['cataract_group'] == 0][var]
    smd_b = calculate_smd(case_b, ctrl_b)
    case_a = df_matched[df_matched['cataract_group'] == 1][var]
    ctrl_a = df_matched[df_matched['cataract_group'] == 0][var]
    smd_a = calculate_smd(case_a, ctrl_a)
    print(f'{var}: Before SMD={smd_b:.4f}, After SMD={smd_a:.4f}')

df_matched.to_csv('matched_data.csv', index=False)
print('\nMatched data saved to matched_data.csv')
print(f'Total matched samples: {len(df_matched):,}')
print(f'Columns: {df_matched.columns.tolist()}')
