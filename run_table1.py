import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, chi2_contingency, fisher_exact

df_matched = pd.read_csv('matched_data.csv')

# 需要重新读取原始数据来构建 df_psm
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

cataract_days_col = 'cataract_time_to_event_days'
df_step1 = df.loc[df['followup_duration_cataract'].notna() & (df['followup_duration_cataract'] >= 365)].copy()
df_step15 = df_step1.dropna(subset=['alcohol_frequency_baseline']).copy()
fa_core_cols = ['fatty_acids_total', 'fatty_acids_n3', 'fatty_acids_n6', 'fatty_acids_pufa', 'fatty_acids_mufa', 'fatty_acids_sfa', 'fatty_acids_la', 'fatty_acids_dha']
existing_fa_core = [col for col in fa_core_cols if col in df_step15.columns]
df_fa_complete = df_step15.dropna(subset=existing_fa_core).copy()
valid_mask = (df_fa_complete[cataract_days_col] >= 365) | (df_fa_complete[cataract_days_col].isna())
df_fa_complete = df_fa_complete[valid_mask].copy()
df_fa_complete['cataract_event'] = np.where(df_fa_complete[cataract_days_col] > 0, 1, 0)
df_psm = df_fa_complete.copy()
df_psm['cataract_group'] = df_psm['cataract_event']

def zscore_by_ref(series, mean, std):
    if std == 0 or np.isnan(std):
        return series * np.nan
    return (series - mean) / std

def format_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    return "<0.001" if p < 0.001 else f"{p:.3f}"

def fmt_mean_sd(series):
    series = series.dropna()
    if len(series) == 0:
        return ""
    return f"{series.mean():.2f} ± {series.std(ddof=1):.2f}"

def fmt_n_pct(n, total):
    if total == 0 or np.isnan(total):
        return f"{int(n)} (0.00%)"
    return f"{int(n)} ({n/total*100:.2f}%)"

def p_value_continuous(x1, x2):
    x1 = x1.dropna()
    x2 = x2.dropna()
    if len(x1)<2 or len(x2)<2:
        return np.nan
    try:
        _, p = ttest_ind(x1, x2, equal_var=False, nan_policy='omit')
        return p
    except:
        return np.nan

def p_value_categorical(tab):
    if tab.empty:
        return np.nan
    if tab.shape[0]==2 and tab.shape[1]==2:
        try:
            return fisher_exact(tab.values.astype(int))[1]
        except:
            return chi2_contingency(tab.values.astype(int))[1]
    else:
        try:
            return chi2_contingency(tab.values.astype(int))[1]
        except:
            return np.nan

def build_table1_psm(df_before, df_after, group_before='cataract_group', group_after='is_case', group_labels=('对照组','白内障组')):
    df_before = df_before.loc[:, ~df_before.columns.duplicated()]
    df_after = df_after.loc[:, ~df_after.columns.duplicated()]
    rows = []
    
    for df in [df_before, df_after]:
        if 'education_baseline' in df.columns:
            df['education_bin'] = df['education_baseline'].apply(lambda x: 1 if x == 1 else 0)
        if 'smoker' in df.columns:
            df['Smoking status'] = df['smoker'].apply(lambda x: 1 if x == 1 else 0)
        if 'alcohol_frequency_baseline' in df.columns:
            df['Alcohol use'] = df['alcohol_frequency_baseline'].apply(lambda x: 1 if x == 1 else 0)
        if 'amd_baseline' in df.columns:
            df['amd'] = df['amd_baseline'].apply(lambda x: 0 if x == 0 else 1)
        if 'glaucoma_baseline' in df.columns:
            df['glaucoma'] = df['glaucoma_baseline'].apply(lambda x: 1 if x == 1 else 0)
        if 'diabetic_retinopathy_baseline' in df.columns:
            df['diabetes'] = df['diabetic_retinopathy_baseline'].apply(lambda x: 1 if x == 1 else 0)
    
    continuous_vars = {'age_baseline': 'Age in years, mean (SD)', 'bmi_baseline': 'BMI (SD)', 'cataract_time_to_event_days': '白内障发病时间（天）'}
    categorical_vars = {
        'sex': {0:'Female',1:'Male'}, 'Ethnicity': {'White':'White', 'Others':'Others'},
        'education_bin': {1:'College/University', 0:'Others'}, 'Smoking status': {0:'Never',1:'Current'},
        'Alcohol use': {1:'Never/occasional',0:'Frequent'}, 'amd': {0:'No',1:'Yes'},
        'glaucoma': {0:'No',1:'Yes'}, 'diabetes': {0:'No',1:'Yes'},
        'hypertension_baseline': {0:'No',1:'Yes'}, 'heart_disease_composite': {0:'No',1:'Yes'}, 'depression_baseline': {0:'No',1:'Yes'}
    }
    
    for col, label in continuous_vars.items():
        x0 = df_before[df_before[group_before]==0][col] if col in df_before.columns else pd.Series(dtype=float)
        x1 = df_before[df_before[group_before]==1][col] if col in df_before.columns else pd.Series(dtype=float)
        p_pre = p_value_continuous(x0, x1)
        y0 = df_after[df_after[group_after]==0][col] if col in df_after.columns else pd.Series(dtype=float)
        y1 = df_after[df_after[group_after]==1][col] if col in df_after.columns else pd.Series(dtype=float)
        p_post = p_value_continuous(y0, y1)
        rows.append([f"{label}", fmt_mean_sd(x0), fmt_mean_sd(x1), format_p(p_pre), fmt_mean_sd(y0), fmt_mean_sd(y1), format_p(p_post)])
    
    for df in [df_before, df_after]:
        if 'ethnic' in df.columns:
            df['Ethnicity'] = df['ethnic'].apply(lambda x: 'White' if x==1 else 'Others')
        else:
            df['Ethnicity'] = np.nan
    
    for col, mapping in categorical_vars.items():
        if col not in df_before.columns and col not in df_after.columns:
            continue
        tab_before = pd.crosstab(df_before[col], df_before[group_before]) if col in df_before.columns else pd.DataFrame()
        tab_after = pd.crosstab(df_after[col], df_after[group_after]) if col in df_after.columns else pd.DataFrame()
        p_pre = p_value_categorical(tab_before)
        p_post = p_value_categorical(tab_after)
        rows.append([f"{col}", '', '', format_p(p_pre), '', '', format_p(p_post)])
        total0_pre = (df_before[group_before]==0).sum()
        total1_pre = (df_before[group_before]==1).sum()
        total0_post = (df_after[group_after]==0).sum()
        total1_post = (df_after[group_after]==1).sum()
        for val, label in mapping.items():
            n0_pre = tab_before.loc[val,0] if (0 in tab_before.columns and val in tab_before.index) else 0
            n1_pre = tab_before.loc[val,1] if (1 in tab_before.columns and val in tab_before.index) else 0
            n0_post = tab_after.loc[val,0] if (0 in tab_after.columns and val in tab_after.index) else 0
            n1_post = tab_after.loc[val,1] if (1 in tab_after.columns and val in tab_after.index) else 0
            rows.append([f"  {label}", fmt_n_pct(n0_pre,total0_pre), fmt_n_pct(n1_pre,total1_pre), '', fmt_n_pct(n0_post,total0_post), fmt_n_pct(n1_post,total1_post), ''])
    
    fa_core_cols = ['fatty_acids_total', 'fatty_acids_n3', 'fatty_acids_n6', 'fatty_acids_pufa', 'fatty_acids_mufa', 'fatty_acids_sfa', 'fatty_acids_la', 'fatty_acids_dha']
    for col in fa_core_cols:
        if col in df_before.columns or col in df_after.columns:
            x0 = df_before[df_before[group_before]==0][col] if col in df_before.columns else pd.Series(dtype=float)
            x1 = df_before[df_before[group_before]==1][col] if col in df_before.columns else pd.Series(dtype=float)
            p_pre = p_value_continuous(x0, x1)
            y0 = df_after[df_after[group_after]==0][col] if col in df_after.columns else pd.Series(dtype=float)
            y1 = df_after[df_after[group_after]==1][col] if col in df_after.columns else pd.Series(dtype=float)
            p_post = p_value_continuous(y0, y1)
            rows.append([f"{col}", fmt_mean_sd(x0), fmt_mean_sd(x1), format_p(p_pre), fmt_mean_sd(y0), fmt_mean_sd(y1), format_p(p_post)])
    
    lipid_cols = ['total_cholesterol', 'ldl_cholesterol', 'hdl_cholesterol', 'triglycerides']
    for col in lipid_cols:
        if col in df_before.columns or col in df_after.columns:
            x0 = df_before[df_before[group_before]==0][col] if col in df_before.columns else pd.Series(dtype=float)
            x1 = df_before[df_before[group_before]==1][col] if col in df_before.columns else pd.Series(dtype=float)
            p_pre = p_value_continuous(x0, x1)
            y0 = df_after[df_after[group_after]==0][col] if col in df_after.columns else pd.Series(dtype=float)
            y1 = df_after[df_after[group_after]==1][col] if col in df_after.columns else pd.Series(dtype=float)
            p_post = p_value_continuous(y0, y1)
            rows.append([f"{col}", fmt_mean_sd(x0), fmt_mean_sd(x1), format_p(p_pre), fmt_mean_sd(y0), fmt_mean_sd(y1), format_p(p_post)])
    
    columns = ['变量', f'匹配前 {group_labels[0]}', f'匹配前 {group_labels[1]}', 'P值', f'匹配后 {group_labels[0]}', f'匹配后 {group_labels[1]}', 'P值']
    table_df = pd.DataFrame(rows, columns=columns)
    return table_df

table1 = build_table1_psm(df_psm, df_matched, group_before='cataract_group', group_after='cataract_group', group_labels=('对照组','白内障组'))
print(table1.to_string())
table1.to_excel("Table1_complete_with_FA_and_Ethnicity.xlsx", index=False)
print("\nTable1 saved to Table1_complete_with_FA_and_Ethnicity.xlsx")
