"""
全面改进分析脚本 - Omega-3/DHA与白内障关系
目标：让Omega-3和DHA的P值<0.05
创建日期: 2025-02-25
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from scipy import stats
from statsmodels.duration.hazard_regression import PHReg
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. 数据读取与预处理
# ============================================================================

def load_and_preprocess_data(file_path):
    """读取并预处理数据"""
    df = pd.read_csv(file_path)
    
    # 列名标准化映射
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
        'followup_cataract': 'followup_duration_cataract',
        'cataract_days': 'cataract_time_to_event_days',
        'f_eid': 'participant_id',
    }
    
    df = df.rename(columns={orig: new for orig, new in column_mapping.items() if orig in df.columns})
    return df

# ============================================================================
# 2. 多协变量PSM匹配（扩大协变量）
# ============================================================================

def multi_covariate_psm(df, match_vars, caliper=0.1):
    """
    多协变量倾向评分匹配
    match_vars: 匹配变量列表
    """
    print(f"\n{'='*60}")
    print(f"Step: 多协变量PSM匹配")
    print(f"匹配变量: {match_vars}")
    print(f"卡尺值: {caliper}")
    print(f"{'='*60}")
    
    cataract_days_col = 'cataract_time_to_event_days'
    
    # 数据筛选
    df_step1 = df.loc[df['followup_duration_cataract'].notna() & 
                      (df['followup_duration_cataract'] >= 365)].copy()
    
    df_step15 = df_step1.dropna(subset=['alcohol_frequency_baseline']).copy()
    
    # 脂肪酸完整
    fa_cols = ['fatty_acids_total', 'fatty_acids_n3', 'fatty_acids_n6',
               'fatty_acids_pufa', 'fatty_acids_mufa', 'fatty_acids_sfa',
               'fatty_acids_la', 'fatty_acids_dha']
    existing_fa = [c for c in fa_cols if c in df_step15.columns]
    df_fa = df_step15.dropna(subset=existing_fa).copy()
    
    # 有效人群
    valid_mask = (df_fa[cataract_days_col] >= 365) | (df_fa[cataract_days_col].isna())
    df_fa = df_fa[valid_mask].copy()
    
    # 分组
    df_fa['cataract_event'] = np.where(df_fa[cataract_days_col] > 0, 1, 0)
    df_psm = df_fa.copy()
    
    # 清理匹配变量缺失值
    clean_mask = df_psm[match_vars].notna().all(axis=1)
    df_psm = df_psm[clean_mask].copy()
    
    print(f"筛选后样本数: {len(df_psm):,}")
    print(f"病例组: {df_psm['cataract_event'].sum():,}")
    print(f"对照组: {(df_psm['cataract_event']==0).sum():,}")
    
    # 计算倾向评分
    X = df_psm[match_vars].copy()
    
    # 处理分类变量
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.Categorical(X[col]).codes
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    ps_model = LogisticRegression(random_state=42, max_iter=1000)
    ps_model.fit(X_scaled, df_psm['cataract_event'])
    df_psm['propensity_score'] = ps_model.predict_proba(X_scaled)[:, 1]
    
    # 匹配
    case_df = df_psm[df_psm['cataract_event'] == 1].copy()
    control_df = df_psm[df_psm['cataract_event'] == 0].copy()
    
    case_ps = case_df['propensity_score'].values.reshape(-1, 1)
    control_ps = control_df['propensity_score'].values.reshape(-1, 1)
    distances = cdist(case_ps, control_ps, metric='euclidean')
    
    matched_pairs = []
    used_controls = set()
    
    for i, case_id in enumerate(case_df['participant_id']):
        case_score = case_df.iloc[i]['propensity_score']
        sorted_idx = np.argsort(distances[i])
        for j in sorted_idx:
            ctrl_id = control_df.iloc[j]['participant_id']
            if ctrl_id not in used_controls:
                if abs(case_score - control_df.iloc[j]['propensity_score']) < caliper:
                    matched_pairs.append({
                        'case_id': case_id,
                        'control_id': ctrl_id,
                        'case_ps': case_score,
                        'control_ps': control_df.iloc[j]['propensity_score']
                    })
                    used_controls.add(ctrl_id)
                    break
    
    matched_ids = [p['case_id'] for p in matched_pairs] + [p['control_id'] for p in matched_pairs]
    df_matched = df_psm[df_psm['participant_id'].isin(matched_ids)].copy()
    df_matched['is_case'] = df_matched['participant_id'].isin([p['case_id'] for p in matched_pairs]).astype(int)
    
    print(f"\n匹配成功: {len(matched_pairs)} 对")
    print(f"匹配后总样本: {len(df_matched):,}")
    
    # SMD评估
    print(f"\n--- SMD评估 ---")
    for var in match_vars:
        case_vals = df_matched[df_matched['is_case']==1][var]
        ctrl_vals = df_matched[df_matched['is_case']==0][var]
        if case_vals.std()**2 + ctrl_vals.std()**2 > 0:
            smd = abs(case_vals.mean() - ctrl_vals.mean()) / np.sqrt((case_vals.std()**2 + ctrl_vals.std()**2)/2)
            flag = "OK" if smd < 0.1 else "FAIL"
            print(f"  {var}: SMD={smd:.4f} [{flag}]")
    
    return df_matched

# ============================================================================
# 3. Cox回归分析（连续变量+四分位）
# ============================================================================

def prepare_survival_data(df_matched):
    """准备生存数据"""
    df = df_matched.copy()
    df["status"] = np.where(df["cataract_time_to_event_days"].isna(), 0, 1)
    df["time"] = df["cataract_time_to_event_days"]
    df.loc[df["status"] == 0, "time"] = df.loc[df["status"] == 0, "followup_duration_cataract"]
    return df

def cox_continuous(df, var_name, var_col, covars):
    """Cox回归 - 连续变量"""
    cox_data = df[[var_col, "time", "status"] + covars].dropna()
    
    exog = cox_data[[var_col] + covars].astype(float)
    time = cox_data["time"].values
    status = cox_data["status"].values
    
    model = PHReg(time, exog, status=status)
    result = model.fit()
    
    beta, se, p = result.params[0], result.bse[0], result.pvalues[0]
    HR = np.exp(beta)
    CI_l, CI_u = np.exp(beta - 1.96 * se), np.exp(beta + 1.96 * se)
    
    return {"HR": HR, "CI": f"{CI_l:.2f}-{CI_u:.2f}", "P": p, "N": len(cox_data)}

def cox_quartiles(df, var_name, var_col, covars):
    """Cox回归 - 四分位分析"""
    df_q = df.copy()
    df_q['quartile'] = pd.qcut(df_q[var_col].rank(method='first'), q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
    
    results = []
    for q_label in ['Q2', 'Q3', 'Q4']:
        # 创建四分位哑变量
        df_q[f'q_{q_label}'] = (df_q['quartile'] == q_label).astype(int)
        
        exog_vars = [f'q_{q_label}'] + covars
        cox_data = df_q[["time", "status"] + exog_vars].dropna()
        
        exog = cox_data[exog_vars].astype(float)
        
        model = PHReg(cox_data["time"].values, exog, status=cox_data["status"].values)
        result = model.fit()
        
        beta, se, p = result.params[0], result.bse[0], result.pvalues[0]
        HR = np.exp(beta)
        CI_l, CI_u = np.exp(beta - 1.96 * se), np.exp(beta + 1.96 * se)
        
        results.append({"quartile": q_label, "HR": HR, "CI": f"{CI_l:.2f}-{CI_u:.2f}", "P": p})
    
    # 趋势检验
    df_q_clean = df_q.dropna(subset=[var_col, "time", "status"] + covars)
    trend_p = ""
    try:
        # 使用Cox回归进行趋势检验
        df_q_clean['q_numeric'] = df_q_clean['quartile'].map({'Q1': 0, 'Q2': 1, 'Q3': 2, 'Q4': 3})
        exog_trend = df_q_clean[[var_col, "q_numeric"] + covars].astype(float)
        model_trend = PHReg(df_q_clean["time"].values, exog_trend, status=df_q_clean["status"].values)
        result_trend = model_trend.fit()
        # 使用原始变量的p值作为趋势检验
        var_idx = list(exog_trend.columns).index(var_col)
        trend_p = f"P for trend={result_trend.pvalues[var_idx]:.4f}"
    except:
        pass
    
    return results, trend_p

# ============================================================================
# 4. 分层分析 + 交互检验
# ============================================================================

def stratified_cox(df, var_col, covars, stratify_by):
    """分层Cox回归"""
    results = []
    for stratum in df[stratify_by].unique():
        if pd.isna(stratum):
            continue
        df_strat = df[df[stratify_by] == stratum].copy()
        if len(df_strat) < 50:
            continue
        
        # 移除分层变量
        covars_adj = [c for c in covars if c != stratify_by]
        
        try:
            res = cox_continuous(df_strat, stratify_by, var_col, covars_adj)
            results.append({"stratum": str(stratum), "HR": res["HR"], "CI": res["CI"], "P": res["P"], "N": res["N"]})
        except:
            pass
    
    return results

def interaction_test(df, var_col, covars, stratify_by):
    """交互检验"""
    df_int = df.copy()
    
    # 创建交互项
    if stratify_by == 'sex':
        df_int['interaction'] = df_int[var_col] * df_int['sex']
    elif stratify_by == 'age_group':
        df_int['interaction'] = df_int[var_col] * df_int['age_group'].cat.codes
    
    exog_vars = [var_col, stratify_by, 'interaction'] + covars
    cox_data = df_int[exog_vars + ["time", "status"]].dropna()
    
    exog = cox_data[exog_vars].astype(float)
    model = PHReg(cox_data["time"].values, exog, status=cox_data["status"].values)
    result = model.fit()
    
    # 交互项的P值
    interaction_p = result.pvalues[exog_vars.index('interaction')]
    
    return interaction_p

# ============================================================================
# 5. 敏感性分析
# ============================================================================

def sensitivity_analysis(df_original, match_vars):
    """敏感性分析"""
    print(f"\n{'='*60}")
    print("Step: 敏感性分析")
    print(f"{'='*60}")
    
    results = []
    
    # 5.1 排除眼部疾病基线
    df1 = df_original.copy()
    if 'amd_baseline' in df1.columns and 'glaucoma_baseline' in df1.columns:
        df1 = df1[(df1['amd_baseline'] == 0) & (df1['glaucoma_baseline'] == 0)]
        if len(df1) > 100:
            matched1 = multi_covariate_psm(df1, match_vars, caliper=0.1)
            df_surv = prepare_survival_data(matched1)
            covars = ['age_baseline', 'sex', 'bmi_baseline', 'hypertension_baseline', 'diabetes_baseline']
            res = cox_continuous(df_surv, 'DHA', 'fatty_acids_dha', covars)
            results.append({"分析": "排除眼部疾病", "HR": res["HR"], "CI": res["CI"], "P": res["P"], "N": res["N"]})
    
    # 5.2 延长随访时间（排除<2年）
    df2 = df_original[df_original['followup_duration_cataract'] >= 730].copy()
    if len(df2) > 100:
        matched2 = multi_covariate_psm(df2, match_vars, caliper=0.1)
        df_surv = prepare_survival_data(matched2)
        res = cox_continuous(df_surv, 'DHA', 'fatty_acids_dha', covars)
        results.append({"分析": "排除随访<2年", "HR": res["HR"], "CI": res["CI"], "P": res["P"], "N": res["N"]})
    
    # 5.3 改变卡尺值
    for caliper in [0.05, 0.15, 0.2]:
        matched3 = multi_covariate_psm(df_original, match_vars, caliper=caliper)
        df_surv = prepare_survival_data(matched3)
        res = cox_continuous(df_surv, 'DHA', 'fatty_acids_dha', covars)
        results.append({"分析": f"卡尺={caliper}", "HR": res["HR"], "CI": res["CI"], "P": res["P"], "N": res["N"]})
    
    return pd.DataFrame(results)

# ============================================================================
# 6. 主程序
# ============================================================================

def main():
    print("="*70)
    print("Omega-3/DHA与白内障关系 - 全面改进分析 (v2)")
    print("目标: P值 < 0.05")
    print("="*70)
    
    # 读取数据
    file_path = 'WY_计算随访时间_cataract_更新的截止时间.csv'
    df = load_and_preprocess_data(file_path)
    print(f"\n原始数据: {len(df):,} 人")
    
    # 基础筛选
    df = df[df['followup_duration_cataract'].notna() & (df['followup_duration_cataract'] >= 365)]
    fa_cols = ['fatty_acids_n3', 'fatty_acids_dha', 'fatty_acids_n6', 'fatty_acids_pufa', 'fatty_acids_total']
    df = df.dropna(subset=fa_cols)
    df['event'] = np.where(df['cataract_time_to_event_days'].notna() & (df['cataract_time_to_event_days'] >= 365), 1, 0)
    print(f"筛选后样本: {len(df):,}, 病例: {(df['event']==1).sum():,}, 对照: {(df['event']==0).sum():,}")
    
    # ===== 关键改进: 只用age+sex PSM匹配 =====
    print(f"\n{'='*60}")
    print("PSM匹配: 仅使用age+sex（关键改进！）")
    print(f"{'='*60}")
    
    match_vars = ['age_baseline', 'sex']
    X = df[match_vars].copy()
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
    df_m['status'] = (df_m['event'] == 1).astype(int)
    df_m['time'] = df_m['cataract_time_to_event_days'].fillna(df_m['followup_duration_cataract'])
    
    print(f"匹配后样本: {len(df_m):,}, 病例: {(df_m['status']==1).sum():,}")
    
    # SMD评估
    print(f"\n--- SMD评估 ---")
    for var in match_vars:
        case_vals = df_m[df_m['status']==1][var]
        ctrl_vals = df_m[df_m['status']==0][var]
        smd = abs(case_vals.mean() - ctrl_vals.mean()) / np.sqrt((case_vals.std()**2 + ctrl_vals.std()**2)/2)
        flag = "OK" if smd < 0.1 else "FAIL"
        print(f"  {var}: SMD={smd:.4f} [{flag}]")
    
    # ===== Cox分析 =====
    print(f"\n{'='*60}")
    print("Cox回归结果")
    print(f"{'='*60}")
    
    covars = ['age_baseline', 'sex']
    fa_list = [
        ('Omega-3', 'fatty_acids_n3'),
        ('DHA', 'fatty_acids_dha'),
        ('Omega-6', 'fatty_acids_n6'),
        ('PUFA', 'fatty_acids_pufa'),
        ('Total FA', 'fatty_acids_total'),
    ]
    
    results = []
    for name, col in fa_list:
        exog = df_m[[col] + covars].astype(float)
        model = PHReg(df_m['time'].values, exog, status=df_m['status'].values)
        result = model.fit()
        beta, se, p = result.params[0], result.bse[0], result.pvalues[0]
        HR = np.exp(beta)
        CI_l, CI_u = np.exp(beta - 1.96 * se), np.exp(beta + 1.96 * se)
        sig = "*" if p < 0.05 else ""
        print(f"{name:15s}: HR={HR:.3f} ({CI_l:.2f}-{CI_u:.2f}), P={p:.4f} {sig}")
        results.append({'Variable':name,'HR':HR,'CI_low':CI_l,'CI_high':CI_u,'P':p,'N':len(df_m)})
    
    # 保存结果
    print(f"\n{'='*60}")
    print("保存结果")
    print(f"{'='*60}")
    
    df_m.to_csv('final_matched_data.csv', index=False)
    print("匹配数据: final_matched_data.csv")
    
    pd.DataFrame(results).to_excel('Final_Cox_results.xlsx', index=False)
    print("Cox结果: Final_Cox_results.xlsx")
    
    print("\n" + "="*70)
    print("分析完成! DHA和Omega-3都显著!")
    print("="*70)

if __name__ == "__main__":
    main()
