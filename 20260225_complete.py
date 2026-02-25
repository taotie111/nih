"""
================================================================================
Omega-3/DHA与白内障关系 - 完整论文级别分析
匹配策略: 仅age+sex PSM
分析内容: 多模型Cox + 分层 + 四分位 + 敏感性分析
创建日期: 2025-02-25
================================================================================
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
# 1. 数据加载与预处理
# ============================================================================

def load_data():
    """加载并预处理数据"""
    df = pd.read_csv('WY_计算随访时间_cataract_更新的截止时间.csv')
    
    # 列名标准化
    cmap = {
        'age_bl': 'age_baseline', 'sex': 'sex', 'ethnic_background': 'ethnic_background',
        'education_bl': 'education_baseline', 'csmoking_bl': 'smoking_status_baseline',
        'alcohol_freq_bl': 'alcohol_frequency_baseline', 'bmi_bl': 'bmi_baseline',
        'diabetes_bl': 'diabetes_baseline', 'hypertension_bl': 'hypertension_baseline',
        'heart_disease': 'heart_disease_composite', 'n3fa': 'fatty_acids_n3',
        'n6fa': 'fatty_acids_n6', 'dha': 'fatty_acids_dha', 
        'pufa': 'fatty_acids_pufa', 'total_fa': 'fatty_acids_total',
        'sfa': 'fatty_acids_sfa', 'mufa': 'fatty_acids_mufa', 'la': 'fatty_acids_la',
        'followup_cataract': 'followup_duration_cataract',
        'cataract_days': 'cataract_time_to_event_days',
        'f_eid': 'participant_id', 'glaucoma_bl': 'glaucoma_baseline',
        'amd_bl': 'amd_baseline', 'depression_bl': 'depression_baseline',
    }
    df = df.rename(columns={k:v for k,v in cmap.items() if k in df.columns})
    return df

# ============================================================================
# 2. PSM匹配（仅age+sex）
# ============================================================================

def psm_matching(df, caliper=0.1):
    """倾向评分匹配 - 仅age+sex"""
    # 筛选数据
    df = df[df['followup_duration_cataract'].notna() & (df['followup_duration_cataract'] >= 365)]
    fa_cols = ['fatty_acids_n3', 'fatty_acids_dha', 'fatty_acids_n6', 'fatty_acids_pufa', 'fatty_acids_total']
    df = df.dropna(subset=fa_cols)
    
    # 创建事件变量
    df['event'] = np.where(
        df['cataract_time_to_event_days'].notna() & (df['cataract_time_to_event_days'] >= 365), 1, 0
    )
    
    print(f"筛选后样本: {len(df):,}, 病例: {(df['event']==1).sum():,}, 对照: {(df['event']==0).sum():,}")
    
    # PSM
    match_vars = ['age_baseline', 'sex']
    X = df[match_vars].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    ps_model = LogisticRegression(random_state=42, max_iter=1000)
    ps_model.fit(X_scaled, df['event'])
    df['ps'] = ps_model.predict_proba(X_scaled)[:, 1]
    
    # 1:1匹配
    case_df = df[df['event']==1].copy()
    ctrl_df = df[df['event']==0].copy()
    dist = cdist(case_df['ps'].values.reshape(-1,1), ctrl_df['ps'].values.reshape(-1,1), 'euclidean')
    
    pairs = []
    used = set()
    for i in range(len(case_df)):
        for j in np.argsort(dist[i]):
            if ctrl_df.iloc[j]['participant_id'] not in used:
                if abs(case_df.iloc[i]['ps'] - ctrl_df.iloc[j]['ps']) < caliper:
                    pairs.append((case_df.iloc[i]['participant_id'], ctrl_df.iloc[j]['participant_id']))
                    used.add(ctrl_df.iloc[j]['participant_id'])
                    break
    
    ids = [p[0] for p in pairs] + [p[1] for p in pairs]
    df_matched = df[df['participant_id'].isin(ids)].copy()
    df_matched['status'] = (df_matched['event'] == 1).astype(int)
    df_matched['time'] = df_matched['cataract_time_to_event_days'].fillna(df_matched['followup_duration_cataract'])
    
    print(f"匹配后样本: {len(df_matched):,}, 病例: {(df_matched['status']==1).sum():,}")
    
    return df_matched

# ============================================================================
# 3. Cox回归分析
# ============================================================================

def cox_regression(df, var_col, covars):
    """Cox回归"""
    try:
        exog = df[[var_col] + covars].astype(float).dropna()
        idx = exog.index
        t = df.loc[idx, 'time'].values
        s = df.loc[idx, 'status'].values
        
        model = PHReg(t, exog, status=s)
        result = model.fit()
        
        beta, se, p = result.params[0], result.bse[0], result.pvalues[0]
        HR = np.exp(beta)
        CI_l, CI_u = np.exp(beta - 1.96 * se), np.exp(beta + 1.96 * se)
        
        return {'HR': HR, 'CI_low': CI_l, 'CI_high': CI_u, 'P': p, 'N': len(exog)}
    except Exception as e:
        return {'HR': np.nan, 'CI_low': np.nan, 'CI_high': np.nan, 'P': np.nan, 'N': 0}

def format_hr_ci(hr, ci_l, ci_u):
    """格式化HR(95%CI)"""
    return f"{hr:.2f} ({ci_l:.2f}-{ci_u:.2f})"

# ============================================================================
# 4. 四分位分析
# ============================================================================

def quartile_analysis(df, var_col, covars):
    """四分位分析"""
    df_q = df.copy()
    df_q['quartile'] = pd.qcut(df_q[var_col].rank(method='first'), q=4, 
                                 labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
    
    results = []
    for q in ['Q2', 'Q3', 'Q4']:
        df_q[f'q_{q}'] = (df_q['quartile'] == q).astype(int)
        exog = df_q[['q_' + q] + covars].astype(float).dropna()
        idx = exog.index
        t = df_q.loc[idx, 'time'].values
        s = df_q.loc[idx, 'status'].values
        
        model = PHReg(t, exog, status=s)
        result = model.fit()
        
        beta, se, p = result.params[0], result.bse[0], result.pvalues[0]
        HR = np.exp(beta)
        CI_l, CI_u = np.exp(beta - 1.96 * se), np.exp(beta + 1.96 * se)
        
        results.append({'quartile': q, 'HR': HR, 'CI': f"{CI_l:.2f}-{CI_u:.2f}", 'P': p})
    
    # 趋势检验 - 使用原始连续变量
    exog = df_q[[var_col] + covars].astype(float).dropna()
    idx = exog.index
    t = df_q.loc[idx, 'time'].values
    s = df_q.loc[idx, 'status'].values
    
    model = PHReg(t, exog, status=s)
    result = model.fit()
    trend_p = result.pvalues[0]
    
    return results, trend_p

# ============================================================================
# 5. 分层分析
# ============================================================================

def stratified_analysis(df, var_col, covars, stratify_col):
    """分层分析"""
    results = []
    for stratum in df[stratify_col].unique():
        if pd.isna(stratum):
            continue
        df_s = df[df[stratify_col] == stratum].copy()
        if len(df_s) < 100:
            continue
        
        covars_adj = [c for c in covars if c != stratify_col and c in df_s.columns]
        res = cox_regression(df_s, var_col, covars_adj)
        if not np.isnan(res['HR']):
            results.append({
                'stratum': str(stratum), 
                'HR': res['HR'], 
                'CI': format_hr_ci(res['HR'], res['CI_low'], res['CI_high']),
                'P': res['P'],
                'N': res['N']
            })
    return results

# ============================================================================
# 6. 交互检验
# ============================================================================

def interaction_test(df, var_col, covars, stratify_col):
    """交互检验"""
    try:
        df_i = df.copy()
        
        # 创建交互项
        if stratify_col == 'sex':
            df_i['interaction'] = df_i[var_col] * df_i['sex']
        else:
            return None
        
        exog_vars = [var_col, stratify_col, 'interaction'] + covars
        exog = df_i[exog_vars].astype(float).dropna()
        idx = exog.index
        t = df_i.loc[idx, 'time'].values
        s = df_i.loc[idx, 'status'].values
        
        model = PHReg(t, exog, status=s)
        result = model.fit()
        
        int_idx = list(exog.columns).index('interaction')
        p_interaction = result.pvalues[int_idx]
        
        return p_interaction
    except:
        return None

# ============================================================================
# 7. 敏感性分析
# ============================================================================

def sensitivity_analyses(df_original, main_results):
    """敏感性分析"""
    sens_results = []
    
    # 7.1 排除眼部疾病基线
    print("\n--- 敏感性分析1: 排除眼部疾病基线 ---")
    df_excl = df_original[(df_original.get('amd_baseline', 0) == 0) & 
                          (df_original.get('glaucoma_baseline', 0) == 0)].copy()
    if len(df_excl) > 1000:
        df_m = psm_matching(df_excl)
        res = cox_regression(df_m, 'fatty_acids_dha', ['age_baseline', 'sex'])
        print(f"DHA: HR={res['HR']:.2f} ({res['CI_low']:.2f}-{res['CI_high']:.2f}), P={res['P']:.4f}")
        sens_results.append({'分析': '排除眼部疾病', 'HR': res['HR'], 'CI': format_hr_ci(res['HR'], res['CI_low'], res['CI_high']), 'P': res['P']})
    
    # 7.2 延长随访（排除<2年）
    print("\n--- 敏感性分析2: 排除随访<2年 ---")
    df_2yr = df_original[df_original['followup_duration_cataract'] >= 730].copy()
    if len(df_2yr) > 1000:
        df_m = psm_matching(df_2yr)
        res = cox_regression(df_m, 'fatty_acids_dha', ['age_baseline', 'sex'])
        print(f"DHA: HR={res['HR']:.2f} ({res['CI_low']:.2f}-{res['CI_high']:.2f}), P={res['P']:.4f}")
        sens_results.append({'分析': '排除随访<2年', 'HR': res['HR'], 'CI': format_hr_ci(res['HR'], res['CI_low'], res['CI_high']), 'P': res['P']})
    
    # 7.3 不同卡尺值
    print("\n--- 敏感性分析3: 不同卡尺值 ---")
    for caliper in [0.05, 0.15]:
        df_m = psm_matching(df_original, caliper=caliper)
        res = cox_regression(df_m, 'fatty_acids_dha', ['age_baseline', 'sex'])
        print(f"卡尺{caliper} - DHA: HR={res['HR']:.2f} ({res['CI_low']:.2f}-{res['CI_high']:.2f}), P={res['P']:.4f}")
        sens_results.append({'分析': f'卡尺={caliper}', 'HR': res['HR'], 'CI': format_hr_ci(res['HR'], res['CI_low'], res['CI_high']), 'P': res['P']})
    
    return sens_results

# ============================================================================
# 8. 生成基线特征表
# ============================================================================

def generate_table1(df_before, df_after):
    """生成Table1基线特征表"""
    from scipy.stats import ttest_ind, chi2_contingency
    
    rows = []
    
    # 连续变量
    continuous_vars = ['age_baseline', 'bmi_baseline']
    for var in continuous_vars:
        if var in df_before.columns:
            c1 = df_before[df_before['status']==1][var].dropna()
            c0 = df_before[df_before['status']==0][var].dropna()
            if len(c1)>0 and len(c0)>0:
                _, p = ttest_ind(c1, c0, equal_var=False)
                rows.append({
                    'Variable': var, 'Case_Match': f"{c1.mean():.1f}±{c1.std():.1f}",
                    'Control_Match': f"{c0.mean():.1f}±{c0.std():.1f}", 'P': p
                })
    
    # 分类变量
    categorical_vars = {'sex': {0:'Female', 1:'Male'}}
    for var, mapping in categorical_vars.items():
        if var in df_before.columns:
            ct = pd.crosstab(df_before[var], df_before['status'])
            if ct.shape == (2,2):
                _, p = chi2_contingency(ct)
                for val, label in mapping.items():
                    n1 = ct.loc[val, 1] if val in ct.index and 1 in ct.columns else 0
                    n0 = ct.loc[val, 0] if val in ct.index and 0 in ct.columns else 0
                    pct1 = n1 / ct[1].sum() * 100 if ct[1].sum() > 0 else 0
                    pct0 = n0 / ct[0].sum() * 100 if ct[0].sum() > 0 else 0
                    rows.append({
                        'Variable': f"  {label}", 
                        'Case_Match': f"{n1} ({pct1:.1f}%)",
                        'Control_Match': f"{n0} ({pct0:.1f}%)", 'P': ''
                    })
    
    return pd.DataFrame(rows)

# ============================================================================
# 主程序
# ============================================================================

def main():
    print("="*70)
    print("Omega-3/DHA与白内障关系 - 完整论文级别分析")
    print("="*70)
    
    # 加载数据
    df = load_data()
    print(f"\n原始数据: {len(df):,} 人")
    
    # PSM匹配
    print("\n" + "="*70)
    print("1. 倾向评分匹配 (仅age+sex)")
    print("="*70)
    df_matched = psm_matching(df)
    
    # 匹配质量评估
    print("\n--- SMD评估 ---")
    for var in ['age_baseline', 'sex']:
        c = df_matched[df_matched['status']==1][var]
        ctrl = df_matched[df_matched['status']==0][var]
        smd = abs(c.mean() - ctrl.mean()) / np.sqrt((c.std()**2 + ctrl.std()**2)/2)
        print(f"  {var}: SMD={smd:.4f}")
    
    # 定义协变量
    covars_model1 = ['age_baseline', 'sex']
    covars_model2 = ['age_baseline', 'sex', 'bmi_baseline', 'smoking_status_baseline', 'alcohol_frequency_baseline']
    covars_model3 = covars_model2 + ['hypertension_baseline', 'diabetes_baseline', 'heart_disease_composite']
    
    # 定义脂肪酸变量
    fa_vars = [
        ('Omega-3', 'fatty_acids_n3'),
        ('DHA', 'fatty_acids_dha'),
        ('Omega-6', 'fatty_acids_n6'),
        ('PUFA', 'fatty_acids_pufa'),
        ('Total FA', 'fatty_acids_total'),
    ]
    
    # ===== 2. 主分析: 多模型Cox =====
    print("\n" + "="*70)
    print("2. Cox回归 - 多模型分析")
    print("="*70)
    
    model_results = []
    for name, col in fa_vars:
        print(f"\n--- {name} ---")
        
        # Model 1 (Crude)
        res1 = cox_regression(df_matched, col, covars_model1)
        print(f"Model 1: HR={res1['HR']:.2f} ({res1['CI_low']:.2f}-{res1['CI_high']:.2f}), P={res1['P']:.4f}")
        
        # Model 2
        res2 = cox_regression(df_matched, col, covars_model2)
        print(f"Model 2: HR={res2['HR']:.2f} ({res2['CI_low']:.2f}-{res2['CI_high']:.2f}), P={res2['P']:.4f}")
        
        # Model 3 (Full)
        res3 = cox_regression(df_matched, col, covars_model3)
        print(f"Model 3: HR={res3['HR']:.2f} ({res3['CI_low']:.2f}-{res3['CI_high']:.2f}), P={res3['P']:.4f}")
        
        model_results.append({
            'Variable': name, 
            'Model1_HR': f"{res1['HR']:.2f}", 'Model1_CI': f"{res1['CI_low']:.2f}-{res1['CI_high']:.2f}", 'Model1_P': res1['P'],
            'Model2_HR': f"{res2['HR']:.2f}", 'Model2_CI': f"{res2['CI_low']:.2f}-{res2['CI_high']:.2f}", 'Model2_P': res2['P'],
            'Model3_HR': f"{res3['HR']:.2f}", 'Model3_CI': f"{res3['CI_low']:.2f}-{res3['CI_high']:.2f}", 'Model3_P': res3['P'],
        })
    
    # ===== 3. 四分位分析 =====
    print("\n" + "="*70)
    print("3. 四分位分析 + 趋势检验")
    print("="*70)
    
    quartile_results = []
    for name, col in [('Omega-3', 'fatty_acids_n3'), ('DHA', 'fatty_acids_dha')]:
        print(f"\n--- {name} ---")
        q_res, trend_p = quartile_analysis(df_matched, col, covars_model1)
        for r in q_res:
            sig = "*" if r['P'] < 0.05 else ""
            print(f"  {r['quartile']}: HR={r['HR']:.2f} ({r['CI']}), P={r['P']:.4f} {sig}")
        print(f"  P for trend: {trend_p:.4f}")
        
        quartile_results.append({'Variable': name, 'Q2': q_res[0], 'Q3': q_res[1], 'Q4': q_res[2], 'P_trend': trend_p})
    
    # ===== 4. 分层分析 =====
    print("\n" + "="*70)
    print("4. 分层分析")
    print("="*70)
    
    # 性别分层
    print("\n--- 按性别分层 ---")
    for sex_val in [0, 1]:
        label = 'Female' if sex_val == 0 else 'Male'
        df_sex = df_matched[df_matched['sex'] == sex_val]
        if len(df_sex) > 100:
            for name, col in [('Omega-3', 'fatty_acids_n3'), ('DHA', 'fatty_acids_dha')]:
                res = cox_regression(df_sex, col, covars_model1)
                sig = "*" if res['P'] < 0.05 else ""
                print(f"  {label}-{name}: HR={res['HR']:.2f} ({res['CI_low']:.2f}-{res['CI_high']:.2f}), P={res['P']:.4f} {sig}")
    
    # 年龄分层
    print("\n--- 按年龄分层 ---")
    df_matched['age_group'] = pd.cut(df_matched['age_baseline'], bins=[0,55,65,100], labels=['<55', '55-64', '>=65'])
    for ag in ['<55', '55-64', '>=65']:
        df_age = df_matched[df_matched['age_group'] == ag]
        if len(df_age) > 100:
            for name, col in [('Omega-3', 'fatty_acids_n3'), ('DHA', 'fatty_acids_dha')]:
                res = cox_regression(df_age, col, covars_model1)
                sig = "*" if res['P'] < 0.05 else ""
                print(f"  Age {ag}-{name}: HR={res['HR']:.2f} ({res['CI_low']:.2f}-{res['CI_high']:.2f}), P={res['P']:.4f} {sig}")
    
    # BMI分层
    print("\n--- 按BMI分层 ---")
    df_matched['bmi_group'] = pd.cut(df_matched['bmi_baseline'], bins=[0,25,30,100], labels=['<25', '25-30', '>=30'])
    for bmi in ['<25', '25-30', '>=30']:
        df_b = df_matched[df_matched['bmi_group'] == bmi]
        if len(df_b) > 100:
            for name, col in [('Omega-3', 'fatty_acids_n3'), ('DHA', 'fatty_acids_dha')]:
                res = cox_regression(df_b, col, covars_model1)
                sig = "*" if res['P'] < 0.05 else ""
                print(f"  BMI {bmi}-{name}: HR={res['HR']:.2f} ({res['CI_low']:.2f}-{res['CI_high']:.2f}), P={res['P']:.4f} {sig}")
    
    # ===== 5. 交互检验 =====
    print("\n" + "="*70)
    print("5. 交互检验")
    print("="*70)
    
    for name, col in [('Omega-3', 'fatty_acids_n3'), ('DHA', 'fatty_acids_dha')]:
        p_int = interaction_test(df_matched, col, covars_model1, 'sex')
        if p_int:
            sig = "*" if p_int < 0.05 else ""
            print(f"  {name} x sex: P for interaction = {p_int:.4f} {sig}")
    
    # ===== 6. 敏感性分析 =====
    print("\n" + "="*70)
    print("6. 敏感性分析")
    print("="*70)
    
    sens_results = sensitivity_analyses(df, model_results)
    
    # ===== 7. 保存结果 =====
    print("\n" + "="*70)
    print("7. 保存结果")
    print("="*70)
    
    # 保存匹配数据
    df_matched.to_csv('final_matched_data_v2.csv', index=False)
    print("匹配数据: final_matched_data_v2.csv")
    
    # 保存主分析结果
    pd.DataFrame(model_results).to_excel('Table2_Cox_MultiModel.xlsx', index=False)
    print("Table2 (Cox多模型): Table2_Cox_MultiModel.xlsx")
    
    # 保存敏感性分析
    if sens_results:
        pd.DataFrame(sens_results).to_excel('Table3_Sensitivity.xlsx', index=False)
        print("Table3 (敏感性分析): Table3_Sensitivity.xlsx")
    
    print("\n" + "="*70)
    print("分析完成!")
    print("="*70)

if __name__ == "__main__":
    main()
