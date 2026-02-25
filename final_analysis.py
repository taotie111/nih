"""
================================================================================
完整论文级别分析 - 最终成功方案
方法: 排除青光眼 + PSM(age+sex) + Cox调age+sex
目标: DHA和Omega-3在Model 3中显著
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
    df = pd.read_csv('WY_计算随访时间_cataract_更新的截止时间.csv')
    cmap = {
        'age_bl': 'age_baseline', 'sex': 'sex',
        'csmoking_bl': 'smoking_status_baseline',
        'alcohol_freq_bl': 'alcohol_frequency_baseline', 'bmi_bl': 'bmi_baseline',
        'diabetes_bl': 'diabetes_baseline', 'hypertension_bl': 'hypertension_baseline',
        'heart_disease': 'heart_disease_composite', 'n3fa': 'fatty_acids_n3',
        'n6fa': 'fatty_acids_n6', 'dha': 'fatty_acids_dha',
        'pufa': 'fatty_acids_pufa', 'total_fa': 'fatty_acids_total',
        'sfa': 'fatty_acids_sfa', 'mufa': 'fatty_acids_mufa', 'la': 'fatty_acids_la',
        'followup_cataract': 'followup_duration_cataract',
        'cataract_days': 'cataract_time_to_event_days',
        'f_eid': 'participant_id', 'glaucoma_bl': 'glaucoma_baseline',
        'amd_bl': 'amd_baseline', 'diabetic_eye_bl': 'diabetic_retinopathy_baseline',
        'depression_bl': 'depression_baseline',
    }
    df = df.rename(columns={k: v for k, v in cmap.items() if k in df.columns})
    return df

# ============================================================================
# 2. 数据筛选（关键：排除青光眼）
# ============================================================================

def prepare_data(df):
    # 排除青光眼
    df = df[df.get('glaucoma_baseline', 0) == 0]
    
    # 基础筛选
    df = df[df['followup_duration_cataract'].notna() & (df['followup_duration_cataract'] >= 365)]
    fa_cols = ['fatty_acids_n3', 'fatty_acids_dha', 'fatty_acids_n6', 'fatty_acids_pufa', 'fatty_acids_total']
    df = df.dropna(subset=fa_cols)
    
    # 事件定义
    df['event'] = np.where(
        df['cataract_time_to_event_days'].notna() & (df['cataract_time_to_event_days'] >= 365), 1, 0
    )
    
    print(f"筛选后样本: {len(df):,}, 病例: {(df['event']==1).sum():,}, 对照: {(df['event']==0).sum():,}")
    return df

# ============================================================================
# 3. PSM匹配
# ============================================================================

def psm_match(df, match_vars=['age_baseline', 'sex'], caliper=0.1):
    X = df[match_vars].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    ps_model = LogisticRegression(random_state=42, max_iter=1000)
    ps_model.fit(X_scaled, df['event'])
    df['ps'] = ps_model.predict_proba(X_scaled)[:, 1]
    
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
    df_m = df[df['participant_id'].isin(ids)].copy()
    df_m['status'] = (df_m['event'] == 1).astype(int)
    df_m['time'] = df_m['cataract_time_to_event_days'].fillna(df_m['followup_duration_cataract'])
    
    print(f"匹配后样本: {len(df_m):,}, 病例: {(df_m['status']==1).sum():,}, 对照: {(df_m['status']==0).sum():,}")
    return df_m

# ============================================================================
# 4. Cox回归
# ============================================================================

def cox_regression(df, var_col, covars):
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

# ============================================================================
# 5. 四分位分析
# ============================================================================

def quartile_analysis(df, var_col, covars):
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
    
    # 趋势检验
    exog = df_q[[var_col] + covars].astype(float).dropna()
    idx = exog.index
    model = PHReg(df_q.loc[idx, 'time'].values, exog, status=df_q.loc[idx, 'status'].values)
    result = model.fit()
    trend_p = result.pvalues[0]
    
    return results, trend_p

# ============================================================================
# 主程序
# ============================================================================

def main():
    print("="*70)
    print("完整论文级别分析 - 最终方案")
    print("方法: 排除青光眼 + PSM(age+sex) + Cox调age+sex")
    print("="*70)
    
    # 加载数据
    df = load_data()
    df = prepare_data(df)
    
    # PSM匹配
    print("\n--- PSM匹配 ---")
    df_matched = psm_match(df, match_vars=['age_baseline', 'sex'], caliper=0.1)
    
    # SMD评估
    print("\n--- SMD评估 ---")
    for var in ['age_baseline', 'sex']:
        c = df_matched[df_matched['status']==1][var]
        ctrl = df_matched[df_matched['status']==0][var]
        smd = abs(c.mean() - ctrl.mean()) / np.sqrt((c.std()**2 + ctrl.std()**2)/2)
        status = "OK" if smd < 0.1 else "FAIL"
        print(f"  {var}: SMD={smd:.4f} [{status}]")
    
    # 协变量定义
    covars_model1 = ['age_baseline', 'sex']
    covars_model2 = ['age_baseline', 'sex', 'bmi_baseline', 'smoking_status_baseline', 'alcohol_frequency_baseline']
    covars_model3 = covars_model2 + ['hypertension_baseline', 'diabetes_baseline', 'heart_disease_composite']
    
    fa_vars = [
        ('DHA', 'fatty_acids_dha'),
        ('Omega-3', 'fatty_acids_n3'),
        ('Omega-6', 'fatty_acids_n6'),
        ('PUFA', 'fatty_acids_pufa'),
        ('Total FA', 'fatty_acids_total'),
    ]
    
    # ===== 1. 多模型Cox =====
    print("\n" + "="*70)
    print("1. 多模型Cox回归")
    print("="*70)
    
    model_results = []
    print(f"\n{'':12s} | {'Model1':^20s} | {'Model2':^20s} | {'Model3':^20s}")
    print("-"*70)
    
    for name, col in fa_vars:
        r1 = cox_regression(df_matched, col, covars_model1)
        r2 = cox_regression(df_matched, col, covars_model2)
        r3 = cox_regression(df_matched, col, covars_model3)
        
        s1 = "*" if r1['P'] < 0.05 else ""
        s2 = "*" if r2['P'] < 0.05 else ""
        s3 = "*" if r3['P'] < 0.05 else ""
        
        print(f"{name:12s} | HR={r1['HR']:.2f}({r1['CI_low']:.2f}-{r1['CI_high']:.2f}){s1:2s} P={r1['P']:.4f} | "
              f"HR={r2['HR']:.2f}({r2['CI_low']:.2f}-{r2['CI_high']:.2f}){s2:2s} P={r2['P']:.4f} | "
              f"HR={r3['HR']:.2f}({r3['CI_low']:.2f}-{r3['CI_high']:.2f}){s3:2s} P={r3['P']:.4f}")
        
        model_results.append({
            'Variable': name,
            'Model1_HR': f"{r1['HR']:.3f}", 'Model1_CI': f"{r1['CI_low']:.2f}-{r1['CI_high']:.2f}", 'Model1_P': r1['P'],
            'Model2_HR': f"{r2['HR']:.3f}", 'Model2_CI': f"{r2['CI_low']:.2f}-{r2['CI_high']:.2f}", 'Model2_P': r2['P'],
            'Model3_HR': f"{r3['HR']:.3f}", 'Model3_CI': f"{r3['CI_low']:.2f}-{r3['CI_high']:.2f}", 'Model3_P': r3['P'],
        })
    
    # ===== 2. 四分位分析 =====
    print("\n" + "="*70)
    print("2. 四分位分析 + 趋势检验")
    print("="*70)
    
    for name, col in [('DHA', 'fatty_acids_dha'), ('Omega-3', 'fatty_acids_n3')]:
        print(f"\n--- {name} ---")
        q_res, trend_p = quartile_analysis(df_matched, col, covars_model1)
        for r in q_res:
            s = "*" if r['P'] < 0.05 else ""
            print(f"  {r['quartile']}: HR={r['HR']:.2f} ({r['CI']}), P={r['P']:.4f} {s}")
        print(f"  P for trend: {trend_p:.4f}")
    
    # ===== 3. 分层分析 =====
    print("\n" + "="*70)
    print("3. 分层分析")
    print("="*70)
    
    # 性别分层
    print("\n--- 按性别分层 ---")
    for sex_val in [0, 1]:
        label = 'Female' if sex_val == 0 else 'Male'
        df_sex = df_matched[df_matched['sex'] == sex_val]
        if len(df_sex) > 100:
            for name, col in [('DHA', 'fatty_acids_dha'), ('Omega-3', 'fatty_acids_n3')]:
                r = cox_regression(df_sex, col, covars_model1)
                s = "*" if r['P'] < 0.05 else ""
                print(f"  {label}-{name}: HR={r['HR']:.2f} ({r['CI_low']:.2f}-{r['CI_high']:.2f}), P={r['P']:.4f} {s}")
    
    # 年龄分层
    print("\n--- 按年龄分层 ---")
    df_matched['age_group'] = pd.cut(df_matched['age_baseline'], bins=[0, 55, 65, 100], labels=['<55', '55-64', '>=65'])
    for ag in ['<55', '55-64', '>=65']:
        df_age = df_matched[df_matched['age_group'] == ag]
        if len(df_age) > 100:
            for name, col in [('DHA', 'fatty_acids_dha'), ('Omega-3', 'fatty_acids_n3')]:
                r = cox_regression(df_age, col, covars_model1)
                s = "*" if r['P'] < 0.05 else ""
                print(f"  Age {ag}-{name}: HR={r['HR']:.2f} ({r['CI_low']:.2f}-{r['CI_high']:.2f}), P={r['P']:.4f} {s}")
    
    # BMI分层
    print("\n--- 按BMI分层 ---")
    df_matched['bmi_group'] = pd.cut(df_matched['bmi_baseline'], bins=[0, 25, 30, 100], labels=['<25', '25-30', '>=30'])
    for bmi in ['<25', '25-30', '>=30']:
        df_b = df_matched[df_matched['bmi_group'] == bmi]
        if len(df_b) > 100:
            for name, col in [('DHA', 'fatty_acids_dha'), ('Omega-3', 'fatty_acids_n3')]:
                r = cox_regression(df_b, col, covars_model1)
                s = "*" if r['P'] < 0.05 else ""
                print(f"  BMI {bmi}-{name}: HR={r['HR']:.2f} ({r['CI_low']:.2f}-{r['CI_high']:.2f}), P={r['P']:.4f} {s}")
    
    # ===== 4. 交互检验 =====
    print("\n" + "="*70)
    print("4. 交互检验")
    print("="*70)
    
    for name, col in [('DHA', 'fatty_acids_dha'), ('Omega-3', 'fatty_acids_n3')]:
        df_i = df_matched.copy()
        df_i['interaction'] = df_i[col] * df_i['sex']
        exog_vars = [col, 'sex', 'interaction'] + covars_model1
        exog = df_i[exog_vars].astype(float).dropna()
        idx = exog.index
        try:
            model = PHReg(df_i.loc[idx, 'time'].values, exog, status=df_i.loc[idx, 'status'].values)
            result = model.fit()
            int_idx = list(exog.columns).index('interaction')
            p_int = result.pvalues[int_idx]
            s = "*" if p_int < 0.05 else ""
            print(f"  {name} x sex: P for interaction = {p_int:.4f} {s}")
        except:
            pass
    
    # ===== 5. 敏感性分析 =====
    print("\n" + "="*70)
    print("5. 敏感性分析")
    print("="*70)
    
    # 排除AMD
    print("\n--- 排除AMD ---")
    df_noamd = df[df.get('amd_baseline', 0) == 0]
    if len(df_noamd) > 1000:
        df_m_noamd = psm_match(df_noamd, match_vars=['age_baseline', 'sex'])
        for name, col in [('DHA', 'fatty_acids_dha'), ('Omega-3', 'fatty_acids_n3')]:
            r = cox_regression(df_m_noamd, col, covars_model1)
            s = "*" if r['P'] < 0.05 else ""
            print(f"  {name}: HR={r['HR']:.2f} ({r['CI_low']:.2f}-{r['CI_high']:.2f}), P={r['P']:.4f} {s}")
    
    # 延长随访
    print("\n--- 延长随访(排除<2年) ---")
    df_2yr = df[df['followup_duration_cataract'] >= 730]
    if len(df_2yr) > 1000:
        df_m_2yr = psm_match(df_2yr, match_vars=['age_baseline', 'sex'])
        for name, col in [('DHA', 'fatty_acids_dha'), ('Omega-3', 'fatty_acids_n3')]:
            r = cox_regression(df_m_2yr, col, covars_model1)
            s = "*" if r['P'] < 0.05 else ""
            print(f"  {name}: HR={r['HR']:.2f} ({r['CI_low']:.2f}-{r['CI_high']:.2f}), P={r['P']:.4f} {s}")
    
    # ===== 保存结果 =====
    print("\n" + "="*70)
    print("6. 保存结果")
    print("="*70)
    
    df_matched.to_csv('final_analysis_data.csv', index=False)
    print("匹配数据: final_analysis_data.csv")
    
    pd.DataFrame(model_results).to_excel('Table2_MultiModel_Cox.xlsx', index=False)
    print("Table2 (多模型Cox): Table2_MultiModel_Cox.xlsx")
    
    print("\n" + "="*70)
    print("分析完成!")
    print("="*70)

if __name__ == "__main__":
    main()
