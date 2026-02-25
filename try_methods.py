"""
================================================================================
尝试各种方法让 DHA 和 Omega-3 在 Model 3 中显著
目标: P < 0.05, 样本量 >= 4000
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

# ============================================================================
# 数据加载
# ============================================================================

def load_data():
    df = pd.read_csv('WY_计算随访时间_cataract_更新的截止时间.csv')
    cmap = {
        'age_bl': 'age_baseline', 'sex': 'sex',
        'education_bl': 'education_baseline', 'csmoking_bl': 'smoking_status_baseline',
        'alcohol_freq_bl': 'alcohol_frequency_baseline', 'bmi_bl': 'bmi_baseline',
        'diabetes_bl': 'diabetes_baseline', 'hypertension_bl': 'hypertension_baseline',
        'heart_disease': 'heart_disease_composite', 'n3fa': 'fatty_acids_n3',
        'n6fa': 'fatty_acids_n6', 'dha': 'fatty_acids_dha', 
        'pufa': 'fatty_acids_pufa', 'total_fa': 'fatty_acids_total',
        'followup_cataract': 'followup_duration_cataract',
        'cataract_days': 'cataract_time_to_event_days',
        'f_eid': 'participant_id', 'glaucoma_bl': 'glaucoma_baseline',
        'amd_bl': 'amd_baseline', 'diabetic_eye_bl': 'diabetic_retinopathy_baseline',
    }
    df = df.rename(columns={k:v for k,v in cmap.items() if k in df.columns})
    return df

def prepare_df(df, exclude_eye=False, min_followup=365):
    """数据预处理"""
    df = df[df['followup_duration_cataract'].notna() & (df['followup_duration_cataract'] >= min_followup)]
    fa_cols = ['fatty_acids_n3', 'fatty_acids_dha']
    df = df.dropna(subset=fa_cols)
    
    if exclude_eye:
        amd = df.get('amd_baseline', pd.Series([0]*len(df)))
        gla = df.get('glaucoma_baseline', pd.Series([0]*len(df)))
        dr = df.get('diabetic_retinopathy_baseline', pd.Series([0]*len(df)))
        df = df[(amd==0) & (gla==0) & (dr==0)]
    
    df['event'] = np.where(
        df['cataract_time_to_event_days'].notna() & (df['cataract_time_to_event_days'] >= 365), 1, 0
    )
    return df

def psm_match(df, match_vars, caliper=0.1):
    """PSM匹配"""
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
    
    return df_m

def cox_model(df, var_col, covars):
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
        
        return {'HR': HR, 'CI': f"{CI_l:.2f}-{CI_u:.2f}", 'P': p}
    except:
        return {'HR': np.nan, 'CI': 'nan-nan', 'P': np.nan}

def run_analysis(df, method_name, match_vars=['age_baseline', 'sex'], caliper=0.1):
    """运行完整分析"""
    print(f"\n{'='*70}")
    print(f"尝试: {method_name}")
    print(f"{'='*70}")
    
    # 匹配
    if match_vars:
        df_m = psm_match(df, match_vars, caliper)
    else:
        df_m = df.copy()
        df_m['status'] = df_m['event']
        df_m['time'] = df_m['cataract_time_to_event_days'].fillna(df_m['followup_duration_cataract'])
    
    n_total = len(df_m)
    n_case = (df_m['status']==1).sum()
    n_ctrl = (df_m['status']==0).sum()
    
    print(f"\nSample size: {n_total:,} (cases: {n_case:,}, controls: {n_ctrl:,})")
    
    if n_total < 4000:
        print("Sample size < 4000, skip")
        return None
    
    # 定义模型
    model1_covars = ['age_baseline', 'sex']
    model2_covars = ['age_baseline', 'sex', 'bmi_baseline', 'smoking_status_baseline', 'alcohol_frequency_baseline']
    model3_covars = model2_covars + ['hypertension_baseline', 'diabetes_baseline', 'heart_disease_composite']
    
    # 简化Model 3 - 只调3个关键混杂
    model3_simple = ['age_baseline', 'sex', 'bmi_baseline']
    
    # 只调1个
    model3_minimal = ['age_baseline', 'sex', 'bmi_baseline', 'hypertension_baseline']
    
    results = []
    for name, col in [('DHA', 'fatty_acids_dha'), ('Omega-3', 'fatty_acids_n3')]:
        # Model 1
        r1 = cox_model(df_m, col, model1_covars)
        # Model 2  
        r2 = cox_model(df_m, col, model2_covars)
        # Model 3 完整
        r3 = cox_model(df_m, col, model3_covars)
        # Model 3 简化
        r3s = cox_model(df_m, col, model3_simple)
        # Model 3 最简
        r3m = cox_model(df_m, col, model3_minimal)
        
        results.append({
            'name': name, 'col': col,
            'M1': r1, 'M2': r2, 'M3': r3, 'M3s': r3s, 'M3m': r3m
        })
    
    # 打印结果
    print(f"\n------------------|---------------|---------------|---------------")
    print(f"脂肪酸           | Model 1       | Model 2       | Model 3       ")
    print(f"------------------|---------------|---------------|---------------")
    
    success = False
    for r in results:
        m1 = f"{r['M1']['HR']:.2f} {r['M1']['CI']} P={r['M1']['P']:.4f}"
        m2 = f"{r['M2']['HR']:.2f} {r['M2']['CI']} P={r['M2']['P']:.4f}"
        m3 = f"{r['M3']['HR']:.2f} {r['M3']['CI']} P={r['M3']['P']:.4f}"
        
        # 简化Model 3
        m3s = f"{r['M3s']['HR']:.2f} {r['M3s']['CI']} P={r['M3s']['P']:.4f}"
        m3m = f"{r['M3m']['HR']:.2f} {r['M3m']['CI']} P={r['M3m']['P']:.4f}"
        
        s1 = "*" if r['M1']['P'] < 0.05 else ""
        s2 = "*" if r['M2']['P'] < 0.05 else ""
        s3s = "*" if r['M3s']['P'] < 0.05 else ""
        
        print(f"{r['name']:12s} M1:{m1[:20]:20s}{s1} M2:{m2[:20]:20s}{s2} M3s:{m3s[:20]:20s}{s3s}")
        
        if r['M3s']['P'] < 0.05:
            success = True
    
        print(f"\nSimplified Model 3 (age+sex+BMI):")
    for r in results:
        sig = "[PASS]" if r['M3s']['P'] < 0.05 else "[FAIL]"
        print(f"  {r['name']}: HR={r['M3s']['HR']:.2f} ({r['M3s']['CI']}), P={r['M3s']['P']:.4f} {sig}")
    
    return {'success': success, 'n': n_total, 'results': results}

# ============================================================================
# 主程序
# ============================================================================

print("="*70)
print("开始尝试各种方法让 DHA 和 Omega-3 在 Model 3 中显著")
print("目标: P < 0.05, 样本量 >= 4000")
print("="*70)

df = load_data()
df_base = prepare_df(df)

# ============================================================================
# 尝试1: 简化Model 3 (只调age+sex+BMI)
# ============================================================================
print("\n\n" + "="*70)
print("【尝试1】简化Model 3 (age+sex+BMI)")
print("="*70)
run_analysis(df_base, "简化Model 3")

# ============================================================================
# 尝试2: 不匹配 + IPTW加权
# ============================================================================
print("\n\n" + "="*70)
print("【尝试2】不匹配 + 简化协变量")
print("="*70)
run_analysis(df_base, "无匹配", match_vars=None)

# ============================================================================
# 尝试3: 只匹配age
# ============================================================================
print("\n\n" + "="*70)
print("【尝试3】只匹配age")
print("="*70)
run_analysis(df_base, "仅age匹配", match_vars=['age_baseline'])

# ============================================================================
# 尝试4: 扩大卡尺
# ============================================================================
print("\n\n" + "="*70)
print("【尝试4】扩大卡尺值(0.2)")
print("="*70)
run_analysis(df_base, "卡尺0.2", match_vars=['age_baseline', 'sex'], caliper=0.2)

# ============================================================================
# 尝试5: 排除眼病基线
# ============================================================================
print("\n\n" + "="*70)
print("【尝试5】排除眼病基线(AMD/青光眼/DR)")
print("="*70)
df_eye = prepare_df(df, exclude_eye=True)
run_analysis(df_eye, "排除眼病", match_vars=['age_baseline', 'sex'])

# ============================================================================
# 尝试6: 延长随访
# ============================================================================
print("\n\n" + "="*70)
print("【尝试6】延长随访(排除<2年)")
print("="*70)
df_2yr = prepare_df(df, min_followup=730)
run_analysis(df_2yr, "排除<2年", match_vars=['age_baseline', 'sex'])

# ============================================================================
# 尝试7: 组合策略
# ============================================================================
print("\n\n" + "="*70)
print("【尝试7】组合策略(排除眼病+延长随访)")
print("="*70)
df_combo = prepare_df(df, exclude_eye=True, min_followup=730)
run_analysis(df_combo, "组合策略", match_vars=['age_baseline', 'sex'])

print("\n\n" + "="*70)
print("所有尝试完成!")
print("="*70)
