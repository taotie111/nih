#!/usr/bin/env python3
"""
Extended analysis for n3fa_grs and other GRS exposures on cataract outcome.
No PS(M). Cox PH model across thresholds.
Supports: 365, 730, 1095 day thresholds; multiple GRS exposures (n3fa_grs, n6fa_grs, pufa_grs, tfa_grs),
 layer-based exposures (n3grs_layer4, n3grs_layer5, n3grs_layer2, n3grs_layer1_23_4, n3grs_1_234, etc).
Resolve covariates from dataset dynamically among: age_baseline, sex, bmi_baseline, smoking_status_baseline,
alcohol_frequency_baseline, hypertension_baseline.
"""
import pandas as pd
import numpy as np
from statsmodels.duration.hazard_regression import PHReg
import warnings
warnings.filterwarnings('ignore')
import sys

def load_and_prepare(df_path='WY_计算随访时间_cataract_更新的截止时间.csv'):
    df = pd.read_csv(df_path)
    # Normalize potential column names
    cmap = {
        'age_bl': 'age_baseline', 'sex': 'sex',
        'csmoking_bl': 'smoking_status_baseline',
        'alcohol_freq_bl': 'alcohol_frequency_baseline', 'bmi_bl': 'bmi_baseline',
        'diabetes_bl': 'diabetes_baseline', 'hypertension_bl': 'hypertension_baseline',
        'heart_disease': 'heart_disease_composite', 'n3fa': 'fatty_acids_n3',
        'n6fa': 'fatty_acids_n6', 'pufa': 'fatty_acids_pufa', 'tfa': 'fatty_acids_tfa',
        'followup_cataract': 'followup_duration_cataract',
        'cataract_days': 'cataract_time_to_event_days', 'f_eid': 'participant_id',
        'glaucoma_bl': 'glaucoma_baseline', 'amd_bl': 'amd_baseline'
    }
    df = df.rename(columns={k: v for k, v in cmap.items() if k in df.columns})
    # 基线筛选：排除青光眼，保留随访信息足够的样本
    if 'glaucoma_baseline' in df.columns:
        df = df[df['glaucoma_baseline'] == 0]
    df = df[df['followup_duration_cataract'].notna()]
    return df

def resolve_covariates(df, prefer=None):
    if prefer is None:
        prefer = ['age_baseline','sex','bmi_baseline','smoking_status_baseline','alcohol_frequency_baseline','hypertension_baseline']
    covs = [c for c in prefer if c in df.columns]
    if len(covs) == 0:
        # fallback
        for c in ['age_bl','sex','bmi_bl','csmoking_bl','alcohol_bl','hypertension_bl']:
            if c in df.columns:
                covs.append(c)
    return covs

def run_cox_for_exposure(df, exposure, covars, min_days):
    if exposure not in df.columns:
        return None
    df2 = df.dropna(subset=[exposure])
    df2 = df2.copy()
    df2['event'] = (df2['cataract_time_to_event_days'].notna()) & (df2['cataract_time_to_event_days'] >= min_days)
    df2['event'] = df2['event'].astype(int)
    df2['time'] = df2['cataract_time_to_event_days'].fillna(df2['followup_duration_cataract'])
    cols = [exposure] + covars
    exog = df2[cols].astype(float).dropna()
    if exog.shape[0] == 0:
        return None
    idx = exog.index
    t = df2.loc[idx, 'time'].values
    s = df2.loc[idx, 'event'].values
    model = PHReg(t, exog, status=s)
    try:
        res = model.fit()
        beta, se, p = res.params[0], res.bse[0], res.pvalues[0]
        HR = np.exp(beta)
        CI_l, CI_u = np.exp(beta - 1.96 * se), np.exp(beta + 1.96 * se)
        return {'Exposure': exposure, 'MinDays': min_days, 'N': len(exog),
                'HR': HR, 'CI_low': CI_l, 'CI_high': CI_u, 'P': p}
    except Exception:
        return {'Exposure': exposure, 'MinDays': min_days, 'N': len(exog),
                'HR': np.nan, 'CI_low': np.nan, 'CI_high': np.nan, 'P': np.nan}

def run_all(df, exposures, mins=[365,730,1095]):
    covars = resolve_covariates(df)
    if len(covars) == 0:
        raise RuntimeError('No covariates found to use in Cox model')
    results = []
    for exposure in exposures:
        for m in mins:
            r = run_cox_for_exposure(df, exposure, covars, m)
            if r:
                results.append(r)
    return pd.DataFrame(results)

def stratified_by_sex(df, exposure, covars, min_days=730):
    """Return Cox results stratified by sex (Female/Male) for a single exposure."""
    out = []
    for sex_val, label in [(0, 'Female'), (1, 'Male')]:
        sub = df[(df.get('sex') == sex_val) & (df[exposure].notna())]
        if sub.empty:
            continue
        r = run_cox_for_exposure(sub, exposure, covars, min_days)
        if r is not None:
            r['Group'] = label
            out.append(r)
    if not out:
        return None
    return pd.DataFrame(out)

def quartile_analysis_exposure(df, exposure, covars, min_days=730):
    """NT: Non-linear analysis via quartiles (Q2-Q4) and trend."""
    df_q = df.copy()
    # ensure time column exists for quartile analysis
    if 'time' not in df_q.columns:
        df_q['time'] = df_q['cataract_time_to_event_days'].fillna(df_q['followup_duration_cataract'])
    if 'event' not in df_q.columns:
        df_q['event'] = (df_q['cataract_time_to_event_days'].notna()) & (df_q['cataract_time_to_event_days'] >= min_days)
        df_q['event'] = df_q['event'].astype(int)
    # ensure event column exists for quartile analysis
    if 'event' not in df_q.columns:
        df_q['event'] = (df_q['cataract_time_to_event_days'].notna()) & (df_q['cataract_time_to_event_days'] >= min_days)
        df_q['event'] = df_q['event'].astype(int)
    df_q['quartile'] = pd.qcut(df_q[exposure].rank(method='first'), q=4, labels=['Q1','Q2','Q3','Q4'], duplicates='drop')
    q_results = []
    for q in ['Q2','Q3','Q4']:
        df_q[f'q_{q}'] = (df_q['quartile'] == q).astype(int)
        exog = df_q[[ 'q_'+q] + covars].astype(float).dropna()
        if exog.shape[0] == 0:
            continue
        idx = exog.index
        t = df_q.loc[idx, 'time'].values
        s = df_q.loc[idx, 'event'].values
        model = PHReg(t, exog, status=s)
        res = model.fit()
        beta, se, p = res.params[0], res.bse[0], res.pvalues[0]
        HR = np.exp(beta); CI_l, CI_u = np.exp(beta - 1.96 * se), np.exp(beta + 1.96 * se)
        q_results.append({'quartile': q, 'HR': HR, 'CI': f"{CI_l:.2f}-{CI_u:.2f}", 'P': p})
    # 趋势检验（连续暴露）
    exog_all = df_q[[exposure] + covars].astype(float).dropna()
    idx = exog_all.index
    t_trend = df_q.loc[idx, 'time'].values
    s_trend = df_q.loc[idx, 'event'].values
    try:
        model_trend = PHReg(t_trend, exog_all, status=s_trend)
        res_trend = model_trend.fit()
        trend_p = float(res_trend.pvalues[0])
    except Exception:
        trend_p = np.nan
    return q_results, trend_p

def main():
    df = load_and_prepare()
    exposures = ['n3fa_grs','n6fa_grs','pufa_grs','tfa_grs',
                 'n3grs_layer4','n3grs_layer5','n3grs_layer2','n3grs_layer1_23_4','n3grs_1_234']
    print('Running extended GRS Cox analyses without PS(M) across exposures and thresholds...')
    res = run_all(df, exposures, mins=[365,730,1095])
    print(res.head())
    res.to_csv('grs_ext_results.csv', index=False)
    print('Results saved to grs_ext_results.csv')
    # 额外分析：分层、非线性与交互等（示例性触发点）
    for exposure in exposures:
        if exposure not in df.columns:
            continue
        covars = resolve_covariates(df)
        strat = stratified_by_sex(df, exposure, covars, min_days=730)
        if strat is not None and not strat.empty:
            strat.to_csv(f'grs_ext_{exposure}_stratified_sex.csv', index=False)
            print(f'Results saved to grs_ext_{exposure}_stratified_sex.csv')
        qres, trend_p = quartile_analysis_exposure(df, exposure, covars, min_days=730)
        if qres:
            import json
            out = {'Exposure': exposure, 'Quartiles': qres, 'TrendP': trend_p}
            with open(f'grs_ext_{exposure}_quartile.json','w') as jf:
                json.dump(out, jf, indent=2, ensure_ascii=False)
            print(f'Quartile analysis for {exposure} saved to grs_ext_{exposure}_quartile.json')

if __name__ == '__main__':
    main()
