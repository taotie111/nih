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
import json
warnings.filterwarnings('ignore')

def load_and_prepare(df_path='WY_计算随访时间_cataract_更新的截止时间.csv'):
    df = pd.read_csv(df_path)
    rename_map = {
        'age_bl':'age_baseline', 'sex':'sex',
        'csmoking_bl':'smoking_status_baseline',
        'alcohol_freq_bl':'alcohol_frequency_baseline', 'bmi_bl':'bmi_baseline',
        'diabetes_bl':'diabetes_baseline', 'hypertension_bl':'hypertension_baseline',
        'heart_disease':'heart_disease_composite', 'n3fa':'fatty_acids_n3',
        'n6fa':'fatty_acids_n6', 'pufa':'fatty_acids_pufa', 'tfa':'fatty_acids_tfa',
        'followup_cataract':'followup_duration_cataract',
        'cataract_days':'cataract_time_to_event_days', 'f_eid':'participant_id',
        'glaucoma_bl':'glaucoma_baseline', 'amd_bl':'amd_baseline'
    }
    df = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns})
    if 'glaucoma_baseline' in df.columns:
        df = df[df['glaucoma_baseline'] == 0]
    df = df[df['followup_duration_cataract'].notna()]
    return df

def resolve_covariates(df, prefer=None):
    if prefer is None:
        prefer = ['age_baseline','sex','bmi_baseline','smoking_status_baseline','alcohol_frequency_baseline','hypertension_baseline']
    covs = [c for c in prefer if c in df.columns]
    if len(covs) == 0:
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
    out = []
    for sex_val, label in [(0, 'Female'), (1, 'Male')]:
        sub = df[(df.get('sex') == sex_val) & (df[exposure].notna())]
        if sub.empty:
            continue
        r = run_cox_for_exposure(sub, exposure, covars, min_days)
        if r is not None:
            r['Group'] = label
            r['StratType'] = 'Sex'
            out.append(r)
    if not out:
        return None
    return pd.DataFrame(out)

def stratified_by_age(df, exposure, covars, min_days=730):
    out = []
    age_bins = [(0, 55, '<55'), (55, 65, '55-64'), (65, 100, '>=65')]
    for low, high, label in age_bins:
        age_col = 'age_baseline' if 'age_baseline' in df.columns else 'age_bl'
        sub = df[(df.get(age_col, 0) >= low) & (df.get(age_col, 0) < high) & (df[exposure].notna())]
        if sub.empty:
            continue
        r = run_cox_for_exposure(sub, exposure, covars, min_days)
        if r is not None:
            r['Group'] = label
            r['StratType'] = 'Age'
            out.append(r)
    if not out:
        return None
    return pd.DataFrame(out)

def stratified_by_bmi(df, exposure, covars, min_days=730):
    out = []
    bmi_bins = [(0, 25, '<25'), (25, 30, '25-30'), (30, 100, '>=30')]
    for low, high, label in bmi_bins:
        bmi_col = 'bmi_baseline' if 'bmi_baseline' in df.columns else 'bmi_bl'
        sub = df[(df.get(bmi_col, 0) >= low) & (df.get(bmi_col, 0) < high) & (df[exposure].notna())]
        if sub.empty:
            continue
        r = run_cox_for_exposure(sub, exposure, covars, min_days)
        if r is not None:
            r['Group'] = label
            r['StratType'] = 'BMI'
            out.append(r)
    if not out:
        return None
    return pd.DataFrame(out)

def quartile_analysis_exposure(df, exposure, covars, min_days=730):
    df_q = df.copy()
    if 'time' not in df_q.columns:
        df_q['time'] = df_q['cataract_time_to_event_days'].fillna(df_q['followup_duration_cataract'])
    if 'event' not in df_q.columns:
        df_q['event'] = (df_q['cataract_time_to_event_days'].notna()) & (df_q['cataract_time_to_event_days'] >= min_days)
        df_q['event'] = df_q['event'].astype(int)
    df_q['quartile'] = pd.qcut(df_q[exposure].rank(method='first'), q=4, labels=['Q1','Q2','Q3','Q4'], duplicates='drop')
    q_results = []
    for q in ['Q2','Q3','Q4']:
        df_q[f'q_{q}'] = (df_q['quartile'] == q).astype(int)
        exog = df_q[['q_'+q] + covars].astype(float).dropna()
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

def interaction_analysis(df, exposure, covars, min_days=730):
    out = []
    df_i = df.dropna(subset=[exposure]).copy()
    df_i['time'] = df_i['cataract_time_to_event_days'].fillna(df_i['followup_duration_cataract'])
    df_i['event'] = ((df_i['cataract_time_to_event_days'].notna()) & (df_i['cataract_time_to_event_days'] >= min_days)).astype(int)
    
    if 'sex' in df_i.columns:
        df_i['int_sex'] = df_i[exposure] * df_i['sex']
        try:
            exog_vars = [exposure, 'sex', 'int_sex'] + covars
            exog = df_i[exog_vars].astype(float).dropna()
            idx = exog.index
            t = df_i.loc[idx, 'time'].values
            s = df_i.loc[idx, 'event'].values
            model = PHReg(t, exog, status=s)
            result = model.fit()
            int_idx = list(exog.columns).index('int_sex')
            beta, se, p = result.params[int_idx], result.bse[int_idx], result.pvalues[int_idx]
            HR = np.exp(beta)
            CI_l, CI_u = np.exp(beta - 1.96 * se), np.exp(beta + 1.96 * se)
            out.append({'Interaction': f'{exposure}*sex', 'HR': HR, 'CI': f'{CI_l:.2f}-{CI_u:.2f}', 'P': p})
        except Exception as e:
            pass
    age_col = 'age_baseline' if 'age_baseline' in df_i.columns else 'age_bl'
    if age_col in df_i.columns:
        df_i['age_group'] = pd.cut(df_i[age_col], bins=[0, 55, 65, 100], labels=['<55', '55-64', '>=65'])
        for ag in ['<55', '55-64', '>=65']:
            df_i[f'int_age_{ag}'] = (df_i['age_group'] == ag).astype(int) * df_i[exposure]
            try:
                exog_vars = [exposure, f'int_age_{ag}'] + covars
                exog = df_i[exog_vars].astype(float).dropna()
                idx = exog.index
                t = df_i.loc[idx, 'time'].values
                s = df_i.loc[idx, 'event'].values
                model = PHReg(t, exog, status=s)
                result = model.fit()
                int_idx = list(exog.columns).index(f'int_age_{ag}')
                beta, se, p = result.params[int_idx], result.bse[int_idx], result.pvalues[int_idx]
                HR = np.exp(beta)
                CI_l, CI_u = np.exp(beta - 1.96 * se), np.exp(beta + 1.96 * se)
                out.append({'Interaction': f'{exposure}*age_{ag}', 'HR': HR, 'CI': f'{CI_l:.2f}-{CI_u:.2f}', 'P': p})
            except Exception as e:
                pass
    bmi_col = 'bmi_baseline' if 'bmi_baseline' in df_i.columns else 'bmi_bl'
    if bmi_col in df_i.columns:
        df_i['bmi_group'] = pd.cut(df_i[bmi_col], bins=[0, 25, 30, 100], labels=['<25', '25-30', '>=30'])
        for bg in ['<25', '25-30', '>=30']:
            df_i[f'int_bmi_{bg}'] = (df_i['bmi_group'] == bg).astype(int) * df_i[exposure]
            try:
                exog_vars = [exposure, f'int_bmi_{bg}'] + covars
                exog = df_i[exog_vars].astype(float).dropna()
                idx = exog.index
                t = df_i.loc[idx, 'time'].values
                s = df_i.loc[idx, 'event'].values
                model = PHReg(t, exog, status=s)
                result = model.fit()
                int_idx = list(exog.columns).index(f'int_bmi_{bg}')
                beta, se, p = result.params[int_idx], result.bse[int_idx], result.pvalues[int_idx]
                HR = np.exp(beta)
                CI_l, CI_u = np.exp(beta - 1.96 * se), np.exp(beta + 1.96 * se)
                out.append({'Interaction': f'{exposure}*bmi_{bg}', 'HR': HR, 'CI': f'{CI_l:.2f}-{CI_u:.2f}', 'P': p})
            except Exception as e:
                pass
    if not out:
        return None
    return pd.DataFrame(out)

def main():
    df = load_and_prepare()
    exposures = ['n3fa_grs','n6fa_grs','pufa_grs','tfa_grs','n3grs_layer4','n3grs_layer5','n3grs_layer2','n3grs_layer1_23_4','n3grs_1_234']
    covars = resolve_covariates(df)
    print('=== C1: Robustness Analysis - All GRS (Stratified + Quartile + Interaction) ===')
    all_stratified = []
    all_quartile = []
    all_interactions = []
    for exposure in exposures:
        if exposure not in df.columns:
            continue
        print(f'Processing: {exposure}')
        # Sex stratification
        strat_sex = stratified_by_sex(df, exposure, covars, min_days=730)
        if strat_sex is not None:
            all_stratified.append(strat_sex)
        # Age stratification
        strat_age = stratified_by_age(df, exposure, covars, min_days=730)
        if strat_age is not None:
            all_stratified.append(strat_age)
        # BMI stratification
        strat_bmi = stratified_by_bmi(df, exposure, covars, min_days=730)
        if strat_bmi is not None:
            all_stratified.append(strat_bmi)
        # Quartile analysis
        qres, trend_p = quartile_analysis_exposure(df, exposure, covars, min_days=730)
        if qres:
            all_quartile.append({'Exposure': exposure, 'Quartiles': qres, 'TrendP': trend_p})
        # Interaction analysis
        intres = interaction_analysis(df, exposure, covars, min_days=730)
        if intres is not None:
            intres['Exposure'] = exposure
            all_interactions.append(intres)
    if all_stratified:
        pd.concat(all_stratified, ignore_index=True).to_csv('grs_ext_all_stratified_all.csv', index=False)
        print('Saved:grs_ext_all_stratified_all.csv')
    if all_quartile:
        with open('grs_ext_all_quartile.json','w') as f:
            json.dump(all_quartile, f, indent=2)
        print('Saved:grs_ext_all_quartile.json')
    if all_interactions:
        pd.concat(all_interactions, ignore_index=True).to_csv('grs_ext_all_interactions.csv', index=False)
        print('Saved:grs_ext_all_interactions.csv')
    print('=== C1 Complete ===')
    
    print('=== D1: Sensitivity Analysis ===')
    sens_results = []
    for exclude_name, exclude_col in [('no_amd', 'amd_baseline'), ('no_diabetes', 'diabetes_baseline'), ('no_dr', 'diabetic_eye_baseline')]:
        if exclude_col in df.columns:
            df_sens = df[df[exclude_col] == 0]
            for exposure in ['n3fa_grs', 'n6fa_grs', 'pufa_grs']:
                if exposure not in df_sens.columns:
                    continue
                r = run_cox_for_exposure(df_sens, exposure, covars, min_days=730)
                if r:
                    r['Exclusion'] = exclude_name
                    sens_results.append(r)
    if sens_results:
        pd.DataFrame(sens_results).to_csv('grs_ext_results_sensitivity.csv', index=False)
        print('Saved:grs_ext_results_sensitivity.csv')
    print('=== D1 Complete ===')
    
    print('=== Main Results ===')
    res = run_all(df, exposures, mins=[365,730,1095])
    print(res.head())
    res.to_csv('grs_ext_results.csv', index=False)
    print('Results saved togrs_ext_results.csv')

if __name__ == '__main__':
    main()
