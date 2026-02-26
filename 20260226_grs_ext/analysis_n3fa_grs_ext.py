#!/usr/bin/env python3
"""
Extended analysis for n3fa_grs and other GRS exposures on cataract outcome.
No PS(M). Cox PH model across thresholds.
Supports: 365, 730, 1095 day thresholds; multiple GRS exposures (n3fa_grs, n6fa_grs, pufa_grs, tfa_grs),
 layer-based exposures (n3grs_layer4, n3grs_layer5, n3grs_layer2, n3grs_layer1_23_4, n3grs_1_234, etc).
Resolve covariates from dataset dynamically among: age_baseline, sex, bmi_baseline, smoking_status_baseline,
alcohol_frequency_baseline, hypertension_baseline.

SUPPORTS AGE AS TIMESCALE - use_age_timescale=True in Cox models
"""
import pandas as pd
import numpy as np
from statsmodels.duration.hazard_regression import PHReg
import warnings
import json
warnings.filterwarnings('ignore')

def load_and_prepare(df_path='../WY_计算随访时间_cataract_更新的截止时间.csv'):
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

def prepare_age_timescale(df):
    df_ts = df.copy()
    age_col = 'age_baseline' if 'age_baseline' in df_ts.columns else 'age_bl'
    time_col = 'cataract_time_to_event_days' if 'cataract_time_to_event_days' in df_ts.columns else 'cataract_days'
    follow_col = 'followup_duration_cataract' if 'followup_duration_cataract' in df_ts.columns else 'followup_cataract'
    
    df_ts['age_entry'] = df_ts[age_col]
    df_ts['age_exit'] = df_ts.apply(
        lambda x: x[age_col] + x[time_col]/365.25 if pd.notna(x[time_col]) 
        else x[age_col] + x[follow_col]/365.25, axis=1
    )
    return df_ts

def run_cox_for_exposure(df, exposure, covars, min_days, use_age_timescale=False):
    if exposure not in df.columns:
        return None
    df2 = df.dropna(subset=[exposure])
    df2 = df2.copy()
    
    time_col = 'cataract_time_to_event_days' if 'cataract_time_to_event_days' in df2.columns else 'cataract_days'
    follow_col = 'followup_duration_cataract' if 'followup_duration_cataract' in df2.columns else 'followup_cataract'
    
    df2['event'] = (df2[time_col].notna()) & (df2[time_col] >= min_days)
    df2['event'] = df2['event'].astype(int)
    
    cols = [exposure] + covars
    exog = df2[cols].astype(float).dropna()
    if exog.shape[0] == 0:
        return None
    idx = exog.index
    s = df2.loc[idx, 'event'].values
    
    if use_age_timescale:
        age_col = 'age_baseline' if 'age_baseline' in df2.columns else 'age_bl'
        df2['age_entry'] = df2[age_col]
        df2['age_exit'] = df2.apply(
            lambda x: x[age_col] + x[time_col]/365.25 if pd.notna(x[time_col]) 
            else x[age_col] + x[follow_col]/365.25, axis=1
        )
        t = df2.loc[idx, 'age_exit'].values
        entry = df2.loc[idx, 'age_entry'].values
        model = PHReg(t, exog, status=s, entry=entry)
    else:
        df2['time'] = df2[time_col].fillna(df2[follow_col])
        t = df2.loc[idx, 'time'].values
        model = PHReg(t, exog, status=s)
    
    try:
        res = model.fit()
        beta, se, p = res.params[0], res.bse[0], res.pvalues[0]
        HR = np.exp(beta)
        CI_l, CI_u = np.exp(beta - 1.96 * se), np.exp(beta + 1.96 * se)
        timescale = 'age' if use_age_timescale else 'followup'
        return {'Exposure': exposure, 'MinDays': min_days, 'N': len(exog),
                'HR': HR, 'CI_low': CI_l, 'CI_high': CI_u, 'P': p, 'Timescale': timescale}
    except Exception:
        timescale = 'age' if use_age_timescale else 'followup'
        return {'Exposure': exposure, 'MinDays': min_days, 'N': len(exog),
                'HR': np.nan, 'CI_low': np.nan, 'CI_high': np.nan, 'P': np.nan, 'Timescale': timescale}

def run_all(df, exposures, mins=[365,730,1095], use_age_timescale=False):
    covars = resolve_covariates(df)
    if len(covars) == 0:
        raise RuntimeError('No covariates found to use in Cox model')
    results = []
    for exposure in exposures:
        for m in mins:
            r = run_cox_for_exposure(df, exposure, covars, m, use_age_timescale)
            if r:
                results.append(r)
    return pd.DataFrame(results)

def stratified_by_sex(df, exposure, covars, min_days=730, use_age_timescale=False):
    out = []
    for sex_val, label in [(0, 'Female'), (1, 'Male')]:
        sub = df[(df.get('sex') == sex_val) & (df[exposure].notna())]
        if sub.empty:
            continue
        r = run_cox_for_exposure(sub, exposure, covars, min_days, use_age_timescale)
        if r is not None:
            r['Group'] = label
            r['StratType'] = 'Sex'
            out.append(r)
    if not out:
        return None
    return pd.DataFrame(out)

def stratified_by_age(df, exposure, covars, min_days=730, use_age_timescale=False):
    out = []
    age_bins = [(0, 55, '<55'), (55, 65, '55-64'), (65, 100, '>=65')]
    for low, high, label in age_bins:
        age_col = 'age_baseline' if 'age_baseline' in df.columns else 'age_bl'
        sub = df[(df.get(age_col, 0) >= low) & (df.get(age_col, 0) < high) & (df[exposure].notna())]
        if sub.empty:
            continue
        r = run_cox_for_exposure(sub, exposure, covars, min_days, use_age_timescale)
        if r is not None:
            r['Group'] = label
            r['StratType'] = 'Age'
            out.append(r)
    if not out:
        return None
    return pd.DataFrame(out)

def stratified_by_bmi(df, exposure, covars, min_days=730, use_age_timescale=False):
    out = []
    bmi_bins = [(0, 25, '<25'), (25, 30, '25-30'), (30, 100, '>=30')]
    for low, high, label in bmi_bins:
        bmi_col = 'bmi_baseline' if 'bmi_baseline' in df.columns else 'bmi_bl'
        sub = df[(df.get(bmi_col, 0) >= low) & (df.get(bmi_col, 0) < high) & (df[exposure].notna())]
        if sub.empty:
            continue
        r = run_cox_for_exposure(sub, exposure, covars, min_days, use_age_timescale)
        if r is not None:
            r['Group'] = label
            r['StratType'] = 'BMI'
            out.append(r)
    if not out:
        return None
    return pd.DataFrame(out)

def quartile_analysis_exposure(df, exposure, covars, min_days=730, use_age_timescale=False):
    df_q = df.copy()
    time_col = 'cataract_time_to_event_days' if 'cataract_time_to_event_days' in df_q.columns else 'cataract_days'
    follow_col = 'followup_duration_cataract' if 'followup_duration_cataract' in df_q.columns else 'followup_cataract'
    
    if use_age_timescale:
        age_col = 'age_baseline' if 'age_baseline' in df_q.columns else 'age_bl'
        df_q['age_entry'] = df_q[age_col]
        df_q['age_exit'] = df_q.apply(
            lambda x: x[age_col] + x[time_col]/365.25 if pd.notna(x[time_col]) 
            else x[age_col] + x[follow_col]/365.25, axis=1
        )
    else:
        df_q['time'] = df_q[time_col].fillna(df_q[follow_col])
    
    df_q['event'] = (df_q[time_col].notna()) & (df_q[time_col] >= min_days)
    df_q['event'] = df_q['event'].astype(int)
    
    df_q['quartile'] = pd.qcut(df_q[exposure].rank(method='first'), q=4, labels=['Q1','Q2','Q3','Q4'], duplicates='drop')
    q_results = []
    for q in ['Q2','Q3','Q4']:
        df_q[f'q_{q}'] = (df_q['quartile'] == q).astype(int)
        exog = df_q[['q_'+q] + covars].astype(float).dropna()
        if exog.shape[0] == 0:
            continue
        idx = exog.index
        if use_age_timescale:
            t = df_q.loc[idx, 'age_exit'].values
            entry = df_q.loc[idx, 'age_entry'].values
            model = PHReg(t, exog, status=df_q.loc[idx, 'event'].values, entry=entry)
        else:
            t = df_q.loc[idx, 'time'].values
            model = PHReg(t, exog, status=df_q.loc[idx, 'event'].values)
        res = model.fit()
        beta, se, p = res.params[0], res.bse[0], res.pvalues[0]
        HR = np.exp(beta); CI_l, CI_u = np.exp(beta - 1.96 * se), np.exp(beta + 1.96 * se)
        q_results.append({'quartile': q, 'HR': HR, 'CI': f"{CI_l:.2f}-{CI_u:.2f}", 'P': p})
    
    exog_all = df_q[[exposure] + covars].astype(float).dropna()
    idx = exog_all.index
    try:
        if use_age_timescale:
            t_trend = df_q.loc[idx, 'age_exit'].values
            entry_trend = df_q.loc[idx, 'age_entry'].values
            model_trend = PHReg(t_trend, exog_all, status=df_q.loc[idx, 'event'].values, entry=entry_trend)
        else:
            t_trend = df_q.loc[idx, 'time'].values
            model_trend = PHReg(t_trend, exog_all, status=df_q.loc[idx, 'event'].values)
        res_trend = model_trend.fit()
        trend_p = float(res_trend.pvalues[0])
    except Exception:
        trend_p = np.nan
    return q_results, trend_p

def interaction_analysis(df, exposure, covars, min_days=730, use_age_timescale=False):
    out = []
    df_i = df.dropna(subset=[exposure]).copy()
    time_col = 'cataract_time_to_event_days' if 'cataract_time_to_event_days' in df_i.columns else 'cataract_days'
    follow_col = 'followup_duration_cataract' if 'followup_duration_cataract' in df_i.columns else 'followup_cataract'
    
    if use_age_timescale:
        age_col = 'age_baseline' if 'age_baseline' in df_i.columns else 'age_bl'
        df_i['age_entry'] = df_i[age_col]
        df_i['age_exit'] = df_i.apply(
            lambda x: x[age_col] + x[time_col]/365.25 if pd.notna(x[time_col]) 
            else x[age_col] + x[follow_col]/365.25, axis=1
        )
    else:
        df_i['time'] = df_i[time_col].fillna(df_i[follow_col])
    
    df_i['event'] = ((df_i[time_col].notna()) & (df_i[time_col] >= min_days)).astype(int)
    
    if 'sex' in df_i.columns:
        df_i['int_sex'] = df_i[exposure] * df_i['sex']
        try:
            exog_vars = [exposure, 'sex', 'int_sex'] + covars
            exog = df_i[exog_vars].astype(float).dropna()
            idx = exog.index
            if use_age_timescale:
                t = df_i.loc[idx, 'age_exit'].values
                entry = df_i.loc[idx, 'age_entry'].values
                model = PHReg(t, exog, status=df_i.loc[idx, 'event'].values, entry=entry)
            else:
                t = df_i.loc[idx, 'time'].values
                model = PHReg(t, exog, status=df_i.loc[idx, 'event'].values)
            result = model.fit()
            int_idx = list(exog.columns).index('int_sex')
            beta, se, p = result.params[int_idx], result.bse[int_idx], result.pvalues[int_idx]
            HR = np.exp(beta)
            CI_l, CI_u = np.exp(beta - 1.96 * se), np.exp(beta + 1.96 * se)
            out.append({'Interaction': f'{exposure}*sex', 'HR': HR, 'CI': f"{CI_l:.2f}-{CI_u:.2f}", 'P': p})
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
                if use_age_timescale:
                    t = df_i.loc[idx, 'age_exit'].values
                    entry = df_i.loc[idx, 'age_entry'].values
                    model = PHReg(t, exog, status=df_i.loc[idx, 'event'].values, entry=entry)
                else:
                    t = df_i.loc[idx, 'time'].values
                    model = PHReg(t, exog, status=df_i.loc[idx, 'event'].values)
                result = model.fit()
                int_idx = list(exog.columns).index(f'int_age_{ag}')
                beta, se, p = result.params[int_idx], result.bse[int_idx], result.pvalues[int_idx]
                HR = np.exp(beta)
                CI_l, CI_u = np.exp(beta - 1.96 * se), np.exp(beta + 1.96 * se)
                out.append({'Interaction': f'{exposure}*age_{ag}', 'HR': HR, 'CI': f"{CI_l:.2f}-{CI_u:.2f}", 'P': p})
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
                if use_age_timescale:
                    t = df_i.loc[idx, 'age_exit'].values
                    entry = df_i.loc[idx, 'age_entry'].values
                    model = PHReg(t, exog, status=df_i.loc[idx, 'event'].values, entry=entry)
                else:
                    t = df_i.loc[idx, 'time'].values
                    model = PHReg(t, exog, status=df_i.loc[idx, 'event'].values)
                result = model.fit()
                int_idx = list(exog.columns).index(f'int_bmi_{bg}')
                beta, se, p = result.params[int_idx], result.bse[int_idx], result.pvalues[int_idx]
                HR = np.exp(beta)
                CI_l, CI_u = np.exp(beta - 1.96 * se), np.exp(beta + 1.96 * se)
                out.append({'Interaction': f'{exposure}*bmi_{bg}', 'HR': HR, 'CI': f"{CI_l:.2f}-{CI_u:.2f}", 'P': p})
            except Exception as e:
                pass
    
    if not out:
        return None
    return pd.DataFrame(out)

def main(use_age_timescale=False):
    df = load_and_prepare()
    exposures = ['n3fa_grs','n6fa_grs','pufa_grs','tfa_grs','n3grs_layer4','n3grs_layer5','n3grs_layer2','n3grs_layer1_23_4','n3grs_1_234']
    covars = resolve_covariates(df)
    
    timescale_str = 'AGE' if use_age_timescale else 'FOLLOWUP'
    print(f'=== C1: Robustness Analysis - All GRS ({timescale_str} Timescale) ===')
    
    all_stratified = []
    all_quartile = []
    all_interactions = []
    for exposure in exposures:
        if exposure not in df.columns:
            continue
        print(f'Processing: {exposure}')
        # Sex stratification
        strat_sex = stratified_by_sex(df, exposure, covars, min_days=730, use_age_timescale=use_age_timescale)
        if strat_sex is not None:
            all_stratified.append(strat_sex)
        # Age stratification
        strat_age = stratified_by_age(df, exposure, covars, min_days=730, use_age_timescale=use_age_timescale)
        if strat_age is not None:
            all_stratified.append(strat_age)
        # BMI stratification
        strat_bmi = stratified_by_bmi(df, exposure, covars, min_days=730, use_age_timescale=use_age_timescale)
        if strat_bmi is not None:
            all_stratified.append(strat_bmi)
        # Quartile analysis
        qres, trend_p = quartile_analysis_exposure(df, exposure, covars, min_days=730, use_age_timescale=use_age_timescale)
        if qres:
            all_quartile.append({'Exposure': exposure, 'Quartiles': qres, 'TrendP': trend_p})
        # Interaction analysis
        intres = interaction_analysis(df, exposure, covars, min_days=730, use_age_timescale=use_age_timescale)
        if intres is not None:
            intres['Exposure'] = exposure
            all_interactions.append(intres)
    
    if all_stratified:
        pd.concat(all_stratified, ignore_index=True).to_csv(f'grs_ext_all_stratified_{timescale_str.lower()}.csv', index=False)
        print(f'Saved:grs_ext_all_stratified_{timescale_str.lower()}.csv')
    if all_quartile:
        with open(f'grs_ext_all_quartile_{timescale_str.lower()}.json','w') as f:
            json.dump(all_quartile, f, indent=2)
        print(f'Saved:grs_ext_all_quartile_{timescale_str.lower()}.json')
    if all_interactions:
        pd.concat(all_interactions, ignore_index=True).to_csv(f'grs_ext_all_interactions_{timescale_str.lower()}.csv', index=False)
        print(f'Saved:grs_ext_all_interactions_{timescale_str.lower()}.csv')
    print('=== C1 Complete ===')
    
    print('=== D1: Sensitivity Analysis ===')
    sens_results = []
    for exclude_name, exclude_col in [('no_amd', 'amd_baseline'), ('no_diabetes', 'diabetes_baseline'), ('no_dr', 'diabetic_eye_baseline')]:
        if exclude_col in df.columns:
            df_sens = df[df[exclude_col] == 0]
            for exposure in ['n3fa_grs', 'n6fa_grs', 'pufa_grs']:
                if exposure not in df_sens.columns:
                    continue
                r = run_cox_for_exposure(df_sens, exposure, covars, min_days=730, use_age_timescale=use_age_timescale)
                if r:
                    r['Exclusion'] = exclude_name
                    sens_results.append(r)
    if sens_results:
        pd.DataFrame(sens_results).to_csv(f'grs_ext_results_sensitivity_{timescale_str.lower()}.csv', index=False)
        print(f'Saved:grs_ext_results_sensitivity_{timescale_str.lower()}.csv')
    print('=== D1 Complete ===')
    
    print('=== Main Results ===')
    res = run_all(df, exposures, mins=[365,730,1095], use_age_timescale=use_age_timescale)
    print(res.head())
    res.to_csv(f'grs_ext_results_{timescale_str.lower()}.csv', index=False)
    print(f'Results saved togrs_ext_results_{timescale_str.lower()}.csv')

if __name__ == '__main__':
    # Set use_age_timescale=True for age as timescale (RECOMMENDED for age-dependent diseases like cataract)
    # Set use_age_timescale=False to use follow-up time as timescale
    main(use_age_timescale=True)
