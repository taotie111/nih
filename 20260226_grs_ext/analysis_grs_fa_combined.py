#!/usr/bin/env python3
"""
Combined analysis: GRS + Fatty Acid Levels on cataract outcome.
Tests whether GRS effect is independent of measured FA levels.
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
        prefer = ['age_baseline','sex','bmi_baseline','smoking_status_baseline',
                  'alcohol_frequency_baseline','hypertension_baseline']
    covs = [c for c in prefer if c in df.columns]
    if len(covs) == 0:
        for c in ['age_bl','sex','bmi_bl','csmoking_bl','alcohol_bl','hypertension_bl']:
            if c in df.columns:
                covs.append(c)
    return covs

def run_cox_single_exposure(df, exposure, covars, min_days=730, use_age_timescale=False):
    if exposure not in df.columns:
        return None
    df2 = df.dropna(subset=[exposure]).copy()
    
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
        return {'Exposure': exposure, 'N': len(exog), 'Events': int(s.sum()),
                'HR': HR, 'CI_low': CI_l, 'CI_high': CI_u, 'P': p}
    except Exception as e:
        return {'Exposure': exposure, 'N': len(exog), 'Events': 0,
                'HR': np.nan, 'CI_low': np.nan, 'CI_high': np.nan, 'P': np.nan}

def run_cox_combined(df, exposure1, exposure2, covars, min_days=730, use_age_timescale=False):
    if exposure1 not in df.columns or exposure2 not in df.columns:
        return None
    df2 = df.dropna(subset=[exposure1, exposure2]).copy()
    
    time_col = 'cataract_time_to_event_days' if 'cataract_time_to_event_days' in df2.columns else 'cataract_days'
    follow_col = 'followup_duration_cataract' if 'followup_duration_cataract' in df2.columns else 'followup_cataract'
    
    df2['event'] = (df2[time_col].notna()) & (df2[time_col] >= min_days)
    df2['event'] = df2['event'].astype(int)
    
    cols = [exposure1, exposure2] + covars
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
        results = []
        for i, exp in enumerate([exposure1, exposure2]):
            beta, se, p = res.params[i], res.bse[i], res.pvalues[i]
            HR = np.exp(beta)
            CI_l, CI_u = np.exp(beta - 1.96 * se), np.exp(beta + 1.96 * se)
            results.append({
                'GRS': exposure1, 'FA': exposure2,
                'Variable': exp, 'N': len(exog), 'Events': int(s.sum()),
                'HR': HR, 'CI_low': CI_l, 'CI_high': CI_u, 'P': p
            })
        return results
    except Exception as e:
        return None

def stratified_by_fa(df, exposure1, exposure2, covars, min_days=730):
    if exposure1 not in df.columns or exposure2 not in df.columns:
        return None
    
    df2 = df.dropna(subset=[exposure1, exposure2]).copy()
    cutoff = df2[exposure2].median()
    
    results = []
    for label, condition in [('Low', df2[exposure2] <= cutoff), 
                               ('High', df2[exposure2] > cutoff)]:
        sub = df2[condition]
        if len(sub) < 50:
            continue
        r = run_cox_single_exposure(sub, exposure1, covars, min_days)
        if r:
            r['FA_Stratum'] = label
            r['FA_Median'] = cutoff
            results.append(r)
    
    return pd.DataFrame(results) if results else None

def quartile_analysis(df, exposure, covars, min_days=730, use_age_timescale=False):
    """Quartile analysis (Q1-Q4) for any exposure"""
    if exposure not in df.columns:
        return None
    
    df2 = df.dropna(subset=[exposure]).copy()
    
    time_col = 'cataract_time_to_event_days' if 'cataract_time_to_event_days' in df2.columns else 'cataract_days'
    follow_col = 'followup_duration_cataract' if 'followup_duration_cataract' in df2.columns else 'followup_cataract'
    
    df2['event'] = (df2[time_col].notna()) & (df2[time_col] >= min_days)
    df2['event'] = df2['event'].astype(int)
    
    df2['quartile'] = pd.qcut(df2[exposure].rank(method='first'), q=4, labels=['Q1','Q2','Q3','Q4'], duplicates='drop')
    
    results = []
    for q in ['Q2','Q3','Q4']:
        df_q = df2.copy()
        df_q[f'is_{q}'] = (df_q['quartile'] == q).astype(int)
        
        exog = df_q[[f'is_{q}'] + covars].astype(float).dropna()
        if exog.shape[0] == 0:
            continue
        idx = exog.index
        
        if use_age_timescale:
            age_col = 'age_baseline' if 'age_baseline' in df_q.columns else 'age_bl'
            df_q['age_entry'] = df_q[age_col]
            df_q['age_exit'] = df_q.apply(
                lambda x: x[age_col] + x[time_col]/365.25 if pd.notna(x[time_col]) 
                else x[age_col] + x[follow_col]/365.25, axis=1
            )
            t = df_q.loc[idx, 'age_exit'].values
            entry = df_q.loc[idx, 'age_entry'].values
            model = PHReg(t, exog, status=df_q.loc[idx, 'event'].values, entry=entry)
        else:
            df_q['time'] = df_q[time_col].fillna(df_q[follow_col])
            t = df_q.loc[idx, 'time'].values
            model = PHReg(t, exog, status=df_q.loc[idx, 'event'].values)
        
        try:
            res = model.fit()
            beta, se, p = res.params[0], res.bse[0], res.pvalues[0]
            HR = np.exp(beta)
            CI_l, CI_u = np.exp(beta - 1.96 * se), np.exp(beta + 1.96 * se)
            results.append({
                'Exposure': exposure, 'Comparison': f'{q} vs Q1',
                'HR': HR, 'CI_low': CI_l, 'CI_high': CI_u, 'P': p
            })
        except:
            pass
    
    exog_trend = df2[[exposure] + covars].astype(float).dropna()
    if exog_trend.shape[0] > 0:
        idx = exog_trend.index
        if use_age_timescale:
            age_col = 'age_baseline' if 'age_baseline' in df2.columns else 'age_bl'
            df2['age_entry'] = df2[age_col]
            df2['age_exit'] = df2.apply(
                lambda x: x[age_col] + x[time_col]/365.25 if pd.notna(x[time_col]) 
                else x[age_col] + x[follow_col]/365.25, axis=1
            )
            t_trend = df2.loc[idx, 'age_exit'].values
            entry_trend = df2.loc[idx, 'age_entry'].values
            model_trend = PHReg(t_trend, exog_trend, status=df2.loc[idx, 'event'].values, entry=entry_trend)
        else:
            df2['time'] = df2[time_col].fillna(df2[follow_col])
            t_trend = df2.loc[idx, 'time'].values
            model_trend = PHReg(t_trend, exog_trend, status=df2.loc[idx, 'event'].values)
        
        try:
            res_trend = model_trend.fit()
            trend_p = float(res_trend.pvalues[0])
            results.append({
                'Exposure': exposure, 'Comparison': 'P for trend',
                'HR': np.nan, 'CI_low': np.nan, 'CI_high': np.nan, 'P': trend_p
            })
        except:
            pass
    
    return pd.DataFrame(results) if results else None

def joint_effect_analysis(df, exposure1, exposure2, covars, min_days=730, use_age_timescale=False):
    """Joint effect analysis: 2x2 (GRS high/low x FA high/low)"""
    if exposure1 not in df.columns or exposure2 not in df.columns:
        return None
    
    df2 = df.dropna(subset=[exposure1, exposure2]).copy()
    
    time_col = 'cataract_time_to_event_days' if 'cataract_time_to_event_days' in df2.columns else 'cataract_days'
    follow_col = 'followup_duration_cataract' if 'followup_duration_cataract' in df2.columns else 'followup_cataract'
    
    df2['event'] = (df2[time_col].notna()) & (df2[time_col] >= min_days)
    df2['event'] = df2['event'].astype(int)
    
    cutoff1 = df2[exposure1].median()
    cutoff2 = df2[exposure2].median()
    
    df2['grp1'] = (df2[exposure1] >= cutoff1).map({True: 'High', False: 'Low'})
    df2['grp2'] = (df2[exposure2] >= cutoff2).map({True: 'High', False: 'Low'})
    df2['joint_group'] = df2['grp1'] + '_' + df2['grp2']
    
    results = []
    reference = 'Low_Low'
    
    for group in ['Low_High', 'High_Low', 'High_High']:
        df_g = df2[df2['joint_group'] == group].copy()
        df_ref = df2[df2['joint_group'] == reference].copy()
        
        if len(df_g) < 10 or len(df_ref) < 10:
            continue
        
        df_g['is_group'] = 1
        df_ref['is_group'] = 0
        df_comb = pd.concat([df_g, df_ref])
        
        exog = df_comb[['is_group'] + covars].astype(float).dropna()
        if exog.shape[0] == 0:
            continue
        idx = exog.index
        
        if use_age_timescale:
            age_col = 'age_baseline' if 'age_baseline' in df_comb.columns else 'age_bl'
            df_comb['age_entry'] = df_comb[age_col]
            df_comb['age_exit'] = df_comb.apply(
                lambda x: x[age_col] + x[time_col]/365.25 if pd.notna(x[time_col]) 
                else x[age_col] + x[follow_col]/365.25, axis=1
            )
            t = df_comb.loc[idx, 'age_exit'].values
            entry = df_comb.loc[idx, 'age_entry'].values
            model = PHReg(t, exog, status=df_comb.loc[idx, 'event'].values, entry=entry)
        else:
            df_comb['time'] = df_comb[time_col].fillna(df_comb[follow_col])
            t = df_comb.loc[idx, 'time'].values
            model = PHReg(t, exog, status=df_comb.loc[idx, 'event'].values)
        
        try:
            res = model.fit()
            beta, se, p = res.params[0], res.bse[0], res.pvalues[0]
            HR = np.exp(beta)
            CI_l, CI_u = np.exp(beta - 1.96 * se), np.exp(beta + 1.96 * se)
            results.append({
                'GRS': exposure1, 'FA': exposure2,
                'Group': group, 'N': len(df_comb),
                'Events': int(df_comb['event'].sum()),
                'HR': HR, 'CI_low': CI_l, 'CI_high': CI_u, 'P': p
            })
        except:
            pass
    
    return pd.DataFrame(results) if results else None

def interaction_analysis(df, exposure1, exposure2, covars, min_days=730, use_age_timescale=False):
    """Interaction effect: test GRS x FA interaction"""
    if exposure1 not in df.columns or exposure2 not in df.columns:
        return None
    
    df2 = df.dropna(subset=[exposure1, exposure2]).copy()
    
    time_col = 'cataract_time_to_event_days' if 'cataract_time_to_event_days' in df2.columns else 'cataract_days'
    follow_col = 'followup_duration_cataract' if 'followup_duration_cataract' in df2.columns else 'followup_cataract'
    
    df2['event'] = (df2[time_col].notna()) & (df2[time_col] >= min_days)
    df2['event'] = df2['event'].astype(int)
    
    df2['interaction'] = df2[exposure1] * df2[exposure2]
    
    cols = [exposure1, exposure2, 'interaction'] + covars
    exog = df2[cols].astype(float).dropna()
    if exog.shape[0] == 0:
        return None
    idx = exog.index
    
    if use_age_timescale:
        age_col = 'age_baseline' if 'age_baseline' in df2.columns else 'age_bl'
        df2['age_entry'] = df2[age_col]
        df2['age_exit'] = df2.apply(
            lambda x: x[age_col] + x[time_col]/365.25 if pd.notna(x[time_col]) 
            else x[age_col] + x[follow_col]/365.25, axis=1
        )
        t = df2.loc[idx, 'age_exit'].values
        entry = df2.loc[idx, 'age_entry'].values
        model = PHReg(t, exog, status=df2.loc[idx, 'event'].values, entry=entry)
    else:
        df2['time'] = df2[time_col].fillna(df2[follow_col])
        t = df2.loc[idx, 'time'].values
        model = PHReg(t, exog, status=df2.loc[idx, 'event'].values)
    
    try:
        res = model.fit()
        results = []
        for i, var in enumerate([exposure1, exposure2, 'interaction']):
            beta, se, p = res.params[i], res.bse[i], res.pvalues[i]
            HR = np.exp(beta)
            CI_l, CI_u = np.exp(beta - 1.96 * se), np.exp(beta + 1.96 * se)
            results.append({
                'GRS': exposure1, 'FA': exposure2,
                'Variable': var, 'N': len(exog),
                'HR': HR, 'CI_low': CI_l, 'CI_high': CI_u, 'P': p
            })
        return results
    except:
        return None

def main(use_age_timescale=True):
    df = load_and_prepare()
    covars = resolve_covariates(df)
    
    fa_exposures = ['n3fa', 'n6fa', 'pufa', 'dha', 'la']
    pairs = [('n3fa_grs', 'n3fa'), ('n6fa_grs', 'n6fa'), ('pufa_grs', 'pufa')]
    
    print('='*60)
    print('ANALYSIS 1: FA Levels Only')
    print('='*60)
    
    fa_results = []
    for fa in fa_exposures:
        if fa in df.columns:
            r = run_cox_single_exposure(df, fa, covars, min_days=730, use_age_timescale=use_age_timescale)
            if r:
                fa_results.append(r)
    
    if fa_results:
        pd.DataFrame(fa_results).to_csv('grs_fa/fa_only_results.csv', index=False)
        print(pd.DataFrame(fa_results).to_string())
    
    print('\n' + '='*60)
    print('ANALYSIS 2: Combined GRS + FA')
    print('='*60)
    
    combined_results = []
    for grsf, fa in pairs:
        r = run_cox_combined(df, grsf, fa, covars, min_days=730, use_age_timescale=use_age_timescale)
        if r:
            combined_results.extend(r)
    
    if combined_results:
        pd.DataFrame(combined_results).to_csv('grs_fa/combined_grs_fa_results.csv', index=False)
        print(pd.DataFrame(combined_results).to_string())
    
    print('\n' + '='*60)
    print('ANALYSIS 3: Stratified by FA Level')
    print('='*60)
    
    strat_results = []
    for grsf, fa in pairs:
        r = stratified_by_fa(df, grsf, fa, covars, min_days=730)
        if r is not None:
            strat_results.append(r)
    
    if strat_results:
        pd.concat(strat_results, ignore_index=True).to_csv('grs_fa/stratified_by_fa.csv', index=False)
        print(pd.concat(strat_results, ignore_index=True).to_string())
    
    # ANALYSIS 4: Quartile Analysis for GRS
    print('\n' + '='*60)
    print('ANALYSIS 4: GRS Quartile Analysis (Q1-Q4)')
    print('='*60)
    
    all_quartile = []
    for grsf in ['n3fa_grs', 'n6fa_grs', 'pufa_grs']:
        if grsf in df.columns:
            r = quartile_analysis(df, grsf, covars, min_days=730, use_age_timescale=use_age_timescale)
            if r is not None:
                all_quartile.append(r)
    
    if all_quartile:
        pd.concat(all_quartile, ignore_index=True).to_csv('grs_fa/grs_quartile_results.csv', index=False)
        print(pd.concat(all_quartile, ignore_index=True).to_string())
    
    # ANALYSIS 5: Quartile Analysis for FA
    print('\n' + '='*60)
    print('ANALYSIS 5: FA Quartile Analysis (Q1-Q4)')
    print('='*60)
    
    all_fa_quartile = []
    for fa in ['n3fa', 'n6fa', 'pufa']:
        if fa in df.columns:
            r = quartile_analysis(df, fa, covars, min_days=730, use_age_timescale=use_age_timescale)
            if r is not None:
                all_fa_quartile.append(r)
    
    if all_fa_quartile:
        pd.concat(all_fa_quartile, ignore_index=True).to_csv('grs_fa/fa_quartile_results.csv', index=False)
        print(pd.concat(all_fa_quartile, ignore_index=True).to_string())
    
    # ANALYSIS 6: Joint Effect (2x2)
    print('\n' + '='*60)
    print('ANALYSIS 6: Joint Effect (GRS x FA)')
    print('='*60)
    
    joint_results = []
    for grsf, fa in pairs:
        r = joint_effect_analysis(df, grsf, fa, covars, min_days=730, use_age_timescale=use_age_timescale)
        if r is not None:
            joint_results.append(r)
    
    if joint_results:
        pd.concat(joint_results, ignore_index=True).to_csv('grs_fa/joint_effect_results.csv', index=False)
        print(pd.concat(joint_results, ignore_index=True).to_string())
    
    # ANALYSIS 7: Interaction Effect
    print('\n' + '='*60)
    print('ANALYSIS 7: Interaction Effect (n3fa_grs x n3fa)')
    print('='*60)
    
    int_results = interaction_analysis(df, 'n3fa_grs', 'n3fa', covars, min_days=730, use_age_timescale=use_age_timescale)
    if int_results:
        pd.DataFrame(int_results).to_csv('grs_fa/interaction_results.csv', index=False)
        print(pd.DataFrame(int_results).to_string())
    
    print('\n=== COMPLETE ===')

if __name__ == '__main__':
    main(use_age_timescale=True)
