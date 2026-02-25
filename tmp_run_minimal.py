import pandas as pd
import numpy as np

def main():
    p = 'WY_计算随访时间_cataract_更新的截止时间.csv'
    df = pd.read_csv(p)
    print(f"原始样本: {len(df):,}")
    df = df[df['followup_duration_cataract'].notna() & (df['followup_duration_cataract'] >= 365)].copy()
    fa_cols = ['fatty_acids_n3', 'fatty_acids_dha', 'fatty_acids_n6', 'fatty_acids_pufa', 'fatty_acids_total']
    df = df.dropna(subset=fa_cols).copy()
    df['event'] = np.where(df['cataract_time_to_event_days'].notna() & (df['cataract_time_to_event_days'] >= 365), 1, 0)
    print(f"筛选后样本: {len(df):,}, 病例: {(df['event']==1).sum():,}, 对照: {(df['event']==0).sum():,}")

if __name__ == '__main__':
    main()
