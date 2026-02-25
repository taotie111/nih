# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from statsmodels.duration.hazard_regression import PHReg
import warnings
warnings.filterwarnings('ignore')

df_matched = pd.read_csv('matched_data.csv')

fa_analysis_vars = [
    "Total fatty acid",
    "Omega 3 fatty acid",
    "Omega 6 fatty acid",
    "Polyunsaturated fatty acids",
    "Monounsaturated fatty acids",
    "Saturated fatty acids",
    "Linoleic acid",
    "Docosahexaenoic acid"
]

actual_fa_columns = [
    "fatty_acids_total",
    "fatty_acids_n3",
    "fatty_acids_n6",
    "fatty_acids_pufa",
    "fatty_acids_mufa",
    "fatty_acids_sfa",
    "fatty_acids_la",
    "fatty_acids_dha"
]

ethnic_dummies = pd.get_dummies(
    df_matched["ethnic_background"],
    prefix="ethnic",
    drop_first=True
)

df_matched = pd.concat([df_matched, ethnic_dummies], axis=1)
ethnic_vars = ethnic_dummies.columns.tolist()

if 'alcohol_frequency_baseline' in df_matched.columns:
    df_matched['Alcohol use'] = df_matched['alcohol_frequency_baseline'].apply(
        lambda x: 1 if x == 1 else 0
    )
if 'smoking_status_baseline' in df_matched.columns:
    df_matched['Smoking status'] = df_matched['smoking_status_baseline'].apply(
        lambda x: 1 if x == 1 else 0
    )
if 'education_baseline' in df_matched.columns:
    df_matched['education_bin'] = df_matched['education_baseline'].apply(
        lambda x: 1 if x == 1 else 0
    )

if 'glaucoma_baseline' in df_matched.columns:
    df_matched['glaucoma'] = df_matched['glaucoma_baseline'].apply(
        lambda x: 1 if x == 1 else 0
    )
if 'depression_baseline' in df_matched.columns:
    df_matched['diabetes'] = df_matched['depression_baseline'].apply(
        lambda x: 1 if x == 1 else 0
    )           

model1_vars = ["age_baseline", "sex", "ethnic_background"]
model2_vars = model1_vars + [
    "bmi_baseline",
    "Alcohol use",
    "Smoking status",
    "education_baseline"
]
model3_vars = model2_vars + [
    "diabetes_baseline",
    "hypertension_baseline",
    "heart_disease_composite",
    "glaucoma",
    "depression_baseline"
]

all_models = {
    "Model1": model1_vars,
    "Model2": model2_vars,
    "Model3": model3_vars
}

df_matched["cataract_group"] = np.where(df_matched["cataract_time_to_event_days"].isna(), 0, 1)
df_matched["time"] = df_matched["cataract_time_to_event_days"]
df_matched.loc[df_matched["cataract_group"] == 0, "time"] = df_matched.loc[df_matched["cataract_group"] == 0, "followup_duration_cataract"]
df_matched["status"] = df_matched["cataract_group"].astype(int)

results_all = []
all_covariates = list(set(model1_vars + model2_vars + model3_vars))

for fa_var, actual_col in zip(fa_analysis_vars, actual_fa_columns):
    cox_data = df_matched[[actual_col, "time", "status"] + all_covariates]
    print(f"\n{fa_var} - Total for Cox analysis: {len(cox_data):,} people")
    
    if len(cox_data) < 20:
        print("Insufficient sample size, skip")
        continue
    
    for model_name, model_vars in all_models.items():
        exog_vars = [actual_col] + [v for v in model_vars if v in cox_data.columns]
        exog = cox_data[exog_vars].astype(float)
        time = cox_data["time"].values
        status = cox_data["status"].values
        
        model = PHReg(time, exog, status=status)
        result = model.fit()
        
        beta = result.params[0]
        se = result.bse[0]
        HR = np.exp(beta)
        CI_l = np.exp(beta - 1.96 * se)
        CI_u = np.exp(beta + 1.96 * se)
        p = result.pvalues[0]
        
        results_all.append({
            "Variable": fa_var,
            "Model": model_name,
            "HR": HR,
            "CI": f"{CI_l:.2f}-{CI_u:.2f}",
            "P_value": p,
            "N": len(cox_data)
        })
        
        print(f"{fa_var} | {model_name}: N={len(cox_data)}, HR={HR:.2f} ({CI_l:.2f}-{CI_u:.2f}), p={p:.4g}")

df_results = pd.DataFrame(results_all)
df_results.to_excel("Cox_matched_results_models.xlsx", index=False)
print("\nCox analysis completed, results saved to Cox_matched_results_models.xlsx")
