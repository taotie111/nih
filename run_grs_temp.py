import pandas as pd
import numpy as np
from statsmodels.duration.hazard_regression import PHReg
import warnings
warnings.filterwarnings('ignore')

df_matched = pd.read_csv('matched_data.csv')

GRS_VARS = {
    "Omega-3 fatty acid GRS": "n3fa_grs",
    "Omega-6 fatty acid GRS": "n6fa_grs",
    "PUFA GRS": "pufa_grs",
    "Total fatty acid GRS": "tfa_grs",
    "n3grs_layer": "n3grs_layer",
    "n6grs_layer": "n6grs_layer",
    "pufagrs_layer": "pufagrs_layer",
    "n3grs_layer2": "n3grs_layer2"
}

if "alcohol_frequency_baseline" in df_matched.columns:
    df_matched["Alcohol use"] = (df_matched["alcohol_frequency_baseline"] == 1).astype(int)
if "smoking_status_baseline" in df_matched.columns:
    df_matched["Smoking status"] = (df_matched["smoking_status_baseline"] == 1).astype(int)
if "education_baseline" in df_matched.columns:
    df_matched["education_bin"] = (df_matched["education_baseline"] == 1).astype(int)
if "glaucoma_baseline" in df_matched.columns:
    df_matched["glaucoma"] = (df_matched["glaucoma_baseline"] == 1).astype(int)
if "diabetes_baseline" in df_matched.columns:
    df_matched["diabetes"] = (df_matched["diabetes_baseline"] == 1).astype(int)

model1_vars = ["age_baseline", "sex", "ethnic_background"]
model2_vars = model1_vars + ["bmi_baseline", "Alcohol use", "Smoking status", "education_baseline"]
model3_vars = model2_vars + ["diabetes", "hypertension_baseline", "heart_disease_composite", "glaucoma", "depression_baseline"]
all_models = {"Model1": model1_vars, "Model2": model2_vars, "Model3": model3_vars}

df_matched["status"] = np.where(df_matched["cataract_time_to_event_days"].isna(), 0, 1)
df_matched["time"] = df_matched["cataract_time_to_event_days"]
df_matched.loc[df_matched["status"] == 0, "time"] = df_matched.loc[df_matched["status"] == 0, "followup_duration_cataract"]
results_all = []
all_covariates = list(set(model1_vars + model2_vars + model3_vars))

for key, val in GRS_VARS.items():
    if val not in df_matched.columns:
        print(f"{val} not found")
        continue
    z_col = f"{val}_z"
    df_matched[z_col] = (df_matched[val] - df_matched[val].mean()) / df_matched[val].std()
    cox_data = df_matched[[z_col, "time", "status"] + all_covariates]
    print(f"\n{key} - N={len(cox_data)}")
    if len(cox_data) < 50: 
        continue
    for model_name, model_vars in all_models.items():
        exog_vars = [z_col] + [v for v in model_vars if v in cox_data.columns]
        exog = cox_data[exog_vars].astype(float)
        time = cox_data["time"].values
        status = cox_data["status"].values
        model = PHReg(time, exog, status=status)
        result = model.fit()
        beta, se, p = result.params[0], result.bse[0], result.pvalues[0]
        HR, CI_l, CI_u = np.exp(beta), np.exp(beta-1.96*se), np.exp(beta+1.96*se)
        results_all.append({"GRS":key,"Model":model_name,"HR":HR,"95% CI":f"{CI_l:.2f}-{CI_u:.2f}","P_value":p,"N":len(cox_data)})
        print(f"{key} | {model_name}: HR={HR:.2f} ({CI_l:.2f}-{CI_u:.2f}), P={p:.4g}")

pd.DataFrame(results_all).to_excel("Cox_4GRS_cataract_results.xlsx",index=False)
print("\nGRS Cox analysis completed")
