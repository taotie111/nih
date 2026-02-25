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

forgrs_name,grs_col in GRS_VARS.items():
    ifgrs_col not in df_matched.columns: