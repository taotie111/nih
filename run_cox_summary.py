import pandas as pd
import numpy as np

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

df_results = pd.read_excel("Cox_matched_results_models.xlsx")

def format_p_table(p):
    if p < 0.001:
        return "<0.001*"
    elif p < 0.05:
        return f"{p:.3f}*"
    else:
        return f"{p:.3f}"

summary_rows = []
for fa_var in fa_analysis_vars:
    row = {"Serum levels (per 1-SD increase)": fa_var}
    for model_name in ["Model1", "Model2", "Model3"]:
        df_sub = df_results[(df_results["Variable"]==fa_var) & (df_results["Model"]==model_name)]
        if not df_sub.empty:
            HR = df_sub["HR"].values[0]
            CI = df_sub["CI"].values[0]
            P = df_sub["P_value"].values[0]
            P_str = format_p_table(P)
            row[f"Adjusted HR{model_name[-1]}"] = round(HR,3)
            row[f"95%CI{model_name[-1]}"] = f"({CI})"
            row[f"P value{model_name[-1]}"] = P_str
        else:
            row[f"Adjusted HR{model_name[-1]}"] = ""
            row[f"95%CI{model_name[-1]}"] = ""
            row[f"P value{model_name[-1]}"] = ""
    summary_rows.append(row)

cox_summary_table = pd.DataFrame(summary_rows)
print(cox_summary_table.to_string())
cox_summary_table.to_excel("Cox_summary_table.xlsx", index=False)
print("\nCox summary table saved to Cox_summary_table.xlsx")
