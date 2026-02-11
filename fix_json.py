#!/usr/bin/env python3
import json
import re

with open('prs_cox_rcs.ipynb', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace problematic lines directly
# Line 1207-1208: Fix the multiline comment + code
content = content.replace(
    '    "    # 修复：RCS 列已包含 age 信息，删除 age_baseline 避免共线性\n    model_cols = [\'survival_time\', \'cataract_incident\', \'sex\', \'bmi_baseline\', \'n3fa_grs\'] + rcs_cols\n",',
    '    "    # 修复：RCS 列已包含 age 信息，删除 age_baseline 避免共线性\\n",\n    "    model_cols = [\'survival_time\', \'cataract_incident\', \'sex\', \'bmi_baseline\', \'n3fa_grs\'] + rcs_cols\\n",'
)

# Line 1216: Fix the multiline comment + code
content = content.replace(
    '    "        # 修复：标准化时不包含 RCS 列（设计矩阵已正交化，标准化会破坏结构）\n    continuous_cols = [\'n3fa_grs\', \'bmi_baseline\']\n",',
    '    "    # 修复：标准化时不包含 RCS 列（设计矩阵已正交化，标准化会破坏结构）\\n",\n    "    continuous_cols = [\'n3fa_grs\', \'bmi_baseline\']\\n",'
)

with open('prs_cox_rcs.ipynb', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed!")

# Validate
try:
    json.loads(content)
    print("JSON is valid!")
except json.JSONDecodeError as e:
    print(f"JSON error: {e}")
