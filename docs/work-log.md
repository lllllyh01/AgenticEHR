## 7/4/2026
### MIMIC IV
#### Model Evaluating Results

| Task                  | Group   | Kind       | AUROC | AUPRC | MAE   | R2    |
| --------------------- | ------- | ---------- | ----- | ----- | ----- | ----- |
| mortality_1y          | forward | binary     | 0.829 | 0.495 |       |       |
| readmission_30d       | forward | binary     | 0.630 | 0.246 |       |       |
| prolonged_stay        | forward | binary     | 0.778 | 0.750 |       |       |
| los_days              | forward | regression |       |       | 5.539 | 0.185 |
| dx_diabetes           | chronic | binary     | 0.816 | 0.693 |       |       |
| dx_hypertension       | chronic | binary     | 0.839 | 0.901 |       |       |
| dx_hyperlipidemia     | chronic | binary     | 0.809 | 0.784 |       |       |
| dx_cardiovascular     | chronic | binary     | 0.843 | 0.860 |       |       |
| dx_respiratory        | chronic | binary     | 0.795 | 0.812 |       |       |
| dx_depression_anxiety | chronic | binary     | 0.727 | 0.582 |       |       |

Models generally perform well for both chronic disease predictions and forward tasks (especially hypertension and cardiovascular with AUROC >0.83, AUPRC >0.86), and overall, chronic predictions are more accurate than forward. 30-day readmission has lower performance, indicating possible prediction challenges.



1. Verified mimic-iv preprocessing is correct. Processed mimic data are in `artifacts/mimic`, processing code: `agentic-ehr/src/agentic_ehr/data/mimic/build.py`
    To verify the processed mimic dataset: Run `pytest tests/test_mimic_build_validation.py -v -s`
2. Add window to control the input time range for different prediction tasks; Changed length of stay input time window from end of stay to the first 24h of admission to prevent training data leakage
3. 