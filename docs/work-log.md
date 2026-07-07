# Work Log

## 7/4/2026
### MIMIC-IV Model Evaluating Results

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

### Progress
1. Verified mimic-iv preprocessing is correct. Processed mimic data are in `artifacts/mimic`, processing code: `agentic-ehr/src/agentic_ehr/data/mimic/build.py`

    To verify the processed mimic dataset: Run `pytest tests/test_mimic_build_validation.py -v -s`

2. Add per-task observation time window `TaskSpec.window` to control the input time range; Changed length of stay input time window from end of stay to the first 24h of admission to prevent training data leakage
3. Full rebuild + retrain + sampling-based validation on the whole cohort confirmed the length-of-stay tasks stay at leak-free performance (prolonged_stay AUROC 0.778, los_days R² 0.185)
4. Change the default LLM from Gemini to Claude to ensure data privacy compliance. MIMIC-IV strictly prohibits sharing data with third parties, including transmitting it via APIs or online platforms (see: https://physionet.org/news/post/llm-responsible-use/). Of Gemini, GPT, and Claude, only Claude guarantees that user inputs and outputs are excluded from model training by default. (https://physionet.org/news/post/gpt-responsible-use/)
   
    ***MIMIC-IV-Note**. MIMIC-IV-Note is a collection of deidentified free-text clinical notes comprising of discharge summaries and radiology reports, and some LLM-integrated researches has been based on it ([Benchmarking Large Language Models for MIMIC-IV Clinical Note Summarization](https://pmc.ncbi.nlm.nih.gov/articles/PMC12872987/)). However, these notes are already high-level narrative text rather than raw signals. Moreover, a discharge summary is made at discharge and encodes the admission's outcomes, so using it as model input would leak the very targets we predict. All these makes MIMIC-IV-Note not suitable for this project.*


## 6/30/2026
1. Prevent data leakage in training:
    Some columns in MIMIC-IV, especially the historical diagnosis records and relevant lab tests results, leak the answers to certain prediction tasks, leading to overclaimed model accuracies. For example, HbA1c, glucose, and diabetes codes could decide whether a patient has diabetes. To prevent such lekage, these directly related columns should be dropped for certain tasks.
2. Add time window setting for prediction tasks. Most tasks are trained on full time range records. However, for length of stay and prolonged stay, the number of records will reveal the length of stay. Therefore, only the data in the first 24 hour of an admission are capped for training for these two tasks.

## 6/24/2026
1. Add training and evaluating on MIMIC-IV
2. Prediction tasks for MIMIC-IV:
    The paper [Towards a General Intelligence and Interface for Wearable Health Data](https://arxiv.org/pdf/2605.22759) describes 35 distinct health and behavior prediction tasks, organized into six categories: cardiovascular health, metabolic health, mental health, sleep, demographics (age, BMI, height, weight), and lifestyle & treatment. Based on the available data in MIMIC-IV, I selected six prediction tasks from where: diabetes, hypertension, hyperlipidemia, cardiovascular disease, respiratory disease, and depression/anxiety. Additionally, four commonly used probability prediction tasks were included: 1-year mortality, 30-day readmission, length of stay, and prolonged stay, bringing the total number of MIMIC-IV prediction tasks to ten.
3. Consolidated the multi-task inference into a `HealthRiskProfile` panel that shares one patient snapshot, and had the same `SummaryAgent` consume it to produce the patient health report.
4. Added a MIMIC concept map so raw feature codes read as plain language for the report.
5. Split `XGBoostRiskModel` to `XGBoostClassifierModel` and `XGBoostRegressionModel` for classification and regression
6. Add feature: dump risk profile, model prediction results and generated summaries into files for human review. Supports json/md/txt

## 6/14/2026 — Foundation Structure Build up
1. Built an agentic user health analysis system over EHR risk models. The high-level workflow is prediction model -> RiskProfile -> Summary Agent.

   1. Prediction model: complete prediction tasks based on EHR data. The current model is xgboost, but it can be swapped seamlessly for other models (e.g. MOTOR-T or other foundation models).
   2. RiskProfile: to fully decouple predictive model and summary agent, as well as provide richer context to summary agent, I implement a RiskProfile as an interface between two.
       - RiskProfile includes: (1) model prediction; (2) per-feature attributions computed with SHAP; (3) ConceptMap, which translates numeric metrics into more understandable natural language explanation; (4) snapshot of original EHR record

   3. Summary agent: generate health summary based on RiskProfile. For model it supports gemini, gpt and claude. It has two mode: with template and no template. 
       - If with template, it produces five fixed sections (What we found / What may be contributing / What this means / What to do next / When to seek care urgently); without one, a single free-form narrative.

   4. To keep outputs rigorous, every summary is validated before it's returned: no diagnostic/overclaiming language (BANNED_PHRASES), the probability must be stated faithfully, and low confidence must be flagged with caution.
   5. Evaluation: predictive model evaluation, and rule-based summary-quality checks (factual consistency, no unsupported claims, uncertainty faithfulness, clarity).

2. Waiting for approval of EHR-shot, used synthetic EHR-like FEMR successfully run through the whole process