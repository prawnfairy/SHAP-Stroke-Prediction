# Explaining a Stroke Classifier with SHAP

Assignment project (Master of Artificial Intelligence, Explainable AI): explain black box models trained on the Kaggle **Stroke Prediction Dataset** using SHAP, from a data scientist, doctor and patient point of view.

> **Educational use only.** The models here are not clinically validated and must not be used for medical decisions. SHAP values describe how a model uses its inputs, not what causes a stroke.

## Data

- Source: [Stroke Prediction Dataset (fedesoriano, Kaggle)](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset), file `healthcare-dataset-stroke-data.csv`.
- 5,110 patients, 12 columns, 249 recorded strokes (4.87%).
- SHA-256 of the file used: `644d473b05d2797006bd94865e4f8bb057f0c721617911613c82c8fcfc707420`.
- The Kaggle page does not state the collection period, country or original source, so representativeness is unknown.
- Download the CSV from Kaggle yourself and place it where `utils.py` expects it (check the path in `utils.py`). Check the dataset's licence terms before committing the file to a public repository.

## Setup

Tested with Python 3.9.6 (macOS, arm64):

```
numpy 2.0.2
pandas 2.3.3
scikit-learn 1.6.1
imbalanced-learn 0.12.4
shap 0.49.1
matplotlib 3.9.4
```

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy==2.0.2 pandas==2.3.3 scikit-learn==1.6.1 imbalanced-learn==0.12.4 shap==0.49.1 matplotlib==3.9.4
```

## How to run

```bash
python 00_data_exploration.py     # Task 1: EDA
python 03_shap.py                 # Tasks 2-4: Random Forest + TreeSHAP, force and summary plots
python 06_kernel_shap.py          # Task 6: MLP + Kernel SHAP (takes a few minutes)
python 07_report_numbers.py       # all report numbers (report_numbers.json) and figures
```

Run the scripts from the project folder so that `utils.py` can be imported. The scripts are written as `# %%` cells, so they also run cell by cell in VS Code or Spyder.

## Method summary

- One-hot encoding (all levels kept) and missing BMI set to 0 by the supplied `utils.py` (before the split; no leakage, but 0 acts as a missing-value flag).
- 80/20 split (seed 2021, not stratified): 4,088 training and 1,022 test rows (50 test strokes).
- Random oversampling of the **training portion only** (3,889 rows per class).
- **Random Forest** (defaults, `random_state=2021`), explained with `shap.TreeExplainer` on the first 500 held-out rows.
- **MLP** (`StandardScaler` + two hidden layers of 64 and 32 units, `max_iter=500`), explained with `shap.KernelExplainer` (50-row background, `nsamples=500`) on the first 50 held-out rows.
- Explained output: predicted probability of stroke (class 1). The SHAP base value (about 0.50) reflects the oversampled training data, not the 4.9% test prevalence.

## Main results (test set: 1,022 patients, 50 strokes)

| Metric | Random Forest | MLP |
|--------|---------------|-----|
| Accuracy | 0.942 | 0.904 |
| Stroke recall | 0.08 (4/50) | 0.20 (10/50) |
| Stroke precision | 0.235 | 0.147 |
| Macro F1 | 0.545 | 0.557 |
| ROC AUC | 0.737 | 0.706 |

- Always predicting "no stroke" gives accuracy 0.951, so accuracy alone is misleading.
- Age, average glucose level and BMI are the most influential features of the Random Forest.
- The two models agree only moderately on global feature importance (Spearman 0.45 across 21 features).
- The Task 3 example (held-out row 293) is a **false positive**: score 0.63 against a 0.50 base value.

## Limitations

- Only 50 test strokes: recall has wide uncertainty (95% Wilson intervals 0.03 to 0.19 for the forest and 0.11 to 0.33 for the MLP, overlapping).
- Scores are not calibrated probabilities (oversampling shifts them upwards; Brier score is worse than a constant prediction).
- Kernel SHAP uses 500 samples and its default regularisation, which keeps at most 10 non-zero attributions per patient, so MLP attributions are approximate.
- Correlated predictors and paired one-hot columns share attribution.
- BMI missingness is informative (16.1% of stroke cases versus 3.3% of others), and zero imputation probably explains the counter-intuitive BMI direction.
- `03_shap.py` and `06_kernel_shap.py` use the unseeded oversampling in `utils.py`, so their numbers differ slightly between runs; the reported numbers come from `07_report_numbers.py`, which uses seeded oversampling (seed 2021).

## Repository contents

```
figures/                      # force plots, summary plots, EDA panel
00_data_exploration.py
01_interpretable.models.py
02_lime.py
03_shap.py
04_counterfactuals.py
05_lrp.py
06_kernelSHAP.py
07_report_numbers.py
report_numbers.json           # all numbers used in the report
utils.py                      # supplied by the lecturer, unmodified
```

## AI assistance

An AI assistant (Claude, Anthropic) helped with structuring the scripts and computing the reported statistics.

## References

1. World Health Organization. *The top 10 causes of death.* https://www.who.int/news-room/fact-sheets/detail/the-top-10-causes-of-death
2. Ponce-Bobadilla, A. V. et al. (2024). Practical guide to SHAP analysis. *Clinical and Translational Science*, 17(11). https://doi.org/10.1111/cts.70056
3. fedesoriano (2021). Stroke Prediction Dataset. Kaggle.
4. SHAP, scikit-learn and imbalanced-learn documentation (see the report's reference list for details).

## Author

Sofia Batrisyia Binti Mohamad Faris. Course: Master of Artificial Intelligence, lecturer Professor Dr. Loo Chu Kiong.