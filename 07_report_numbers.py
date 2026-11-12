# %% Imports
# Produces every number and figure that the report template asks for, in ONE
# run, so that all reported values come from the same fitted models.
# Output: report_numbers.json and report_fig*.png in the current folder.
import json
import platform
import sys
import time
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import sklearn
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from utils import DataLoader

warnings.filterwarnings("ignore")
SEED = 2021
N_TREE_EXPLAIN = 500      # held-out rows explained with TreeSHAP
N_KERNEL_EXPLAIN = 50     # held-out rows explained with Kernel SHAP
N_BACKGROUND = 50         # Kernel SHAP background rows (from training data)
KERNEL_NSAMPLES = 500
OUT = {}


def jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    return str(o)


# Utility functions: extract class 1 (stroke) SHAP values and base value,
# compatible with both old (list) and new (3D array) SHAP output formats
def class1_values(sv):
    if isinstance(sv, list):
        return sv[1]
    if sv.ndim == 3:
        return sv[:, :, 1]
    return sv


def class1_base_value(ev):
    ev = np.atleast_1d(ev)
    return float(ev[1] if len(ev) > 1 else ev[0])


# %% Versions
OUT["environment"] = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scikit_learn": sklearn.__version__,
    "shap": shap.__version__,
    "matplotlib": matplotlib.__version__,
}
try:
    import imblearn
    OUT["environment"]["imbalanced_learn"] = imblearn.__version__
except Exception:
    pass

# %% TASK 1 numbers: data quality and descriptive statistics (raw file)
raw_loader = DataLoader()
raw_loader.load_dataset()
raw = raw_loader.data.copy()


def summ(s):
    s = s.dropna()
    return {"n": int(s.count()), "mean": float(s.mean()), "sd": float(s.std()),
            "median": float(s.median()), "q1": float(s.quantile(0.25)),
            "q3": float(s.quantile(0.75)), "min": float(s.min()),
            "max": float(s.max())}


cat_cols = ["gender", "ever_married", "work_type", "Residence_type",
            "smoking_status", "hypertension", "heart_disease"]
eda = {
    "rows": int(raw.shape[0]),
    "columns": int(raw.shape[1]),
    "column_names": list(raw.columns),
    "duplicate_rows": int(raw.duplicated().sum()),
    "duplicate_ids": int(raw["id"].duplicated().sum()),
    "missing": {c: {"n": int(raw[c].isna().sum()),
                    "pct": float(100 * raw[c].isna().mean())} for c in raw.columns},
    "smoking_unknown": {"n": int((raw["smoking_status"] == "Unknown").sum()),
                        "pct": float(100 * (raw["smoking_status"] == "Unknown").mean())},
    "stroke_counts": {"n0": int((raw["stroke"] == 0).sum()),
                      "n1": int((raw["stroke"] == 1).sum()),
                      "pct_positive": float(100 * raw["stroke"].mean())},
    "numeric": {c: {"overall": summ(raw[c]),
                    "stroke0": summ(raw.loc[raw.stroke == 0, c]),
                    "stroke1": summ(raw.loc[raw.stroke == 1, c])}
                for c in ["age", "avg_glucose_level", "bmi"]},
    "categorical": {},
    "bmi_missing_among_stroke1": int(raw.loc[raw.stroke == 1, "bmi"].isna().sum()),
}
for c in cat_cols:
    g = raw.groupby(c)["stroke"].agg(["count", "sum", "mean"])
    eda["categorical"][c] = {
        str(k): {"n": int(v["count"]), "pct": float(100 * v["count"] / len(raw)),
                 "stroke_n": int(v["sum"]), "stroke_rate_pct": float(100 * v["mean"])}
        for k, v in g.iterrows()}
OUT["eda"] = eda

# EDA figure panel
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
cnt = raw["stroke"].value_counts().sort_index()
ax[0, 0].bar(["No stroke (0)", "Stroke (1)"], cnt.values, color=["#1E88E5", "#FF0D57"])
for i, v in enumerate(cnt.values):
    ax[0, 0].text(i, v, f"{v}\n({100 * v / len(raw):.1f}%)", ha="center", va="bottom")
ax[0, 0].set_title("Class counts")
ax[0, 0].set_ylim(0, cnt.max() * 1.2)
for a, col, title in [(ax[0, 1], "age", "Age (years)"),
                      (ax[0, 2], "avg_glucose_level", "Average glucose level"),
                      (ax[1, 0], "bmi", "BMI")]:
    for lab, color in [(0, "#1E88E5"), (1, "#FF0D57")]:
        a.hist(raw.loc[raw.stroke == lab, col].dropna(), bins=30, alpha=0.6,
               density=True, color=color, label=f"stroke = {lab}")
    a.set_title(f"{title} by stroke status")
    a.set_ylabel("Density")
    a.legend()
miss = pd.Series({"bmi (missing)": raw["bmi"].isna().sum(),
                  "smoking_status (Unknown)": (raw["smoking_status"] == "Unknown").sum()})
ax[1, 1].barh(miss.index, miss.values, color="#6B7689")
for i, v in enumerate(miss.values):
    ax[1, 1].text(v, i, f" {v} ({100 * v / len(raw):.1f}%)", va="center")
ax[1, 1].set_xlim(0, miss.max() * 1.3)
ax[1, 1].set_title("Missing or unavailable values")
rate = raw.groupby("smoking_status")["stroke"].mean() * 100
ax[1, 2].bar(rate.index, rate.values, color="#6B7689")
ax[1, 2].set_title("Stroke rate (%) by smoking status")
ax[1, 2].tick_params(axis="x", rotation=20)
plt.tight_layout()
plt.savefig("report_fig1_eda_panel.png", dpi=200, bbox_inches="tight")
plt.close()

# %% Prepare the modelling data (same steps as 03_shap.py and 06_kernel_shap.py)
dl = DataLoader()
dl.load_dataset()
dl.preprocess_data()
X_train, X_test, y_train, y_test = dl.get_data_split()
split = {"train_rows_before_oversampling": int(len(X_train)),
         "train_positive_before_oversampling": int(y_train.sum()),
         "test_rows": int(len(X_test)),
         "test_positive": int(y_test.sum()),
         "test_prevalence_pct": float(100 * y_test.mean()),
         "test_size_fraction": 0.20, "split_seed": SEED, "stratified": False,
         "predictors": list(X_train.columns), "n_predictors": int(X_train.shape[1])}
# Same oversampling as utils.DataLoader.oversample, but with a fixed seed so the
# report numbers are reproducible (utils.py itself is left untouched)
from imblearn.over_sampling import RandomOverSampler
_ros = RandomOverSampler(sampling_strategy="minority", random_state=SEED)
_x, _y = _ros.fit_resample(X_train.to_numpy(), y_train.to_numpy())
X_train = pd.DataFrame(_x, columns=X_train.columns).astype(float)
y_train = pd.Series(_y, name=y_train.name)
X_test = X_test.astype(float)
split["train_rows_after_oversampling"] = int(len(X_train))
split["train_positive_after_oversampling"] = int(y_train.sum())
OUT["split"] = split
features = list(X_train.columns)


def evaluate(model):
    proba = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    return {
        "classes": [int(c) for c in model.classes_],
        "accuracy": accuracy_score(y_test, pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "precision_stroke": precision_score(y_test, pred, zero_division=0),
        "recall_stroke": recall_score(y_test, pred, zero_division=0),
        "f1_stroke": f1_score(y_test, pred, zero_division=0),
        "f1_macro": f1_score(y_test, pred, average="macro"),
        "roc_auc": roc_auc_score(y_test, proba),
        "average_precision": average_precision_score(y_test, proba),
        "brier_score": brier_score_loss(y_test, proba),
        "mean_predicted_probability": float(proba.mean()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "threshold_rule": "model.predict (class with higher predicted probability; 0.5 for two classes)",
    }, proba, pred


def global_table(sv, X):
    mean_abs = np.abs(sv).mean(axis=0)
    rows = []
    for j, f in enumerate(features):
        col = X[f].to_numpy(dtype=float)
        s = sv[:, j]
        if np.nanstd(col) == 0 or np.nanstd(s) == 0:
            rho = None
        else:
            rho = float(spearmanr(col, s)[0])
        binary = set(np.unique(col)).issubset({0.0, 1.0})
        if binary:
            hi, lo = col == 1, col == 0
        else:
            med = np.median(col)
            hi, lo = col > med, col <= med
        rows.append({"feature": f, "mean_abs_shap": float(mean_abs[j]),
                     "spearman_value_vs_shap": rho,
                     "mean_shap_high_or_1": float(s[hi].mean()) if hi.any() else None,
                     "mean_shap_low_or_0": float(s[lo].mean()) if lo.any() else None,
                     "binary": bool(binary)})
    rows.sort(key=lambda r: -r["mean_abs_shap"])
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows


def case_table(sv_row, x_row):
    rows = [{"feature": f, "value": float(x_row[f]), "shap": float(sv_row[j])}
            for j, f in enumerate(features)]
    rows.sort(key=lambda r: -abs(r["shap"]))
    return rows


def save_force(base, sv_row, x_row, fname):
    shap.force_plot(base, sv_row, x_row, matplotlib=True, show=False,
                    figsize=(20, 4), text_rotation=15, contribution_threshold=0.05)
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()


def save_summary(sv, X, fname, bar=False):
    plt.figure()
    shap.summary_plot(sv, X, show=False, max_display=len(features),
                      plot_type="bar" if bar else "dot")
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()


# %% TASK 2 to 4: Random Forest + TreeSHAP
rf = RandomForestClassifier(random_state=SEED)
rf.fit(X_train, y_train)
rf_metrics, rf_proba, rf_pred = evaluate(rf)
OUT["rf"] = {"model": "RandomForestClassifier(random_state=2021), default hyperparameters",
             "metrics": rf_metrics}

t0 = time.time()
tree_expl = shap.TreeExplainer(rf)
X_tree = X_test.iloc[:N_TREE_EXPLAIN]
raw_sv = tree_expl.shap_values(X_tree)
sv_rf = class1_values(raw_sv)
base_rf = class1_base_value(tree_expl.expected_value)
t_tree = time.time() - t0

pred_tree = rf.predict(X_tree)
pos_idx = np.where(pred_tree == 1)[0]
case_pos = int(pos_idx[0]) if len(pos_idx) else 0
case_label = X_tree.index[case_pos]
case_actual = int(y_test.loc[case_label])
case_pred = int(pred_tree[case_pos])
case_type = {(1, 1): "true positive", (0, 1): "false positive",
             (0, 0): "true negative", (1, 0): "false negative"}[(case_actual, case_pred)]
p_case_rf = float(rf.predict_proba(X_tree.iloc[[case_pos]])[0, 1])
resid_rf = rf_proba[: len(X_tree)] - (base_rf + sv_rf.sum(axis=1))

OUT["rf"]["shap"] = {
    "explainer": "shap.TreeExplainer(rf) (feature_perturbation=tree_path_dependent; no background dataset)",
    "raw_output_type": type(raw_sv).__name__,
    "raw_output_shape": list(np.shape(raw_sv)),
    "class_index_used": 1,
    "classes_": rf_metrics["classes"],
    "n_explained": int(len(X_tree)),
    "n_explained_positive_actual": int(y_test.loc[X_tree.index].sum()),
    "n_explained_predicted_positive": int(pred_tree.sum()),
    "base_value": base_rf,
    "output_scale": "predicted probability of stroke = 1 (RandomForest predict_proba)",
    "reconstruction_max_abs_residual": float(np.abs(resid_rf).max()),
    "reconstruction_mean_abs_residual": float(np.abs(resid_rf).mean()),
    "runtime_seconds": t_tree,
    "global": global_table(sv_rf, X_tree),
    "case": {"row_label": int(case_label), "position_in_explained_sample": case_pos,
             "actual_class": case_actual, "predicted_class": case_pred,
             "case_type": case_type, "probability": p_case_rf,
             "base_plus_sum_shap": base_rf + float(sv_rf[case_pos].sum()),
             "table": case_table(sv_rf[case_pos], X_tree.iloc[case_pos]),
             "selection_rule": "first row, in held-out split order, among the first 500 held-out rows that the Random Forest predicts as stroke = 1"},
}
save_force(base_rf, sv_rf[case_pos], X_tree.iloc[case_pos], "report_fig2_rf_force.png")
save_summary(sv_rf, X_tree, "report_fig3_rf_summary.png")
save_summary(sv_rf, X_tree, "report_fig3b_rf_bar.png", bar=True)

# %% TASK 6: MLP + Kernel SHAP (same split, same predictors, same case)
mlp = make_pipeline(StandardScaler(),
                    MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                                  random_state=SEED))
mlp.fit(X_train, y_train)
mlp_metrics, mlp_proba, mlp_pred = evaluate(mlp)
mlp_metrics["mlp_iterations"] = int(mlp[-1].n_iter_)
mlp_metrics["mlp_converged_before_max_iter"] = bool(mlp[-1].n_iter_ < 500)
OUT["mlp"] = {"model": "Pipeline(StandardScaler, MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=2021))",
              "metrics": mlp_metrics}

# Explained sample: first 50 held-out rows, plus the Task 3 case if not among them
X_k = X_test.iloc[:N_KERNEL_EXPLAIN]
if case_label not in X_k.index:
    X_k = pd.concat([X_k, X_test.loc[[case_label]]])
kcase_pos = int(X_k.index.get_loc(case_label))


def mlp_predict_proba(x):
    # Plain function wrapper: Kernel SHAP passes NumPy arrays, so rebuild the
    # DataFrame with the training column names before calling the pipeline
    return mlp.predict_proba(pd.DataFrame(x, columns=features))


def run_kernel(bg_seed):
    background = shap.sample(X_train, N_BACKGROUND, random_state=bg_seed)
    expl = shap.KernelExplainer(mlp_predict_proba, background)
    sv = class1_values(expl.shap_values(X_k, nsamples=KERNEL_NSAMPLES))
    return expl, sv


t0 = time.time()
k_expl, sv_k = run_kernel(SEED)
t_kernel = time.time() - t0
base_k = class1_base_value(k_expl.expected_value)
pred_k = mlp.predict(X_k)
resid_k = mlp.predict_proba(X_k)[:, 1] - (base_k + sv_k.sum(axis=1))
p_case_k = float(mlp.predict_proba(X_k.iloc[[kcase_pos]])[0, 1])

OUT["mlp"]["shap"] = {
    "explainer": f"shap.KernelExplainer(mlp_predict_proba (wrapper of mlp.predict_proba), shap.sample(X_train, {N_BACKGROUND}, random_state={SEED})); shap_values(nsamples={KERNEL_NSAMPLES})",
    "class_index_used": 1,
    "classes_": mlp_metrics["classes"],
    "n_explained": int(len(X_k)),
    "n_explained_positive_actual": int(y_test.loc[X_k.index].sum()),
    "n_explained_predicted_positive": int(pred_k.sum()),
    "background_size": N_BACKGROUND, "background_seed": SEED, "nsamples": KERNEL_NSAMPLES,
    "base_value": base_k,
    "output_scale": "predicted probability of stroke = 1 (MLP predict_proba)",
    "reconstruction_max_abs_residual": float(np.abs(resid_k).max()),
    "reconstruction_mean_abs_residual": float(np.abs(resid_k).mean()),
    "runtime_seconds": t_kernel,
    "global": global_table(sv_k, X_k),
    "case": {"row_label": int(case_label), "position_in_explained_sample": kcase_pos,
             "actual_class": case_actual, "predicted_class": int(pred_k[kcase_pos]),
             "probability": p_case_k,
             "base_plus_sum_shap": base_k + float(sv_k[kcase_pos].sum()),
             "table": case_table(sv_k[kcase_pos], X_k.iloc[kcase_pos]),
             "same_case_as_task_3": True},
}
save_force(base_k, sv_k[kcase_pos], X_k.iloc[kcase_pos], "report_fig4_mlp_force.png")
save_summary(sv_k, X_k, "report_fig5_mlp_summary.png")
save_summary(sv_k, X_k, "report_fig5b_mlp_bar.png", bar=True)

# %% Stability check: repeat Kernel SHAP with a different training background sample
_, sv_k2 = run_kernel(SEED + 1)
m1, m2 = np.abs(sv_k).mean(axis=0), np.abs(sv_k2).mean(axis=0)
top5_1 = set(np.array(features)[np.argsort(-m1)[:5]])
top5_2 = set(np.array(features)[np.argsort(-m2)[:5]])
a, b = sv_k[kcase_pos], sv_k2[kcase_pos]
nz = (np.sign(a) != 0) & (np.sign(b) != 0)
OUT["mlp"]["stability_second_background"] = {
    "second_background_seed": SEED + 1,
    "spearman_rank_corr_mean_abs_shap": float(spearmanr(m1, m2)[0]),
    "top5_first": sorted(top5_1), "top5_second": sorted(top5_2),
    "top5_overlap": len(top5_1 & top5_2),
    "case_sign_agreement_fraction": float((np.sign(a)[nz] == np.sign(b)[nz]).mean()),
    "case_sign_changes": [features[j] for j in np.where(nz & (np.sign(a) != np.sign(b)))[0]],
}

# %% Comparison of the two models
r_rf = np.array([next(r["mean_abs_shap"] for r in OUT["rf"]["shap"]["global"] if r["feature"] == f) for f in features])
r_k = np.array([next(r["mean_abs_shap"] for r in OUT["mlp"]["shap"]["global"] if r["feature"] == f) for f in features])
top3_rf = [r["feature"] for r in OUT["rf"]["shap"]["global"][:3]]
top3_k = [r["feature"] for r in OUT["mlp"]["shap"]["global"][:3]]
top5_rf = {r["feature"] for r in OUT["rf"]["shap"]["global"][:5]}
top5_k = {r["feature"] for r in OUT["mlp"]["shap"]["global"][:5]}
OUT["comparison"] = {
    "spearman_rank_corr_mean_abs_shap_rf_vs_mlp": float(spearmanr(r_rf, r_k)[0]),
    "top3_rf": top3_rf, "top3_mlp": top3_k,
    "top5_overlap": len(top5_rf & top5_k),
    "test_prediction_agreement_fraction": float((rf_pred == mlp_pred).mean()),
    "both_predict_positive": int(((rf_pred == 1) & (mlp_pred == 1)).sum()),
    "rf_only_positive": int(((rf_pred == 1) & (mlp_pred == 0)).sum()),
    "mlp_only_positive": int(((rf_pred == 0) & (mlp_pred == 1)).sum()),
    "note": "Compare ranks, not raw SHAP magnitudes: the explained samples differ (500 vs. 50 rows) and the models differ.",
}

# %% Save
with open("report_numbers.json", "w") as fh:
    json.dump(OUT, fh, indent=2, default=jsonable)

print("EDA rows/cols:", eda["rows"], eda["columns"], "| positives:", eda["stroke_counts"])
print("RF  :", {k: round(v, 4) for k, v in rf_metrics.items() if isinstance(v, float)})
print("MLP :", {k: round(v, 4) for k, v in mlp_metrics.items() if isinstance(v, float)})
print("RF case:", OUT["rf"]["shap"]["case"]["row_label"], case_type,
      "p =", round(p_case_rf, 3), "| residual max:", OUT["rf"]["shap"]["reconstruction_max_abs_residual"])
print("Kernel residual max:", OUT["mlp"]["shap"]["reconstruction_max_abs_residual"])
print("Saved report_numbers.json and report_fig*.png")
