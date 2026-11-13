# %% Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from utils import DataLoader
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score
import shap

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# %% Load and preprocess data
data_loader = DataLoader()
data_loader.load_dataset()
data_loader.preprocess_data()

# Split the data for evaluation
X_train, X_test, y_train, y_test = data_loader.get_data_split()

# Oversample the train data
X_train, y_train = data_loader.oversample(X_train, y_train)

# Same numeric dytpe as the (oversampled) training data
X_test = X_test.astype(float)
print(X_train.shape)
print(X_test.shape)

# %% Fit a different blacbox model: Neural Network (MLP)
# Scaling is part of the pipeline as neural networks are sensitive to feature scale
MLP = make_pipeline(StandardScaler(),
               MLPClassifier(hidden_layer_sizes=(64, 32),
                             max_iter=500,
                             random_state=2021))

MLP.fit(X_train, y_train)
y_pred = MLP.predict(X_test)

print(f"F1 Score: {f1_score(y_test, y_pred, average='macro')}")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# %% TASK 6: Create Kernel SHAP explainer
# Kernel SHAP is model-agnostic and can be used with any blackbox model, including neural networks
# Use a small background sample and explain few patients as Kernel SHAP is slow
background = shap.sample(X_train, 50, random_state=2021)

def predict_proba(data):
    return MLP.predict_proba(pd.DataFrame(data, columns=X_train.columns))

explainer = shap.KernelExplainer(predict_proba, background)

# Utility functions: extract class 1 (stroke) SHAP values and base value
def class1_values(sv):
    # SHAP values for class 1 (stroke)
    if isinstance(sv, list): # old SHAP versions return a list of arrays for each class
        return sv[1]
    if sv.ndim == 3: # new SHAP versions return a 3D array for multi-class problems
        return sv[:, :, 1]
    return sv

def class1_base_value(ev):
    # Expected (base) value for class 1 (stroke)
    ev = np.atleast_1d(ev)
    return ev[1] if len(ev)>1 else ev[0]

base_value = class1_base_value(explainer.expected_value)

# Calculate shapley values for a small subset of test data
X_explain = X_test.iloc[:50] # 50 rows to keep the computation time reasonable
raw_shap_values = explainer.shap_values(X_explain, nsamples=500)
shap_values = class1_values(raw_shap_values)
X_explain.head()

print("Raw SHAP output:", type(raw_shap_values), np.shape(raw_shap_values))
print("Stroke-class SHAP values:", shap_values.shape)

# %% Visualize local predictions (Force plot)
prediction = MLP.predict(X_explain)

stroke_index = np.where(prediction == 1)[0]
patient = int(stroke_index[0]) if len(stroke_index) > 0 else 0 # pick the first patient predicted as stroke
print(f"The MLP predicted: {prediction[patient]}")

shap.force_plot(base_value,
                shap_values[patient],
                X_explain.iloc[patient],
                matplotlib=True,
                show=False)

# Save Force plot as image
plt.savefig(FIGURES_DIR / "KernelSHAP_force_plot.png", bbox_inches="tight")
plt.show()

# %% Visualize global features (Summary plot)
shap.summary_plot(shap_values, X_explain, show=False)

plt.savefig(FIGURES_DIR / "KernelSHAP_summary_plot.png", bbox_inches="tight")
plt.show()

shap.summary_plot(shap_values, X_explain, plot_type="bar", show=False)
plt.savefig(FIGURES_DIR / "KernelSHAP_summary_bar_plot.png", bbox_inches="tight")
plt.show()
# %%
