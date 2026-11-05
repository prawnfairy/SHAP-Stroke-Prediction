# %% Imports
from utils import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
import shap
# Added libraries
import numpy as np 
import matplotlib.pyplot as plt

# %% Load and preprocess data
data_loader = DataLoader()
data_loader.load_dataset()
data_loader.preprocess_data()
# Split the data for evaluation
X_train, X_test, y_train, y_test = data_loader.get_data_split()
# Oversample the train data
X_train, y_train = data_loader.oversample(X_train, y_train)
print(X_train.shape)
print(X_test.shape)

# %% Fit blackbox model
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(f"F1 Score {f1_score(y_test, y_pred, average='macro')}")
print(f"Accuracy {accuracy_score(y_test, y_pred)}")

# %% Create SHAP explainer
explainer = shap.TreeExplainer(rf)

# TASK 2: Added code
# Utility functions: extract class 1 (stroke) SHAP values and base value,
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

# Calculate shapley values for test data
X_explain = X_test.iloc[:500] # 500 rows to keep the computation time reasonable
raw_shap_values = explainer.shap_values(X_explain)
shap_values = class1_values(raw_shap_values)
X_explain.head()

# %% Investigating the values (classification problem)
# Raw output: one set of values per class
# class 0 = no stroke, class 1 = stroke
print("Raw SHAP output:", type(raw_shap_values), raw_shap_values.shape)
print("Stroke-class SHAP values:", shap_values.shape)                   
print("Base value (stroke):", base_value)

# %% >> Visualize local predictions
# TASK 3: Visualize local Explainer using Force plot
shap.initjs()
# Force plot
prediction = rf.predict(X_explain)

stroke_index = np.where(prediction == 1)[0]
patient = int(stroke_index[0]) if len(stroke_index) > 0 else 0 # pick the first patient predicted as stroke
print(f"The RF predicted: {prediction[patient]}")

force = shap.force_plot(base_value,
                        shap_values[patient],
                        X_explain.iloc[patient])
force

# Save Force plot as image
shap.force_plot(base_value,
                shap_values[patient],
                X_explain.iloc[patient],
                matplotlib=True,
                show=False)
plt.savefig("force_plot.png", bbox_inches="tight")

# %% >> Visualize global features
# TASK 4: Visualize global Explainer using Summary plot
# Feature summary
shap.summary_plot(shap_values, X_explain, show=False)
plt.savefig("summary_plot.png", bbox_inches="tight")
plt.show()

# Bar version (mean |SHAP|) for slides
shap.summary_plot(shap_values, X_explain, plot_type="bar", show=False)
plt.savefig("summary_bar_plot.png", bbox_inches="tight")


# %%
