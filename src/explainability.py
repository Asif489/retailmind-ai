import shap
import matplotlib.pyplot as plt
import numpy as np


# ==========================================
# CREATE SHAP VALUES
# ==========================================
def explain(model, X):

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Binary classifier output handling
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_values = np.array(shap_values)

    # If still 3D -> convert to 2D
    if len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 1]

    return explainer, shap_values, explainer.expected_value


# ==========================================
# WATERFALL (FIXED)
# ==========================================
def plot_waterfall(explainer, shap_values, X, index=0):

    # SINGLE CUSTOMER ONLY
    single_shap = np.array(shap_values[index]).reshape(-1)
    single_x = X.iloc[index]

    # Scalar expected value
    expected_value = explainer.expected_value

    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = np.array(expected_value).flatten()[-1]

    fig = plt.figure(figsize=(8,5))

    shap.plots._waterfall.waterfall_legacy(
        expected_value=expected_value,
        shap_values=single_shap,
        features=single_x,
        show=False
    )

    return fig


# ==========================================
# SUMMARY
# ==========================================
def plot_summary(shap_values, X):

    fig = plt.figure(figsize=(8,5))
    shap.summary_plot(shap_values, X, show=False)
    return fig


# ==========================================
# FEATURE IMPORTANCE
# ==========================================
def plot_feature_importance(shap_values, X):

    fig = plt.figure(figsize=(8,5))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    return fig


# ==========================================
# TOP FEATURES
# ==========================================
def get_top_features(shap_values, X, index=0):

    vals = shap_values[index]
    names = X.columns

    pairs = list(zip(names, vals))
    pairs = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)

    return pairs[:3]