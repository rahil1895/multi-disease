import pandas as pd
import numpy as np
import shap
import lime
import lime.lime_tabular
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Create output folders for charts
os.makedirs('static/images/shap', exist_ok=True)
os.makedirs('static/images/lime', exist_ok=True)

# ============================================================
# HELPER — SHAP Summary Plot
# ============================================================
def generate_shap(disease, model, X_train, X_test, feature_names):
    print(f"  Generating SHAP for {disease}...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test[:200])

    # Handle multi-class (use class 1 or first class)
    if isinstance(shap_values, list):
        sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        sv = shap_values

    # Summary bar plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(sv, X_test[:200],
                      feature_names=feature_names,
                      plot_type='bar', show=False)
    plt.title(f'SHAP Feature Importance — {disease.title()}',
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'static/images/shap/{disease}_shap_bar.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    # Summary dot plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(sv, X_test[:200],
                      feature_names=feature_names,
                      show=False)
    plt.title(f'SHAP Summary Plot — {disease.title()}',
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'static/images/shap/{disease}_shap_dot.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✅ SHAP charts saved for {disease}")
    return explainer, shap_values


# ============================================================
# HELPER — LIME Explanation
# ============================================================
def generate_lime(disease, model, X_train, X_test,
                  feature_names, class_names, sample_idx=0):
    print(f"  Generating LIME for {disease}...")

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=feature_names,
        class_names=class_names,
        mode='classification',
        random_state=42
    )

    # Explain a sample patient
    instance = np.array(X_test)[sample_idx]
    explanation = explainer.explain_instance(
        instance,
        model.predict_proba,
        num_features=10
    )

    # Save LIME plot
    fig = explanation.as_pyplot_figure()
    fig.suptitle(f'LIME Explanation — {disease.title()} '
                 f'(Sample Patient #{sample_idx})',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'static/images/lime/{disease}_lime.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✅ LIME chart saved for {disease}")
    return explainer


# ============================================================
# ❤️ HEART
# ============================================================
def explain_heart():
    print("\n❤️  Explaining Heart Disease...")
    X_train, X_test, y_train, y_test = joblib.load(
        'models/heart/heart_data.pkl')
    model = joblib.load('models/heart/heart_xgb.pkl')
    features = joblib.load('models/heart/heart_features.pkl')

    explainer, shap_values = generate_shap(
        'heart', model, X_train, X_test, features)
    lime_exp = generate_lime(
        'heart', model, X_train, X_test, features,
        class_names=['No Disease', 'Disease'])

    joblib.dump(explainer, 'models/heart/heart_shap_explainer.pkl')
    print("✅ Heart explainability done!")


# ============================================================
# 🩸 DIABETES
# ============================================================
def explain_diabetes():
    print("\n🩸  Explaining Diabetes...")
    X_train, X_test, y_train, y_test = joblib.load(
        'models/diabetes/diabetes_data.pkl')
    model = joblib.load('models/diabetes/diabetes_xgb.pkl')
    features = joblib.load('models/diabetes/diabetes_features.pkl')

    explainer, shap_values = generate_shap(
        'diabetes', model, X_train, X_test, features)
    lime_exp = generate_lime(
        'diabetes', model, X_train, X_test, features,
        class_names=['No Diabetes', 'Pre-Diabetes', 'Diabetes'])

    joblib.dump(explainer, 'models/diabetes/diabetes_shap_explainer.pkl')
    print("✅ Diabetes explainability done!")


# ============================================================
# 🫀 LIVER
# ============================================================
def explain_liver():
    print("\n🫀  Explaining Liver Disease...")
    X_train, X_test, y_train, y_test = joblib.load(
        'models/liver/liver_data.pkl')
    model = joblib.load('models/liver/liver_rf.pkl')
    features = joblib.load('models/liver/liver_features.pkl')

    explainer, shap_values = generate_shap(
        'liver', model, X_train, X_test, features)
    lime_exp = generate_lime(
        'liver', model, X_train, X_test, features,
        class_names=['No Disease', 'Disease'])

    joblib.dump(explainer, 'models/liver/liver_shap_explainer.pkl')
    print("✅ Liver explainability done!")


# ============================================================
# 🫘 KIDNEY
# ============================================================
def explain_kidney():
    print("\n🫘  Explaining Kidney Disease...")
    X_train, X_test, y_train, y_test = joblib.load(
        'models/kidney/kidney_data.pkl')
    model = joblib.load('models/kidney/kidney_rf.pkl')
    features = joblib.load('models/kidney/kidney_features.pkl')

    explainer, shap_values = generate_shap(
        'kidney', model, X_train, X_test, features)
    lime_exp = generate_lime(
        'kidney', model, X_train, X_test, features,
        class_names=['High Risk', 'Low Risk',
                     'Moderate Risk', 'No Disease', 'Severe Disease'])

    joblib.dump(explainer, 'models/kidney/kidney_shap_explainer.pkl')
    print("✅ Kidney explainability done!")


# ============================================================
# RUN ALL
# ============================================================
if __name__ == '__main__':
    print("🚀 Generating SHAP & LIME explanations...\n")
    print("=" * 55)
    explain_heart()
    print("=" * 55)
    explain_diabetes()
    print("=" * 55)
    explain_liver()
    print("=" * 55)
    explain_kidney()
    print("=" * 55)
    print("\n🎉 All SHAP & LIME explanations generated!")
    print("📁 SHAP charts saved in static/images/shap/")
    print("📁 LIME charts saved in static/images/lime/")