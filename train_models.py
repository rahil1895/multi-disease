import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from xgboost import XGBClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

def evaluate_model(name, model, X_test, y_test, multi_class=False):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='weighted')
    if multi_class:
        y_prob = model.predict_proba(X_test)
        auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
    else:
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    print(f"  {name}:")
    print(f"    Accuracy  : {acc*100:.2f}%")
    print(f"    F1 Score  : {f1*100:.2f}%")
    print(f"    ROC-AUC   : {auc*100:.2f}%")
    return acc, f1, auc

# ============================================================
# ❤️ HEART DISEASE
# ============================================================
def train_heart():
    print("Training Heart Disease models...")
    X_train, X_test, y_train, y_test = joblib.load('models/heart/heart_data.pkl')

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    joblib.dump(rf, 'models/heart/heart_rf.pkl')

    # XGBoost
    xgb = XGBClassifier(n_estimators=100, random_state=42,
                         use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    joblib.dump(xgb, 'models/heart/heart_xgb.pkl')

    print("  Results:")
    rf_scores  = evaluate_model("Random Forest", rf, X_test, y_test)
    xgb_scores = evaluate_model("XGBoost      ", xgb, X_test, y_test)
    print(f"✅ Heart models saved!\n")
    return rf_scores, xgb_scores

# ============================================================
# 🩸 DIABETES
# ============================================================
def train_diabetes():
    print("Training Diabetes models...")
    X_train, X_test, y_train, y_test = joblib.load(
        'models/diabetes/diabetes_data.pkl')

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    joblib.dump(rf, 'models/diabetes/diabetes_rf.pkl')

    # XGBoost
    xgb = XGBClassifier(
        n_estimators=100, random_state=42,
        eval_metric='logloss')
    xgb.fit(X_train, y_train)
    joblib.dump(xgb, 'models/diabetes/diabetes_xgb.pkl')

    print("  Results:")
    # Binary class now — not multi_class
    rf_scores  = evaluate_model(
        "Random Forest", rf, X_test, y_test,
        multi_class=False)
    xgb_scores = evaluate_model(
        "XGBoost      ", xgb, X_test, y_test,
        multi_class=False)
    print(f"✅ Diabetes models saved!\n")
    return rf_scores, xgb_scores

# ============================================================
# 🫀 LIVER DISEASE
# ============================================================
def train_liver():
    print("Training Liver Disease models...")
    X_train, X_test, y_train, y_test = joblib.load(
        'models/liver/liver_data.pkl')

    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    rf.fit(X_train, y_train)
    joblib.dump(rf, 'models/liver/liver_rf.pkl')

    xgb = XGBClassifier(
        n_estimators=200,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        learning_rate=0.1,
        max_depth=6
    )
    xgb.fit(X_train, y_train)
    joblib.dump(xgb, 'models/liver/liver_xgb.pkl')

    print("  Results:")
    rf_scores  = evaluate_model(
        "Random Forest", rf, X_test, y_test)
    xgb_scores = evaluate_model(
        "XGBoost      ", xgb, X_test, y_test)
    print(f"✅ Liver models saved!\n")
    return rf_scores, xgb_scores

# ============================================================
# 🫘 KIDNEY DISEASE
# ============================================================
def train_kidney():
    print("Training Kidney Disease models...")
    X_train, X_test, y_train, y_test = joblib.load('models/kidney/kidney_data.pkl')

    # Random Forest with balanced class weight
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    rf.fit(X_train, y_train)
    joblib.dump(rf, 'models/kidney/kidney_rf.pkl')

    # XGBoost
    xgb = XGBClassifier(
        n_estimators=200,
        random_state=42,
        eval_metric='mlogloss',
        learning_rate=0.1,
        max_depth=6
    )
    xgb.fit(X_train, y_train)
    joblib.dump(xgb, 'models/kidney/kidney_xgb.pkl')

    print("  Results:")
    rf_scores  = evaluate_model("Random Forest", rf, X_test, y_test, multi_class=True)
    xgb_scores = evaluate_model("XGBoost      ", xgb, X_test, y_test, multi_class=True)
    print(f"✅ Kidney models saved!\n")
    return rf_scores, xgb_scores

# ============================================================
# RUN ALL
# ============================================================
if __name__ == '__main__':
    print("🚀 Starting model training for all 4 diseases...\n")
    print("=" * 55)
    train_heart()
    print("=" * 55)
    train_diabetes()
    print("=" * 55)
    train_liver()
    print("=" * 55)
    train_kidney()
    print("=" * 55)
    print("🎉 All models trained and saved successfully!")