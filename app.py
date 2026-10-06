from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import shap
import lime
import lime.lime_tabular
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
import io
import warnings
warnings.filterwarnings('ignore')
from database.db import (save_prediction, get_recent_predictions,
                         get_prediction_counts)

app = Flask(__name__)
app.jinja_env.globals.update(zip=zip)

# ============================================================
# LOAD ALL MODELS & FEATURES
# ============================================================
models = {
    'heart'   : joblib.load('models/heart/heart_xgb.pkl'),
    'diabetes': joblib.load('models/diabetes/diabetes_rf.pkl'),
    'liver'   : joblib.load('models/liver/liver_rf.pkl'),
    'kidney'  : joblib.load('models/kidney/kidney_rf.pkl'),
}
features = {
    'heart'   : joblib.load('models/heart/heart_features.pkl'),
    'diabetes': joblib.load('models/diabetes/diabetes_features.pkl'),
    'liver'   : joblib.load('models/liver/liver_features.pkl'),
    'kidney'  : joblib.load('models/kidney/kidney_features.pkl'),
}
train_data = {
    'heart'   : joblib.load('models/heart/heart_data.pkl')[0],
    'diabetes': joblib.load('models/diabetes/diabetes_data.pkl')[0],
    'liver'   : joblib.load('models/liver/liver_data.pkl')[0],
    'kidney'  : joblib.load('models/kidney/kidney_data.pkl')[0],
}

# ============================================================
# LABEL MAPS
# ============================================================
label_maps = {
    'heart'   : {0: 'No Disease', 1: 'Disease Detected'},
    'diabetes': {0: 'No Diabetes', 1: 'Diabetes Detected'},
    'liver'   : {0: 'No Disease', 1: 'Disease Detected'},
    'kidney'  : {
        0: 'High Risk', 1: 'Low Risk',
        2: 'Moderate Risk', 3: 'No Disease',
        4: 'Severe Disease'
    },
}
class_names = {
    'heart'   : ['No Disease', 'Disease Detected'],
    'diabetes': ['No Diabetes', 'Diabetes Detected'],
    'liver'   : ['No Disease', 'Disease Detected'],
    'kidney'  : [
        'High Risk', 'Low Risk', 'Moderate Risk',
        'No Disease', 'Severe Disease'
    ],
}

# ============================================================
# HUMAN READABLE FEATURE NAMES
# ============================================================
human_names = {
    'heart': {
        'age'        : 'Age',
        'gender'     : 'Gender',
        'sys_bp'     : 'Systolic Blood Pressure',
        'dia_bp'     : 'Diastolic Blood Pressure',
        'cholesterol': 'Cholesterol (mg/dL)',
        'glucose'    : 'Glucose (mg/dL)',
        'smoke'      : 'Smoking',
        'alco'       : 'Alcohol Intake',
        'active'     : 'Physical Activity',
    },
    'diabetes': {
        'gender'             : 'Gender',
        'age'                : 'Age',
        'hypertension'       : 'Hypertension',
        'heart_disease'      : 'Heart Disease History',
        'smoking_history'    : 'Smoking History',
        'bmi'                : 'BMI',
        'HbA1c_level'        : 'HbA1c Level',
        'blood_glucose_level': 'Blood Glucose (mg/dL)',
    },
    'liver': {
        'Age'                       : 'Age',
        'Gender'                    : 'Gender',
        'Total_Bilirubin'           : 'Total Bilirubin',
        'Direct_Bilirubin'          : 'Direct Bilirubin',
        'Alkaline_Phosphotase'      : 'Alkaline Phosphotase',
        'Alamine_Aminotransferase'  : 'SGPT (ALT)',
        'Aspartate_Aminotransferase': 'SGOT (AST)',
        'Total_Proteins'            : 'Total Proteins',
        'Albumin'                   : 'Albumin',
        'Albumin_Globulin_Ratio'    : 'A/G Ratio',
    },
    'kidney': {
        'Age'               : 'Age',
        'Blood_Pressure'    : 'Blood Pressure',
        'Specific_Gravity'  : 'Specific Gravity',
        'Albumin'           : 'Albumin in Urine',
        'Sugar'             : 'Sugar in Urine',
        'Red_Blood_Cells'   : 'Red Blood Cells',
        'Pus_Cells'         : 'Pus Cells',
        'Pus_Cell_Clumps'   : 'Pus Cell Clumps',
        'Bacteria'          : 'Bacteria',
        'Blood_Glucose'     : 'Blood Glucose',
        'Blood_Urea'        : 'Blood Urea',
        'Serum_Creatinine'  : 'Serum Creatinine',
        'Sodium'            : 'Sodium',
        'Potassium'         : 'Potassium',
        'Haemoglobin'       : 'Haemoglobin',
        'Packed_Cell_Volume': 'Packed Cell Volume',
        'WBC_Count'         : 'WBC Count',
        'RBC_Count'         : 'RBC Count',
        'Hypertension'      : 'Hypertension',
        'Diabetes_Mellitus' : 'Diabetes',
        'Coronary_Artery'   : 'Coronary Artery Disease',
        'Appetite'          : 'Appetite',
        'Pedal_Edema'       : 'Pedal Edema',
        'Anaemia'           : 'Anaemia',
        'eGFR'              : 'Kidney Filtration Rate',
        'Urine_Protein'     : 'Protein in Urine',
        'Urine_Output'      : 'Urine Output',
        'Serum_Albumin'     : 'Serum Albumin',
        'Cholesterol'       : 'Cholesterol',
        'PTH_Level'         : 'PTH Level',
        'Serum_Calcium'     : 'Calcium Level',
        'Serum_Phosphate'   : 'Phosphate Level',
        'Family_History'    : 'Family History',
        'Smoking'           : 'Smoking',
        'BMI'               : 'BMI',
        'Physical_Activity' : 'Physical Activity',
        'Diabetes_Duration' : 'Years with Diabetes',
        'Hypertension_Duration': 'Years with Hypertension',
        'Cystatin_C'        : 'Cystatin C',
        'Urinary_Sediment'  : 'Urinary Sediment',
        'CRP_Level'         : 'CRP Level',
        'IL6_Level'         : 'IL-6 Level',
    }
}

# ============================================================
# SMART CONVERSION FUNCTIONS
# ============================================================
def convert_heart_inputs(data):
    """Convert patient friendly inputs to model format"""

    # Age
    age = int(data['age'])

    # Gender
    gender = int(data['gender'])

    # Blood Pressure — actual values
    sys_bp = int(data['sys_bp'])
    dia_bp = int(data['dia_bp'])

    # Cholesterol — actual mg/dL value
    chol_value = float(data['cholesterol'])
    if chol_value < 200:
        cholesterol = 180   # Normal range
    elif chol_value < 240:
        cholesterol = 220   # Above normal
    else:
        cholesterol = 280   # Well above normal

    # Glucose — actual mg/dL value
    gluc_value = float(data['glucose'])
    if gluc_value < 100:
        glucose = 85    # Normal
    elif gluc_value < 126:
        glucose = 110   # Above normal
    else:
        glucose = 160   # Well above normal

    # Smoking — cigarettes per week
    cigs_per_week = int(data.get('cigs_per_week', 0))
    smoke = 1 if cigs_per_week > 0 else 0

    # Alcohol — units per week
    alcohol_units = int(data.get('alcohol_units', 0))
    alco = 1 if alcohol_units > 14 else 0

    # Physical activity — minutes per week
    exercise_mins = int(data.get('exercise_mins', 0))
    active = 1 if exercise_mins >= 150 else 0

    return [age, gender, sys_bp, dia_bp,
            cholesterol, glucose, smoke, alco, active]


def convert_diabetes_inputs(data):
    """Convert patient friendly inputs to model format"""

    # Gender
    gender = 1 if data['gender'] == 'Male' else 0

    # Age
    age = float(data['age'])

    # Hypertension
    hypertension = int(data['hypertension'])

    # Heart disease history
    heart_disease = int(data['heart_disease'])

    # Smoking history
    smoking_map = {
        'never'      : 0,
        'former'     : 1,
        'current'    : 2,
        'No Info'    : 0
    }
    smoking_history = smoking_map.get(
        data.get('smoking_history', 'never'), 0)

    # BMI — calculate from height and weight if provided
    if 'height' in data and 'weight' in data:
        height_m = float(data['height']) / 100
        weight_kg = float(data['weight'])
        bmi = round(weight_kg / (height_m ** 2), 2)
    else:
        bmi = float(data.get('bmi', 25.0))

    # HbA1c — actual value
    hba1c = float(data['HbA1c_level'])

    # Blood glucose — actual mg/dL
    blood_glucose = float(data['blood_glucose_level'])

    return [gender, age, hypertension, heart_disease,
            smoking_history, bmi, hba1c, blood_glucose]


def convert_liver_inputs(data):
    """Convert patient friendly inputs to model format"""

    age = int(data['Age'])
    gender = 1 if data['Gender'] == 'Male' else 0

    # Actual lab values
    total_bilirubin   = float(data['Total_Bilirubin'])
    direct_bilirubin  = float(data['Direct_Bilirubin'])
    alk_phosphotase   = float(data['Alkaline_Phosphotase'])
    alt               = float(data['Alamine_Aminotransferase'])
    ast               = float(data['Aspartate_Aminotransferase'])
    total_proteins    = float(data['Total_Proteins'])
    albumin           = float(data['Albumin'])
    ag_ratio          = float(data['Albumin_Globulin_Ratio'])

    return [age, gender, total_bilirubin, direct_bilirubin,
            alk_phosphotase, alt, ast,
            total_proteins, albumin, ag_ratio]


# ============================================================
# SYMPTOM RISK BOOSTER
# ============================================================
def calculate_symptom_risk(disease, data):
    """
    Calculate additional risk score from symptoms
    and patient history — these are not in the dataset
    but help give more accurate overall assessment
    """
    symptom_score = 0
    symptom_messages = []

    if disease == 'heart':
        if data.get('chest_pain') == 'yes':
            symptom_score += 3
            symptom_messages.append(
                "You have chest pain — this is a key warning sign of heart disease")
        if data.get('shortness_breath') == 'yes':
            symptom_score += 2
            symptom_messages.append(
                "You experience shortness of breath — this may indicate heart stress")
        if data.get('palpitations') == 'yes':
            symptom_score += 2
            symptom_messages.append(
                "You have irregular heartbeat — this needs immediate attention")
        if data.get('family_heart_history') == 'yes':
            symptom_score += 3
            symptom_messages.append(
                "You have family history of heart disease — genetic risk is significant")
        if data.get('prev_heart_attack') == 'yes':
            symptom_score += 4
            symptom_messages.append(
                "You have had a previous heart attack — this is a major risk factor")
        if data.get('swelling_legs') == 'yes':
            symptom_score += 1
            symptom_messages.append(
                "You have swelling in legs — this can indicate heart problems")

    elif disease == 'diabetes':
        if data.get('frequent_urination') == 'yes':
            symptom_score += 2
            symptom_messages.append(
                "You have frequent urination — this is a common diabetes symptom")
        if data.get('excessive_thirst') == 'yes':
            symptom_score += 2
            symptom_messages.append(
                "You have excessive thirst — this is a key diabetes warning sign")
        if data.get('blurred_vision') == 'yes':
            symptom_score += 2
            symptom_messages.append(
                "You have blurred vision — this can be caused by high blood sugar")
        if data.get('slow_healing') == 'yes':
            symptom_score += 2
            symptom_messages.append(
                "Your wounds heal slowly — this is a sign of diabetes")
        if data.get('family_diabetes') == 'yes':
            symptom_score += 3
            symptom_messages.append(
                "You have family history of diabetes — genetic risk is high")
        if data.get('weight_loss') == 'yes':
            symptom_score += 1
            symptom_messages.append(
                "You have unexplained weight loss — this can indicate diabetes")

    elif disease == 'liver':
        if data.get('jaundice') == 'yes':
            symptom_score += 4
            symptom_messages.append(
                "You have jaundice (yellowing of skin/eyes) — this directly indicates liver disease")
        if data.get('abdominal_pain') == 'yes':
            symptom_score += 3
            symptom_messages.append(
                "You have abdominal pain — this can indicate liver inflammation")
        if data.get('dark_urine') == 'yes':
            symptom_score += 2
            symptom_messages.append(
                "You have dark coloured urine — this is a sign of liver stress")
        if data.get('nausea') == 'yes':
            symptom_score += 1
            symptom_messages.append(
                "You have nausea — this is a common liver disease symptom")
        if data.get('family_liver') == 'yes':
            symptom_score += 2
            symptom_messages.append(
                "You have family history of liver disease — genetic risk is present")
        if data.get('prev_hepatitis') == 'yes':
            symptom_score += 4
            symptom_messages.append(
                "You have had hepatitis before — this significantly increases liver disease risk")

    elif disease == 'kidney':
        if data.get('blood_urine') == 'yes':
            symptom_score += 4
            symptom_messages.append(
                "You have blood in urine — this is a direct sign of kidney damage")
        if data.get('foamy_urine') == 'yes':
            symptom_score += 3
            symptom_messages.append(
                "You have foamy urine — this indicates protein leakage from kidneys")
        if data.get('back_pain') == 'yes':
            symptom_score += 2
            symptom_messages.append(
                "You have back pain near kidneys — this may indicate kidney stress")
        if data.get('frequent_night_urination') == 'yes':
            symptom_score += 2
            symptom_messages.append(
                "You urinate frequently at night — this is a kidney disease symptom")
        if data.get('family_kidney') == 'yes':
            symptom_score += 3
            symptom_messages.append(
                "You have family history of kidney disease — genetic risk is significant")
        if data.get('prev_kidney_stones') == 'yes':
            symptom_score += 2
            symptom_messages.append(
                "You have had kidney stones before — this increases kidney disease risk")

    # Determine risk level from symptoms
    if symptom_score >= 8:
        symptom_risk = "Critical"
    elif symptom_score >= 5:
        symptom_risk = "High"
    elif symptom_score >= 3:
        symptom_risk = "Moderate"
    elif symptom_score >= 1:
        symptom_risk = "Low"
    else:
        symptom_risk = "None"

    return symptom_score, symptom_risk, symptom_messages


# ============================================================
# GENERATE HUMAN READABLE REASONS FROM LIME
# ============================================================
def get_human_reasons(disease, instance, label, pred):
    try:
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.array(train_data[disease]),
            feature_names=features[disease],
            class_names=class_names[disease],
            mode='classification',
            random_state=42
        )
        explanation = explainer.explain_instance(
            np.array(instance),
            models[disease].predict_proba,
            num_features=10
        )
        exp_list = explanation.as_list(label=pred)
        feat_index = {f: i for i, f in enumerate(features[disease])}

        always_bad = {
            'heart'   : ['smoke', 'alco', 'sys_bp', 'dia_bp',
                         'cholesterol', 'glucose'],
            'diabetes': ['hypertension', 'heart_disease',
                         'HbA1c_level', 'blood_glucose_level'],
            'liver'   : ['Total_Bilirubin', 'Direct_Bilirubin',
                         'Alkaline_Phosphotase',
                         'Alamine_Aminotransferase',
                         'Aspartate_Aminotransferase'],
            'kidney'  : ['Blood_Pressure', 'Albumin', 'Sugar',
                         'Blood_Urea', 'Serum_Creatinine',
                         'Hypertension', 'Diabetes_Mellitus',
                         'CRP_Level', 'Anaemia', 'Pedal_Edema'],
        }

        risk_messages = {
            # HEART
            'sys_bp'     : lambda v: f"Your systolic BP is {v} mmHg — this is high and strains your heart" if v > 120 else None,
            'dia_bp'     : lambda v: f"Your diastolic BP is {v} mmHg — this is elevated above normal" if v > 80 else None,
            'cholesterol': lambda v: f"Your cholesterol is {v} mg/dL — high cholesterol blocks arteries" if v > 200 else None,
            'glucose'    : lambda v: f"Your glucose is {v} mg/dL — high glucose damages blood vessels" if v > 100 else None,
            'smoke'      : lambda v: "You are a smoker — smoking is one of the top causes of heart disease" if v == 1 else None,
            'alco'       : lambda v: "You consume heavy alcohol — alcohol weakens the heart muscle" if v == 1 else None,
            'active'     : lambda v: "You are physically inactive — regular exercise strengthens the heart" if v == 0 else None,
            'age'        : lambda v: f"Your age ({v} years) is a natural risk factor for heart disease" if v > 50 else None,
            # DIABETES
            'hypertension'       : lambda v: "You have high blood pressure — strongly linked to diabetes" if v == 1 else None,
            'heart_disease'      : lambda v: "You have history of heart disease — closely linked to diabetes" if v == 1 else None,
            'HbA1c_level'        : lambda v: f"Your HbA1c is {v}% — above 6.5% confirms diabetes" if v >= 6.5 else f"Your HbA1c is {v}% — between 5.7-6.4% indicates pre-diabetes risk" if v >= 5.7 else None,
            'blood_glucose_level': lambda v: f"Your blood glucose is {v} mg/dL — above 126 mg/dL indicates diabetes" if v >= 126 else None,
            'bmi'                : lambda v: f"Your BMI is {v} — being {'overweight' if v <= 30 else 'obese'} significantly increases diabetes risk" if v > 25 else None,
            'smoking_history'    : lambda v: "You are a current smoker — smoking causes insulin resistance" if v == 2 else "You were a former smoker — this still increases diabetes risk" if v == 1 else None,
            # LIVER
            'Total_Bilirubin'           : lambda v: f"Your total bilirubin is {v} mg/dL — elevated bilirubin indicates liver dysfunction" if v > 1.2 else None,
            'Direct_Bilirubin'          : lambda v: f"Your direct bilirubin is {v} mg/dL — this is elevated above normal" if v > 0.3 else None,
            'Alkaline_Phosphotase'      : lambda v: f"Your alkaline phosphotase is {v} — elevated levels indicate liver or bone disease" if v > 147 else None,
            'Alamine_Aminotransferase'  : lambda v: f"Your SGPT (ALT) is {v} — elevated ALT is a direct sign of liver damage" if v > 56 else None,
            'Aspartate_Aminotransferase': lambda v: f"Your SGOT (AST) is {v} — elevated AST indicates liver cell damage" if v > 40 else None,
            'Albumin'                   : lambda v: f"Your albumin is {v} g/dL — low albumin means liver is not making enough protein" if v < 3.5 else None,
            # KIDNEY
            'Blood_Pressure'  : lambda v: f"Your blood pressure is {v} mmHg — high BP is the leading cause of kidney damage" if v > 90 else None,
            'Albumin'         : lambda v: f"Albumin level {v} in urine — protein in urine indicates kidney damage" if v > 0 else None,
            'Blood_Glucose'   : lambda v: f"Your blood glucose is {v} mg/dL — high glucose destroys kidney filters" if v > 140 else None,
            'Blood_Urea'      : lambda v: f"Your blood urea is {v} mg/dL — elevated urea means kidneys are not filtering properly" if v > 40 else None,
            'Serum_Creatinine': lambda v: f"Your creatinine is {v} mg/dL — high creatinine is a direct sign of kidney damage" if v > 1.2 else None,
            'Haemoglobin'     : lambda v: f"Your haemoglobin is {v} gms — low haemoglobin is common in kidney disease" if v < 12 else None,
            'Hypertension'    : lambda v: "You have hypertension — the leading cause of chronic kidney disease" if v == 1 else None,
            'Diabetes_Mellitus': lambda v: "You have diabetes — diabetes is the most common cause of kidney disease" if v == 1 else None,
        }

        good_messages = {
            'heart': {
                'active': lambda v: "You are physically active — exercise is one of the best protections against heart disease" if v == 1 else None,
                'smoke' : lambda v: "You are a non-smoker — this significantly reduces your heart disease risk" if v == 0 else None,
                'alco'  : lambda v: "You do not drink heavily — this is good for your heart health" if v == 0 else None,
            },
            'diabetes': {
                'hypertension'       : lambda v: "You do not have high blood pressure — this reduces your diabetes risk" if v == 0 else None,
                'HbA1c_level'        : lambda v: f"Your HbA1c is {v}% — this is within normal range" if v < 5.7 else None,
                'blood_glucose_level': lambda v: f"Your blood glucose is {v} mg/dL — this is normal" if v < 100 else None,
            },
            'liver': {
                'Total_Bilirubin': lambda v: f"Your total bilirubin is {v} mg/dL — this is within normal range" if v <= 1.2 else None,
                'Albumin'        : lambda v: f"Your albumin is {v} g/dL — this is normal" if v >= 3.5 else None,
            },
            'kidney': {
                'Haemoglobin'     : lambda v: f"Your haemoglobin is {v} gms — this is normal and healthy" if v >= 12 else None,
                'Hypertension'    : lambda v: "You do not have hypertension — this protects your kidneys" if v == 0 else None,
                'Diabetes_Mellitus': lambda v: "You do not have diabetes — this reduces kidney disease risk" if v == 0 else None,
            }
        }

        is_disease = ('No Disease' not in label and
                      'No Diabetes' not in label)
        reasons_risk = []
        reasons_good = []
        seen_risk = set()
        seen_good = set()

        for feature_condition, weight in exp_list:
            for feat_key in features[disease]:
                if (feat_key.lower() in feature_condition.lower()
                        and feat_key not in seen_risk
                        and feat_key not in seen_good):
                    val = instance[feat_index[feat_key]]
                    ab = always_bad.get(disease, [])

                    if feat_key in ab or weight > 0.02:
                        if feat_key in risk_messages:
                            msg = risk_messages[feat_key](val)
                            if msg and feat_key not in seen_risk:
                                reasons_risk.append(msg)
                                seen_risk.add(feat_key)
                    elif weight < -0.02:
                        gm = good_messages.get(disease, {})
                        if feat_key in gm:
                            msg = gm[feat_key](val)
                            if msg and feat_key not in seen_good:
                                reasons_good.append(msg)
                                seen_good.add(feat_key)
                    break

        return reasons_risk[:5], reasons_good[:3]

    except Exception as e:
        print(f"Reason error: {e}")
        return [], []


# ============================================================
# SHAP CHART
# ============================================================
def get_shap_chart(disease, instance):
    try:
        explainer = shap.TreeExplainer(models[disease])
        df_instance = pd.DataFrame(
            [instance], columns=features[disease])
        shap_vals = explainer.shap_values(df_instance)

        if isinstance(shap_vals, list):
            probs = models[disease].predict_proba(df_instance)[0]
            best_class = int(np.argmax(probs))
            sv = shap_vals[best_class]
        elif len(np.array(shap_vals).shape) == 3:
            probs = models[disease].predict_proba(df_instance)[0]
            best_class = int(np.argmax(probs))
            sv = np.array(shap_vals)[0, :, best_class].reshape(1, -1)
        else:
            sv = shap_vals

        sv = np.array(sv)
        if len(sv.shape) == 1:
            sv = sv.reshape(1, -1)

        feat_labels = [
            human_names[disease].get(f, f)
            for f in features[disease]
        ]
        shap_df = pd.DataFrame({
            'feature': feat_labels,
            'value'  : np.abs(sv[0])
        }).sort_values('value', ascending=True).tail(10)

        plt.figure(figsize=(10, 6))
        plt.barh(shap_df['feature'], shap_df['value'],
                 color='#2C7BE5')
        plt.xlabel('Importance Score', fontsize=12)
        plt.title('Which factors influenced this prediction?',
                  fontsize=13)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120,
                    bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
    except Exception as e:
        print(f"SHAP error: {e}")
        return None


# ============================================================
# LIME CHART
# ============================================================
def get_lime_chart(disease, instance):
    try:
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.array(train_data[disease]),
            feature_names=[
                human_names[disease].get(f, f)
                for f in features[disease]
            ],
            class_names=class_names[disease],
            mode='classification',
            random_state=42
        )
        explanation = explainer.explain_instance(
            np.array(instance),
            models[disease].predict_proba,
            num_features=10
        )
        fig = explanation.as_pyplot_figure()
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120,
                    bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
    except Exception as e:
        print(f"LIME error: {e}")
        return None


# ============================================================
# ADVICE
# ============================================================
def get_advice(disease, label):
    advice = {
        'heart': {
            'Disease Detected': [
                'Consult a cardiologist immediately',
                'Monitor your blood pressure daily',
                'Reduce salt and fatty food intake',
                'Avoid smoking and alcohol completely',
                'Do light exercise with doctor guidance'
            ],
            'No Disease': [
                'Maintain a healthy diet rich in vegetables',
                'Exercise at least 30 minutes daily',
                'Get regular heart checkups once a year',
                'Avoid stress, smoking and excess alcohol',
                'Monitor blood pressure periodically'
            ]
        },
        'diabetes': {
            'Diabetes Detected': [
                'Consult an endocrinologist immediately',
                'Monitor blood sugar levels daily',
                'Follow a low sugar and low carb diet',
                'Exercise regularly — at least 30 mins per day',
                'Take prescribed medications on time'
            ],
            'No Diabetes': [
                'Maintain healthy eating habits',
                'Stay physically active every day',
                'Avoid sugary drinks and junk food',
                'Get annual blood sugar checkup',
                'Maintain a healthy body weight'
            ]
        },
        'liver': {
            'Disease Detected': [
                'Consult a hepatologist immediately',
                'Stop alcohol consumption completely',
                'Avoid fatty and oily foods',
                'Get liver function tests done urgently',
                'Take prescribed medications regularly'
            ],
            'No Disease': [
                'Limit alcohol to safe levels',
                'Maintain a healthy weight',
                'Eat plenty of fruits and vegetables',
                'Stay well hydrated every day',
                'Get annual liver checkup'
            ]
        },
        'kidney': {
            'Severe Disease': [
                'Seek immediate medical attention today',
                'Follow strict low protein diet',
                'Monitor fluid intake very carefully',
                'Take all prescribed medications',
                'Discuss dialysis options with your doctor'
            ],
            'High Risk': [
                'Consult a nephrologist urgently',
                'Control blood pressure and diabetes strictly',
                'Reduce salt and protein intake',
                'Stay well hydrated daily',
                'Get kidney function tests monthly'
            ],
            'Moderate Risk': [
                'See your doctor within a week',
                'Control blood sugar and blood pressure',
                'Reduce salt intake significantly',
                'Exercise moderately every day',
                'Get regular kidney checkups'
            ],
            'Low Risk': [
                'Monitor kidney health regularly',
                'Stay well hydrated every day',
                'Maintain healthy blood pressure',
                'Reduce salt and processed food intake',
                'Annual kidney function test recommended'
            ],
            'No Disease': [
                'Stay well hydrated — drink 8 glasses daily',
                'Maintain healthy blood pressure',
                'Eat a balanced and nutritious diet',
                'Exercise regularly for overall health',
                'Annual health checkup recommended'
            ]
        }
    }
    return advice.get(disease, {}).get(label, [])


# ============================================================
# ROUTES
# ============================================================
@app.route('/')
def home():
    counts = get_prediction_counts()
    return render_template('index.html', counts=counts)

@app.route('/heart')
def heart():
    return render_template('heart.html')

@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')

@app.route('/liver')
def liver():
    return render_template('liver.html')

@app.route('/kidney')
def kidney():
    return render_template('kidney.html')

@app.route('/history')
def history():
    records = get_recent_predictions(20)
    return render_template('history.html', records=records)

@app.route('/dashboard')
def dashboard():
    counts = get_prediction_counts()
    records = get_recent_predictions(5)
    return render_template('dashboard.html',
                           counts=counts, records=records)


# ============================================================
# PREDICT HEART
# ============================================================
@app.route('/predict/heart', methods=['POST'])
def predict_heart():
    try:
        data = request.form
        instance = convert_heart_inputs(data)
        df = pd.DataFrame([instance], columns=features['heart'])

        pred = int(models['heart'].predict(df)[0])
        prob = models['heart'].predict_proba(df)[0]
        confidence = round(float(max(prob)) * 100, 2)
        label = label_maps['heart'][pred]

        # Get symptom risk from patient history
        symptom_score, symptom_risk, symptom_messages = \
            calculate_symptom_risk('heart', data)

        shap_chart = get_shap_chart('heart', instance)
        lime_chart = get_lime_chart('heart', instance)
        reasons_risk, reasons_good = get_human_reasons(
            'heart', instance, label, pred)

        # Add symptom messages to risk reasons
        reasons_risk = symptom_messages + reasons_risk

        advice = get_advice('heart', label)

        explanation = ', '.join(reasons_risk[:2]) \
            if reasons_risk else label
        save_prediction('heart', dict(data), label,
                        confidence, explanation)

        return render_template('result.html',
            disease='Heart Disease',
            label=label,
            confidence=confidence,
            symptom_score=symptom_score,
            symptom_risk=symptom_risk,
            shap_chart=shap_chart,
            lime_chart=lime_chart,
            features=features['heart'],
            instance=instance,
            pred=pred,
            reasons_risk=reasons_risk[:6],
            reasons_good=reasons_good,
            advice=advice,
            human_names=human_names['heart'])
    except Exception as e:
        return render_template('error.html', error=str(e))


# ============================================================
# PREDICT DIABETES
# ============================================================
@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes():
    try:
        data = request.form
        instance = convert_diabetes_inputs(data)
        df = pd.DataFrame([instance],
                          columns=features['diabetes'])

        pred = int(models['diabetes'].predict(df)[0])
        prob = models['diabetes'].predict_proba(df)[0]
        confidence = round(float(max(prob)) * 100, 2)
        label = label_maps['diabetes'][pred]

        symptom_score, symptom_risk, symptom_messages = \
            calculate_symptom_risk('diabetes', data)

        shap_chart = get_shap_chart('diabetes', instance)
        lime_chart = get_lime_chart('diabetes', instance)
        reasons_risk, reasons_good = get_human_reasons(
            'diabetes', instance, label, pred)

        reasons_risk = symptom_messages + reasons_risk

        advice = get_advice('diabetes', label)

        explanation = ', '.join(reasons_risk[:2]) \
            if reasons_risk else label
        save_prediction('diabetes', dict(data), label,
                        confidence, explanation)

        return render_template('result.html',
            disease='Diabetes',
            label=label,
            confidence=confidence,
            symptom_score=symptom_score,
            symptom_risk=symptom_risk,
            shap_chart=shap_chart,
            lime_chart=lime_chart,
            features=features['diabetes'],
            instance=instance,
            pred=pred,
            reasons_risk=reasons_risk[:6],
            reasons_good=reasons_good,
            advice=advice,
            human_names=human_names['diabetes'])
    except Exception as e:
        return render_template('error.html', error=str(e))


# ============================================================
# PREDICT LIVER
# ============================================================
@app.route('/predict/liver', methods=['POST'])
def predict_liver():
    try:
        data = request.form
        instance = convert_liver_inputs(data)
        df = pd.DataFrame([instance], columns=features['liver'])

        pred = int(models['liver'].predict(df)[0])
        prob = models['liver'].predict_proba(df)[0]
        confidence = round(float(max(prob)) * 100, 2)
        label = label_maps['liver'][pred]

        symptom_score, symptom_risk, symptom_messages = \
            calculate_symptom_risk('liver', data)

        shap_chart = get_shap_chart('liver', instance)
        lime_chart = get_lime_chart('liver', instance)
        reasons_risk, reasons_good = get_human_reasons(
            'liver', instance, label, pred)

        reasons_risk = symptom_messages + reasons_risk

        advice = get_advice('liver', label)

        explanation = ', '.join(reasons_risk[:2]) \
            if reasons_risk else label
        save_prediction('liver', dict(data), label,
                        confidence, explanation)

        return render_template('result.html',
            disease='Liver Disease',
            label=label,
            confidence=confidence,
            symptom_score=symptom_score,
            symptom_risk=symptom_risk,
            shap_chart=shap_chart,
            lime_chart=lime_chart,
            features=features['liver'],
            instance=instance,
            pred=pred,
            reasons_risk=reasons_risk[:6],
            reasons_good=reasons_good,
            advice=advice,
            human_names=human_names['liver'])
    except Exception as e:
        return render_template('error.html', error=str(e))


# ============================================================
# PREDICT KIDNEY
# ============================================================
@app.route('/predict/kidney', methods=['POST'])
def predict_kidney():
    try:
        data = request.form
        encoders = joblib.load(
            'models/kidney/kidney_encoders.pkl')

        instance = [
            int(data['Age']),
            int(data['Blood_Pressure']),
            float(data['Specific_Gravity']),
            int(data['Albumin']),
            int(data['Sugar']),
            int(encoders['Red_Blood_Cells'].transform(
                [data['Red_Blood_Cells']])[0]),
            int(encoders['Pus_Cells'].transform(
                [data['Pus_Cells']])[0]),
            int(encoders['Pus_Cell_Clumps'].transform(
                [data['Pus_Cell_Clumps']])[0]),
            int(encoders['Bacteria'].transform(
                [data['Bacteria']])[0]),
            int(data['Blood_Glucose']),
            float(data['Blood_Urea']),
            float(data['Serum_Creatinine']),
            float(data['Sodium']),
            float(data['Potassium']),
            float(data['Haemoglobin']),
            int(data['Packed_Cell_Volume']),
            int(data['WBC_Count']),
            float(data['RBC_Count']),
            int(encoders['Hypertension'].transform(
                [data['Hypertension']])[0]),
            int(encoders['Diabetes_Mellitus'].transform(
                [data['Diabetes_Mellitus']])[0]),
            int(encoders['Coronary_Artery'].transform(
                [data['Coronary_Artery']])[0]),
            int(encoders['Appetite'].transform(
                [data['Appetite']])[0]),
            int(encoders['Pedal_Edema'].transform(
                [data['Pedal_Edema']])[0]),
            int(encoders['Anaemia'].transform(
                [data['Anaemia']])[0]),
            float(data['eGFR']),
            float(data['Urine_Protein']),
            int(data['Urine_Output']),
            float(data['Serum_Albumin']),
            int(data['Cholesterol']),
            float(data['PTH_Level']),
            float(data['Serum_Calcium']),
            float(data['Serum_Phosphate']),
            int(encoders['Family_History'].transform(
                [data['Family_History']])[0]),
            int(encoders['Smoking'].transform(
                [data['Smoking']])[0]),
            float(data['BMI']),
            int(encoders['Physical_Activity'].transform(
                [data['Physical_Activity']])[0]),
            int(data['Diabetes_Duration']),
            int(data['Hypertension_Duration']),
            float(data['Cystatin_C']),
            int(encoders['Urinary_Sediment'].transform(
                [data['Urinary_Sediment']])[0]),
            float(data['CRP_Level']),
            float(data['IL6_Level']),
        ]

        df = pd.DataFrame([instance],
                          columns=features['kidney'])
        pred = int(models['kidney'].predict(df)[0])
        prob = models['kidney'].predict_proba(df)[0]
        confidence = round(float(max(prob)) * 100, 2)
        label = label_maps['kidney'][pred]

        symptom_score, symptom_risk, symptom_messages = \
            calculate_symptom_risk('kidney', data)

        shap_chart = get_shap_chart('kidney', instance)
        lime_chart = get_lime_chart('kidney', instance)
        reasons_risk, reasons_good = get_human_reasons(
            'kidney', instance, label, pred)

        reasons_risk = symptom_messages + reasons_risk

        advice = get_advice('kidney', label)

        explanation = ', '.join(reasons_risk[:2]) \
            if reasons_risk else label
        save_prediction('kidney', dict(data), label,
                        confidence, explanation)

        return render_template('result.html',
            disease='Kidney Disease',
            label=label,
            confidence=confidence,
            symptom_score=symptom_score,
            symptom_risk=symptom_risk,
            shap_chart=shap_chart,
            lime_chart=lime_chart,
            features=features['kidney'],
            instance=instance,
            pred=pred,
            reasons_risk=reasons_risk[:6],
            reasons_good=reasons_good,
            advice=advice,
            human_names=human_names['kidney'])
    except Exception as e:
        return render_template('error.html', error=str(e))


# ============================================================
# RUN
# ============================================================
if __name__ == '__main__':
    app.run(debug=True)