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
from database.db import save_prediction, get_recent_predictions, get_prediction_counts

app = Flask(__name__)
app.jinja_env.globals.update(zip=zip)

# ============================================================
# LOAD ALL MODELS & FEATURES
# ============================================================
models = {
    'heart'   : joblib.load('models/heart/heart_xgb.pkl'),
    'diabetes': joblib.load('models/diabetes/diabetes_xgb.pkl'),
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
    'diabetes': {0: 'No Diabetes', 1: 'Pre-Diabetes', 2: 'Diabetes'},
    'liver'   : {0: 'No Disease', 1: 'Disease Detected'},
    'kidney'  : {0: 'High Risk', 1: 'Low Risk',
                 2: 'Moderate Risk', 3: 'No Disease', 4: 'Severe Disease'},
}
class_names = {
    'heart'   : ['No Disease', 'Disease Detected'],
    'diabetes': ['No Diabetes', 'Pre-Diabetes', 'Diabetes'],
    'liver'   : ['No Disease', 'Disease Detected'],
    'kidney'  : ['High Risk', 'Low Risk',
                 'Moderate Risk', 'No Disease', 'Severe Disease'],
}

# ============================================================
# HUMAN READABLE FEATURE NAMES
# ============================================================
human_names = {
    'heart': {
        'age': 'Age', 'gender': 'Gender', 'height': 'Height',
        'weight': 'Weight', 'ap_hi': 'Systolic Blood Pressure',
        'ap_lo': 'Diastolic Blood Pressure', 'cholesterol': 'Cholesterol Level',
        'gluc': 'Glucose Level', 'smoke': 'Smoking',
        'alco': 'Alcohol Intake', 'active': 'Physical Activity',
    },
    'diabetes': {
        'HighBP': 'High Blood Pressure', 'HighChol': 'High Cholesterol',
        'CholCheck': 'Cholesterol Check', 'BMI': 'BMI',
        'Smoker': 'Smoking', 'Stroke': 'Stroke History',
        'HeartDiseaseorAttack': 'Heart Disease History',
        'PhysActivity': 'Physical Activity', 'Fruits': 'Fruits Consumption',
        'Veggies': 'Vegetables Consumption',
        'HvyAlcoholConsump': 'Heavy Alcohol Consumption',
        'AnyHealthcare': 'Healthcare Coverage',
        'NoDocbcCost': 'Avoided Doctor due to Cost',
        'GenHlth': 'General Health', 'MentHlth': 'Mental Health Days',
        'PhysHlth': 'Physical Health Days', 'DiffWalk': 'Difficulty Walking',
        'Sex': 'Gender', 'Age': 'Age Group',
        'Education': 'Education Level', 'Income': 'Income Level',
    },
    'liver': {
        'Age': 'Age', 'Gender': 'Gender', 'BMI': 'BMI',
        'AlcoholConsumption': 'Alcohol Consumption',
        'Smoking': 'Smoking', 'GeneticRisk': 'Genetic Risk',
        'PhysicalActivity': 'Physical Activity',
        'Diabetes': 'Diabetes', 'Hypertension': 'Hypertension',
        'LiverFunctionTest': 'Liver Function Test Score',
    },
    'kidney': {
        'Age': 'Age', 'Blood_Pressure': 'Blood Pressure',
        'Specific_Gravity': 'Specific Gravity',
        'Albumin': 'Albumin in Urine', 'Sugar': 'Sugar in Urine',
        'Red_Blood_Cells': 'Red Blood Cells', 'Pus_Cells': 'Pus Cells',
        'Pus_Cell_Clumps': 'Pus Cell Clumps', 'Bacteria': 'Bacteria',
        'Blood_Glucose': 'Blood Glucose', 'Blood_Urea': 'Blood Urea',
        'Serum_Creatinine': 'Serum Creatinine', 'Sodium': 'Sodium',
        'Potassium': 'Potassium', 'Haemoglobin': 'Haemoglobin',
        'Packed_Cell_Volume': 'Packed Cell Volume',
        'WBC_Count': 'WBC Count', 'RBC_Count': 'RBC Count',
        'Hypertension': 'Hypertension', 'Diabetes_Mellitus': 'Diabetes',
        'Coronary_Artery': 'Coronary Artery Disease',
        'Appetite': 'Appetite', 'Pedal_Edema': 'Pedal Edema',
        'Anaemia': 'Anaemia', 'eGFR': 'Kidney Filtration Rate',
        'Urine_Protein': 'Protein in Urine',
        'Urine_Output': 'Urine Output', 'Serum_Albumin': 'Serum Albumin',
        'Cholesterol': 'Cholesterol', 'PTH_Level': 'PTH Level',
        'Serum_Calcium': 'Calcium Level',
        'Serum_Phosphate': 'Phosphate Level',
        'Family_History': 'Family History of Kidney Disease',
        'Smoking': 'Smoking', 'BMI': 'BMI',
        'Physical_Activity': 'Physical Activity Level',
        'Diabetes_Duration': 'Years with Diabetes',
        'Hypertension_Duration': 'Years with Hypertension',
        'Cystatin_C': 'Cystatin C Level',
        'Urinary_Sediment': 'Urinary Sediment',
        'CRP_Level': 'Inflammation Level (CRP)',
        'IL6_Level': 'Inflammation Marker (IL-6)',
    }
}

# ============================================================
# GENERATE HUMAN READABLE REASONS
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

        # ── These features are ALWAYS bad regardless of LIME weight ──
        always_bad = {
            'heart'   : ['smoke', 'alco', 'ap_hi', 'ap_lo',
                         'cholesterol', 'gluc'],
            'diabetes': ['HighBP', 'HighChol', 'Smoker', 'Stroke',
                         'HeartDiseaseorAttack', 'HvyAlcoholConsump',
                         'DiffWalk'],
            'liver'   : ['AlcoholConsumption', 'Smoking', 'GeneticRisk',
                         'Diabetes', 'Hypertension', 'LiverFunctionTest'],
            'kidney'  : ['Blood_Pressure', 'Albumin', 'Sugar',
                         'Blood_Urea', 'Serum_Creatinine', 'Hypertension',
                         'Diabetes_Mellitus', 'CRP_Level', 'Anaemia',
                         'Pedal_Edema'],
        }

        # ── These features are ALWAYS good regardless of LIME weight ──
        always_good = {
            'heart'   : ['active'],
            'diabetes': ['PhysActivity', 'Fruits', 'Veggies'],
            'liver'   : ['PhysicalActivity'],
            'kidney'  : ['Haemoglobin', 'eGFR', 'Appetite'],
        }

        risk_messages = {
            # HEART
            'ap_hi'      : lambda v: f"Your systolic blood pressure is {v} mmHg — this is high and puts strain on your heart" if v > 120 else None,
            'ap_lo'      : lambda v: f"Your diastolic blood pressure is {v} mmHg — this is elevated above normal" if v > 80 else None,
            'cholesterol': lambda v: f"Your cholesterol is {'above normal' if v==2 else 'well above normal'} — high cholesterol blocks arteries" if v > 1 else None,
            'gluc'       : lambda v: f"Your glucose is {'above normal' if v==2 else 'well above normal'} — high glucose damages blood vessels" if v > 1 else None,
            'smoke'      : lambda v: "You are a smoker — smoking is one of the top causes of heart disease" if v==1 else None,
            'alco'       : lambda v: "You consume alcohol — alcohol weakens the heart muscle over time" if v==1 else None,
            'active'     : lambda v: "You are physically inactive — regular exercise strengthens the heart" if v==0 else None,
            'weight'     : lambda v: f"Your weight is {v} kg — being overweight increases heart disease risk" if v > 85 else None,
            'age'        : lambda v: f"Your age ({v} years) is a natural risk factor for heart disease" if v > 50 else None,
            # DIABETES
            'HighBP'              : lambda v: "You have high blood pressure — this is strongly linked to diabetes" if v==1 else None,
            'HighChol'            : lambda v: "You have high cholesterol — this increases your risk of diabetes" if v==1 else None,
            'BMI'                 : lambda v: f"Your BMI is {v} — being {'overweight' if v<=30 else 'obese'} is the top risk factor for diabetes" if v>25 else None,
            'Smoker'              : lambda v: "You are a smoker — smoking causes insulin resistance" if v==1 else None,
            'Stroke'              : lambda v: "You have a history of stroke — this is closely linked to diabetes" if v==1 else None,
            'HeartDiseaseorAttack': lambda v: "You have a history of heart disease — strongly associated with diabetes" if v==1 else None,
            'PhysActivity'        : lambda v: "You are physically inactive — exercise is the best way to control blood sugar" if v==0 else None,
            'HvyAlcoholConsump'   : lambda v: "You are a heavy alcohol drinker — alcohol damages the pancreas which controls insulin" if v==1 else None,
            'GenHlth'             : lambda v: f"Your general health is rated {int(v)}/5 — poor health increases diabetes risk" if v>=4 else None,
            'DiffWalk'            : lambda v: "You have difficulty walking — this indicates physical health problems linked to diabetes" if v==1 else None,
            # LIVER
            'AlcoholConsumption': lambda v: f"You consume {v} units of alcohol per week — alcohol is the number one cause of liver damage" if v>5 else f"You consume {v} units of alcohol per week — even moderate alcohol affects the liver" if v>0 else None,
            'Smoking'           : lambda v: "You smoke — smoking produces toxins that damage liver cells directly" if v==1 else None,
            'GeneticRisk'       : lambda v: f"You have {'medium' if v==1 else 'high'} genetic risk — family history significantly raises liver disease risk" if v>0 else None,
            'Diabetes'          : lambda v: "You have diabetes — diabetic patients are 2x more likely to develop liver disease" if v==1 else None,
            'Hypertension'      : lambda v: "You have hypertension — high blood pressure directly strains the liver" if v==1 else None,
            'LiverFunctionTest' : lambda v: f"Your liver function score is {v} — an elevated score means your liver is under stress" if v>55 else None,
            'BMI'               : lambda v: f"Your BMI is {v} — excess body fat causes fatty liver disease" if v>25 else None,
            'PhysicalActivity'  : lambda v: "You are physically inactive — exercise reduces fat buildup in the liver" if v<2 else None,
            # KIDNEY
            'Blood_Pressure'  : lambda v: f"Your blood pressure is {v} mmHg — high blood pressure is the #1 cause of kidney damage" if v>90 else None,
            'Albumin'         : lambda v: f"Albumin level {v} found in your urine — protein in urine is a direct sign of kidney damage" if v>0 else None,
            'Sugar'           : lambda v: f"Sugar level {v} found in your urine — glucose in urine means kidneys are struggling to filter" if v>0 else None,
            'Blood_Glucose'   : lambda v: f"Your blood glucose is {v} mg/dl — high glucose destroys kidney filters over time" if v>140 else None,
            'Blood_Urea'      : lambda v: f"Your blood urea is {v} mg/dl — high urea means your kidneys are not cleaning blood properly" if v>40 else None,
            'Serum_Creatinine': lambda v: f"Your creatinine is {v} mg/dl — this is the most direct indicator of kidney damage" if v>1.2 else None,
            'Haemoglobin'     : lambda v: f"Your haemoglobin is {v} gms — low haemoglobin (anaemia) is very common in kidney disease" if v<12 else None,
            'Hypertension'    : lambda v: "You have hypertension — uncontrolled blood pressure is the leading cause of kidney failure" if v==1 else None,
            'Diabetes_Mellitus': lambda v: "You have diabetes — diabetes is the most common cause of chronic kidney disease" if v==1 else None,
            'eGFR'            : lambda v: f"Your kidney filtration rate (eGFR) is {v} — below 60 means your kidneys are not working well" if v<60 else None,
            'Family_History'  : lambda v: "You have a family history of kidney disease — genetic risk is a significant factor" if v==1 else None,
            'CRP_Level'       : lambda v: f"Your inflammation marker (CRP) is {v} — high inflammation actively damages kidney tissue" if v>3 else None,
            'Anaemia'         : lambda v: "You have anaemia — anaemia is both a cause and sign of kidney disease" if v==1 else None,
            'Pedal_Edema'     : lambda v: "You have swollen feet (pedal edema) — this indicates your kidneys are not removing excess fluid" if v==1 else None,
        }

        # ── Good health messages (only show when genuinely good) ──
        good_messages = {
            'heart': {
                'active'     : lambda v: "You are physically active — exercise is one of the best protections against heart disease" if v==1 else None,
                'smoke'      : lambda v: "You are a non-smoker — this significantly reduces your heart disease risk" if v==0 else None,
                'alco'       : lambda v: "You do not drink alcohol — this is good for your heart health" if v==0 else None,
            },
            'diabetes': {
                'PhysActivity': lambda v: "You are physically active — regular exercise helps control blood sugar levels" if v==1 else None,
                'HighBP'      : lambda v: "You do not have high blood pressure — this reduces your diabetes risk" if v==0 else None,
                'Smoker'      : lambda v: "You are a non-smoker — this reduces your diabetes risk" if v==0 else None,
            },
            'liver': {
                'Smoking'           : lambda v: "You do not smoke — this protects your liver from toxic damage" if v==0 else None,
                'PhysicalActivity'  : lambda v: f"You are physically active ({v} hrs/week) — exercise prevents fatty liver disease" if v>=3 else None,
                'Diabetes'          : lambda v: "You do not have diabetes — this reduces your liver disease risk" if v==0 else None,
            },
            'kidney': {
                'Haemoglobin'      : lambda v: f"Your haemoglobin is {v} gms — this is normal and healthy" if v>=12 else None,
                'Hypertension'     : lambda v: "You do not have hypertension — this protects your kidneys" if v==0 else None,
                'Diabetes_Mellitus': lambda v: "You do not have diabetes — this significantly reduces kidney disease risk" if v==0 else None,
            }
        }

        reasons_risk = []
        reasons_good = []
        seen_risk = set()
        seen_good = set()

        for feature_condition, weight in exp_list:
            for feat_key in features[disease]:
                if feat_key.lower() in feature_condition.lower() \
                        and feat_key not in seen_risk \
                        and feat_key not in seen_good:

                    val = instance[feat_index[feat_key]]
                    ab = always_bad.get(disease, [])
                    ag = always_good.get(disease, [])

                    # Risk reasons — feature is always bad OR LIME says risky
                    if feat_key in ab or weight > 0.02:
                        if feat_key in risk_messages:
                            msg = risk_messages[feat_key](val)
                            if msg and feat_key not in seen_risk:
                                reasons_risk.append(msg)
                                seen_risk.add(feat_key)

                    # Good reasons — feature is always good OR LIME says safe
                    elif feat_key in ag or weight < -0.02:
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
    explainer = shap.TreeExplainer(models[disease])
    df_instance = pd.DataFrame([instance], columns=features[disease])
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

    feat_labels = [human_names[disease].get(f, f) for f in features[disease]]
    shap_df = pd.DataFrame({
        'feature': feat_labels,
        'value': np.abs(sv[0])
    }).sort_values('value', ascending=True).tail(10)

    plt.figure(figsize=(10, 6))
    plt.barh(shap_df['feature'], shap_df['value'], color='#2C7BE5')
    plt.xlabel('Importance Score', fontsize=12)
    plt.title('Which factors influenced this prediction?', fontsize=13)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# ============================================================
# LIME CHART
# ============================================================
def get_lime_chart(disease, instance):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(train_data[disease]),
        feature_names=[human_names[disease].get(f, f)
                       for f in features[disease]],
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
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

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
            'Diabetes': [
                'Consult an endocrinologist immediately',
                'Monitor blood sugar levels daily',
                'Follow a low sugar and low carb diet',
                'Exercise regularly — at least 30 mins/day',
                'Take prescribed medications on time'
            ],
            'Pre-Diabetes': [
                'Consult your doctor soon',
                'Reduce sugar and processed food intake',
                'Increase physical activity daily',
                'Lose weight if you are overweight',
                'Get blood sugar tested every 3 months'
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
# PREDICT ROUTES
# ============================================================
@app.route('/predict/heart', methods=['POST'])
def predict_heart():
    try:
        data = request.form
        instance = [
            int(data['age']), int(data['gender']),
            int(data['height']), float(data['weight']),
            int(data['ap_hi']), int(data['ap_lo']),
            int(data['cholesterol']), int(data['gluc']),
            int(data['smoke']), int(data['alco']),
            int(data['active']),
        ]
        df = pd.DataFrame([instance], columns=features['heart'])
        pred = int(models['heart'].predict(df)[0])
        prob = models['heart'].predict_proba(df)[0]
        confidence = round(float(max(prob)) * 100, 2)
        label = label_maps['heart'][pred]
        shap_chart = get_shap_chart('heart', instance)
        lime_chart = get_lime_chart('heart', instance)
        reasons_risk, reasons_good = get_human_reasons(
            'heart', instance, label, pred)
        advice = get_advice('heart', label)
        save_prediction('heart', dict(data), label, confidence,
            ', '.join(reasons_risk[:2]) if reasons_risk else label)
        return render_template('result.html',
            disease='Heart Disease', label=label,
            confidence=confidence, shap_chart=shap_chart,
            lime_chart=lime_chart, features=features['heart'],
            instance=instance, pred=pred,
            reasons_risk=reasons_risk,
            reasons_good=reasons_good,
            advice=advice,
            human_names=human_names['heart'])
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes():
    try:
        data = request.form
        instance = [
            float(data['HighBP']), float(data['HighChol']),
            float(data['CholCheck']), float(data['BMI']),
            float(data['Smoker']), float(data['Stroke']),
            float(data['HeartDiseaseorAttack']),
            float(data['PhysActivity']), float(data['Fruits']),
            float(data['Veggies']), float(data['HvyAlcoholConsump']),
            float(data['AnyHealthcare']), float(data['NoDocbcCost']),
            float(data['GenHlth']), float(data['MentHlth']),
            float(data['PhysHlth']), float(data['DiffWalk']),
            float(data['Sex']), float(data['Age']),
            float(data['Education']), float(data['Income']),
        ]
        df = pd.DataFrame([instance], columns=features['diabetes'])
        pred = int(models['diabetes'].predict(df)[0])
        prob = models['diabetes'].predict_proba(df)[0]
        confidence = round(float(max(prob)) * 100, 2)
        label = label_maps['diabetes'][pred]
        shap_chart = get_shap_chart('diabetes', instance)
        lime_chart = get_lime_chart('diabetes', instance)
        reasons_risk, reasons_good = get_human_reasons(
            'diabetes', instance, label, pred)
        advice = get_advice('diabetes', label)
        save_prediction('diabetes', dict(data), label, confidence,
            ', '.join(reasons_risk[:2]) if reasons_risk else label)
        return render_template('result.html',
            disease='Diabetes', label=label,
            confidence=confidence, shap_chart=shap_chart,
            lime_chart=lime_chart, features=features['diabetes'],
            instance=instance, pred=pred,
            reasons_risk=reasons_risk,
            reasons_good=reasons_good,
            advice=advice,
            human_names=human_names['diabetes'])
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/predict/liver', methods=['POST'])
def predict_liver():
    try:
        data = request.form
        instance = [
            int(data['Age']), int(data['Gender']),
            float(data['BMI']), float(data['AlcoholConsumption']),
            int(data['Smoking']), int(data['GeneticRisk']),
            float(data['PhysicalActivity']), int(data['Diabetes']),
            int(data['Hypertension']), float(data['LiverFunctionTest']),
        ]
        df = pd.DataFrame([instance], columns=features['liver'])
        pred = int(models['liver'].predict(df)[0])
        prob = models['liver'].predict_proba(df)[0]
        confidence = round(float(max(prob)) * 100, 2)
        label = label_maps['liver'][pred]
        shap_chart = get_shap_chart('liver', instance)
        lime_chart = get_lime_chart('liver', instance)
        reasons_risk, reasons_good = get_human_reasons(
            'liver', instance, label, pred)
        advice = get_advice('liver', label)
        save_prediction('liver', dict(data), label, confidence,
            ', '.join(reasons_risk[:2]) if reasons_risk else label)
        return render_template('result.html',
            disease='Liver Disease', label=label,
            confidence=confidence, shap_chart=shap_chart,
            lime_chart=lime_chart, features=features['liver'],
            instance=instance, pred=pred,
            reasons_risk=reasons_risk,
            reasons_good=reasons_good,
            advice=advice,
            human_names=human_names['liver'])
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/predict/kidney', methods=['POST'])
def predict_kidney():
    try:
        data = request.form
        encoders = joblib.load('models/kidney/kidney_encoders.pkl')
        instance = [
            int(data['Age']), int(data['Blood_Pressure']),
            float(data['Specific_Gravity']),
            int(data['Albumin']), int(data['Sugar']),
            int(encoders['Red_Blood_Cells'].transform([data['Red_Blood_Cells']])[0]),
            int(encoders['Pus_Cells'].transform([data['Pus_Cells']])[0]),
            int(encoders['Pus_Cell_Clumps'].transform([data['Pus_Cell_Clumps']])[0]),
            int(encoders['Bacteria'].transform([data['Bacteria']])[0]),
            int(data['Blood_Glucose']), float(data['Blood_Urea']),
            float(data['Serum_Creatinine']), float(data['Sodium']),
            float(data['Potassium']), float(data['Haemoglobin']),
            int(data['Packed_Cell_Volume']), int(data['WBC_Count']),
            float(data['RBC_Count']),
            int(encoders['Hypertension'].transform([data['Hypertension']])[0]),
            int(encoders['Diabetes_Mellitus'].transform([data['Diabetes_Mellitus']])[0]),
            int(encoders['Coronary_Artery'].transform([data['Coronary_Artery']])[0]),
            int(encoders['Appetite'].transform([data['Appetite']])[0]),
            int(encoders['Pedal_Edema'].transform([data['Pedal_Edema']])[0]),
            int(encoders['Anaemia'].transform([data['Anaemia']])[0]),
            float(data['eGFR']), float(data['Urine_Protein']),
            int(data['Urine_Output']), float(data['Serum_Albumin']),
            int(data['Cholesterol']), float(data['PTH_Level']),
            float(data['Serum_Calcium']), float(data['Serum_Phosphate']),
            int(encoders['Family_History'].transform([data['Family_History']])[0]),
            int(encoders['Smoking'].transform([data['Smoking']])[0]),
            float(data['BMI']),
            int(encoders['Physical_Activity'].transform([data['Physical_Activity']])[0]),
            int(data['Diabetes_Duration']),
            int(data['Hypertension_Duration']),
            float(data['Cystatin_C']),
            int(encoders['Urinary_Sediment'].transform([data['Urinary_Sediment']])[0]),
            float(data['CRP_Level']), float(data['IL6_Level']),
        ]
        df = pd.DataFrame([instance], columns=features['kidney'])
        pred = int(models['kidney'].predict(df)[0])
        prob = models['kidney'].predict_proba(df)[0]
        confidence = round(float(max(prob)) * 100, 2)
        label = label_maps['kidney'][pred]
        shap_chart = get_shap_chart('kidney', instance)
        lime_chart = get_lime_chart('kidney', instance)
        reasons_risk, reasons_good = get_human_reasons(
            'kidney', instance, label, pred)
        advice = get_advice('kidney', label)
        save_prediction('kidney', dict(data), label, confidence,
            ', '.join(reasons_risk[:2]) if reasons_risk else label)
        return render_template('result.html',
            disease='Kidney Disease', label=label,
            confidence=confidence, shap_chart=shap_chart,
            lime_chart=lime_chart, features=features['kidney'],
            instance=instance, pred=pred,
            reasons_risk=reasons_risk,
            reasons_good=reasons_good,
            advice=advice,
            human_names=human_names['kidney'])
    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)