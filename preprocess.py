import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# ❤️ HEART DISEASE — COMBINE BOTH DATASETS
# ============================================================
def preprocess_heart():
    print("Processing Heart dataset...")

    # ── Load cardio_train ────────────────────────────────────
    cardio = pd.read_csv('datasets/heart/heart.csv', sep=';')
    cardio['age'] = (cardio['age'] / 365).astype(int)
    cardio = cardio[(cardio['ap_hi'] >= 60) & (cardio['ap_hi'] <= 250)]
    cardio = cardio[(cardio['ap_lo'] >= 40) & (cardio['ap_lo'] <= 200)]
    cardio.drop(columns=['id'], inplace=True, errors='ignore')

    # Standardize column names
    cardio = cardio.rename(columns={
        'cardio': 'target'
    })

    # Convert cholesterol 1,2,3 to actual approx values
    cardio['cholesterol'] = cardio['cholesterol'].map({
        1: 180, 2: 220, 3: 280
    })
    # Convert glucose 1,2,3 to actual approx values
    cardio['gluc'] = cardio['gluc'].map({
        1: 85, 2: 110, 3: 160
    })

    cardio_clean = cardio[[
        'age', 'gender', 'ap_hi', 'ap_lo',
        'cholesterol', 'gluc', 'smoke',
        'alco', 'active', 'target'
    ]].copy()
    cardio_clean.columns = [
        'age', 'gender', 'sys_bp', 'dia_bp',
        'cholesterol', 'glucose', 'smoke',
        'alco', 'active', 'target'
    ]

    # ── Load Framingham ──────────────────────────────────────
    fram = pd.read_csv('datasets/heart/framingham.csv')

    # Handle missing values
    for col in fram.columns:
        if fram[col].dtype != 'object':
            fram[col] = fram[col].fillna(fram[col].median())

    # Map currentSmoker to smoke
    # cigsPerDay > 0 means smoker
    fram['smoke'] = (fram['cigsPerDay'] > 0).astype(int)
    fram['alco']  = 0  # not in framingham
    fram['active'] = 0  # not in framingham

    fram_clean = fram[[
        'age', 'male', 'sysBP', 'diaBP',
        'totChol', 'glucose', 'smoke',
        'alco', 'active', 'TenYearCHD'
    ]].copy()
    fram_clean.columns = [
        'age', 'gender', 'sys_bp', 'dia_bp',
        'cholesterol', 'glucose', 'smoke',
        'alco', 'active', 'target'
    ]

    # ── Combine both ─────────────────────────────────────────
    combined = pd.concat([cardio_clean, fram_clean],
                         ignore_index=True)
    combined.dropna(inplace=True)

    print(f"  Cardio records    : {len(cardio_clean)}")
    print(f"  Framingham records: {len(fram_clean)}")
    print(f"  Combined total    : {len(combined)}")
    print(f"  Missing after fix : {combined.isnull().sum().sum()}")

    X = combined.drop(columns=['target'])
    y = combined['target']

    print(f"  Target distribution: {y.value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    joblib.dump((X_train, X_test, y_train, y_test),
                'models/heart/heart_data.pkl')
    joblib.dump(X.columns.tolist(),
                'models/heart/heart_features.pkl')

    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples : {len(X_test)}")
    print(f"✅ Heart preprocessing done!\n")


# ============================================================
# 🩸 DIABETES — NEW DATASET
# ============================================================
def preprocess_diabetes():
    print("Processing Diabetes dataset...")

    df = pd.read_csv(
        'datasets/diabetes/diabetes_prediction_dataset.csv')

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Encode gender
    df['gender'] = df['gender'].map({
        'Male': 1, 'Female': 0, 'Other': 0
    })

    # Encode smoking history
    # never=0, former=1, current=2, not current=1, No Info=0
    smoking_map = {
        'never'      : 0,
        'No Info'    : 0,
        'former'     : 1,
        'not current': 1,
        'ever'       : 1,
        'current'    : 2
    }
    df['smoking_history'] = df['smoking_history'].map(
        smoking_map).fillna(0).astype(int)

    # Drop rows with missing values
    df.dropna(inplace=True)

    print(f"  Shape            : {df.shape}")
    print(f"  Missing values   : {df.isnull().sum().sum()}")
    print(f"  Target distribution: "
          f"{df['diabetes'].value_counts().to_dict()}")

    X = df.drop(columns=['diabetes'])
    y = df['diabetes']

    # Apply SMOTE — only 8,500 diabetic vs 91,500 non diabetic
    print(f"  Applying SMOTE for class balance...")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    print(f"  After SMOTE: {pd.Series(y_res).value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2,
        random_state=42, stratify=y_res)

    joblib.dump((X_train, X_test, y_train, y_test),
                'models/diabetes/diabetes_data.pkl')
    joblib.dump(X.columns.tolist(),
                'models/diabetes/diabetes_features.pkl')

    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples : {len(X_test)}")
    print(f"✅ Diabetes preprocessing done!\n")


# ============================================================
# 🫀 LIVER DISEASE — NEW LARGE DATASET
# ============================================================
def preprocess_liver():
    print("Processing Liver dataset...")

    df = pd.read_csv('datasets/liver/liver.csv',
                     encoding='latin1')

    df.columns = [
        'Age', 'Gender', 'Total_Bilirubin',
        'Direct_Bilirubin', 'Alkaline_Phosphotase',
        'Alamine_Aminotransferase',
        'Aspartate_Aminotransferase',
        'Total_Proteins', 'Albumin',
        'Albumin_Globulin_Ratio', 'Result'
    ]

    df = df.assign(
        Gender=df['Gender'].fillna(df['Gender'].mode()[0]))

    num_cols = [
        'Age', 'Total_Bilirubin', 'Direct_Bilirubin',
        'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
        'Aspartate_Aminotransferase', 'Total_Proteins',
        'Albumin', 'Albumin_Globulin_Ratio'
    ]
    for col in num_cols:
        median_val = df[col].median()
        missing = df[col].isnull().sum()
        df[col] = df[col].fillna(median_val)
        if missing > 0:
            print(f"  Fixed: {col} → {missing} missing")

    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])
    joblib.dump(le, 'models/liver/liver_gender_encoder.pkl')

    df['Result'] = df['Result'].map({1: 1, 2: 0})
    df = df.dropna()

    # Cap outliers using IQR
    num_cols_iqr = [
        'Total_Bilirubin', 'Direct_Bilirubin',
        'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
        'Aspartate_Aminotransferase'
    ]
    for col in num_cols_iqr:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = df[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)

    print(f"  Shape            : {df.shape}")
    print(f"  Missing after fix: {df.isnull().sum().sum()}")
    print(f"  Target distribution: "
          f"{df['Result'].value_counts().to_dict()}")

    X = df.drop(columns=['Result'])
    y = df['Result']

    # NO SMOTE — use class weight instead
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    joblib.dump((X_train, X_test, y_train, y_test),
                'models/liver/liver_data.pkl')
    joblib.dump(X.columns.tolist(),
                'models/liver/liver_features.pkl')

    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples : {len(X_test)}")
    print(f"✅ Liver preprocessing done!\n")


# ============================================================
# 🫘 KIDNEY DISEASE — KEEP CURRENT
# ============================================================
def preprocess_kidney():
    print("Processing Kidney dataset...")

    df = pd.read_csv('datasets/kidney/kidney.csv')

    df.columns = [
        'Age', 'Blood_Pressure', 'Specific_Gravity',
        'Albumin', 'Sugar', 'Red_Blood_Cells', 'Pus_Cells',
        'Pus_Cell_Clumps', 'Bacteria', 'Blood_Glucose',
        'Blood_Urea', 'Serum_Creatinine', 'Sodium',
        'Potassium', 'Haemoglobin', 'Packed_Cell_Volume',
        'WBC_Count', 'RBC_Count', 'Hypertension',
        'Diabetes_Mellitus', 'Coronary_Artery', 'Appetite',
        'Pedal_Edema', 'Anaemia', 'eGFR', 'Urine_Protein',
        'Urine_Output', 'Serum_Albumin', 'Cholesterol',
        'PTH_Level', 'Serum_Calcium', 'Serum_Phosphate',
        'Family_History', 'Smoking', 'BMI',
        'Physical_Activity', 'Diabetes_Duration',
        'Hypertension_Duration', 'Cystatin_C',
        'Urinary_Sediment', 'CRP_Level', 'IL6_Level', 'Target'
    ]

    le_dict = {}
    str_cols = df.select_dtypes(include='str').columns.tolist()
    print(f"  Text columns to encode: {str_cols}")

    for col in str_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
        if col == 'Target':
            mapping = dict(zip(le.classes_,
                               le.transform(le.classes_)))
            print(f"  Target encoding: {mapping}")

    joblib.dump(le_dict, 'models/kidney/kidney_encoders.pkl')

    X = df.drop(columns=['Target'])
    y = df['Target']

    print(f"  Class distribution before SMOTE: "
          f"{y.value_counts().to_dict()}")

    smote = SMOTE(random_state=42, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X, y)

    print(f"  Class distribution after SMOTE : "
          f"{pd.Series(y_res).value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2,
        random_state=42, stratify=y_res)

    joblib.dump((X_train, X_test, y_train, y_test),
                'models/kidney/kidney_data.pkl')
    joblib.dump(X.columns.tolist(),
                'models/kidney/kidney_features.pkl')

    print(f"  Shape        : {df.shape}")
    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples : {len(X_test)}")
    print(f"✅ Kidney preprocessing done!\n")


# ============================================================
# RUN ALL
# ============================================================
if __name__ == '__main__':
    print("🚀 Starting preprocessing for all 4 diseases...\n")
    print("=" * 55)
    preprocess_heart()
    print("=" * 55)
    preprocess_diabetes()
    print("=" * 55)
    preprocess_liver()
    print("=" * 55)
    preprocess_kidney()
    print("=" * 55)
    print("🎉 All datasets preprocessed and saved successfully!")