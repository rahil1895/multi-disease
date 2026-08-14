import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

# ============================================================
# ❤️ HEART DISEASE
# ============================================================
def preprocess_heart():
    print("Processing Heart dataset...")
    df = pd.read_csv('datasets/heart/heart.csv', sep=';')

    # Convert age from days to years
    df['age'] = (df['age'] / 365).astype(int)

    # Drop ID column
    df.drop(columns=['id'], inplace=True)

    # Remove blood pressure outliers
    df = df[(df['ap_hi'] >= 60) & (df['ap_hi'] <= 250)]
    df = df[(df['ap_lo'] >= 40) & (df['ap_lo'] <= 200)]

    # Separate features and target
    X = df.drop(columns=['cardio'])
    y = df['cardio']

    # Split 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Save
    joblib.dump((X_train, X_test, y_train, y_test),
                'models/heart/heart_data.pkl')
    joblib.dump(X.columns.tolist(),
                'models/heart/heart_features.pkl')

    print(f"  Shape after cleaning : {df.shape}")
    print(f"  Train samples        : {len(X_train)}")
    print(f"  Test samples         : {len(X_test)}")
    print(f"  Target distribution  : {y.value_counts().to_dict()}")
    print(f"✅ Heart preprocessing done!\n")


# ============================================================
# 🩸 DIABETES
# ============================================================
def preprocess_diabetes():
    print("Processing Diabetes dataset...")
    df = pd.read_csv('datasets/diabetes/diabetes.csv')

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Convert target to int (0, 1, 2)
    df['Diabetes_012'] = df['Diabetes_012'].astype(int)

    # Separate features and target
    X = df.drop(columns=['Diabetes_012'])
    y = df['Diabetes_012']

    # Split 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Save
    joblib.dump((X_train, X_test, y_train, y_test),
                'models/diabetes/diabetes_data.pkl')
    joblib.dump(X.columns.tolist(),
                'models/diabetes/diabetes_features.pkl')

    print(f"  Shape after cleaning : {df.shape}")
    print(f"  Train samples        : {len(X_train)}")
    print(f"  Test samples         : {len(X_test)}")
    print(f"  Target distribution  : {y.value_counts().to_dict()}")
    print(f"✅ Diabetes preprocessing done!\n")


# ============================================================
# 🫀 LIVER DISEASE
# ============================================================
def preprocess_liver():
    print("Processing Liver dataset...")
    df = pd.read_csv('datasets/liver/liver.csv')

    # No missing values — no fix needed
    print(f"  Shape                : {df.shape}")
    print(f"  Missing values       : {df.isnull().sum().sum()}")

    # All columns already numeric — no encoding needed
    X = df.drop(columns=['Diagnosis'])
    y = df['Diagnosis']

    print(f"  Class distribution   : {y.value_counts().to_dict()}")

    # Apply SMOTE if imbalanced
    min_class = y.value_counts().min()
    maj_class = y.value_counts().max()
    ratio = min_class / maj_class

    if ratio < 0.7:
        print(f"  Imbalanced (ratio={ratio:.2f}) → Applying SMOTE...")
        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y)
        print(f"  After SMOTE          : {pd.Series(y).value_counts().to_dict()}")
    else:
        print(f"  Balanced (ratio={ratio:.2f}) → No SMOTE needed")

    # Split 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    joblib.dump((X_train, X_test, y_train, y_test),
                'models/liver/liver_data.pkl')
    joblib.dump(X.columns.tolist(),
                'models/liver/liver_features.pkl')

    print(f"  Train samples        : {len(X_train)}")
    print(f"  Test samples         : {len(X_test)}")
    print(f"✅ Liver preprocessing done!\n")



# ============================================================
# 🫘 KIDNEY DISEASE
# ============================================================
def preprocess_kidney():
    print("Processing Kidney dataset...")
    df = pd.read_csv('datasets/kidney/kidney.csv')

    df.columns = [
        'Age', 'Blood_Pressure', 'Specific_Gravity', 'Albumin', 'Sugar',
        'Red_Blood_Cells', 'Pus_Cells', 'Pus_Cell_Clumps', 'Bacteria',
        'Blood_Glucose', 'Blood_Urea', 'Serum_Creatinine', 'Sodium',
        'Potassium', 'Haemoglobin', 'Packed_Cell_Volume', 'WBC_Count',
        'RBC_Count', 'Hypertension', 'Diabetes_Mellitus', 'Coronary_Artery',
        'Appetite', 'Pedal_Edema', 'Anaemia', 'eGFR', 'Urine_Protein',
        'Urine_Output', 'Serum_Albumin', 'Cholesterol', 'PTH_Level',
        'Serum_Calcium', 'Serum_Phosphate', 'Family_History', 'Smoking',
        'BMI', 'Physical_Activity', 'Diabetes_Duration',
        'Hypertension_Duration', 'Cystatin_C', 'Urinary_Sediment',
        'CRP_Level', 'IL6_Level', 'Target'
    ]

    # Encode all text columns
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

    print(f"  Class distribution before SMOTE: {y.value_counts().to_dict()}")

    # Apply SMOTE to balance all 5 classes
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X, y)

    print(f"  Class distribution after SMOTE : {pd.Series(y_res).value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)

    joblib.dump((X_train, X_test, y_train, y_test),
                'models/kidney/kidney_data.pkl')
    joblib.dump(X.columns.tolist(), 'models/kidney/kidney_features.pkl')

    print(f"  Shape                : {df.shape}")
    print(f"  Train samples        : {len(X_train)}")
    print(f"  Test samples         : {len(X_test)}")
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