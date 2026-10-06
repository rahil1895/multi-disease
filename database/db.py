from datetime import datetime
import json
import os

# ============================================================
# TRY MONGODB FIRST
# ============================================================
MONGO_URI = "mongodb+srv://project:project@cluster0.tbyeg2j.mongodb.net/?appName=Cluster0"
DB_NAME   = "xai_healthcare_db"
DATA_FILE = "database/predictions.json"
USE_MONGO = False

try:
    import certifi
    from pymongo import MongoClient
    client = MongoClient(
        MONGO_URI,
        tlsCAFile=certifi.where(),
        serverSelectionTimeoutMS=3000,
        connectTimeoutMS=3000,
    )
    client.admin.command('ping')
    db_mongo = client[DB_NAME]
    USE_MONGO = True
    print("✅ MongoDB connected!")
except Exception as e:
    print(f"⚠️ MongoDB not available — using local storage")
    db_mongo = None

# ============================================================
# LOCAL JSON STORAGE
# ============================================================
def init_local():
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'w') as f:
            json.dump([], f)

def get_local_data():
    init_local()
    try:
        with open(DATA_FILE, 'r') as f:
            return json.load(f)
    except:
        return []

def save_local_data(data):
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f, default=str, indent=2)

# ============================================================
# SAVE PREDICTION
# ============================================================
def save_prediction(disease, patient_data, prediction,
                    confidence, explanation_text):
    try:
        record = {
            "disease"         : disease,
            "patient_data"    : dict(patient_data),
            "prediction"      : prediction,
            "confidence"      : confidence,
            "explanation_text": explanation_text,
            "timestamp"       : str(datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"))
        }

        if USE_MONGO and db_mongo is not None:
            result = db_mongo["predictions"].insert_one(record)
            print(f"✅ Saved to MongoDB: {result.inserted_id}")
            return str(result.inserted_id)
        else:
            data = get_local_data()
            record["id"] = len(data) + 1
            data.insert(0, record)
            save_local_data(data)
            print(f"✅ Saved locally: {disease} — {prediction}")
            return record["id"]
    except Exception as e:
        print(f"❌ Save error: {e}")
        return None

# ============================================================
# GET RECENT PREDICTIONS
# ============================================================
def get_recent_predictions(limit=20):
    try:
        if USE_MONGO and db_mongo is not None:
            records = db_mongo["predictions"].find(
                {}, {"_id": 0}
            ).sort("timestamp", -1).limit(limit)
            return list(records)
        else:
            data = get_local_data()
            return data[:limit]
    except Exception as e:
        print(f"❌ Fetch error: {e}")
        return []

# ============================================================
# GET ALL PREDICTIONS
# ============================================================
def get_all_predictions(disease=None):
    try:
        if USE_MONGO and db_mongo is not None:
            if disease:
                records = db_mongo["predictions"].find(
                    {"disease": disease}, {"_id": 0}
                ).sort("timestamp", -1)
            else:
                records = db_mongo["predictions"].find(
                    {}, {"_id": 0}
                ).sort("timestamp", -1)
            return list(records)
        else:
            data = get_local_data()
            if disease:
                return [r for r in data
                        if r['disease'] == disease]
            return data
    except Exception as e:
        print(f"❌ Fetch error: {e}")
        return []

# ============================================================
# GET PREDICTION COUNTS
# ============================================================
def get_prediction_counts():
    try:
        if USE_MONGO and db_mongo is not None:
            return {
                "heart"   : db_mongo["predictions"].count_documents(
                    {"disease": "heart"}),
                "diabetes": db_mongo["predictions"].count_documents(
                    {"disease": "diabetes"}),
                "liver"   : db_mongo["predictions"].count_documents(
                    {"disease": "liver"}),
                "kidney"  : db_mongo["predictions"].count_documents(
                    {"disease": "kidney"}),
                "total"   : db_mongo["predictions"].count_documents({})
            }
        else:
            data = get_local_data()
            return {
                "heart"   : len([r for r in data
                                 if r['disease'] == 'heart']),
                "diabetes": len([r for r in data
                                 if r['disease'] == 'diabetes']),
                "liver"   : len([r for r in data
                                 if r['disease'] == 'liver']),
                "kidney"  : len([r for r in data
                                 if r['disease'] == 'kidney']),
                "total"   : len(data)
            }
    except Exception as e:
        print(f"❌ Count error: {e}")
        return {
            "heart": 0, "diabetes": 0,
            "liver": 0, "kidney": 0, "total": 0
        }

# ============================================================
# TEST
# ============================================================
if __name__ == '__main__':
    print(f"Storage mode: {'MongoDB' if USE_MONGO else 'Local JSON'}")
    test_id = save_prediction(
        disease='heart',
        patient_data={'age': 45},
        prediction='Disease Detected',
        confidence=87.5,
        explanation_text='High blood pressure detected'
    )
    print(f"Saved with ID: {test_id}")
    counts = get_prediction_counts()
    print(f"Counts: {counts}")
    recent = get_recent_predictions(5)
    print(f"Recent records: {len(recent)}")
    print("✅ Storage working!")