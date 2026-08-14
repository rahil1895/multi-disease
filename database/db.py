from pymongo import MongoClient
from datetime import datetime

MONGO_URI = "mongodb+srv://raoufmanhas046_db_user:raoufmanhas18@cluster0.laytouq.mongodb.net/?appName=Cluster0"
DB_NAME = "xai_healthcare_db"

def get_db():
    try:
        client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=3000,
            connectTimeoutMS=3000,
            socketTimeoutMS=3000,
            directConnection=False
        )
        return client[DB_NAME]
    except Exception as e:
        print(f"DB connection failed: {e}")
        return None

def save_prediction(disease, patient_data, prediction,
                    confidence, explanation_text):
    try:
        db = get_db()
        if db is None:
            return None
        record = {
            "disease"         : disease,
            "patient_data"    : patient_data,
            "prediction"      : prediction,
            "confidence"      : confidence,
            "explanation_text": explanation_text,
            "timestamp"       : datetime.now()
        }
        result = db["predictions"].insert_one(record)
        return str(result.inserted_id)
    except Exception as e:
        print(f"DB save error: {e}")
        return None

def get_recent_predictions(limit=10):
    try:
        db = get_db()
        if db is None:
            return []
        records = db["predictions"].find(
            {}, {"_id": 0}
        ).sort("timestamp", -1).limit(limit)
        return list(records)
    except Exception as e:
        print(f"DB fetch error: {e}")
        return []

def get_prediction_counts():
    try:
        db = get_db()
        if db is None:
            return {
                "heart": 0, "diabetes": 0,
                "liver": 0, "kidney": 0, "total": 0
            }
        counts = {
            "heart"   : db["predictions"].count_documents(
                {"disease": "heart"}),
            "diabetes": db["predictions"].count_documents(
                {"disease": "diabetes"}),
            "liver"   : db["predictions"].count_documents(
                {"disease": "liver"}),
            "kidney"  : db["predictions"].count_documents(
                {"disease": "kidney"}),
            "total"   : db["predictions"].count_documents({})
        }
        return counts
    except Exception as e:
        print(f"DB count error: {e}")
        return {
            "heart": 0, "diabetes": 0,
            "liver": 0, "kidney": 0, "total": 0
        }

def get_all_predictions(disease=None):
    try:
        db = get_db()
        if db is None:
            return []
        if disease:
            records = db["predictions"].find(
                {"disease": disease}, {"_id": 0}
            ).sort("timestamp", -1)
        else:
            records = db["predictions"].find(
                {}, {"_id": 0}
            ).sort("timestamp", -1)
        return list(records)
    except Exception as e:
        print(f"DB fetch error: {e}")
        return []