# api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
from typing import List, Optional
from datetime import datetime, timedelta
from bson.objectid import ObjectId
from stock_predictor import StockPredictor # Import the prediction logic

# --- Pydantic Schemas for API ---

# Helper class to handle MongoDB's default _id field
class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")

class MongoBaseModel(BaseModel):
    # This configuration is necessary to handle MongoDB's BSON types in Pydantic
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str, datetime: lambda dt: dt.isoformat()}

class InsightResponse(MongoBaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    company: str
    date: str
    category: str
    metrics: dict

class PredictionResponse(MongoBaseModel):
    company: str
    predicted_change_pct: float
    prediction_date: datetime
    predicted_direction: str # UP/DOWN string can be derived here or in the model

# --- FastAPI App Setup ---
app = FastAPI(title="BSE AI Analysis API")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "bse_data")

# Connect to MongoDB
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]
insights_collection = db["insights"]
predictions_collection = db["predictions"]
announcements_collection = db["announcements"]


# Initialize Prediction Logic (Needs to be trained before use!)
predictor = StockPredictor()

@app.on_event("startup")
def startup_db_client():
    # Placeholder: In a production environment, you would trigger model training here.
    # predictor.train_model_across_all_companies() 
    print("FastAPI app started. Remember to train your model before using /predict.")

# --- Endpoints ---

@app.get("/")
def read_root():
    return {"message": "BSE AI Analysis API is running."}

@app.get("/companies")
def get_companies():
    """Get list of all companies with insights."""
    companies = insights_collection.distinct("company")
    return {"companies": sorted(companies)}

@app.get("/insights/{company}", response_model=List[InsightResponse])
def get_company_insights(company: str, limit: int = 10):
    """Get AI insights for a company."""
    insights = list(
        insights_collection.find({"company": company})
        .sort("date", -1)
        .limit(limit)
    )
    if not insights:
        raise HTTPException(status_code=404, detail=f"No insights found for {company}")
    return insights

@app.get("/predictions/{company}")
def get_prediction(company: str):
    """Get latest stock prediction, or run one if needed."""
    
    # 1. Retrieve latest prediction
    prediction = predictions_collection.find_one(
        {"company": company},
        sort=[("prediction_date", -1)]
    )
    
    if not prediction:
        # If no prediction exists, trigger a manual prediction run (assuming model is trained)
        predicted_change = predictor.predict_latest(company)
        
        if predicted_change is None:
            raise HTTPException(status_code=404, detail="Prediction model is not trained or no data available.")
            
        # Retrieve the newly created prediction
        prediction = predictions_collection.find_one(
            {"company": company},
            sort=[("prediction_date", -1)]
        )

    # Convert BSON types for API response
    prediction['id'] = str(prediction.pop('_id'))
    prediction['predicted_direction'] = "UP" if prediction['predicted_change_pct'] > 0 else "DOWN"
    prediction['confidence'] = abs(prediction['predicted_change_pct']) # Simple confidence proxy
    
    return prediction

@app.get("/dashboard/summary")
def get_dashboard_summary():
    """Get dashboard summary data."""
    
    total_announcements = announcements_collection.count_documents({})
    total_insights = insights_collection.count_documents({})
    
    # Top predicted movers
    top_predictions = list(
        predictions_collection.find()
        .sort("predicted_change_pct", -1)
        .limit(10)
    )
    
    return {
        "stats": {
            "total_announcements": total_announcements,
            "insights_generated": total_insights,
        },
        "top_predictions": top_predictions
    }