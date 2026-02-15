
from fastapi import FastAPI, Request, Form

from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pickle
import numpy as np
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd 
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Load trained model using pickle
with open("heart_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

client = AsyncIOMotorClient("mongodb://localhost:27017")
db = client["health_db"]
collection = db["predictions"]

app = FastAPI(title="Heart Disease Prediction API")

templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

# Define input format
class HeartData(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict_form", response_class=HTMLResponse)
async def predict_form(
    request: Request,
    name: str = Form(...),
    age: float = Form(...),
    sex: float = Form(...),
    cp: float = Form(...),
    trestbps: float = Form(...),
    chol: float = Form(...),
    fbs: float = Form(...),
    restecg: float = Form(...),
    thalach: float = Form(...),
    exang: float = Form(...),
    oldpeak: float = Form(...),
    slope: float = Form(...),
    ca: float = Form(...),
    thal: float = Form(...)
):

    input_data = np.array([[age, sex, cp, trestbps,
                            chol, fbs, restecg, thalach,
                            exang, oldpeak, slope, ca, thal]])

    prediction = model.predict(input_data)[0]

    result = "Disease Detected" if prediction == 1 else "No Disease"

    record = {
        "input": {
            "name":name,
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal
        },
        "prediction": result,
        "timestamp": datetime.utcnow()
    }

    await collection.insert_one({
        "patient_name": name,
    "input": {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    },
    "prediction": result,
    "timestamp": datetime.utcnow()
    })

    images = await get_visualizations()

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "result": result,
            "feature_importance": images["feature_importance"],
            "heatmap": images["heatmap"],
            "distribution": images["distribution"],
            "name": name
        }
    )



@app.get("/history")
async def get_history(request: Request):

    records = []

    cursor = collection.find().sort("timestamp", -1)

    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        records.append(doc)

    return templates.TemplateResponse(
        "history.html",
        {
            "request": request,
            "records": records
        }
    )

@app.get("/visualizations")
async def get_visualizations():

    images = {}

    # Feature names
    features = [
        "age","sex","cp","trestbps","chol",
        "fbs","restecg","thalach","exang",
        "oldpeak","slope","ca","thal"
    ]

    # ---------------------------
    # 1. Feature Importance Plot
    # ---------------------------
    plt.figure(figsize=(8,6))

    importances = model.named_steps["model"].feature_importances_

    sns.barplot(x=importances, y=features)

    plt.title("Feature Importance")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)

    images["feature_importance"] = base64.b64encode(buf.read()).decode("utf-8")

    plt.close()

    # ---------------------------
    # 2. Correlation Heatmap
    # ---------------------------

    # Load dataset (same dataset used for training)
    df = pd.read_csv("processed.cleveland.data")

    df.rename(columns={"num\t":"target"}, inplace=True)

    df.replace("?", pd.NA, inplace=True)
    df = df.dropna()
    df = df.astype(float)

    plt.figure(figsize=(10,8))

    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")

    plt.title("Correlation Heatmap")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)

    images["heatmap"] = base64.b64encode(buf.read()).decode("utf-8")

    plt.close()

    # ---------------------------
    # 3. Distribution Plot
    # ---------------------------

    plt.figure(figsize=(8,6))

    sns.histplot(df["age"], bins=20, kde=True)

    plt.title("Age Distribution")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)

    images["distribution"] = base64.b64encode(buf.read()).decode("utf-8")

    plt.close()

    return images



