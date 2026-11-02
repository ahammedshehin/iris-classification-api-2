from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict
import pandas as pd
import joblib

app = FastAPI(
    title="API Iris project",
    description="API for prrdicting Iris flower",
    version="1.0"
)

model = joblib.load("model/model.pkl")
class IrisData(BaseModel):

    model_config = ConfigDict(populate_by_name=True)
    
    sepal_length : float = Field(alias="sepal length (cm)")
    sepal_width : float = Field(alias="sepal width (cm)")
    petal_length : float = Field(alias="petal length (cm)")
    petal_width : float = Field(alias="petal width (cm)")

@app.get("/")
def home():
    return{
        "massege": "Iris API is running"
    }

@app.post("/predict")
def predict_iris(data: Iris_data):
    input_data = pd.DataFrame({
        "sepal length (cm)": [data.sepal_length],
        "sepal width (cm)": [data.sepal_width],
        "petal length (cm)": [data.petal_length], 
        "petal width (cm)": [data.petal_width] 
    })

    prediction = model.predict(input_data)

    return{
        "predicted": str(prediction[0])
    }

