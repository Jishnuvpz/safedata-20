# backend/app/routers/anonymize.py
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from app.utils.data_processing import load_dataset, anonymize_data


router = APIRouter()

# Load dataset once
df = load_dataset("data/sample_data.csv")

@router.get("/anonymize")
def anonymize_endpoint(columns: list[str] = ["Name", "Email"]):
    """
    Anonymizes specified columns in the dataset.
    By default, anonymizes "Name" and "Email".
    """
    anonymized_df = anonymize_data(df, columns_to_anonymize=columns)
    return JSONResponse(content=anonymized_df.to_dict(orient="records"))
