from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
import os
from pathlib import Path

from src import forecast_demand_workers

WORKING_DIR = Path.cwd()
STATIC_DIR = WORKING_DIR.joinpath("static")

app = FastAPI(title="Excel File Processor")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_upload_page():
    with open("static/index.html", "r") as file:
        return file.read()

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        if not (file.filename.endswith('.xlsx') or file.filename.endswith('.xls')):
            raise HTTPException(status_code=400, detail="Please upload an Excel file (.xlsx or .xls)")
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents))
        print(df.shape)
        try:
            result_df, insample_df, future_df = forecast_demand_workers(df)
            dates = result_df["Month"].tolist()
            workers = result_df["Workers Required"].tolist()
            demand = result_df["Forecasted Demand"].tolist()
            insample = {
                "dates": insample_df["Month"].tolist(),
                "Actual": insample_df["Actual"].tolist(),
                "SARIMA": insample_df["SARIMA"].tolist(),
                "Prophet": insample_df["Prophet"].tolist(),
                "Holt_Winters": insample_df["Holt-Winters"].tolist(),
                "Combined": insample_df["Combined"].tolist()
            }
            future = {
                "dates": future_df["Month"].tolist(),
                "SARIMA": future_df["SARIMA"].tolist(),
                "Prophet": future_df["Prophet"].tolist(),
                "Holt_Winters": future_df["Holt-Winters"].tolist(),
                "Combined": future_df["Combined"].tolist()
            }
            
            return {"dates":dates, "workers": workers, "demand": demand, "insample": insample, "future": future}
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing Excel data: {str(e)}")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)