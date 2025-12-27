import os
import sys
from contextlib import asynccontextmanager
from urllib.parse import quote_plus as urlquote

import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from source_main.exception.exception import BankException
from source_main.logging.logging import logging
from source_main.utlis.main_utlis.utlis import load_object
from source_main.utlis.model.estimator import BankModel

from cloud.azure_blob_syncer import AzureBlobSync


# =========================
# Helpers
# =========================
def is_local() -> bool:
    return os.getenv("ENV", "local") == "local"


# =========================
# DB config (optional)
# =========================
MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")

engine = None
SessionLocal = None

if MYSQL_HOST and MYSQL_USER and MYSQL_PASSWORD and MYSQL_DATABASE:
    safe_user = urlquote(MYSQL_USER)
    safe_pass = urlquote(MYSQL_PASSWORD)

    DATABASE_URL = (
        f"mysql+mysqlconnector://{safe_user}:{safe_pass}@{MYSQL_HOST}/{MYSQL_DATABASE}"
    )

    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_recycle=1800,
        future=True,
    )

    SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
        future=True,
    )


# =========================
# Azure-safe model path
# =========================
MODEL_DIR = "/home/site/wwwroot/models"
os.makedirs(MODEL_DIR, exist_ok=True)

model = None
preprocessor = None


# =========================
# Lifespan (startup / shutdown)
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, preprocessor
    try:
        if is_local():
            logging.info("Running locally → skipping Azure Blob sync")
        else:
            blob_sync = AzureBlobSync()
            blob_sync.sync_folder_from_blob(
                local_folder=MODEL_DIR,
                blob_prefix="final_model"
            )

        model_path = os.path.join(MODEL_DIR, "model.pkl")
        preprocessor_path = os.path.join(MODEL_DIR, "preprocessor.pkl")

        if os.path.exists(model_path) and os.path.exists(preprocessor_path):
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            logging.info("Model loaded successfully")
        else:
            logging.warning("Model files not found (expected in local mode)")

        yield

    except Exception as e:
        raise BankException(e, sys)


# =========================
# FastAPI app (MUST be here)
# =========================
app = FastAPI(
    title="Bank Term Deposit Prediction API",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Optional DB write
# =========================
def write_df(df: pd.DataFrame, table_name: str) -> None:
    if engine is None:
        return
    try:
        with engine.connect() as connection:
            df.to_sql(
                name=table_name,
                con=connection,
                if_exists="replace",
                index=False,
            )
    except Exception as e:
        raise BankException(e, sys)


# =========================
# Routes
# =========================
@app.get("/")
async def health():
    return {"status": "FastAPI running locally / Azure-ready"}


@app.post("/predict", tags=["predict"])
async def predict_route(file: UploadFile = File(...)):
    try:
        if model is None or preprocessor is None:
            return Response(
                "Model not loaded (expected in local mode without model files)",
                status_code=503
            )

        df = pd.read_csv(file.file)

        bank_model = BankModel(
            preprocessor=preprocessor,
            model=model,
        )

        preds = bank_model.predict(df)
        df["predicted_column"] = ["yes" if int(i) == 1 else "no" for i in preds]

        try:
            write_df(df, "predictions")
        except Exception:
            logging.warning("DB write skipped")

        return {"rows": len(df), "message": "Prediction completed"}

    except Exception as e:
        raise BankException(e, sys)


# =========================
# Local run
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
