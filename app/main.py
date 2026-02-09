from fastapi import FastAPI
from dotenv import load_dotenv

load_dotenv()

from app.routes.cashflow import router as cashflow_router
from app.routes.qa import router as qa_router

app = FastAPI(title="Cashflow Forecast Service")

app.include_router(cashflow_router, prefix="/cashflow", tags=["cashflow"])
app.include_router(qa_router, prefix="/cashflow", tags=["cashflow-qa"])
