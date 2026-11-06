"""FastAPI application entrypoint."""

import os
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from api.routers import dashboard, predict
from src.utils import get_logger

log = get_logger("api")
app = FastAPI(
    title="Vantara Customer Intelligence API",
    version="1.0.0",
    description="Churn risk, customer lifetime value, segments and explanations for every customer.",
)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "*").split(","), allow_methods=["*"], allow_headers=["*"]
)
app.include_router(predict.router)
app.include_router(dashboard.router)


@app.middleware("http")
async def access_log(request: Request, call_next):
    """Structured access log with latency for every request."""
    start = time.perf_counter()
    response = await call_next(request)
    log.info(
        "request",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "ms": round((time.perf_counter() - start) * 1000, 1),
        },
    )
    return response
