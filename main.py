# main.py
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from opentelemetry.trace.status import Status, StatusCode
from pydantic import BaseModel

import model
import otel_setup as otel


class EmbedRequest(BaseModel):
    text: str | list[str] = ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    model.load()          # blocking, before the port opens
    app.state.ready = True
    yield                 # <-- serving happens here
    app.state.ready = False
    otel.flush()          # once, on SIGTERM


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        o for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o
    ],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


@app.exception_handler(RequestValidationError)
async def bad_request(request: Request, exc: RequestValidationError):
    """Preserve the Lambda contract: malformed input is a 400, and it still
    increments the bad_request counter the Grafana dashboard charts."""
    otel.request_counter.add(1, {"outcome": "bad_request"})
    return JSONResponse(status_code=400, content={"error": "invalid request body"})


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    if not getattr(app.state, "ready", False):
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ready", "providers": model.providers()}


@app.post("/embed")
def embed(req: EmbedRequest):
    start = time.perf_counter()
    with otel.tracer.start_as_current_span("embed.request") as span:
        try:
            n = 1 if isinstance(req.text, str) else len(req.text)
            otel.batch_hist.record(n)
            span.set_attribute("batch.size", n)

            embedding = model.encode(req.text)

            otel.request_counter.add(1, {"outcome": "ok"})
            return {"embedding": embedding.tolist()}
        except Exception as e:
            otel.request_counter.add(1, {"outcome": "error"})
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
        finally:
            otel.duration_hist.record(time.perf_counter() - start)