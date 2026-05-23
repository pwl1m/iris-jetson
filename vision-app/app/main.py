from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .settings import settings
from .vision_runtime import VisionRuntime

runtime = VisionRuntime(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    runtime.start()
    try:
        yield
    finally:
        runtime.stop()


app = FastAPI(title="Vision App", version="0.2.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return runtime.health()


@app.get("/subjects")
def subjects() -> dict:
    return runtime.subjects()


@app.post("/enroll")
async def enroll(subject: str = Form(...), file: UploadFile = File(...)) -> dict:
    try:
        return runtime.enroll(subject=subject, filename=file.filename, payload=await file.read())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/recognize")
async def recognize(file: UploadFile = File(...)) -> dict:
    try:
        return runtime.recognize_bytes(await file.read())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
