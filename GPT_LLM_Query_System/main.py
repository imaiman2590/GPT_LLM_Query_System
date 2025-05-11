from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from logging_loki import LokiQueueHandler
from queue import Queue
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter
import tempfile
import logging
import asyncio
import os

from utils import preprocess_text, extract_named_entities, extract_text, prepare_layoutlm_input, infer_with_layoutlmv3
from services import async_process_file, process_file
from llm import generate_response, init_models

app = FastAPI()
instrumentator = Instrumentator().instrument(app)

# Logging setup
handler = LokiQueueHandler(Queue(-1), url="http://localhost:3100/loki/api/v1/push", version="1")
handler.setLevel(logging.INFO)
logger = logging.getLogger("fastapi_app")
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# Prometheus request count
REQUEST_COUNT = Counter('app_request_count', 'Application Request Count', ['method', 'endpoint', 'http_status'])

@app.on_event("startup")
async def startup_event():
    instrumentator.expose(app)
    init_models()

@app.post("/chat/")
async def chat_with_file(query: str = Form(...), file: UploadFile = File(None), background_tasks: BackgroundTasks = None):
    try:
        context = ""
        if file:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(await file.read())
                tmp_path = tmp.name
            processed_text, entities = await async_process_file(tmp_path)

            if file.filename.lower().endswith(('.pdf', '.jpg', '.png')):
                image, words, boxes = prepare_layoutlm_input(tmp_path)
                layout_output = infer_with_layoutlmv3(image, words, boxes)
                layout_entities = [f"{word}:{label}" for word, label in layout_output if label != "O"]
                context += f"\n\nLayoutLM Entities:\n{layout_entities}"

            context += f"\n\nSummary:\n{processed_text[:300]}...\nEntities: {entities}"
            background_tasks.add_task(process_file, tmp_path)

        prompt = f"{context}\n\nQuestion: {query}" if context else f"Question: {query}"
        answer = await asyncio.to_thread(generate_response, prompt)
        return {"answer": answer}

    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
def root():
    return {"message": "Enhanced LLM Chat API with async concurrency is live!"}
