import os
import asyncio
from utils import extract_text, preprocess_text, extract_named_entities
import logging

logger = logging.getLogger("fastapi_app")

async def async_process_file(file_path: str):
    return await asyncio.to_thread(process_file_sync, file_path)

def process_file_sync(file_path: str):
    suffix = os.path.splitext(file_path)[-1].lower()
    text = extract_text(file_path, suffix)
    processed = preprocess_text(text)
    entities = extract_named_entities(processed)
    return processed, entities

def process_file(file_path: str):
    try:
        _, _ = process_file_sync(file_path)
        os.unlink(file_path)
    except Exception as e:
        logger.error(f"Background task failed for {file_path}: {str(e)}")
