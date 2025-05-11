

---

# 🧠 GPT-LLM Query System

A modular, async-enabled FastAPI application for processing and querying documents using both LLMs and LayoutLMv3 for structured data extraction. Designed for scalability, monitoring, and deployment with Docker.

---

## 🚀 Features

* **Asynchronous File Processing** with `asyncio.to_thread`
* **Named Entity Recognition** using spaCy & LayoutLMv3
* **Multi-format File Support**: `.txt`, `.pdf`, `.csv`, `.json`, `.jpg`, `.png`, etc.
* **Prometheus Monitoring** & **Loki Logging**
* **Background File Cleanup Tasks**
* **LLM Integration** via `llama-cpp`
* **Containerized** with Docker

---

## 📁 Project Structure

```
gpt_llm_query_system/
├── main.py             # FastAPI entrypoint
├── utils.py            # OCR, NER, preprocessing utilities
├── services.py         # Async processing logic
├── llm.py              # LLM & LayoutLMv3 loading and inference
├── Dockerfile          # Docker container definition
├── requirements.txt    # Python dependencies
```

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/gpt-llm-query-system.git
cd gpt-llm-query-system
```

### 2. Install Dependencies

Make sure to have Python 3.10+ and `poppler`, `tesseract-ocr`, etc., installed.

```bash
pip install -r requirements.txt
python -m nltk.downloader punkt stopwords
python -m spacy download en_core_web_sm
```

### 3. Run Locally

```bash
uvicorn main:app --reload
```

Visit `http://localhost:8000`

---

## 🐳 Docker Deployment

Build and run in a container:

```bash
docker build -t gpt-llm-app .
docker run -p 8000:8000 gpt-llm-app
```

---

## 🔌 API Endpoints

### `POST /chat/`

Upload a file and ask a question.

**Form fields:**

* `query` (str): Your question
* `file` (UploadFile): Optional file for context

**Response:**

```json
{
  "answer": "Generated response from the LLM."
}
```

### `GET /`

Health check/root endpoint.

---

## 📊 Monitoring & Logging

* **Prometheus Metrics** at `/metrics` (auto-instrumented)
* **Loki Logging** via `logging_loki.LokiQueueHandler` (configure URL in `main.py`)

---

## 📌 Model Initialization

* Uses `llama-cpp-python` to load a local GGUF model (`tinyllama.gguf`)
* Uses `transformers` to load `microsoft/layoutlmv3-base` for layout-aware NER

Ensure the model paths are correctly set in `llm.py`.

---

## 🧹 Background Tasks

Files are processed and deleted asynchronously using FastAPI `BackgroundTasks`.

---

## 🛡️ Notes

* CPU-heavy operations are offloaded using `asyncio.to_thread` to avoid blocking the event loop.
* `extract_text()` uses multiple strategies including `pdf2image` and `pytesseract` fallback for scanned PDFs.
* LayoutLM inputs are constructed using `pytesseract.image_to_data`.

---

## 📬 Contact / Contribute

Feel free to open issues or PRs for improvements. For deployment help (e.g., AWS/GCP), monitoring dashboards, or CI/CD setup, [reach out](mailto:rishiitsme1245790@gmail.com).

---


