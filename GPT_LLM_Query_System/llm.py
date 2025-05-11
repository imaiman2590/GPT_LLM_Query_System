from llama_cpp import Llama
from transformers import AutoModelForTokenClassification
from utils import init_nlp

llm_model = None
layout_model = None

def init_models():
    global llm_model, layout_model
    llm_model = Llama(model_path="path/to/tinyllama.gguf", n_ctx=2048)
    layout_model = AutoModelForTokenClassification.from_pretrained("microsoft/layoutlmv3-base")
    init_nlp()

def generate_response(prompt: str) -> str:
    return llm_model(prompt)["choices"][0]["text"]
