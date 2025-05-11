import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import spacy
import pytesseract
from PIL import Image
import json
import pandas as pd
import fitz
from pdf2image import convert_from_path
from transformers import AutoProcessor

nlp = None
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

def init_nlp():
    global nlp
    nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    tokens = word_tokenize(text)
    return " ".join(stemmer.stem(word) for word in tokens if word not in stop_words)

def extract_named_entities(text):
    return [(ent.text, ent.label_) for ent in nlp(text).ents]

def extract_text(file_path, suffix):
    if suffix == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    elif suffix == '.pdf':
        text = "".join([page.get_text() for page in fitz.open(file_path)])
        if not text.strip():
            images = convert_from_path(file_path)
            return "".join([pytesseract.image_to_string(img) for img in images])
        return text
    elif suffix == '.csv':
        return pd.read_csv(file_path).to_string()
    elif suffix == '.xlsx':
        return pd.read_excel(file_path).to_string()
    elif suffix == '.json':
        with open(file_path, 'r') as f:
            return json.dumps(json.load(f))
    elif suffix in ['.jpg', '.png']:
        return pytesseract.image_to_string(Image.open(file_path))
    else:
        raise ValueError("Unsupported file type")

def prepare_layoutlm_input(image_path):
    image = Image.open(image_path).convert("RGB")
    words, boxes = [], []
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 60:
            word = data['text'][i]
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            words.append(word)
            boxes.append([x, y, x + w, y + h])
    return image, words, boxes

def infer_with_layoutlmv3(image, words, boxes):
    from llm import layout_model
    encoding = processor(image, words, boxes=boxes, return_tensors="pt")
    outputs = layout_model(**encoding)
    logits = outputs.logits
    predictions = logits.argmax(dim=-1)
    labels = [layout_model.config.id2label[p.item()] for p in predictions[0]]
    return list(zip(encoding.tokens(), labels))
