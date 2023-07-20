#import FastAPI
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Form
import json

#import NLP model
import torch
import numpy as np
import pandas as pd
device = torch.device("cpu")
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AdamW, get_scheduler
model_name = "sberbank-ai/sbert_large_nlu_ru"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2648).to(device)
from sklearn import preprocessing
Label_encoder = preprocessing.LabelEncoder()
Label_encoder.classes_ = np.load('./cl_classes1307.npy', allow_pickle=True)
model.load_state_dict(torch.load("../../best_model2306/pytorch_model10d.bin", map_location=device))
dscr = pd.read_csv("../data/desc.csv", sep=';', names=['id', 'label'], dtype={'id': str, 'label': str})

#prediction func for one most likely class (argmax)
def predict_class(text, desc=True):
    model.to(torch.device('cpu'))
    inputs = tokenizer(text, truncation = True, max_length=100, padding='max_length', return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_class_id = logits.argmax().item()
        if desc==True:
            result = dict()
            id = Label_encoder.inverse_transform([predicted_class_id])[0]
            result[id] = dscr[dscr['id']==id[:4]].iloc[0]['label']
            return result
        else:
            return Label_encoder.inverse_transform([predicted_class_id])[0]

#prediction func for bunch of most likely classes
def predict_prob(text, qtty=5):
    model.to(torch.device('cpu'))
    inputs = tokenizer(text, truncation = True, max_length=100, padding='max_length', return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    result = dict()
    p = torch.nn.functional.softmax(logits, dim=1)
    for i in range(qtty):
        a = p.argmax().item()
        result[Label_encoder.inverse_transform([a])[0]] = p[0][a].item()
        p[0][a] = 0
    return result

#predict_prob func with description
def predict_prob_with_descr(text, qtty=5):
    probs = predict_prob(text, qtty=qtty)
    #result = np.array()
    result = list()
    for each in probs:
        result.append([each, dscr[dscr['id']==each[:4]].iloc[0]['label'], probs[each]])
    return result
	
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.post("/sendrq/")
async def sendrq(qtty: str = Form(), desc: str = Form()):
    print(desc, qtty)
    response = predict_prob_with_descr(desc, int(qtty))
    print(response)
    return {"data": response}


@app.get("/", response_class=HTMLResponse)
async def index_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
	
