#import FastAPI
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Form
from pydantic import BaseModel

class Item(BaseModel):
    id: str

#import NLP model
import requests, json
import numpy as np
import pandas as pd

dscr = pd.read_csv("../data/desc10d.csv", sep=';', names=['id', 'label', 'till'], dtype={'id': str, 'label': str, 'till': str})
dscr['valid'] = dscr['till'].isna()
#addr = 'https://api.tnved.info/api/Search/Search'


def get_sample(qtty = 5):
    result = list()
    for i, each in enumerate(dscr.sample(n = qtty).to_numpy()):
        #print(each[0])
        #valid = 1 if dscr[dscr['id']==each].iloc[0]['valid']==True else 0 #result.append([each, dscr[dscr['id']==each].iloc[0]['label'], round(probs[each], 3)])
        #result.append([each, dscr[dscr['id']==each].iloc[0]['label'], round(probs[each], 3), dscr[dscr['id']==each].iloc[0]['till']])
        valid = 1 if dscr[dscr['id']==each[0]].iloc[0]['valid']== True else 0 #result.append([each, dscr[dscr['id']==each].iloc[0]['label'], round(probs[each], 3)])
        result.append([each[0], each[1], i, valid])
    return result
#get_sample()

#predict_prob func with description
def predict_prob_with_descr(text, qtty=5):
    probs = predict_prob(text, qtty=qtty)
    #result = np.array()
    result = list()
    for each in probs:
        #valid = 1 if dscr[dscr['id']==each].iloc[0]['valid']==True else 0 #result.append([each, dscr[dscr['id']==each].iloc[0]['label'], round(probs[each], 3)])
        result.append([each, dscr[dscr['id']==each].iloc[0]['label'], round(probs[each], 3), dscr[dscr['id']==each].iloc[0]['till']])
    return result

#train data
df = pd.read_csv("../data/mergedcleared0407.csv", sep=';', names=['id', 'label'], dtype={'id': str, 'label': str})

#Function return training data for exact code or group
def whats_data(id, pandas=False):
    if pandas == True:
        return df[df.id.str.slice(start=0, stop=len(str(id)))==str(id)]
    else:
        return df[df.id.str.slice(start=0, stop=len(str(id)))==str(id)].to_numpy().tolist()
	
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.post("/sendrq/")
async def sendrq(qtty: str = Form(), desc: str = Form()):
    print(desc, qtty)
    response = get_sample(int(qtty))
    #j_response = ''
    #for each in response:
        #j_response = j_response + '["id": "' + each[0] + '", "label": "' + each[1] + '", "prob": ' + str(each[2]) + ', "valid": ' + str(each[3]) + '],'
    #j_response = j_response + '}'
    print(response)
    return {'data': response}


@app.get("/", response_class=HTMLResponse)
async def index_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
	
@app.post("/code/")
async def code(item: Item):
    return {'data': parse(item.id)}

@app.post("/whatdata/")
async def whatdata(item: Item):
	return {'data': whats_data(item.id)}
	
