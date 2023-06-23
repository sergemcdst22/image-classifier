import os

from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from googletrans import Translator

from classifier import get_all_models_dict, Classifier



app = FastAPI()
translator = Translator(service_urls=['translate.googleapis.com'])
templates = Jinja2Templates(directory="templates")
all_models = get_all_models_dict()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "models": all_models})


@app.post("/upload")
async def upload_file(file: UploadFile, model_name: str):
    contents = await file.read()
    with open(file.filename, "wb") as f:
        f.write(contents)

    try:
        model, weights = all_models[model_name]
        classifier = Classifier(model, weights)
        result = {}
        result['en'] = classifier.classify(file.filename)
        result['ru'] = translator.translate(result['en'], dest='ru', src='en').text
    except Exception as e:
        return {"result": "", "err": str(e)}
    finally:
        os.remove(file.filename)

    return {"result": result, "err": ""}
