from fastapi import FastAPI,File,UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import requests
from background.u2net_test import *
from resource_files.plant_info import *
import re

app = FastAPI()

def read_file_as_image(data) -> np.ndarray:
    img = Image.open(BytesIO(data))
    return [np.array(img),img]



def getJsonData(image):
    img_batch = np.expand_dims(image,0)
    return {
        "instances" : img_batch.tolist()
    }
    
def getPrediction(species,json_data):
    endpoint = "http://localhost:8100/v1/models/plant_models_" + species + "/versions/1:predict"
    response = requests.post(endpoint,json=json_data)

    prediction = np.array(response.json()['predictions'][0])
    CLASS_NAMES = species_diseases[species]
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)
    return {
        'class' : predicted_class,
        'confidence' : float(confidence)
    }


@app.post("/predict/{species}")
async def predict(species : str,file: UploadFile = File(...)):
    [image,imgFile] = read_file_as_image(await file.read()) 
    image = removeBg(imgFile,species)
    json_data = getJsonData(image)
    prediction = getPrediction(species,json_data)
    
    if(re.search('([Hh])ealthy',prediction['class'])):
        prediction['plant_info'] = disease_info[prediction['class']]
        return prediction

    prediction['supplement'] = supplements[prediction['class']]
    prediction['disease_brief'] = disease_info[prediction['class']]
    return prediction
    
if __name__ == '__main__':
    loadModel()
    uvicorn.run(app,host='localhost',port=8000)