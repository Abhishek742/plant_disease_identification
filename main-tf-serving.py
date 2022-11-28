from fastapi import FastAPI,File,UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import requests
import tensorflow as tf
from background.u2net_test import *
app = FastAPI()
species_diseases = {
    "apple" : ['Apple_Black_rot', 'Apple_Cedar_Apple_Rust', 'Apple_Healthy', 'Apple_Scab'],
    "potato" : ['Potato_Early_blight', 'Potato_Late_blight', 'Potato_healthy'],
    "corn" : ['Corn_Cercospora_leaf_spot','Corn_Common_rust','Corn_Healthy','Corn_Northern_Leaf_Blight'],
    "tomato" : ['Tomato_Bacterial_spot','Tomato_Early_blight','Tomato_Late_blight','Tomato_Leaf_Mold','Tomato_Septoria_leaf_spot',
                'Tomato_Spider_mites','Tomato_Target_Spot','Tomato_Tomato_Yellow_Leaf_Curl_Virus','Tomato_Tomato_mosaic_virus','Tomato_healthy']
}
disease_info = {
    "Apple__Apple_scab":{
        "supplement":{
        "name": "Katyayani Prozol Propiconazole Systematic Fungicide",
        "image_url": "https://encrypted-tbn1.gstatic.com/shopping?q=tbn:ANd9GcRfq9MLrPL9tFkuFbGb98fMGDdl67v4I2iDLYCVprdsdGaXURCl9UNEr8v_65X1hKrYF5NjSvB01HOGexg-3CJxjkVSu9zPNJ2AunP09vPa0gjEILskTILx&usqp=CAE",
        "buy_link": "https://agribegri.com/products/buy-propiconazole--25-ec-systematic-fungicide-online-.php",
        },
        "disease_brief":"Apple scab is the most common disease of apple and crabapple trees in Minnesota.\nScab is caused by a fungus that infects both leaves and fruit.\nScabby fruit are often unfit for eating.\nInfected leaves have olive green to brown spots.\nLeaves with many leaf spots turn yellow and fall off early.\nLeaf loss weakens the tree when it occurs many years in a row.\nPlanting disease resistant varieties is the best way to manage scab."
    },
    "Apple__Black_rot" :{ 
        "supplement" :{
        "name": "Magic FungiX For Fungal disease",
        "image_url": "https://encrypted-tbn3.gstatic.com/shopping?q=tbn:ANd9GcTZZH2SUe7Hufpd49iFoF_04c96J-fZeywsYQXDb0gFerGOYyL7xPBxLN05LIx6s36u6C_qMvtescsDbrTEzniJp0yhfEsvJoTCMD7FDnI&usqp=CAE",
        "buy_link": "https://agribegri.com/products/buy-fungicide-online-india--organic-fungicide--yield-enhancer.php"
        },
        "disease_brief":"Leaf symptoms first occur early in the spring when the leaves are unfolding.\nThey appear as small, purple specks on the upper surface of the leaves that enlarge into circular lesions 1/8 to 1/4 inch (3-6 mm) in diameter.\nThe margin of the lesions remains purple, while the center turns tan to brown. In a few weeks, secondary enlargement of these leaf spots occurs.\nHeavily infected leaves become chlorotic and defoliation occurs.\nAs the rotted area enlarges, a series of concentric bands of uniform width form which alternate in color from black to brown. The flesh of the rotted area remains firm and leathery. Black pycnidia are often seen on the surface of the infected fruit."
    },
    "Apple__Cedar_apple_rust" : {
        "supplement":{
        "name": "Katyayani All in 1 Organic Fungicide",
        "image_url": "https://encrypted-tbn2.gstatic.com/shopping?q=tbn:ANd9GcT2JQ-fAdtrzzmkSespqEpKwop3BnWntsLioVSgjy79mpxQVPSqoD4v9yfL0mtiFJvFnPqeE7EcefadhdEpc9uUTCZbROwOPsklL_XDMSLTpxOGvIcBMMFiBQ&usqp=CAE",
        "buy_link": "https://agribegri.com/products/buy-organic-fungicide-all-in-1-online--organic-fungicide-.php"
        },
        "disease_brief":"Cedar apple rust (Gymnosporangium juniperi-virginianae) is a fungal disease that requires juniper plants to complete its complicated two year life-cycle. Spores overwinter as a reddish-brown gall on young twigs of various juniper species. In early spring, during wet weather, these galls swell and bright orange masses of spores are blown by the wind where they infect susceptible apple and crab-apple trees. The spores that develop on these trees will only infect junipers the following year. From year to year, the disease must pass from junipers to apples to junipers again; it cannot spread between apple trees."
    },
    "Apple__healthy" : {
        "supplement":{
        "name": "Tapti Booster Organic Fertilizer",
        "image_url": "https://rukminim1.flixcart.com/image/416/416/kc6jyq80/soil-manure/6/y/v/500-tapti-booster-500-ml-green-yantra-original-imaftd6rrgfhvshc.jpeg?q=70",
        "buy_link": "https://agribegri.com/products/tapti-booster-500-ml--organic-fertilizer-online-in-india.php"
        },
        "disease_brief":"As with most fruit, apples produce best when grown in full sun, which means six or more hours of direct summer Sun daily.The best exposure for apples is a north side of a house, tree line, or rise rather than the south.Apple trees need well-drained soil, but should be able to retain some moisture."
    }
}

def read_file_as_image(data) -> np.ndarray:
    img = Image.open(BytesIO(data))
    return [np.array(img),img]
def removeBg(image):
    image.save("/Users/abhishek-pt5840/Desktop/College/mini-project/code/plant-disease/background/src/image.jpeg")
    return removeBgColor()
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
def getSupplement(class_name):
    return disease_info[class_name]
@app.post("/predict/{species}")
async def predict(species : str,file: UploadFile = File(...)):   
    [image,imgFile] = read_file_as_image(await file.read()) 
    image = removeBg(imgFile)
    json_data = getJsonData(image)
    prediction = getPrediction(species,json_data)
    # prediction['supplement'] = getSupplement(prediction['class'])['supplement']
    # prediction['disease_brief'] = getSupplement(prediction['class'])['disease_brief']
    return prediction
    
if __name__ == '__main__':
    loadModel()
    uvicorn.run(app,host='localhost',port=8000)