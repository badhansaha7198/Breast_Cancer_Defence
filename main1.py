from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

bchp_model = tf.keras.models.load_model("../models/bc_hp_cnn")
bcmri_model = tf.keras.models.load_model("../models/bc_mri_cnn")

bchp_class_name = ["Benign", "Malignant"]
bcmri_class_name = ["Benign", "Malignant", "Normal"]


def read_file_as_hpimage(data) -> np.ndarray:
    hpimage = np.array(Image.open(BytesIO(data)))
    return hpimage

def read_file_as_mriimage(data) -> np.ndarray:
    mriimage = np.array(Image.open(BytesIO(data)))
    return mriimage


@app.post("/breast_cancer_hp")
async def breast_cancer_hp(hpfile: UploadFile = File(...)):
    hpimage = read_file_as_hpimage(await hpfile.read())
    hpimg_batch = np.expand_dims(hpimage, 0)
    hp_prediction = bchp_model.predict(hpimg_batch)
    hp_prediction_class = bchp_class_name[np.argmax(hp_prediction[0])]
    hp_confidence = np.max(hp_prediction[0])
    return {
        'hp_class': hp_prediction_class,
        'hp_confidence': float(hp_confidence)
    }



@app.post("/breast_cancer_mri")
async def breast_cancer_mri(mrifile: UploadFile = File(...)):
    mriimage = read_file_as_mriimage(await mrifile.read())
    mri_img_batch = np.expand_dims(mriimage, 0)
    mri_prediction = bcmri_model.predict(mri_img_batch)
    mri_prediction_class = bcmri_class_name[np.argmax(mri_prediction[0])]
    mri_confidence = np.max(mri_prediction[0])
    return {
        'mri_class': mri_prediction_class,
        'mri_confidence': float(mri_confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)