from fastapi import FastAPI,UploadFile,File
import uvicorn
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
#
# app=FastAPI()
#
# MODEL=tf.keras.models.load_model("../Model/potatoes.h5")
# CLASS_NAMES=["Early Blight","Late Blight","Healthy"]
# @app.get("/ping")
# async def ping():
#     return "Hello, i am alive"
# def read_file_as_image(data) -> np.ndarray:
#     image=np.array(Image.open(BytesIO(data)))
#     return image
#
# @app.post("/predict")
# async def predict(
#         file:UploadFile = File(...)
# ):
#     image=read_file_as_image(await file.read())
#     img_batch=np.expand_dims(image,0)
#     predictions=MODEL.predict(image,0)
#
#     pass
# if __name__=="__main__":
#     uvicorn.run(app,host='localhost',port=8000)

#
# from fastapi import FastAPI, UploadFile, File
# import uvicorn
# from io import BytesIO
# from PIL import Image
# import numpy as np
# import tensorflow as tf
#
# app = FastAPI()
#
# origins = [
#     "http://localhost",
#     "http://localhost:3000",
# ]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
#
# # Load trained model
# MODEL = tf.keras.models.load_model("../Model/potatoes.h5")
# CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
#
# @app.get("/ping")
# async def ping():
#     return {"message": "Hello, I am alive"}
#
# # Preprocess image
# def read_file_as_image(data) -> np.ndarray:
#     image = Image.open(BytesIO(data)).convert("RGB")
#     image = image.resize((256, 256))  # Ensure shape matches model input
#     image = np.array(image)
#     image = image / 255.0  # Normalize pixel values
#     return image
#
# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     image = read_file_as_image(await file.read())
#     img_batch = np.expand_dims(image, 0)  # Shape: (1, 256, 256, 3)
#
#     predictions = MODEL.predict(img_batch)
#     predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
#     confidence = float(np.max(predictions[0]))
#
#     return {
#         "class": predicted_class,
#         "confidence": confidence
#     }
#
# if __name__ == "__main__":
#     uvicorn.run(app, host='localhost', port=8000)

#
# from fastapi import FastAPI, File, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# import uvicorn
# import numpy as np
# from io import BytesIO
# from PIL import Image
# import tensorflow as tf
#
# app = FastAPI()
#
# origins = [
#     "http://localhost",
#     "http://localhost:3000",
# ]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
#
# MODEL = tf.keras.models.load_model("../Model/final_model.keras")
#
# CLASS_NAMES = ["i", "ii", "iii"]
#
#
# @app.get("/ping")
# async def ping():
#     return "Hello, I am alive"
#
#
# def read_file_as_image(data) -> np.ndarray:
#     image = np.array(Image.open(BytesIO(data)))
#     return image
#
#
# @app.post("/predict")
# async def predict(
#         file: UploadFile = File(...)
# ):
#     image = read_file_as_image(await file.read())
#     img_batch = np.expand_dims(image, 0)
#
#     predictions = MODEL.predict(img_batch)
#
#     predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
#     confidence = np.max(predictions[0])
#     return {
#         'class': predicted_class,
#         'confidence': float(confidence)
#     }
#
#
# if __name__ == "__main__":
#     uvicorn.run(app, host='localhost', port=8000)
#
#



from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load the trained model (ensure correct path)
MODEL = tf.keras.models.load_model("../Model/final_model.keras")

# ✅ List of class names in correct order
CLASS_NAMES = [
  "Class I: Fine lower bainite with elongated M-A islands",
  "Class II: Inhomogeneous mix of large and small M-A islands",
  "Class III: Coarse bainite with largest M-A islands visible"
];


# ✅ Health check route
@app.get("/ping")
async def ping():
    return "Hello, I am alive"

# ✅ Image preprocessing function
def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")     # ensure 3 channels
    image = image.resize((224, 224))                     # resize to model input shape
    image_array = np.array(image) / 255.0                # normalize to [0,1]
    return image_array

# ✅ Inference route
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)                 # shape: (1, 224, 224, 3)

    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))

    return {
        'class': predicted_class,
        'confidence': confidence
    }

# ✅ Main entry point
if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
