import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import streamlit as st
import tensorflow_hub as hub
import tf_keras
from PIL import Image

IMG_SIZE = 224

@st.cache_resource
def load():
    model =tf_keras.models.load_model(
    "dog_vision_model.h5",
    custom_objects = {"KerasLayer":hub.KerasLayer)
    
    with open("labels.txt") as f:
        labels = [l.strip() for l in f if l.strip()]
    return model,labels

model, labels = load()
st.title("Dog Vision")
st.write("Upload a dog photo and the model predicts its breed(120 breeds).")
file = st.file_uploader("Upload a dog photo", type=["jpg","jpeg", "png"])
if file:
    img = Image.open(file).convert("RGB")
    st.image(img)
    arr = np.array(img.resize((IMG_SIZE, IMG_SIZE)), dtype = np.float32)/255.0
    probs = model.predict(np.expand_dims(arr, axis =0))[0]
    for i in np.argsort(probs)[::-1][:5]:
        st.write(f"{labels[i].replace("_", " ")}: {probs[i]:.1%}")
        st.progress(float(probs[i]))








