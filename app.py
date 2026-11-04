import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

APP_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="Leaf Disease Classifier", layout="centered")
st.title("Leaf Disease Classifier")
st.write("Upload a clear photo of one leaf to get a model prediction.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(APP_DIR / "leaf_disease_final.keras", compile=False)

class_names = json.loads((APP_DIR / "class_names.json").read_text())

def display_name(name):
    return name.replace("___", " — ").replace("_", " ")

upload = st.file_uploader("Choose a leaf photo", type=["jpg", "jpeg", "png"])

if upload is not None:
    image = Image.open(upload).convert("RGB")
    st.image(image, caption=upload.name, use_container_width=True)

    if st.button("Classify leaf", type="primary"):
        with st.spinner("Loading model and analyzing the image..."):
            model = load_model()
            resized = image.resize((224, 224))
            batch = np.expand_dims(np.asarray(resized, dtype=np.float32), axis=0)
            scores = model.predict(batch, verbose=0)[0]

        order = np.argsort(scores)[::-1]
        best = int(order[0])
        confidence = float(scores[best])
        st.subheader(display_name(class_names[best]))
        st.metric("Model score", f"{confidence:.1%}")
        if confidence < 0.5:
            st.warning("The model is unsure. Try a clear, close photo of one leaf; this result is not a confirmed diagnosis.")
        st.caption("Other top predictions")
        for index in order[1:3]:
            st.write(f"{display_name(class_names[int(index)])}: {scores[index]:.1%}")
