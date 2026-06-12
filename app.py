import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

model = load_model("mnist_cnn_model.keras")

st.title("Handwritten Digit Recognition")

uploaded_file = st.file_uploader(
    "Upload Digit Image",
    type=["png","jpg","jpeg"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    image = image.convert("L")
    image = image.resize((28,28))

    img = np.array(image)

    img = img/255.0
    img = img.reshape(1,28,28,1)

    pred = model.predict(img)

    digit = np.argmax(pred)

    st.image(image)
    st.success(f"Predicted Digit : {digit}")
