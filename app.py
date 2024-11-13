import base64
from io import BytesIO
import torch
import numpy as np
from PIL import Image
import streamlit as st

import inference

st.title("Image Colorizer with GAN")
st.markdown("### Colorize black and white images using a pre-trained model")
st.write("Made by Alikhan Nurkamal for the course ELCE455 Machine Learning with Python.")

# Radio buttons for output format and ResNet backbone
output_format = st.radio("Select output format:", ('rgb', 'ab'))
resnet_backbone = st.radio("Use ResNet backbone?", (False, True))

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
c1, c2, c3 = st.columns([1, 0.3, 1])
start_colorization = False

if uploaded_file:
    gray_image = Image.open(uploaded_file).resize((256, 256)).convert("L")
    
    # Convert image to a base64 string for HTML embedding
    buffered = BytesIO()
    gray_image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    
    c1.markdown(
        f"""
        <div style="text-align: center;">
            <h2>Input Image</h2>
            <img src="data:image/png;base64,{img_base64}" alt="Grayscale Image" style="display: block; margin: 0 auto;"/>
            <p style="font-size: 14px; margin-top: 10px;">Shape: {np.array(gray_image).shape}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    for i in range(10):
        c2.write("")
    start_colorization = c2.button("Colorize Image")

if start_colorization:
    with c3:
        with st.spinner("Colorizing..."):
            model = inference.prepare_model(output_format, resnet_backbone)
            gray_image = np.array(gray_image).astype(np.float32)
            gray_image = gray_image / 127.5 - 1.0  # Normalize the image to [-1, 1]
            gray_image = torch.from_numpy(gray_image).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
            colored = inference.colorize_image(model, gray_image, output_format)
            
            # Convert image to a base64 string for HTML embedding
            colored = Image.fromarray(colored)
            buffered = BytesIO()
            colored.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
    c3.markdown(
        f"""
        <div style="text-align: center;">
            <h2>Output Image</h2>
            <img src="data:image/png;base64,{img_base64}" alt="Colorized Image" style="display: block; margin: 0 auto;"/>
            <p style="font-size: 14px; margin-top: 10px;">Shape: {np.array(colored).shape}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
