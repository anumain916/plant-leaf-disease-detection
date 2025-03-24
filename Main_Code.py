import os
import streamlit as st
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import CNN  # Your CNN model

# Load model
model = CNN.CNN(39)
model.load_state_dict(torch.load(r"C:\Users\VICTUS\Plant-Disease-Detection\Flask Deployed App\plant_disease_model_1_latest.pt"))
model.eval()

# Load CSV files
disease_info = pd.read_csv(r'C:\Users\VICTUS\Plant-Disease-Detection\Flask Deployed App\disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv(r'C:\Users\VICTUS\Plant-Disease-Detection\Flask Deployed App\supplement_info.csv', encoding='cp1252')

# Function for prediction
def prediction(image):
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index

# Streamlit UI
st.title("🌿 Plant Disease Detection AI")
st.write("Upload an image or use your camera to detect plant diseases.")

# Image upload OR Camera input
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
camera_image = st.camera_input("Take a picture")

# Process the image
image = None
if uploaded_image:
    image = Image.open(uploaded_image)
elif camera_image:
    image = Image.open(camera_image)

if image:
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Predict
    with st.spinner("Analyzing Image..."):
        pred = prediction(image)
        
        # Fetch details
        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]

    # Show results
    st.success(f"**Detected Disease:** {title}")
    st.write(f"📝 **Description:** {description}")
    st.write(f"🛑 **Prevention Steps:** {prevent}")

    # Supplement info
    st.subheader("Recommended Supplement")
    st.write(f"**{supplement_name}**")
    st.image(supplement_image_url, caption="Supplement", use_column_width=True)
    st.markdown(f"[🛒 Buy Here]({supplement_buy_link})", unsafe_allow_html=True)
