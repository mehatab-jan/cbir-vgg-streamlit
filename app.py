import streamlit as st
import os
from vgg_features import extract_features, load_model
from utils import get_top_k_similar, load_image_paths, show_images
import numpy as np
import cv2

st.title("🔍 CBIR with VGG16 + Streamlit")

uploaded_file = st.file_uploader("Upload a query image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Save uploaded image
    with open("query.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.image("query.jpg", caption="Uploaded Query Image", width=300)
    
    model = load_model()
    
    st.write("🔄 Extracting features...")
    query_feat = extract_features("query.jpg", model)
    
    st.write("📂 Loading dataset features...")
    image_paths = load_image_paths("images")  # Point to your image dataset
    dataset_features = [extract_features(p, model) for p in image_paths]
    
    st.write("🔎 Searching for similar images...")
    top_k = get_top_k_similar(query_feat, dataset_features, image_paths, k=5)
    
    st.write("🎯 Top 5 Most Similar Images:")
    show_images(top_k)

