import streamlit as st
import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

# Load model
base_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)

# Load features
features = np.load("features/deep_features.npy")
filenames = np.load("features/filenames.npy", allow_pickle=True)

def extract_deep_features_img(img_array):
    img = cv2.resize(img_array, (224, 224))
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    return model.predict(x).flatten()

def get_similar_images(query_img, top_k=5):
    query_feat = extract_deep_features_img(query_img)
    sims = cosine_similarity([query_feat], features)[0]
    indices = np.argsort(sims)[::-1][:top_k]
    return indices

# Streamlit UI
st.title("🔍 CBIR System (VGG16 Deep Features)")
uploaded_file = st.file_uploader("Upload a query image", type=["jpg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    query_img = cv2.imdecode(file_bytes, 1)
    query_img_rgb = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)

    st.image(query_img_rgb, caption="Query Image", use_column_width=True)

    st.markdown("---")
    st.subheader("Top-k Retrieved Images")

    indices = get_similar_images(query_img, top_k=5)
    for i, idx in enumerate(indices):
        img = cv2.imread(filenames[idx])
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"Rank {i+1}", use_column_width=True)
